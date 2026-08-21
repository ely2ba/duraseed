"""Authenticate reconciled actual lifetime spend before the dose launch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any

from duraseed.calibration_stage_a_terminal import existing_stage_a_terminal
from duraseed.capability_dose_budget import DOSE_CAP_USD
from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import (
    RunStatus,
    read_run_record,
    write_run_record,
)
from duraseed.runners import RunnerGateError


CORRECTION_RUN_ID = "acquisition-calibration-stage-a-correction-20260818T134223Z"
CORRECTION_SESSION_ID = "11c608b5-339f-5253-85be-d505ba751a36"
CORRECTION_PREFLIGHT_SHA256 = (
    "sha256:c28f9935a7fb5cd3d8d84d6a867eb1c790f81d95397579d2639db2364d4e4d61"
)
CORRECTION_TERMINAL_SHA256 = (
    "sha256:d2ae05880965a96f386e93465514abac5299796151651d1aaacb4d4f9c368bef"
)
CORRECTION_JOURNAL_SHA256 = (
    "sha256:d495ecbea0566745ef82792f3b88bd2f554de721e139d336b1b06ec8d5e83f30"
)
INTERRUPTED_DOSE_RUN_ID = "capability-dose-20260820T185147Z"
INTERRUPTED_DOSE_SESSION_ID = "6ece3007-254a-554a-9903-f0445290e17e"
INTERRUPTED_DOSE_PREFLIGHT_SHA256 = (
    "sha256:5b086de4e4bef0e1073ed6afc55d79c3ae960a6dc1fcc8b2bf4f6c44d58e58ac"
)
INTERRUPTED_DOSE_RUN_SHA256 = (
    "sha256:6bbcb28bffccf9242eba5423136e098a60b858868ebdb3e6c11d32d5c6157aa2"
)
INTERRUPTED_DOSE_JOURNAL_SHA256 = (
    "sha256:13a22bdca2c1a8d337b975c518f7739e7c9de742cc32416cb014c0c0cafbeb88"
)
PRE_ACQUISITION_CONSOLE_SPEND_USD = Decimal("66.19")
LIFETIME_CALIBRATION_CAP_USD = Decimal("300")
LIFETIME_SESSION_IDS = (
    "9e923e2f-e347-54b9-8787-111aa907dd39",
    "6e54fc35-bcfd-5985-960d-dd1e79e5e4ce",
    "2530f903-4064-5207-9fdc-82855e0e7419",
    "250a790b-5b89-5649-a120-e1702087a478",
    CORRECTION_SESSION_ID,
    INTERRUPTED_DOSE_SESSION_ID,
)


@dataclass(frozen=True, slots=True)
class CapabilityDoseBilling:
    actual_lifetime_spend_usd: float
    remaining_balance_usd: float
    console_evidence_sha256: str
    lineage: dict[str, Any]


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid capability-dose {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"capability-dose {label} is not an object")
    return value, raw


def _amount(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise RunnerGateError(f"{label} is not a finite amount") from error
    if not result.is_finite() or result < 0:
        raise RunnerGateError(f"{label} is not a nonnegative amount")
    return result


def _time(value: Any, label: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise RunnerGateError(f"{label} is not an aware timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise RunnerGateError(f"{label} is not an aware timestamp")
    return result.astimezone(UTC)


def load_capability_dose_billing(
    correction_root: Path,
    console_evidence_path: Path,
    *,
    project_id: str,
) -> CapabilityDoseBilling:
    """Require the authenticated console delta and exact local run lineages."""

    if correction_root.name != CORRECTION_RUN_ID:
        raise RunnerGateError("capability-dose billing uses the wrong terminal run")
    preflight_raw = (correction_root / "preflight.json").read_bytes()
    terminal_raw = (correction_root / "stage-a-terminal.json").read_bytes()
    journal_path = (
        correction_root
        / "stage-a-arms/complete-bounded-stage-a/attempt-0001/remote-call-state.json"
    )
    journal, journal_raw = _object(journal_path, "correction journal")
    run = read_run_record(correction_root)
    interrupted_root = correction_root.parent / INTERRUPTED_DOSE_RUN_ID
    interrupted_preflight = (interrupted_root / "preflight.json").read_bytes()
    interrupted_run_raw = (interrupted_root / "run.json").read_bytes()
    interrupted_run = read_run_record(interrupted_root)
    interrupted_journal, interrupted_journal_raw = _object(
        interrupted_root
        / "capability-dose-arms/b-s-capability-dose/attempt-0001/remote-call-state.json",
        "interrupted dose journal",
    )
    existing_stage_a_terminal(correction_root, CORRECTION_PREFLIGHT_SHA256)
    evidence, evidence_raw = _object(console_evidence_path, "console evidence")
    before = _amount(
        evidence.get("pre_acquisition_console_spend_usd"),
        "pre-acquisition console spend",
    )
    account_spend = _amount(evidence.get("account_spend_usd"), "account console spend")
    actual = _amount(evidence.get("actual_lifetime_spend_usd"), "actual lifetime spend")
    token_cost = _amount(evidence.get("token_cost_usd"), "token cost")
    storage_cost = _amount(evidence.get("storage_cost_usd"), "storage cost")
    remaining = _amount(evidence.get("remaining_balance_usd"), "balance")
    grant = _amount(evidence.get("grant_total_usd"), "grant total")
    observed = _time(evidence.get("observed_at_utc"), "console observation")
    sessions = evidence.get("lifetime_session_ids")
    checkpoints = evidence.get("session_checkpoint_counts")
    expected_checkpoints = {
        LIFETIME_SESSION_IDS[0]: 7,
        LIFETIME_SESSION_IDS[1]: 5,
        LIFETIME_SESSION_IDS[2]: 5,
        LIFETIME_SESSION_IDS[3]: 6,
        LIFETIME_SESSION_IDS[4]: 4,
        LIFETIME_SESSION_IDS[5]: 0,
    }
    interrupted_pending = interrupted_journal.get("pending")
    if (
        sha256_bytes(preflight_raw) != CORRECTION_PREFLIGHT_SHA256
        or sha256_bytes(terminal_raw) != CORRECTION_TERMINAL_SHA256
        or sha256_bytes(journal_raw) != CORRECTION_JOURNAL_SHA256
        or journal.get("pending") is not None
        or run.status is not RunStatus.FAILED
        or run.project_id != project_id
        or run.tinker_session_id != CORRECTION_SESSION_ID
        or run.finished_at is None
        or sha256_bytes(interrupted_preflight) != INTERRUPTED_DOSE_PREFLIGHT_SHA256
        or sha256_bytes(interrupted_run_raw) != INTERRUPTED_DOSE_RUN_SHA256
        or sha256_bytes(interrupted_journal_raw) != INTERRUPTED_DOSE_JOURNAL_SHA256
        or interrupted_run.status is not RunStatus.INTERRUPTED
        or interrupted_run.project_id != project_id
        or interrupted_run.tinker_session_id != INTERRUPTED_DOSE_SESSION_ID
        or interrupted_run.finished_at is None
        or interrupted_journal.get("completed_count") != 831
        or not isinstance(interrupted_pending, dict)
        or interrupted_pending.get("sequence") != 831
        or interrupted_pending.get("operation") != "stage-a-monitor-group"
        or evidence.get("schema_version") != "duraseed-tinker-console-lifetime-spend-v1"
        or evidence.get("source") != "authenticated_tinker_web_console"
        or evidence.get("billing_period") != "2026-08"
        or evidence.get("project_id") != project_id
        or evidence.get("project_name") != "DuraSeed"
        or tuple(sessions or ()) != LIFETIME_SESSION_IDS
        or checkpoints != expected_checkpoints
        or observed
        < max(
            run.finished_at.astimezone(UTC),
            interrupted_run.finished_at.astimezone(UTC),
        )
        or before != PRE_ACQUISITION_CONSOLE_SPEND_USD
        or account_spend - before != actual
        or token_cost + storage_cost != account_spend
        or account_spend + remaining != grant
        or grant != Decimal("5000")
        or actual + DOSE_CAP_USD > LIFETIME_CALIBRATION_CAP_USD
    ):
        raise RunnerGateError(
            "actual lifetime spend is not reconciled for the capability-dose launch"
        )
    evidence_sha256 = sha256_bytes(evidence_raw)
    lineage = {
        "correction_run_id": CORRECTION_RUN_ID,
        "correction_preflight_sha256": CORRECTION_PREFLIGHT_SHA256,
        "correction_terminal_sha256": CORRECTION_TERMINAL_SHA256,
        "correction_journal_sha256": CORRECTION_JOURNAL_SHA256,
        "interrupted_dose_run_id": INTERRUPTED_DOSE_RUN_ID,
        "interrupted_dose_preflight_sha256": INTERRUPTED_DOSE_PREFLIGHT_SHA256,
        "interrupted_dose_run_sha256": INTERRUPTED_DOSE_RUN_SHA256,
        "interrupted_dose_journal_sha256": INTERRUPTED_DOSE_JOURNAL_SHA256,
        "interrupted_dose_pending_sequence": 831,
        "session_ids": LIFETIME_SESSION_IDS,
        "observed_at_utc": evidence["observed_at_utc"],
        "pre_acquisition_console_spend_usd": float(before),
        "account_spend_usd": float(account_spend),
        "actual_lifetime_spend_usd": float(actual),
        "remaining_balance_usd": float(remaining),
        "console_evidence_sha256": evidence_sha256,
    }
    return CapabilityDoseBilling(
        float(actual),
        float(remaining),
        evidence_sha256,
        lineage,
    )


def reconcile_capability_dose_billing(
    root: Path, reconciliation_path: Path, raw_billing_path: Path
) -> dict[str, Any]:
    """Close a terminal dose run against its raw usage and actual lifetime spend."""

    required, _ = _object(
        root / "billing-reconciliation-required.json", "billing requirement"
    )
    reconciliation, reconciliation_raw = _object(
        reconciliation_path, "final reconciliation"
    )
    raw, raw_bytes = _object(raw_billing_path, "final raw billing")
    preflight, preflight_raw = _object(root / "preflight.json", "preflight")
    run = read_run_record(root)
    terminal_raw = (root / "capability-dose-terminal.json").read_bytes()
    from duraseed.capability_dose_provenance import (
        existing_capability_dose_terminal as _existing,
    )

    _existing(root, sha256_bytes(preflight_raw))
    events = raw.get("data")
    sessions = required.get("session_ids")
    if not isinstance(events, list) or not isinstance(sessions, list):
        raise RunnerGateError("capability-dose billing omits session-bound events")
    event_sessions = {
        row.get("session_id")
        for row in events
        if isinstance(row, dict) and isinstance(row.get("session_id"), str)
    }
    action_costs = reconciliation.get("action_billed_usd")
    if not isinstance(action_costs, dict) or set(action_costs) != {"capability-dose"}:
        raise RunnerGateError("capability-dose billed action is missing")
    action = _amount(action_costs["capability-dose"], "dose billed spend")
    aggregate = _amount(reconciliation.get("aggregate_billed_usd"), "dose aggregate")
    before = _amount(
        reconciliation.get("pre_acquisition_console_spend_usd"),
        "pre-acquisition console spend",
    )
    after = _amount(
        reconciliation.get("post_run_console_spend_usd"),
        "post-run console spend",
    )
    lifetime = _amount(
        reconciliation.get("post_run_actual_lifetime_spend_usd"),
        "post-run actual lifetime spend",
    )
    remaining = _amount(reconciliation.get("remaining_balance_usd"), "balance")
    cutoff = _time(reconciliation.get("raw_usage_cutoff_utc"), "raw cutoff")
    reconciled = _time(reconciliation.get("reconciled_at_utc"), "reconciliation")
    finished = run.finished_at
    prelaunch = _amount(
        required.get("prelaunch_actual_lifetime_spend_usd"),
        "prelaunch actual lifetime spend",
    )
    if (
        required.get("schema_version") != "duraseed-capability-dose-billing-required-v1"
        or required.get("status") != "pending"
        or reconciliation.get("schema_version")
        != "duraseed-calibration-final-reconciliation-v1"
        or reconciliation.get("status") != "billing_reconciled"
        or reconciliation.get("run_id") != root.name
        or reconciliation.get("project_id") != run.project_id
        or reconciliation.get("session_ids") != sessions
        or not set(sessions).issubset(event_sessions)
        or reconciliation.get("raw_billing_sha256") != sha256_bytes(raw_bytes)
        or reconciliation.get("raw_billing_entry_count") != len(events)
        or finished is None
        or cutoff < finished.astimezone(UTC)
        or reconciled < cutoff
        or required.get("terminal_sha256") != sha256_bytes(terminal_raw)
        or preflight.get("actual_lifetime_billing")
        != required.get("prelaunch_billing_lineage")
        or action != aggregate
        or action > _amount(required.get("action_cap_usd"), "dose action cap")
        or before != PRE_ACQUISITION_CONSOLE_SPEND_USD
        or after - before != lifetime
        or lifetime < prelaunch
        or lifetime > LIFETIME_CALIBRATION_CAP_USD
        or remaining < 0
    ):
        raise RunnerGateError("capability-dose billing reconciliation is incomplete")
    destination = root / "billing-reconciliation.json"
    payload = canonical_json_bytes(reconciliation)
    if destination.exists() and destination.read_bytes() != payload:
        raise RunnerGateError("capability-dose billing reconciliation changed")
    atomic_write_bytes(destination, payload)
    write_run_record(
        root,
        run.model_copy(
            update={
                "updated_at": reconciled,
                "cost_usd": float(aggregate),
                "deviations": [],
            }
        ),
    )
    return {
        "artifact_sha256": sha256_bytes(reconciliation_raw),
        "raw_billing_sha256": sha256_bytes(raw_bytes),
        "aggregate_billed_usd": float(aggregate),
        "actual_lifetime_spend_usd": float(lifetime),
    }


__all__ = [
    "CapabilityDoseBilling",
    "load_capability_dose_billing",
    "reconcile_capability_dose_billing",
]
