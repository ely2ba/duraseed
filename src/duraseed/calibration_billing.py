"""Final billing reconciliation contract for the `$300` calibration gate."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any

from duraseed.calibration_parent import PARENT_BILLED_USD, PARENT_RUN_ID
from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import RunStatus, read_run_record, write_run_record
from duraseed.runners import RunnerGateError
from duraseed.teacher_exposure_spec import (
    AMENDED_AGGREGATE_CAP_USD,
    AMENDED_STAGE_A_CAP_USD,
    DIRECT_M0_AGGREGATE_CAP_USD,
    DIRECT_M0_STAGE_A_CAP_USD,
    LIFETIME_CALIBRATION_CAP_USD,
    M1_TEACHER_CAP_USD,
    ORIGINAL_TEACHER_CAP_USD,
    PRIOR_DIRECT_STAGE_A_CHARGE_USD,
    PRIOR_REPAIR_TEACHER_CAP_USD,
)


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid calibration {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"calibration {label} is not an object")
    return value, raw


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise RunnerGateError(f"{label} must be a finite amount")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise RunnerGateError(f"{label} must be a finite amount") from error
    if not result.is_finite() or result < 0:
        raise RunnerGateError(f"{label} must be a nonnegative finite amount")
    return result


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise RunnerGateError(f"{label} must be an aware timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RunnerGateError(f"{label} must be an aware timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise RunnerGateError(f"{label} must be an aware timestamp")
    return result.astimezone(UTC)


def _run_has_authorized_terminal(
    root: Path, required: dict[str, Any], run: Any
) -> bool:
    if run.status is RunStatus.COMPLETED:
        return (
            required.get("run_status") == RunStatus.COMPLETED.value
            and required.get("terminal_status") is None
            and required.get("terminal_sha256") is None
        )
    if run.status is not RunStatus.FAILED:
        return False
    stage_a_path = root / "stage-a-terminal.json"
    preflight_path = root / "preflight.json"
    if stage_a_path.exists() and preflight_path.exists():
        terminal, terminal_raw = _object(stage_a_path, "Stage-A terminal")
        preflight_sha256 = sha256_bytes(preflight_path.read_bytes())
        from duraseed.calibration_stage_a_terminal import existing_stage_a_terminal

        return (
            required.get("run_status") == RunStatus.FAILED.value
            and required.get("terminal_status") == terminal.get("status")
            and required.get("terminal_sha256") == sha256_bytes(terminal_raw)
            and terminal.get("preflight_sha256") == preflight_sha256
            and existing_stage_a_terminal(root, preflight_sha256) is not None
        )
    terminal_path = root / "teacher-dose-terminal.json"
    if not terminal_path.exists() or not preflight_path.exists():
        return False
    terminal, terminal_raw = _object(terminal_path, "teacher-exposure terminal")
    from duraseed.calibration_teacher_terminal import (
        existing_teacher_exposure_terminal,
    )

    return (
        required.get("run_status") == RunStatus.FAILED.value
        and required.get("terminal_status") == "no_stable_checkpoint"
        and required.get("terminal_sha256") == sha256_bytes(terminal_raw)
        and terminal.get("preflight_sha256")
        == sha256_bytes(preflight_path.read_bytes())
        and existing_teacher_exposure_terminal(root, terminal["preflight_sha256"])
        is not None
    )


def reconcile_calibration_billing(
    run_directory: str | Path,
    reconciliation_path: str | Path,
    raw_billing_path: str | Path,
) -> dict[str, Any]:
    """Authenticate lag-cleared raw usage, caps, sessions, and protected reserve."""

    root = Path(run_directory)
    required, _ = _object(
        root / "billing-reconciliation-required.json", "billing requirement"
    )
    preflight, _ = _object(root / "preflight.json", "preflight")
    reconciliation, reconciliation_raw = _object(
        Path(reconciliation_path), "billing reconciliation"
    )
    raw, raw_bytes = _object(Path(raw_billing_path), "raw billing export")
    run = read_run_record(root)
    sessions_value, _ = _object(root / "session-lineage.json", "session lineage")
    sessions = sessions_value.get("session_ids")
    events = raw.get("data")
    if not isinstance(sessions, list) or not isinstance(events, list):
        raise RunnerGateError("billing evidence omits session-bound raw events")
    event_sessions = {
        row.get("session_id")
        for row in events
        if isinstance(row, dict) and isinstance(row.get("session_id"), str)
    }
    action_costs = reconciliation.get("action_billed_usd")
    if not isinstance(action_costs, dict) or set(action_costs) != {"stage-a"}:
        raise RunnerGateError("billing reconciliation omits the Stage-A total")
    action_caps = required.get("action_caps_usd")
    parent_lineage = preflight.get("parent_calibration")
    prior_lineage = preflight.get("prior_repair")
    m1_lineage = preflight.get("interrupted_m1")
    prior_stage_a_lineage = preflight.get("prior_direct_stage_a")
    if not isinstance(action_caps, dict) or set(action_caps) != {"stage-a"}:
        raise RunnerGateError("billing requirement omits the Stage-A cap")
    stage_a = _decimal(action_costs["stage-a"], "Stage-A billed spend")
    stage_a_cap = _decimal(action_caps["stage-a"], "Stage-A action cap")
    child_cap = _decimal(required.get("aggregate_cap_usd"), "aggregate cap")
    parent_spend = _decimal(required.get("parent_billed_usd"), "parent spend")
    prior_spend = _decimal(
        required.get("prior_repair_teacher_cap_usd"), "prior repair spend floor"
    )
    m1_spend = _decimal(
        required.get("interrupted_m1_teacher_cap_usd"), "M1 spend floor"
    )
    prior_stage_a_spend = _decimal(
        required.get("prior_direct_stage_a_charge_usd", 0),
        "prior direct Stage-A spend bound",
    )
    lifetime_cap = _decimal(
        required.get("lifetime_calibration_cap_usd"), "lifetime calibration cap"
    )
    required_reserve = _decimal(
        required.get("protected_reserve_usd"), "required protected reserve"
    )
    aggregate = _decimal(reconciliation.get("aggregate_billed_usd"), "aggregate spend")
    balance = _decimal(reconciliation.get("remaining_balance_usd"), "remaining balance")
    reserve = _decimal(reconciliation.get("protected_reserve_usd"), "protected reserve")
    cutoff = _utc(reconciliation.get("raw_usage_cutoff_utc"), "raw usage cutoff")
    reconciled = _utc(reconciliation.get("reconciled_at_utc"), "reconciliation time")
    finished = run.finished_at
    amended = stage_a_cap == Decimal(
        str(AMENDED_STAGE_A_CAP_USD)
    ) and child_cap == Decimal(str(AMENDED_AGGREGATE_CAP_USD))
    legacy = stage_a_cap == Decimal(
        str(DIRECT_M0_STAGE_A_CAP_USD)
    ) and child_cap == Decimal(str(DIRECT_M0_AGGREGATE_CAP_USD))
    prior_stage_a_valid = (
        amended
        and isinstance(prior_stage_a_lineage, dict)
        and required.get("prior_direct_stage_a") == prior_stage_a_lineage
        and prior_stage_a_spend == Decimal(str(PRIOR_DIRECT_STAGE_A_CHARGE_USD))
        and prior_stage_a_lineage.get("charged_stage_a_usd")
        == PRIOR_DIRECT_STAGE_A_CHARGE_USD
        and prior_stage_a_lineage.get("pending_remote_calls") == 0
    ) or (
        legacy
        and prior_stage_a_lineage is None
        and required.get("prior_direct_stage_a") is None
        and prior_stage_a_spend == 0
    )
    if (
        reconciliation.get("schema_version")
        != "duraseed-calibration-final-reconciliation-v1"
        or reconciliation.get("status") != "billing_reconciled"
        or not _run_has_authorized_terminal(root, required, run)
        or finished is None
        or required.get("schema_version") != "duraseed-calibration-billing-required-v1"
        or required.get("status") != "pending"
        or reconciliation.get("run_id") != root.name
        or reconciliation.get("project_id") != run.project_id
        or reconciliation.get("session_ids") != sessions
        or not set(sessions).issubset(event_sessions)
        or reconciliation.get("raw_billing_sha256") != sha256_bytes(raw_bytes)
        or reconciliation.get("raw_billing_entry_count") != len(events)
        or cutoff < finished.astimezone(UTC)
        or reconciled < cutoff
        or not (amended or legacy)
        or child_cap != stage_a_cap
        or parent_spend != Decimal(str(PARENT_BILLED_USD))
        or required.get("parent_run_id") != PARENT_RUN_ID
        or not isinstance(parent_lineage, dict)
        or required.get("parent_billing_sha256") != parent_lineage.get("billing_sha256")
        or not isinstance(prior_lineage, dict)
        or required.get("prior_repair") != prior_lineage
        or prior_spend != Decimal(str(PRIOR_REPAIR_TEACHER_CAP_USD))
        or prior_lineage.get("charged_teacher_cap_usd") != PRIOR_REPAIR_TEACHER_CAP_USD
        or not isinstance(m1_lineage, dict)
        or required.get("interrupted_m1") != m1_lineage
        or m1_spend != Decimal(str(M1_TEACHER_CAP_USD))
        or m1_lineage.get("charged_teacher_cap_usd") != M1_TEACHER_CAP_USD
        or not prior_stage_a_valid
        or lifetime_cap != Decimal(str(LIFETIME_CALIBRATION_CAP_USD))
        or parent_spend + prior_spend + m1_spend + prior_stage_a_spend + child_cap
        > lifetime_cap
        or parent_spend + prior_spend + m1_spend
        > Decimal(str(ORIGINAL_TEACHER_CAP_USD))
        or stage_a > stage_a_cap
        or aggregate != stage_a
        or aggregate > child_cap
        or parent_spend + prior_spend + m1_spend + prior_stage_a_spend + aggregate
        > lifetime_cap
        or reconciliation.get("protected_reserve_survives") is not True
        or reserve != required_reserve
        or balance < required_reserve
    ):
        raise RunnerGateError("calibration billing reconciliation is incomplete")
    destination = root / "billing-reconciliation.json"
    payload = canonical_json_bytes(reconciliation)
    if destination.exists() and destination.read_bytes() != payload:
        raise RunnerGateError("calibration billing reconciliation changed")
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
    }


__all__ = ["reconcile_calibration_billing"]
