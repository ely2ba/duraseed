"""Local lineage and actual-spend gates for a separately authorized Pilot pair."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any

from duraseed.calibration_sources import CalibrationSourceEvidence
from duraseed.capability_dose_provenance import existing_capability_dose_terminal
from duraseed.pilot0_contract import (
    STAGE_B_DECISION_SHA256,
    STAGE_B_LEARNING_RATE,
    STAGE_B_PROFILE,
    STAGE_B_PUBLIC_DECISION_SHA256,
    PilotPairSourceAuthentication,
    PilotSeedSources,
)
from duraseed.pilot0_sources import seed_source_ids, visible_leakage_hash
from duraseed.provenance import canonical_json_hash, sha256_bytes
from duraseed.run_records import RunStatus, read_run_record
from duraseed.runners import RunnerGateError


PRE_ACQUISITION_CONSOLE_SPEND_USD = Decimal("66.19")
GRANT_TOTAL_USD = Decimal("5000")


@dataclass(frozen=True, slots=True)
class PilotPairBilling:
    actual_lifetime_spend_usd: float
    remaining_balance_usd: float
    console_evidence_sha256: str
    lineage: dict[str, Any]
    prior_pair_result_sha256: str | None


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid Pilot {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"Pilot {label} is not an object")
    return value, raw


def _amount(value: Any, label: str) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise RunnerGateError(f"Pilot {label} is not an amount") from error
    if not amount.is_finite() or amount < 0:
        raise RunnerGateError(f"Pilot {label} is not nonnegative and finite")
    return amount


def _time(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise RunnerGateError(f"Pilot {label} is not a timestamp") from error
    if parsed.tzinfo is None:
        raise RunnerGateError(f"Pilot {label} is not timezone-aware")
    return parsed.astimezone(UTC)


def authenticate_pilot_sources(
    *,
    base: CalibrationSourceEvidence,
    source: PilotSeedSources,
    dose_root: Path,
    stage_b_recipe_path: Path,
) -> PilotPairSourceAuthentication:
    preflight_raw = (dose_root / "preflight.json").read_bytes()
    preflight_sha256 = sha256_bytes(preflight_raw)
    terminal = existing_capability_dose_terminal(dose_root, preflight_sha256)
    decision, decision_raw = _object(stage_b_recipe_path, "Stage-B recipe")
    if (
        terminal is None
        or terminal.get("status") != "proceed_to_pilot"
        or sha256_bytes(decision_raw) != STAGE_B_PUBLIC_DECISION_SHA256
        or decision.get("status") != "selected"
        or decision.get("profile") != STAGE_B_PROFILE
        or decision.get("selected_learning_rate") != STAGE_B_LEARNING_RATE
        or decision.get("selected_training_step") != 480
    ):
        raise RunnerGateError("Pilot source decisions are not the frozen branch (i)")
    terminal_sha256 = sha256_bytes(
        (dose_root / "capability-dose-terminal.json").read_bytes()
    )
    lineage = {
        "base_calibration_sources": base,
        "dose_run_id": dose_root.name,
        "dose_preflight_sha256": preflight_sha256,
        "dose_terminal_sha256": terminal_sha256,
        "stage_b_recipe_sha256": STAGE_B_DECISION_SHA256,
        "stage_b_public_projection_sha256": STAGE_B_PUBLIC_DECISION_SHA256,
        "seed_sources": seed_source_ids(source),
        "visible_leakage_sha256": visible_leakage_hash((source,)),
    }
    return PilotPairSourceAuthentication(canonical_json_hash(lineage), lineage)


def _prior_pair(
    root: Path | None, *, pair_index: int
) -> tuple[str | None, str | None, datetime | None]:
    if pair_index == 1:
        if root is not None:
            raise RunnerGateError("pair 1 cannot use a prior Pilot pair")
        return None, None, None
    if root is None:
        raise RunnerGateError("pair 2 requires the completed pair-1 root")
    result, result_raw = _object(root / "result.json", "pair-1 result")
    state, _ = _object(root / "run.json", "pair-1 state")
    preflight, _ = _object(root / "preflight.json", "pair-1 preflight")
    _, evidence_index_raw = _object(
        root / "evidence-index.json", "pair-1 evidence index"
    )
    if (
        result.get("status") != "evidence_collected"
        or result.get("pair_index") != 1
        or not isinstance(result.get("F1_F2_cells"), list)
        or len(result["F1_F2_cells"]) != 2
        or not isinstance(result.get("F3_pre_b_profiles"), list)
        or len(result["F3_pre_b_profiles"]) != 2
        or not isinstance(result.get("paired_primary_contrast"), dict)
        or result.get("evidence_index_sha256") != sha256_bytes(evidence_index_raw)
        or state.get("status") != "evidence_collected"
        or state.get("F1_F2_F3_complete") is not True
        or state.get("result_sha256") != sha256_bytes(result_raw)
        or preflight.get("pair_index") != 1
    ):
        raise RunnerGateError("pair 2 requires durable pair-1 F1/F2/F3 evidence")
    session = preflight.get("lineage", {}).get("session_id")
    if not isinstance(session, str) or not session:
        raise RunnerGateError("pair-1 preflight omitted its session")
    return (
        sha256_bytes(result_raw),
        session,
        _time(state.get("updated_at"), "pair-1 finish"),
    )


def load_pilot_pair_billing(
    dose_root: Path,
    console_evidence_path: Path,
    *,
    project_id: str,
    pair_index: int,
    prior_pair_root: Path | None,
) -> PilotPairBilling:
    """Require console actuals after the dose and, for pair 2, after pair 1."""

    evidence, evidence_raw = _object(console_evidence_path, "console evidence")
    dose_run = read_run_record(dose_root)
    prior_hash, prior_session, prior_finished = _prior_pair(
        prior_pair_root, pair_index=pair_index
    )
    before = _amount(
        evidence.get("pre_acquisition_console_spend_usd"), "pre-acquisition spend"
    )
    account = _amount(evidence.get("account_spend_usd"), "account spend")
    actual = _amount(evidence.get("actual_lifetime_spend_usd"), "actual lifetime spend")
    remaining = _amount(evidence.get("remaining_balance_usd"), "remaining balance")
    grant = _amount(evidence.get("grant_total_usd"), "grant total")
    observed = _time(evidence.get("observed_at_utc"), "console observation")
    sessions = evidence.get("lifetime_session_ids")
    required_sessions = {dose_run.tinker_session_id, prior_session} - {None}
    latest = max(
        value for value in (dose_run.finished_at, prior_finished) if value is not None
    ).astimezone(UTC)
    if (
        dose_run.status is not RunStatus.COMPLETED
        or dose_run.project_id != project_id
        or evidence.get("schema_version") != "duraseed-tinker-console-lifetime-spend-v1"
        or evidence.get("source") != "authenticated_tinker_web_console"
        or evidence.get("project_id") != project_id
        or not isinstance(sessions, list)
        or not required_sessions.issubset(set(sessions))
        or observed < latest
        or before != PRE_ACQUISITION_CONSOLE_SPEND_USD
        or account - before != actual
        or account + remaining != grant
        or grant != GRANT_TOTAL_USD
    ):
        raise RunnerGateError("Pilot launch lacks reconciled actual console spend")
    evidence_sha256 = sha256_bytes(evidence_raw)
    lineage = {
        "console_evidence_sha256": evidence_sha256,
        "observed_at_utc": evidence["observed_at_utc"],
        "actual_lifetime_spend_usd": float(actual),
        "remaining_balance_usd": float(remaining),
        "dose_run_id": dose_root.name,
        "dose_session_id": dose_run.tinker_session_id,
        "prior_pair_result_sha256": prior_hash,
        "prior_pair_session_id": prior_session,
    }
    return PilotPairBilling(
        float(actual), float(remaining), evidence_sha256, lineage, prior_hash
    )


__all__ = [
    "PilotPairBilling",
    "authenticate_pilot_sources",
    "load_pilot_pair_billing",
]
