"""Final billing reconciliation contract for the `$300` calibration gate."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import RunStatus, read_run_record, write_run_record
from duraseed.runners import RunnerGateError


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
    if not isinstance(action_costs, dict) or set(action_costs) != {
        "teacher-dose",
        "stage-a",
    }:
        raise RunnerGateError("billing reconciliation omits the two action totals")
    action_caps = required.get("action_caps_usd")
    if not isinstance(action_caps, dict) or set(action_caps) != {
        "teacher-dose",
        "stage-a",
    }:
        raise RunnerGateError("billing requirement omits the two action caps")
    teacher = _decimal(action_costs["teacher-dose"], "teacher-dose billed spend")
    stage_a = _decimal(action_costs["stage-a"], "Stage-A billed spend")
    teacher_cap = _decimal(action_caps["teacher-dose"], "teacher-dose action cap")
    stage_a_cap = _decimal(action_caps["stage-a"], "Stage-A action cap")
    aggregate = _decimal(reconciliation.get("aggregate_billed_usd"), "aggregate spend")
    balance = _decimal(reconciliation.get("remaining_balance_usd"), "remaining balance")
    reserve = _decimal(reconciliation.get("protected_reserve_usd"), "protected reserve")
    cutoff = _utc(reconciliation.get("raw_usage_cutoff_utc"), "raw usage cutoff")
    reconciled = _utc(reconciliation.get("reconciled_at_utc"), "reconciliation time")
    finished = run.finished_at
    if (
        reconciliation.get("schema_version")
        != "duraseed-calibration-final-reconciliation-v1"
        or reconciliation.get("status") != "billing_reconciled"
        or run.status is not RunStatus.COMPLETED
        or finished is None
        or required.get("status") != "pending"
        or reconciliation.get("run_id") != root.name
        or reconciliation.get("project_id") != run.project_id
        or reconciliation.get("session_ids") != sessions
        or not set(sessions).issubset(event_sessions)
        or reconciliation.get("raw_billing_sha256") != sha256_bytes(raw_bytes)
        or reconciliation.get("raw_billing_entry_count") != len(events)
        or cutoff < finished.astimezone(UTC)
        or reconciled < cutoff
        or _decimal(required.get("aggregate_cap_usd"), "aggregate cap")
        != Decimal("300")
        or teacher_cap + stage_a_cap > Decimal("300")
        or teacher > teacher_cap
        or stage_a > stage_a_cap
        or aggregate != teacher + stage_a
        or aggregate > Decimal("300")
        or reconciliation.get("protected_reserve_survives") is not True
        or balance < reserve
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
