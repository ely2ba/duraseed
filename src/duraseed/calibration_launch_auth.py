"""Hash-bound billing evidence required before the `$300` calibration launch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any

from duraseed.provenance import sha256_bytes
from duraseed.run_records import RunStatus, read_run_record
from duraseed.runners import RunnerGateError


@dataclass(frozen=True, slots=True)
class PrecalibrationBillingEvidence:
    artifact_sha256: str
    raw_billing_sha256: str
    raw_billing_entry_count: int
    remaining_balance_usd: Decimal
    protected_reserve_usd: Decimal


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid {label} artifact") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"{label} artifact is not an object")
    return value, raw


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise RunnerGateError(f"{label} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise RunnerGateError(f"{label} must be a finite decimal") from error
    if not result.is_finite():
        raise RunnerGateError(f"{label} must be a finite decimal")
    return result


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise RunnerGateError(f"{label} must be a UTC timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RunnerGateError(f"{label} must be a UTC timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise RunnerGateError(f"{label} must be a UTC timestamp")
    return result.astimezone(UTC)


def authenticate_precalibration_billing(
    reconciliation_path: str | Path,
    raw_billing_path: str | Path,
    *,
    boundary_directory: str | Path,
    project_id: str,
) -> PrecalibrationBillingEvidence:
    """Require lag-cleared usage and a surviving protected reserve."""

    reconciliation, reconciliation_raw = _object(
        Path(reconciliation_path), "pre-calibration billing reconciliation"
    )
    raw, raw_bytes = _object(Path(raw_billing_path), "raw billing export")
    run = read_run_record(boundary_directory)
    remaining = _decimal(
        reconciliation.get("remaining_balance_usd"), "remaining balance"
    )
    reserve = _decimal(reconciliation.get("protected_reserve_usd"), "protected reserve")
    current = _decimal(reconciliation.get("current_spend_usd"), "current spend")
    cutoff = _utc(reconciliation.get("raw_usage_cutoff_utc"), "raw usage cutoff")
    events = raw.get("data")
    raw_hash = sha256_bytes(raw_bytes)
    if (
        reconciliation.get("schema_version") != "duraseed-precalibration-billing-v1"
        or reconciliation.get("status") != "reconciled"
        or reconciliation.get("project_id") != project_id
        or reconciliation.get("boundary_run_id") != Path(boundary_directory).name
        or reconciliation.get("raw_billing_sha256") != raw_hash
        or not isinstance(events, list)
        or reconciliation.get("raw_billing_entry_count") != len(events)
        or run.status is not RunStatus.COMPLETED
        or run.finished_at is None
        or cutoff < run.finished_at.astimezone(UTC)
        or reconciliation.get("calibration_authorization_usd") != 300
        or reconciliation.get("protected_reserve_survives") is not True
        or current < 0
        or remaining < 0
        or reserve < 0
        or remaining - Decimal("300") < reserve
    ):
        raise RunnerGateError("pre-calibration billing reconciliation is incomplete")
    return PrecalibrationBillingEvidence(
        sha256_bytes(reconciliation_raw), raw_hash, len(events), remaining, reserve
    )


__all__ = ["PrecalibrationBillingEvidence", "authenticate_precalibration_billing"]
