"""Post-smoke billing gate used by the paid boundary launcher."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any

from duraseed.post_smoke_billing import (
    SmokeBillingTotals,
    validate_smoke_billing_export,
)
from duraseed.runners import RunnerGateError


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("invalid post-smoke billing reconciliation") from error
    if not isinstance(value, dict):
        raise RunnerGateError("invalid post-smoke billing reconciliation")
    return value


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


def authenticate_post_smoke_billing(
    path: str | Path,
    *,
    smoke_run_id: str,
    smoke_sha256: str,
    smoke_finished_at: datetime,
    smoke_tokens: SmokeBillingTotals,
    project_id: str,
) -> None:
    """Require session-bound, lag-cleared usage and balance evidence."""

    resolved = Path(path).resolve()
    value = _object(resolved)
    raw_path_value = value.get("raw_usage_path")
    if not isinstance(raw_path_value, str) or not raw_path_value.strip():
        raise RunnerGateError("post-smoke billing reconciliation omitted raw usage")
    raw_path = Path(raw_path_value).expanduser()
    if not raw_path.is_absolute():
        raw_path = resolved.parent / raw_path
    try:
        raw_bytes = raw_path.read_bytes()
    except OSError as error:
        raise RunnerGateError("post-smoke raw usage is unreadable") from error
    raw_sha256 = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
    cutoff_value = value.get("raw_usage_cutoff_utc")
    if not isinstance(cutoff_value, str):
        raise RunnerGateError("raw usage cutoff must be a UTC timestamp")
    try:
        cutoff = datetime.fromisoformat(cutoff_value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RunnerGateError("raw usage cutoff must be a UTC timestamp") from error
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise RunnerGateError("raw usage cutoff must be a UTC timestamp")
    balance = _decimal(value.get("remaining_balance_usd"), "remaining balance")
    reserve = _decimal(value.get("protected_reserve_usd"), "protected reserve")
    if (
        value.get("status") != "reconciled"
        or value.get("source_run_id") != smoke_run_id
        or value.get("source_acceptance_sha256") != smoke_sha256
        or value.get("project_id") != project_id
        or value.get("raw_usage_sha256") != raw_sha256
        or cutoff < smoke_finished_at
        or value.get("remaining_balance_verified") is not True
        or value.get("protected_reserve_survives") is not True
        or _decimal(value.get("boundary_authorization_usd"), "boundary authorization")
        != Decimal("120")
        or balance < Decimal("120")
        or reserve < 0
        or balance - Decimal("120") < reserve
    ):
        raise RunnerGateError("post-smoke billing reconciliation is incomplete")
    validate_smoke_billing_export(
        raw_bytes,
        smoke_run_id=smoke_run_id,
        project_id=project_id,
        smoke_finished_at=smoke_finished_at,
        raw_usage_cutoff=cutoff,
        expected=smoke_tokens,
    )


__all__ = ["authenticate_post_smoke_billing"]
