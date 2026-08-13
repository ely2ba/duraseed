"""Strict validation of the raw billing export for the live-smoke gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import math
from typing import Any

from duraseed.runners import RunnerGateError


_EVENT_TYPES = {
    "training",
    "sampling_prefill",
    "sampling_sample",
    "checkpoint",
    "storage",
}


@dataclass(frozen=True, slots=True)
class SmokeBillingTotals:
    prefill_tokens: int
    sample_tokens: int
    train_tokens: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (
                self.prefill_tokens,
                self.sample_tokens,
                self.train_tokens,
            )
        ):
            raise RunnerGateError("local smoke token totals are malformed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("raw billing JSON contains a duplicate key")
        value[key] = item
    return value


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise RunnerGateError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RunnerGateError(f"{label} must be a UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunnerGateError(f"{label} must be a UTC timestamp")
    return parsed.astimezone(UTC)


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RunnerGateError(f"{label} must be a nonnegative integer")
    return value


def _validate_session_table(value: Any) -> tuple[dict[str, Any], set[str]]:
    if not isinstance(value, dict):
        raise RunnerGateError("raw billing sessions must be an object")
    session_ids: set[str] = set()
    for session_id, session in value.items():
        if not isinstance(session_id, str) or not session_id.strip():
            raise RunnerGateError("raw billing session ID is malformed")
        if not isinstance(session, dict):
            raise RunnerGateError("raw billing session metadata is malformed")
        metadata = session.get("user_metadata")
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in metadata.items()
            )
        ):
            raise RunnerGateError("raw billing session metadata is malformed")
        session_ids.add(session_id)
    return value, session_ids


def _event_identity(row: dict[str, Any], kind: str) -> tuple[Any, ...]:
    info = row["event_info"]
    return (
        row["bucket_start"],
        row["bucket_end"],
        row.get("base_model"),
        row.get("user_id"),
        row.get("session_id"),
        row.get("project_id"),
        kind,
        info.get("cached") if kind == "sampling_prefill" else None,
    )


def _validate_event(
    row: Any, *, cutoff: datetime
) -> tuple[dict[str, Any], datetime, datetime, str]:
    if not isinstance(row, dict):
        raise RunnerGateError("raw billing event is malformed")
    start = _utc(row.get("bucket_start"), "billing bucket start")
    end = _utc(row.get("bucket_end"), "billing bucket end")
    if (
        end - start != timedelta(hours=1)
        or any((start.minute, start.second, start.microsecond))
        or end > cutoff
    ):
        raise RunnerGateError("raw billing event window is invalid for the cutoff")
    session_id = row.get("session_id")
    project_id = row.get("project_id")
    if session_id is not None and (
        not isinstance(session_id, str) or not session_id.strip()
    ):
        raise RunnerGateError("raw billing event session ID is malformed")
    if project_id is not None and (
        not isinstance(project_id, str) or not project_id.strip()
    ):
        raise RunnerGateError("raw billing event project ID is malformed")
    info = row.get("event_info")
    if not isinstance(info, dict) or info.get("type") not in _EVENT_TYPES:
        raise RunnerGateError("raw billing event type is malformed")
    kind = info["type"]
    if kind in {"training", "sampling_prefill", "sampling_sample"}:
        _nonnegative_int(info.get("token_count"), "billing token count")
    if kind == "sampling_prefill" and type(info.get("cached")) is not bool:
        raise RunnerGateError("billing prefill cache flag is malformed")
    if kind == "checkpoint":
        _nonnegative_int(info.get("count"), "billing checkpoint count")
    if kind == "storage":
        quantity = info.get("gigabyte_hours")
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, (int, float))
            or not math.isfinite(quantity)
            or quantity < 0
        ):
            raise RunnerGateError("billing storage quantity is malformed")
    return row, start, end, kind


def validate_smoke_billing_export(
    raw_bytes: bytes,
    *,
    smoke_run_id: str,
    project_id: str,
    smoke_finished_at: datetime,
    raw_usage_cutoff: datetime,
    expected: SmokeBillingTotals,
) -> str:
    """Return the unique smoke session after validating its exact token usage."""

    try:
        raw = json.loads(raw_bytes, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RunnerGateError("post-smoke raw billing JSON is malformed") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), list):
        raise RunnerGateError("raw billing export must contain data and sessions")
    sessions, declared_session_ids = _validate_session_table(raw.get("sessions"))
    matching_sessions = [
        session_id
        for session_id, session in sessions.items()
        if session.get("user_metadata")
        == {"phase_label": "live-smoke-gate", "run_id": smoke_run_id}
    ]
    if len(matching_sessions) != 1:
        raise RunnerGateError("raw billing does not identify one exact smoke session")
    smoke_session_id = matching_sessions[0]
    cutoff = raw_usage_cutoff.astimezone(UTC)
    finished = smoke_finished_at.astimezone(UTC)
    if any((cutoff.minute, cutoff.second, cutoff.microsecond)):
        raise RunnerGateError("raw billing cutoff must be aligned to a UTC hour")

    seen: set[tuple[Any, ...]] = set()
    observed_session_ids: set[str] = set()
    totals = {"prefill": 0, "sample": 0, "train": 0}
    selected_count = 0
    for item in raw["data"]:
        row, start, end, kind = _validate_event(item, cutoff=cutoff)
        identity = _event_identity(row, kind)
        if identity in seen:
            raise RunnerGateError("raw billing contains duplicate hourly events")
        seen.add(identity)
        session_id = row.get("session_id")
        if session_id is not None:
            observed_session_ids.add(session_id)
        if session_id != smoke_session_id:
            continue
        selected_count += 1
        if row.get("project_id") != project_id:
            raise RunnerGateError("smoke billing session has a different project ID")
        if not start <= finished < end:
            raise RunnerGateError(
                "smoke billing event window excludes smoke completion"
            )
        info = row["event_info"]
        if kind == "sampling_prefill":
            totals["prefill"] += info["token_count"]
        elif kind == "sampling_sample":
            totals["sample"] += info["token_count"]
        elif kind == "training":
            totals["train"] += info["token_count"]
        # Checkpoint and storage rows are deliberately excluded from token totals.

    if observed_session_ids != declared_session_ids:
        raise RunnerGateError("raw billing sessions table does not match its events")
    if selected_count == 0:
        raise RunnerGateError("raw billing omits events for the smoke session")
    observed = SmokeBillingTotals(totals["prefill"], totals["sample"], totals["train"])
    if observed != expected:
        raise RunnerGateError(
            "raw billing token totals differ from the local smoke run"
        )
    return smoke_session_id


__all__ = ["SmokeBillingTotals", "validate_smoke_billing_export"]
