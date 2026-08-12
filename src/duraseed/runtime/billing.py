"""Pinned Tinker pricing and local usage reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class UsageQuantities:
    prefill_tokens: int = 0
    cached_prefill_tokens: int = 0
    sample_tokens: int = 0
    train_tokens: int = 0
    checkpoint_count: int = 0
    storage_gb_hours: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.prefill_tokens,
            self.cached_prefill_tokens,
            self.sample_tokens,
            self.train_tokens,
            self.checkpoint_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(
                "usage token and checkpoint counts must be nonnegative integers"
            )
        if not math.isfinite(self.storage_gb_hours) or self.storage_gb_hours < 0:
            raise ValueError("storage usage must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    snapshot_id: str
    model_id: str
    effective_date: date
    prefill_per_million_usd: float
    cached_prefill_per_million_usd: float
    sample_per_million_usd: float
    train_per_million_usd: float
    checkpoint_gb_month_usd: float

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip() or not self.model_id.strip():
            raise ValueError("price snapshot identity must be nonempty")
        rates = (
            self.prefill_per_million_usd,
            self.cached_prefill_per_million_usd,
            self.sample_per_million_usd,
            self.train_per_million_usd,
            self.checkpoint_gb_month_usd,
        )
        if any(not math.isfinite(value) or value < 0 for value in rates):
            raise ValueError("price rates must be finite and nonnegative")

    def cost(self, usage: UsageQuantities) -> float:
        if usage.checkpoint_count:
            raise ValueError(
                "checkpoint event pricing is absent from the frozen price snapshot"
            )
        token_cost = (
            usage.prefill_tokens * self.prefill_per_million_usd
            + usage.cached_prefill_tokens * self.cached_prefill_per_million_usd
            + usage.sample_tokens * self.sample_per_million_usd
            + usage.train_tokens * self.train_per_million_usd
        ) / 1_000_000
        storage_cost = usage.storage_gb_hours * self.checkpoint_gb_month_usd / (30 * 24)
        return token_cost + storage_cost


PRICE_SNAPSHOT = PriceSnapshot(
    snapshot_id="tinker-qwen3.5-9b-base-2026-08-09",
    model_id="Qwen/Qwen3.5-9B-Base",
    effective_date=date(2026, 8, 9),
    prefill_per_million_usd=0.66,
    cached_prefill_per_million_usd=0.132,
    sample_per_million_usd=1.995,
    train_per_million_usd=1.463,
    checkpoint_gb_month_usd=0.10,
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def parse_billing_usage(
    usage: Any,
    *,
    session_id: str,
    project_id: str,
) -> UsageQuantities:
    """Aggregate one Tinker billing export for a run and its project storage."""

    counts: dict[str, int | float] = {
        "prefill_tokens": 0,
        "cached_prefill_tokens": 0,
        "sample_tokens": 0,
        "train_tokens": 0,
        "checkpoint_count": 0,
        "storage_gb_hours": 0.0,
    }
    events = _field(usage, "data", ())
    if isinstance(usage, dict) and "raw_quantities" in usage:
        raw = usage["raw_quantities"]
        return UsageQuantities(
            prefill_tokens=int(raw.get("prefill_tokens", 0)),
            cached_prefill_tokens=int(raw.get("cached_prefill_tokens", 0)),
            sample_tokens=int(raw.get("sample_tokens", 0)),
            train_tokens=int(raw.get("train_tokens", 0)),
            checkpoint_count=int(raw.get("checkpoint_count", 0)),
            storage_gb_hours=float(raw.get("storage_gb_hours", 0.0)),
        )
    for event in events:
        info = _field(event, "event_info")
        kind = _field(info, "type")
        run_event = _field(event, "session_id") == session_id
        project_storage = (
            kind == "storage"
            and _field(event, "session_id") is None
            and _field(event, "project_id") == project_id
        )
        if not (run_event or project_storage):
            continue
        if kind == "training":
            counts["train_tokens"] += int(_field(info, "token_count", 0))
        elif kind == "sampling_prefill":
            key = (
                "cached_prefill_tokens"
                if bool(_field(info, "cached", False))
                else "prefill_tokens"
            )
            counts[key] += int(_field(info, "token_count", 0))
        elif kind == "sampling_sample":
            counts["sample_tokens"] += int(_field(info, "token_count", 0))
        elif kind == "checkpoint":
            counts["checkpoint_count"] += int(_field(info, "count", 0))
        elif kind == "storage":
            counts["storage_gb_hours"] += float(_field(info, "gigabyte_hours", 0.0))
    return UsageQuantities(**counts)  # type: ignore[arg-type]


__all__ = ["PRICE_SNAPSHOT", "PriceSnapshot", "UsageQuantities", "parse_billing_usage"]
