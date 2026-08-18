"""Mutable live state for one Stage-A training branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from duraseed.run_records import TrainingMetricRecord
from duraseed.runtime import RuntimeBundle


@dataclass(slots=True)
class Branch:
    method: Literal["B-S", "B-G"]
    learning_rate: float
    runtime: RuntimeBundle
    metrics: list[TrainingMetricRecord] = field(default_factory=list)
    surprisal_by_step: dict[int, float] = field(default_factory=dict)
    unique_completions_by_step: dict[int, set[str]] = field(default_factory=dict)
    valid_families_by_step: dict[int, set[str]] = field(default_factory=dict)
    successful_completions_by_step: dict[int, set[str]] = field(default_factory=dict)


__all__ = ["Branch"]
