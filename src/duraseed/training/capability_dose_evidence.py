"""Evidence models and frozen constants for the B-S capability dose."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Literal

from duraseed.run_records import TrainingMetricRecord
from duraseed.training.stage_a_amended_evidence import (
    StageAAnswerTagPairedItemEvidence,
)
from duraseed.training.teacher_dose import (
    PairedControlChange,
    summarize_paired_control_change,
)


DOSE_LEARNING_RATE = 1e-4
EPOCH_UPDATES = 49
MAX_UPDATES = 294
CADENCE = 10
CADENCE_UPDATES = tuple(range(CADENCE, MAX_UPDATES, CADENCE))
THETA_SUCCESSES = 19
CONFIRMATION_SUCCESSES = 38
MAX_CONFIRMATIONS = 3
MINIMUM_FAMILY_REACHABILITY = 0.60
MAXIMUM_VALID_TAG_DROP = 0.10
MAXIMUM_LENGTH_STOP_RATE = 0.50
MAXIMUM_LOOP_FRACTION = 0.15

DosePhase = Literal["cadence", "confirmation", "epoch_cap"]
DoseAction = Literal[
    "continue",
    "confirm",
    "proceed_to_pilot",
    "tier2_degenerated",
    "dose_limited",
]


def _boolean_rate(
    items: tuple[StageAAnswerTagPairedItemEvidence, ...], name: str
) -> float:
    values = tuple(value for item in items for value in getattr(item, name))
    return fsum(values) / len(values)


@dataclass(frozen=True, slots=True)
class DosePanelEvidence:
    items: tuple[StageAAnswerTagPairedItemEvidence, ...]
    looped_length_stop_count: int
    unique_completion_count: int
    mean_completion_tokens: float
    mean_token_surprisal: float | None
    verified_strategy_count: int

    def __post_init__(self) -> None:
        sample_counts = {item.sample_count for item in self.items}
        if (
            len(self.items) != 96
            or len({item.task_id for item in self.items}) != 96
            or len(sample_counts) != 1
            or next(iter(sample_counts), 0) not in {1, 2}
        ):
            raise ValueError("dose panel requires 96 paired one- or two-draw items")
        counts = (
            self.looped_length_stop_count,
            self.unique_completion_count,
            self.verified_strategy_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("dose panel counts must be nonnegative integers")
        if not isfinite(self.mean_completion_tokens) or self.mean_completion_tokens < 0:
            raise ValueError("dose panel mean completion length is invalid")
        if self.mean_token_surprisal is not None and (
            not isfinite(self.mean_token_surprisal) or self.mean_token_surprisal < 0
        ):
            raise ValueError("dose panel token surprisal is invalid")
        if self.looped_length_stop_count > self.length_stop_count:
            raise ValueError("loop count exceeds the capped-output denominator")

    @property
    def sample_count(self) -> int:
        return sum(item.sample_count for item in self.items)

    @property
    def success_count(self) -> int:
        return sum(sum(item.current_successes) for item in self.items)

    @property
    def paired_gain(self) -> PairedControlChange:
        return summarize_paired_control_change(
            tuple(item.origin_success_rate for item in self.items),
            tuple(item.current_success_rate for item in self.items),
        )

    @property
    def origin_valid_tag_rate(self) -> float:
        return _boolean_rate(self.items, "origin_valid_answer_tags")

    @property
    def current_valid_tag_rate(self) -> float:
        return _boolean_rate(self.items, "current_valid_answer_tags")

    @property
    def valid_tag_retention_passed(self) -> bool:
        return self.current_valid_tag_rate >= (
            self.origin_valid_tag_rate - MAXIMUM_VALID_TAG_DROP
        )

    @property
    def length_stop_count(self) -> int:
        return sum(sum(item.current_length_stops) for item in self.items)

    @property
    def length_stop_rate(self) -> float:
        return self.length_stop_count / self.sample_count

    @property
    def loop_fraction(self) -> float:
        return (
            self.looped_length_stop_count / self.length_stop_count
            if self.length_stop_count
            else 0.0
        )

    @property
    def family_reachability(self) -> float:
        families = {item.family_id for item in self.items}
        reached = {item.family_id for item in self.items if any(item.current_successes)}
        return len(reached) / len(families)


@dataclass(frozen=True, slots=True)
class DoseEvaluationEvidence:
    update: int
    phase: DosePhase
    target: DosePanelEvidence
    sentinel: DosePanelEvidence | None
    metrics: tuple[TrainingMetricRecord, ...]
    leakage_clean: bool

    def __post_init__(self) -> None:
        expected_samples = 192 if self.phase != "cadence" else 96
        phase_coordinate_valid = (
            self.phase in {"cadence", "confirmation"} and self.update in CADENCE_UPDATES
        ) or (self.phase == "epoch_cap" and self.update == MAX_UPDATES)
        if (
            not phase_coordinate_valid
            or self.target.sample_count != expected_samples
            or (self.phase == "cadence") != (self.sentinel is not None)
            or (self.sentinel is not None and self.sentinel.sample_count != 96)
            or type(self.leakage_clean) is not bool
        ):
            raise ValueError("dose evaluation shape differs from its frozen phase")

    @property
    def finite_metrics(self) -> bool:
        return bool(self.metrics) and all(
            isfinite(value) for row in self.metrics for value in row.metrics.values()
        )


@dataclass(frozen=True, slots=True)
class DoseCriterion:
    name: str
    tier: Literal[1, 2, 3]
    decisive: bool
    passed: bool | None
    observed: float | bool | None
    threshold: float | bool | None


@dataclass(frozen=True, slots=True)
class DoseAssessment:
    evidence: DoseEvaluationEvidence
    criteria: tuple[DoseCriterion, ...]

    @property
    def tier1_passed(self) -> bool:
        return all(row.passed is True for row in self.criteria if row.tier == 1)

    @property
    def tier2_passed(self) -> bool:
        return all(
            row.passed is True
            for row in self.criteria
            if row.tier == 2 and row.decisive
        )

    @property
    def cadence_tripwire_passed(self) -> bool:
        rows = tuple(
            row for row in self.criteria if row.name.startswith("cadence_tripwire_")
        )
        return bool(rows) and all(row.passed is True for row in rows)


@dataclass(frozen=True, slots=True)
class DoseDecision:
    action: DoseAction
    update: int
    assessment: DoseAssessment
    confirmation_count: int
    reason: str


@dataclass(frozen=True, slots=True)
class CapabilityDoseLiveEvidence:
    target_panel_id: str
    sentinel_panel_id: str
    origin_sampler_checkpoint_path: str
    origin_state_checkpoint_path: str
    evaluations: tuple[DoseEvaluationEvidence, ...]
    metrics: tuple[TrainingMetricRecord, ...]
    decision: DoseDecision
    retained_sampler_checkpoint_path: str
    retained_state_checkpoint_path: str
    weights_only_restore_validated: Literal[True]
    schedule: Literal["six_replays_of_canonical_steps_1_through_49"] = (
        "six_replays_of_canonical_steps_1_through_49"
    )
    schema_version: Literal["duraseed-capability-dose-v1"] = (
        "duraseed-capability-dose-v1"
    )


__all__ = [name for name in globals() if name.isupper() or name.startswith("Dose")]
__all__.append("CapabilityDoseLiveEvidence")
