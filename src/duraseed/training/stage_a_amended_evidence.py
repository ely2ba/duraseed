"""Evidence and decision records for the verifier-valid Stage-A amendment."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Literal, TypeAlias

from duraseed.run_records import TrainingMetricRecord
from duraseed.training.stage_a_calibration import (
    StageACalibrationEvidenceError,
    StageADurationDecisionStatus,
    StageALearningRateDecisionStatus,
)
from duraseed.training.teacher_dose import PairedControlChange


AmendedStageAMethod: TypeAlias = Literal["B-S", "B-G"]
AMENDED_STAGE_A_LEARNING_RATES: dict[AmendedStageAMethod, float] = {
    "B-S": 1e-4,
    "B-G": 1e-5,
}
STAGE_A_MAXIMUM_VALID_ANSWER_TAG_DROP = 0.10


@dataclass(frozen=True, slots=True)
class StageAAnswerTagPairedItemEvidence:
    """Paired M0/current samples with the verifier's answer-tag observation."""

    task_id: str
    family_id: str
    sampling_seeds: tuple[int, ...]
    origin_successes: tuple[bool, ...]
    current_successes: tuple[bool, ...]
    origin_valid_answer_tags: tuple[bool, ...]
    current_valid_answer_tags: tuple[bool, ...]
    origin_length_stops: tuple[bool, ...]
    current_length_stops: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.family_id.strip():
            raise ValueError("Stage-A paired item identity must be nonempty")
        if (
            not self.sampling_seeds
            or any(type(seed) is not int or seed < 0 for seed in self.sampling_seeds)
            or len(set(self.sampling_seeds)) != len(self.sampling_seeds)
        ):
            raise ValueError("Stage-A paired sampling seeds are invalid")
        for name in (
            "origin_successes",
            "current_successes",
            "origin_valid_answer_tags",
            "current_valid_answer_tags",
            "origin_length_stops",
            "current_length_stops",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or len(values) != len(self.sampling_seeds)
                or any(type(value) is not bool for value in values)
            ):
                raise ValueError(f"{name} must match the paired sampling seeds")

    @property
    def sample_count(self) -> int:
        return len(self.sampling_seeds)

    @property
    def origin_success_rate(self) -> float:
        return fsum(self.origin_successes) / self.sample_count

    @property
    def current_success_rate(self) -> float:
        return fsum(self.current_successes) / self.sample_count

    @property
    def coordinate(self) -> tuple[str, str, tuple[int, ...]]:
        return self.task_id, self.family_id, self.sampling_seeds

    @property
    def origin_observations(self) -> tuple[tuple[bool, ...], ...]:
        return (
            self.origin_successes,
            self.origin_valid_answer_tags,
            self.origin_length_stops,
        )


def _validate_items(
    items: tuple[StageAAnswerTagPairedItemEvidence, ...],
    *,
    label: str,
    samples_per_item: int,
) -> None:
    if (
        len(items) != 96
        or any(
            not isinstance(item, StageAAnswerTagPairedItemEvidence)
            or item.sample_count != samples_per_item
            for item in items
        )
        or len({item.task_id for item in items}) != 96
    ):
        raise StageACalibrationEvidenceError(
            f"{label} requires 96 unique items with {samples_per_item} sample(s) each"
        )


def _validate_metrics(
    metrics: tuple[TrainingMetricRecord, ...], required_steps: tuple[int, ...]
) -> None:
    if (
        not metrics
        or any(not isinstance(row, TrainingMetricRecord) for row in metrics)
        or any(row.phase != "stage_a" for row in metrics)
        or len({row.training_step for row in metrics}) != len(metrics)
        or not set(required_steps).issubset(row.training_step for row in metrics)
    ):
        raise StageACalibrationEvidenceError("amended Stage-A metrics are incomplete")


@dataclass(frozen=True, slots=True)
class AmendedStageAScreenEvidence:
    method: AmendedStageAMethod
    learning_rate: float
    target_panel_id: str
    sentinel_panel_id: str
    origin_sampler_checkpoint_path: str
    candidate_sampler_checkpoint_path: str
    target_items: tuple[StageAAnswerTagPairedItemEvidence, ...]
    sentinel_items: tuple[StageAAnswerTagPairedItemEvidence, ...]
    metrics: tuple[TrainingMetricRecord, ...]
    leakage_clean: bool
    training_step: Literal[10] = 10
    format_estimand: Literal["verifier_valid_answer_tag"] = "verifier_valid_answer_tag"

    def __post_init__(self) -> None:
        if self.method not in AMENDED_STAGE_A_LEARNING_RATES:
            raise ValueError("unsupported amended Stage-A method")
        if self.learning_rate != AMENDED_STAGE_A_LEARNING_RATES[self.method]:
            raise StageACalibrationEvidenceError(
                "amended Stage-A evidence uses an unselected coordinate"
            )
        if (
            not self.target_panel_id.strip()
            or not self.sentinel_panel_id.strip()
            or self.target_panel_id == self.sentinel_panel_id
            or not self.origin_sampler_checkpoint_path.strip()
            or not self.candidate_sampler_checkpoint_path.strip()
            or self.origin_sampler_checkpoint_path
            == self.candidate_sampler_checkpoint_path
            or self.training_step != 10
            or type(self.leakage_clean) is not bool
        ):
            raise StageACalibrationEvidenceError(
                "amended Stage-A screen identity is invalid"
            )
        _validate_items(self.target_items, label="screen target", samples_per_item=1)
        _validate_items(
            self.sentinel_items, label="screen sentinel", samples_per_item=1
        )
        if {item.task_id for item in self.target_items}.intersection(
            item.task_id for item in self.sentinel_items
        ):
            raise StageACalibrationEvidenceError("screen panels overlap")
        _validate_metrics(self.metrics, (10,))


@dataclass(frozen=True, slots=True)
class AmendedStageAFinalEvidence:
    method: AmendedStageAMethod
    learning_rate: float
    target_panel_id: str
    sentinel_panel_id: str
    origin_sampler_checkpoint_path: str
    candidate_sampler_checkpoint_path: str
    target_items: tuple[StageAAnswerTagPairedItemEvidence, ...]
    sentinel_items: tuple[StageAAnswerTagPairedItemEvidence, ...]
    metrics: tuple[TrainingMetricRecord, ...]
    leakage_clean: bool
    training_step: Literal[50] = 50
    format_estimand: Literal["verifier_valid_answer_tag"] = "verifier_valid_answer_tag"

    def __post_init__(self) -> None:
        if self.method not in AMENDED_STAGE_A_LEARNING_RATES:
            raise ValueError("unsupported amended Stage-A method")
        if self.learning_rate != AMENDED_STAGE_A_LEARNING_RATES[self.method]:
            raise StageACalibrationEvidenceError(
                "amended Stage-A final uses an unselected coordinate"
            )
        if (
            not self.target_panel_id.strip()
            or not self.sentinel_panel_id.strip()
            or self.target_panel_id == self.sentinel_panel_id
            or not self.origin_sampler_checkpoint_path.strip()
            or not self.candidate_sampler_checkpoint_path.strip()
            or self.origin_sampler_checkpoint_path
            == self.candidate_sampler_checkpoint_path
            or self.training_step != 50
            or type(self.leakage_clean) is not bool
        ):
            raise StageACalibrationEvidenceError(
                "amended Stage-A final identity is invalid"
            )
        _validate_items(self.target_items, label="final target", samples_per_item=2)
        _validate_items(self.sentinel_items, label="final sentinel", samples_per_item=1)
        if {item.task_id for item in self.target_items}.intersection(
            item.task_id for item in self.sentinel_items
        ):
            raise StageACalibrationEvidenceError("final panels overlap")
        _validate_metrics(self.metrics, (25, 50))


@dataclass(frozen=True, slots=True)
class AmendedStageALiveEvidence:
    bs_screens: tuple[AmendedStageAScreenEvidence, ...]
    bg_screens: tuple[AmendedStageAScreenEvidence, ...]
    final_evidence: tuple[AmendedStageAFinalEvidence, ...]
    schema_version: Literal["duraseed-stage-a-valid-tag-v1"] = (
        "duraseed-stage-a-valid-tag-v1"
    )


@dataclass(frozen=True, slots=True)
class AmendedStageAPanelHealth:
    origin_valid_answer_tag_rate: float
    current_valid_answer_tag_rate: float
    valid_answer_tag_drop: float
    valid_answer_tag_retention_passed: bool
    origin_length_stop_rate: float
    current_length_stop_rate: float
    length_stop_increase: float
    length_health_passed: bool


@dataclass(frozen=True, slots=True)
class AmendedStageAScreenAssessment:
    evidence: AmendedStageAScreenEvidence
    target_paired_gain: PairedControlChange
    target_health: AmendedStageAPanelHealth
    sentinel_health: AmendedStageAPanelHealth
    mean_mixed_group_rate: float | None
    finite_metrics: bool
    leakage_clean: bool
    eligible: bool


@dataclass(frozen=True, slots=True)
class AmendedStageALearningRateDecision:
    method: AmendedStageAMethod
    status: StageALearningRateDecisionStatus
    selected_learning_rate: float | None
    assessments: tuple[AmendedStageAScreenAssessment, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class AmendedStageAFinalCriterion:
    name: str
    passed: bool
    observed: float | bool
    threshold: float | bool


@dataclass(frozen=True, slots=True)
class AmendedStageAFinalAssessment:
    evidence: AmendedStageAFinalEvidence
    target_paired_gain: PairedControlChange
    sentinel_paired_change: PairedControlChange
    family_reachability: float
    target_health: AmendedStageAPanelHealth
    sentinel_health: AmendedStageAPanelHealth
    final_ten_mixed_group_rate: float | None
    criteria: tuple[AmendedStageAFinalCriterion, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class AmendedStageADurationDecision:
    status: StageADurationDecisionStatus
    selected_max_updates: int | None
    assessments: tuple[AmendedStageAFinalAssessment, ...]
    automatic_extension_allowed: Literal[False]
    reason: str


__all__ = [
    "AMENDED_STAGE_A_LEARNING_RATES",
    "AmendedStageADurationDecision",
    "AmendedStageAFinalAssessment",
    "AmendedStageAFinalCriterion",
    "AmendedStageAFinalEvidence",
    "AmendedStageALearningRateDecision",
    "AmendedStageALiveEvidence",
    "AmendedStageAMethod",
    "AmendedStageAPanelHealth",
    "AmendedStageAScreenAssessment",
    "AmendedStageAScreenEvidence",
    "STAGE_A_MAXIMUM_VALID_ANSWER_TAG_DROP",
    "StageAAnswerTagPairedItemEvidence",
]
