"""Pure Stage-A learning-rate and duration decisions for Phase 4.

The reducer deliberately has no Tinker dependency.  It accepts complete,
paired item evidence from the frozen target and sentinel panels and returns the
pre-result calibration decisions.  It never schedules more training.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import fsum, isfinite
from typing import Literal, TypeAlias

from duraseed.run_records import TrainingMetricRecord
from duraseed.training.teacher_dose import (
    PairedControlChange,
    summarize_paired_control_change,
)


StageACalibrationMethod: TypeAlias = Literal["B-S", "B-G"]

STAGE_A_SCREEN_STEP = 10
STAGE_A_CONTINUATION_STEPS = (25, 50)
STAGE_A_SELECTED_MAX_UPDATES = 50
STAGE_A_SCREEN_TARGET_ITEM_COUNT = 96
STAGE_A_FINAL_TARGET_ITEM_COUNT = 96
STAGE_A_FINAL_SENTINEL_ITEM_COUNT = 96
STAGE_A_SCREEN_SAMPLES_PER_ITEM = 1
STAGE_A_FINAL_TARGET_SAMPLES_PER_ITEM = 2
STAGE_A_FINAL_SENTINEL_SAMPLES_PER_ITEM = 1
STAGE_A_REQUIRED_WRAPPER_COMPLIANCE = 0.97
STAGE_A_MAXIMUM_LENGTH_STOP_RATE = 0.50
STAGE_A_MAXIMUM_LENGTH_STOP_INCREASE = 0.10
STAGE_A_MAXIMUM_SENTINEL_DROP = 0.10
STAGE_A_MINIMUM_FAMILY_REACHABILITY = 0.60
STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE = 0.20
STAGE_A_LEARNING_RATE_GRIDS: dict[StageACalibrationMethod, tuple[float, ...]] = {
    "B-S": (1e-4, 3e-4, 1e-3),
    "B-G": (1e-5, 3e-5, 1e-4),
}


class StageACalibrationEvidenceError(ValueError):
    """Raised when Stage-A calibration evidence is incomplete or mixed."""


class StageALearningRateDecisionStatus(StrEnum):
    INCOMPLETE = "incomplete"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"
    SELECTED = "selected"


class StageADurationDecisionStatus(StrEnum):
    SCREENING_NOT_READY = "screening_not_ready"
    FINAL_EVIDENCE_INCOMPLETE = "final_evidence_incomplete"
    NOT_FROZEN = "not_frozen"
    FROZEN = "frozen"


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _strict_bool_tuple(values: tuple[bool, ...], name: str) -> tuple[bool, ...]:
    if not isinstance(values, tuple) or any(
        type(value) is not bool for value in values
    ):
        raise TypeError(f"{name} must be a tuple of bool values")
    return values


def _probability(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite probability")
    return float(value)


def _learning_rate(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError("learning_rate must be positive and finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class StageAPairedItemEvidence:
    """Paired origin/current samples for one immutable panel item."""

    task_id: str
    family_id: str
    sampling_seeds: tuple[int, ...]
    origin_successes: tuple[bool, ...]
    current_successes: tuple[bool, ...]
    origin_wrapper_compliance: tuple[bool, ...]
    current_wrapper_compliance: tuple[bool, ...]
    origin_length_stops: tuple[bool, ...]
    current_length_stops: tuple[bool, ...]

    def __post_init__(self) -> None:
        _nonempty(self.task_id, "task_id")
        _nonempty(self.family_id, "family_id")
        if (
            not isinstance(self.sampling_seeds, tuple)
            or not self.sampling_seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in self.sampling_seeds
            )
            or len(set(self.sampling_seeds)) != len(self.sampling_seeds)
        ):
            raise ValueError(
                "sampling_seeds must be a nonempty tuple of unique nonnegative integers"
            )
        fields = (
            ("origin_successes", self.origin_successes),
            ("current_successes", self.current_successes),
            ("origin_wrapper_compliance", self.origin_wrapper_compliance),
            ("current_wrapper_compliance", self.current_wrapper_compliance),
            ("origin_length_stops", self.origin_length_stops),
            ("current_length_stops", self.current_length_stops),
        )
        for name, values in fields:
            _strict_bool_tuple(values, name)
            if len(values) != len(self.sampling_seeds):
                raise ValueError(f"{name} must match the paired sampling-seed count")

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
            self.origin_wrapper_compliance,
            self.origin_length_stops,
        )


def _validate_items(
    items: tuple[StageAPairedItemEvidence, ...],
    *,
    label: str,
    expected_item_count: int,
    expected_samples_per_item: int,
) -> None:
    if not isinstance(items, tuple) or any(
        not isinstance(item, StageAPairedItemEvidence) for item in items
    ):
        raise TypeError(f"{label} must be a tuple of StageAPairedItemEvidence")
    if len(items) != expected_item_count:
        raise StageACalibrationEvidenceError(
            f"{label} requires exactly {expected_item_count} items"
        )
    if len({item.task_id for item in items}) != len(items):
        raise StageACalibrationEvidenceError(f"{label} contains duplicate task IDs")
    if any(item.sample_count != expected_samples_per_item for item in items):
        raise StageACalibrationEvidenceError(
            f"{label} requires exactly {expected_samples_per_item} samples per item"
        )


def _validate_method(method: StageACalibrationMethod) -> None:
    if method not in STAGE_A_LEARNING_RATE_GRIDS:
        raise ValueError(f"unsupported Stage-A calibration method: {method!r}")


def _validate_metrics(
    metrics: tuple[TrainingMetricRecord, ...], *, required_steps: tuple[int, ...]
) -> None:
    if (
        not isinstance(metrics, tuple)
        or not metrics
        or any(not isinstance(row, TrainingMetricRecord) for row in metrics)
    ):
        raise TypeError("metrics must be a nonempty tuple of TrainingMetricRecord")
    if any(row.phase != "stage_a" for row in metrics):
        raise StageACalibrationEvidenceError("all metrics must be Stage-A records")
    steps = tuple(row.training_step for row in metrics)
    if len(set(steps)) != len(steps):
        raise StageACalibrationEvidenceError("metric training steps must be unique")
    if not set(required_steps).issubset(steps):
        raise StageACalibrationEvidenceError(
            f"metrics omit required training steps {required_steps}"
        )
    if any(not isfinite(value) for row in metrics for value in row.metrics.values()):
        raise StageACalibrationEvidenceError("training metrics must all be finite")


@dataclass(frozen=True, slots=True)
class StageAScreenEvidence:
    """Complete step-10 evidence for one method/LR coordinate."""

    method: StageACalibrationMethod
    learning_rate: float
    target_panel_id: str
    origin_sampler_checkpoint_path: str
    candidate_sampler_checkpoint_path: str
    target_items: tuple[StageAPairedItemEvidence, ...]
    metrics: tuple[TrainingMetricRecord, ...]
    leakage_clean: bool
    training_step: Literal[10] = STAGE_A_SCREEN_STEP

    def __post_init__(self) -> None:
        _validate_method(self.method)
        object.__setattr__(self, "learning_rate", _learning_rate(self.learning_rate))
        _nonempty(self.target_panel_id, "target_panel_id")
        _nonempty(self.origin_sampler_checkpoint_path, "origin_sampler_checkpoint_path")
        _nonempty(
            self.candidate_sampler_checkpoint_path,
            "candidate_sampler_checkpoint_path",
        )
        if (
            self.candidate_sampler_checkpoint_path
            == self.origin_sampler_checkpoint_path
        ):
            raise StageACalibrationEvidenceError(
                "screen candidate checkpoint must differ from its origin"
            )
        if self.training_step != STAGE_A_SCREEN_STEP:
            raise ValueError("Stage-A LR screening is frozen at step 10")
        _validate_items(
            self.target_items,
            label="screen target evidence",
            expected_item_count=STAGE_A_SCREEN_TARGET_ITEM_COUNT,
            expected_samples_per_item=STAGE_A_SCREEN_SAMPLES_PER_ITEM,
        )
        _validate_metrics(self.metrics, required_steps=(STAGE_A_SCREEN_STEP,))
        if type(self.leakage_clean) is not bool:
            raise TypeError("leakage_clean must be bool")


@dataclass(frozen=True, slots=True)
class StageAFinalEvidence:
    """Complete step-50 evidence after the selected arm passed step 25."""

    method: StageACalibrationMethod
    learning_rate: float
    target_panel_id: str
    sentinel_panel_id: str
    origin_sampler_checkpoint_path: str
    candidate_sampler_checkpoint_path: str
    target_items: tuple[StageAPairedItemEvidence, ...]
    sentinel_items: tuple[StageAPairedItemEvidence, ...]
    metrics: tuple[TrainingMetricRecord, ...]
    leakage_clean: bool
    training_step: Literal[50] = STAGE_A_SELECTED_MAX_UPDATES

    def __post_init__(self) -> None:
        _validate_method(self.method)
        object.__setattr__(self, "learning_rate", _learning_rate(self.learning_rate))
        for name in (
            "target_panel_id",
            "sentinel_panel_id",
            "origin_sampler_checkpoint_path",
            "candidate_sampler_checkpoint_path",
        ):
            _nonempty(getattr(self, name), name)
        if self.target_panel_id == self.sentinel_panel_id:
            raise StageACalibrationEvidenceError(
                "target and sentinel panel identities must differ"
            )
        if self.training_step != STAGE_A_SELECTED_MAX_UPDATES:
            raise ValueError("Stage-A final calibration is frozen at step 50")
        _validate_items(
            self.target_items,
            label="final target evidence",
            expected_item_count=STAGE_A_FINAL_TARGET_ITEM_COUNT,
            expected_samples_per_item=STAGE_A_FINAL_TARGET_SAMPLES_PER_ITEM,
        )
        _validate_items(
            self.sentinel_items,
            label="final sentinel evidence",
            expected_item_count=STAGE_A_FINAL_SENTINEL_ITEM_COUNT,
            expected_samples_per_item=STAGE_A_FINAL_SENTINEL_SAMPLES_PER_ITEM,
        )
        if {item.task_id for item in self.target_items}.intersection(
            item.task_id for item in self.sentinel_items
        ):
            raise StageACalibrationEvidenceError(
                "target and sentinel item identities must be disjoint"
            )
        _validate_metrics(self.metrics, required_steps=STAGE_A_CONTINUATION_STEPS)
        if type(self.leakage_clean) is not bool:
            raise TypeError("leakage_clean must be bool")


def _item_rates(
    items: Sequence[StageAPairedItemEvidence], *, current: bool
) -> tuple[float, ...]:
    return tuple(
        item.current_success_rate if current else item.origin_success_rate
        for item in items
    )


def _mean_boolean_field(
    items: Sequence[StageAPairedItemEvidence], name: str, *, current: bool
) -> float:
    values: list[bool] = []
    prefix = "current" if current else "origin"
    field_name = f"{prefix}_{name}"
    for item in items:
        values.extend(getattr(item, field_name))
    return fsum(values) / len(values)


def _target_change(items: Sequence[StageAPairedItemEvidence]) -> PairedControlChange:
    return summarize_paired_control_change(
        _item_rates(items, current=False),
        _item_rates(items, current=True),
    )


@dataclass(frozen=True, slots=True)
class StageAScreenAssessment:
    evidence: StageAScreenEvidence
    target_paired_gain: PairedControlChange
    wrapper_compliance: float
    origin_length_stop_rate: float
    current_length_stop_rate: float
    length_stop_increase: float
    catastrophic_operational_stability_passed: bool
    finite_metrics: bool
    leakage_clean: bool
    eligible: bool


def assess_stage_a_screen(evidence: StageAScreenEvidence) -> StageAScreenAssessment:
    """Apply only the frozen operational gates to one step-10 LR screen."""

    if not isinstance(evidence, StageAScreenEvidence):
        raise TypeError("evidence must be StageAScreenEvidence")
    wrapper = _mean_boolean_field(
        evidence.target_items, "wrapper_compliance", current=True
    )
    origin_length = _mean_boolean_field(
        evidence.target_items, "length_stops", current=False
    )
    current_length = _mean_boolean_field(
        evidence.target_items, "length_stops", current=True
    )
    length_increase = current_length - origin_length
    stable = (
        current_length <= STAGE_A_MAXIMUM_LENGTH_STOP_RATE
        and length_increase <= STAGE_A_MAXIMUM_LENGTH_STOP_INCREASE
    )
    finite_metrics = all(
        isfinite(value) for row in evidence.metrics for value in row.metrics.values()
    )
    eligible = (
        wrapper >= STAGE_A_REQUIRED_WRAPPER_COMPLIANCE
        and stable
        and finite_metrics
        and evidence.leakage_clean
    )
    return StageAScreenAssessment(
        evidence=evidence,
        target_paired_gain=_target_change(evidence.target_items),
        wrapper_compliance=wrapper,
        origin_length_stop_rate=origin_length,
        current_length_stop_rate=current_length,
        length_stop_increase=length_increase,
        catastrophic_operational_stability_passed=stable,
        finite_metrics=finite_metrics,
        leakage_clean=evidence.leakage_clean,
        eligible=eligible,
    )


@dataclass(frozen=True, slots=True)
class StageALearningRateDecision:
    method: StageACalibrationMethod
    status: StageALearningRateDecisionStatus
    selected_learning_rate: float | None
    tied_learning_rates: tuple[float, ...]
    assessments: tuple[StageAScreenAssessment, ...]
    reason: str


def _frozen_screen_signature(evidence: StageAScreenEvidence) -> tuple[object, ...]:
    return (
        evidence.target_panel_id,
        evidence.origin_sampler_checkpoint_path,
        tuple(
            (item.coordinate, item.origin_observations)
            for item in sorted(evidence.target_items, key=lambda value: value.task_id)
        ),
    )


def _candidate_is_within_two_paired_se(
    candidate: StageAScreenAssessment,
    best: StageAScreenAssessment,
) -> bool:
    candidate_by_task = {
        item.task_id: item.current_success_rate
        for item in candidate.evidence.target_items
    }
    best_by_task = {
        item.task_id: item.current_success_rate for item in best.evidence.target_items
    }
    task_ids = tuple(sorted(best_by_task))
    candidate_rates = tuple(candidate_by_task[task_id] for task_id in task_ids)
    best_rates = tuple(best_by_task[task_id] for task_id in task_ids)
    best_minus_candidate = summarize_paired_control_change(
        candidate_rates,
        best_rates,
    )
    return best_minus_candidate.mean_change <= 2.0 * best_minus_candidate.standard_error


def select_stage_a_learning_rate(
    method: StageACalibrationMethod,
    evidence: Sequence[StageAScreenEvidence],
) -> StageALearningRateDecision:
    """Select a method LR on the complete grid with a paired two-SE tie rule."""

    _validate_method(method)
    rows = tuple(evidence)
    if any(not isinstance(row, StageAScreenEvidence) for row in rows):
        raise TypeError("evidence must contain StageAScreenEvidence")
    if any(row.method != method for row in rows):
        raise StageACalibrationEvidenceError("LR screen mixes Stage-A methods")
    by_lr = {row.learning_rate: row for row in rows}
    if len(by_lr) != len(rows):
        raise StageACalibrationEvidenceError("LR screen contains duplicate coordinates")
    grid = STAGE_A_LEARNING_RATE_GRIDS[method]
    if not set(by_lr).issubset(grid):
        raise StageACalibrationEvidenceError("LR screen contains an off-grid candidate")
    assessments = tuple(
        assess_stage_a_screen(by_lr[learning_rate])
        for learning_rate in grid
        if learning_rate in by_lr
    )
    if set(by_lr) != set(grid):
        return StageALearningRateDecision(
            method=method,
            status=StageALearningRateDecisionStatus.INCOMPLETE,
            selected_learning_rate=None,
            tied_learning_rates=(),
            assessments=assessments,
            reason="the complete frozen learning-rate grid has not been assessed",
        )
    candidate_paths = {row.candidate_sampler_checkpoint_path for row in rows}
    if len(candidate_paths) != len(rows):
        raise StageACalibrationEvidenceError(
            "each LR candidate requires a distinct sampler checkpoint"
        )
    signatures = {_frozen_screen_signature(row) for row in rows}
    if len(signatures) != 1:
        raise StageACalibrationEvidenceError(
            "LR candidates do not share identical frozen target/origin evidence"
        )
    eligible = tuple(value for value in assessments if value.eligible)
    if not eligible:
        return StageALearningRateDecision(
            method=method,
            status=StageALearningRateDecisionStatus.NO_ELIGIBLE_CANDIDATE,
            selected_learning_rate=None,
            tied_learning_rates=(),
            assessments=assessments,
            reason="every LR candidate failed an operational or evidence gate",
        )
    best = min(
        eligible,
        key=lambda value: (
            -value.target_paired_gain.mean_change,
            value.evidence.learning_rate,
        ),
    )
    tied = tuple(
        sorted(
            value.evidence.learning_rate
            for value in eligible
            if _candidate_is_within_two_paired_se(value, best)
        )
    )
    selected = tied[0]
    return StageALearningRateDecision(
        method=method,
        status=StageALearningRateDecisionStatus.SELECTED,
        selected_learning_rate=selected,
        tied_learning_rates=tied,
        assessments=assessments,
        reason=(
            "selected the smaller LR among candidates within two paired sampling "
            "standard errors of the best target gain"
        ),
    )


@dataclass(frozen=True, slots=True)
class StageAFinalCriterion:
    name: str
    passed: bool
    observed: float | bool
    threshold: float | bool


@dataclass(frozen=True, slots=True)
class StageAFinalAssessment:
    evidence: StageAFinalEvidence
    target_paired_gain: PairedControlChange
    sentinel_paired_change: PairedControlChange
    family_reachability: float
    wrapper_compliance: float
    origin_length_stop_rate: float
    current_length_stop_rate: float
    length_stop_increase: float
    final_ten_mixed_group_rate: float | None
    criteria: tuple[StageAFinalCriterion, ...]
    passed: bool


def _family_reachability(items: Sequence[StageAPairedItemEvidence]) -> float:
    families = {item.family_id for item in items}
    reached = {item.family_id for item in items if any(item.current_successes)}
    return len(reached) / len(families)


def _final_ten_mixed_group_rate(evidence: StageAFinalEvidence) -> float | None:
    if evidence.method != "B-G":
        return None
    by_step = {row.training_step: row for row in evidence.metrics}
    final_steps = tuple(
        range(
            STAGE_A_SELECTED_MAX_UPDATES - 9,
            STAGE_A_SELECTED_MAX_UPDATES + 1,
        )
    )
    try:
        rates = tuple(by_step[step].metrics["mixed_group_rate"] for step in final_steps)
    except KeyError as error:
        raise StageACalibrationEvidenceError(
            "B-G requires mixed_group_rate for each of the final ten updates"
        ) from error
    return fsum(_probability(value, "mixed_group_rate") for value in rates) / len(rates)


def assess_stage_a_final(evidence: StageAFinalEvidence) -> StageAFinalAssessment:
    """Assess the frozen step-50 target/sentinel calibration evidence."""

    if not isinstance(evidence, StageAFinalEvidence):
        raise TypeError("evidence must be StageAFinalEvidence")
    target_change = _target_change(evidence.target_items)
    sentinel_change = _target_change(evidence.sentinel_items)
    reachability = _family_reachability(evidence.target_items)
    wrapper = _mean_boolean_field(
        evidence.target_items, "wrapper_compliance", current=True
    )
    origin_length = _mean_boolean_field(
        evidence.target_items, "length_stops", current=False
    )
    current_length = _mean_boolean_field(
        evidence.target_items, "length_stops", current=True
    )
    length_increase = current_length - origin_length
    finite_metrics = all(
        isfinite(value) for row in evidence.metrics for value in row.metrics.values()
    )
    mixed_rate = _final_ten_mixed_group_rate(evidence)
    target_lower = target_change.approximate_95_interval[0]
    criteria = [
        StageAFinalCriterion(
            "positive_target_paired_gain",
            target_change.mean_change > 0,
            target_change.mean_change,
            0.0,
        ),
        StageAFinalCriterion(
            "target_gain_approximate_95_lower_bound",
            target_lower > 0,
            target_lower,
            0.0,
        ),
        StageAFinalCriterion(
            "catastrophic_sentinel_drop",
            -sentinel_change.mean_change <= STAGE_A_MAXIMUM_SENTINEL_DROP,
            -sentinel_change.mean_change,
            STAGE_A_MAXIMUM_SENTINEL_DROP,
        ),
        StageAFinalCriterion(
            "family_reachability",
            reachability >= STAGE_A_MINIMUM_FAMILY_REACHABILITY,
            reachability,
            STAGE_A_MINIMUM_FAMILY_REACHABILITY,
        ),
        StageAFinalCriterion(
            "wrapper_compliance",
            wrapper >= STAGE_A_REQUIRED_WRAPPER_COMPLIANCE,
            wrapper,
            STAGE_A_REQUIRED_WRAPPER_COMPLIANCE,
        ),
        StageAFinalCriterion(
            "absolute_length_stop_rate",
            current_length <= STAGE_A_MAXIMUM_LENGTH_STOP_RATE,
            current_length,
            STAGE_A_MAXIMUM_LENGTH_STOP_RATE,
        ),
        StageAFinalCriterion(
            "length_stop_increase",
            length_increase <= STAGE_A_MAXIMUM_LENGTH_STOP_INCREASE,
            length_increase,
            STAGE_A_MAXIMUM_LENGTH_STOP_INCREASE,
        ),
        StageAFinalCriterion("finite_metrics", finite_metrics, finite_metrics, True),
        StageAFinalCriterion(
            "leakage_clean",
            evidence.leakage_clean,
            evidence.leakage_clean,
            True,
        ),
    ]
    if mixed_rate is not None:
        criteria.append(
            StageAFinalCriterion(
                "final_ten_mixed_group_rate",
                mixed_rate >= STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE,
                mixed_rate,
                STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE,
            )
        )
    frozen_criteria = tuple(criteria)
    return StageAFinalAssessment(
        evidence=evidence,
        target_paired_gain=target_change,
        sentinel_paired_change=sentinel_change,
        family_reachability=reachability,
        wrapper_compliance=wrapper,
        origin_length_stop_rate=origin_length,
        current_length_stop_rate=current_length,
        length_stop_increase=length_increase,
        final_ten_mixed_group_rate=mixed_rate,
        criteria=frozen_criteria,
        passed=all(criterion.passed for criterion in frozen_criteria),
    )


@dataclass(frozen=True, slots=True)
class StageADurationDecision:
    status: StageADurationDecisionStatus
    selected_max_updates: int | None
    assessments: tuple[StageAFinalAssessment, ...]
    automatic_extension_allowed: Literal[False]
    reason: str


def _item_identity(
    items: Sequence[StageAPairedItemEvidence],
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((item.task_id, item.family_id) for item in items))


def _paired_origin_signature(
    items: Sequence[StageAPairedItemEvidence],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (item.coordinate, item.origin_observations)
        for item in sorted(items, key=lambda value: value.task_id)
    )


def decide_stage_a_duration(
    learning_rate_decisions: Sequence[StageALearningRateDecision],
    final_evidence: Sequence[StageAFinalEvidence],
) -> StageADurationDecision:
    """Freeze 50 updates only when complete B-S and B-G evidence both pass.

    Failure or missing evidence never triggers an automatic duration extension.
    """

    decisions = tuple(learning_rate_decisions)
    if any(not isinstance(row, StageALearningRateDecision) for row in decisions):
        raise TypeError(
            "learning_rate_decisions must contain StageALearningRateDecision"
        )
    by_method = {row.method: row for row in decisions}
    if len(by_method) != len(decisions):
        raise StageACalibrationEvidenceError("duplicate method LR decisions")
    required_methods = set(STAGE_A_LEARNING_RATE_GRIDS)
    if set(by_method) != required_methods or any(
        row.status is not StageALearningRateDecisionStatus.SELECTED for row in decisions
    ):
        return StageADurationDecision(
            status=StageADurationDecisionStatus.SCREENING_NOT_READY,
            selected_max_updates=None,
            assessments=(),
            automatic_extension_allowed=False,
            reason="both method-specific LR screens must select before duration freezes",
        )

    final_rows = tuple(final_evidence)
    if any(not isinstance(row, StageAFinalEvidence) for row in final_rows):
        raise TypeError("final_evidence must contain StageAFinalEvidence")
    final_by_method = {row.method: row for row in final_rows}
    if len(final_by_method) != len(final_rows):
        raise StageACalibrationEvidenceError("duplicate method final evidence")
    if not set(final_by_method).issubset(required_methods):
        raise StageACalibrationEvidenceError("unknown method final evidence")
    assessments = tuple(
        assess_stage_a_final(final_by_method[method])
        for method in ("B-S", "B-G")
        if method in final_by_method
    )
    if set(final_by_method) != required_methods:
        return StageADurationDecision(
            status=StageADurationDecisionStatus.FINAL_EVIDENCE_INCOMPLETE,
            selected_max_updates=None,
            assessments=assessments,
            automatic_extension_allowed=False,
            reason="step-50 evidence is required for both B-S and B-G",
        )
    for method, evidence in final_by_method.items():
        if evidence.learning_rate != by_method[method].selected_learning_rate:
            raise StageACalibrationEvidenceError(
                f"{method} final evidence does not use its selected learning rate"
            )

    bs = final_by_method["B-S"]
    bg = final_by_method["B-G"]
    if (
        bs.target_panel_id != bg.target_panel_id
        or bs.sentinel_panel_id != bg.sentinel_panel_id
        or bs.origin_sampler_checkpoint_path != bg.origin_sampler_checkpoint_path
        or _paired_origin_signature(bs.target_items)
        != _paired_origin_signature(bg.target_items)
        or _paired_origin_signature(bs.sentinel_items)
        != _paired_origin_signature(bg.sentinel_items)
    ):
        raise StageACalibrationEvidenceError(
            "B-S and B-G final evidence must share frozen paired panel/origin evidence"
        )
    for method, final in final_by_method.items():
        screen = next(
            assessment.evidence
            for assessment in by_method[method].assessments
            if assessment.evidence.learning_rate
            == by_method[method].selected_learning_rate
        )
        if (
            screen.target_panel_id != final.target_panel_id
            or screen.origin_sampler_checkpoint_path
            != final.origin_sampler_checkpoint_path
            or _item_identity(screen.target_items) != _item_identity(final.target_items)
        ):
            raise StageACalibrationEvidenceError(
                f"{method} final evidence changed the frozen screen target panel"
            )

    if all(assessment.passed for assessment in assessments):
        return StageADurationDecision(
            status=StageADurationDecisionStatus.FROZEN,
            selected_max_updates=STAGE_A_SELECTED_MAX_UPDATES,
            assessments=assessments,
            automatic_extension_allowed=False,
            reason="both selected method arms passed the frozen step-50 gates",
        )
    return StageADurationDecision(
        status=StageADurationDecisionStatus.NOT_FROZEN,
        selected_max_updates=None,
        assessments=assessments,
        automatic_extension_allowed=False,
        reason="at least one selected method arm failed; stop without auto-extension",
    )


__all__ = [
    "STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE",
    "STAGE_A_CONTINUATION_STEPS",
    "STAGE_A_FINAL_SENTINEL_ITEM_COUNT",
    "STAGE_A_FINAL_SENTINEL_SAMPLES_PER_ITEM",
    "STAGE_A_FINAL_TARGET_ITEM_COUNT",
    "STAGE_A_FINAL_TARGET_SAMPLES_PER_ITEM",
    "STAGE_A_LEARNING_RATE_GRIDS",
    "STAGE_A_MAXIMUM_LENGTH_STOP_INCREASE",
    "STAGE_A_MAXIMUM_LENGTH_STOP_RATE",
    "STAGE_A_MAXIMUM_SENTINEL_DROP",
    "STAGE_A_MINIMUM_FAMILY_REACHABILITY",
    "STAGE_A_REQUIRED_WRAPPER_COMPLIANCE",
    "STAGE_A_SCREEN_SAMPLES_PER_ITEM",
    "STAGE_A_SCREEN_STEP",
    "STAGE_A_SCREEN_TARGET_ITEM_COUNT",
    "STAGE_A_SELECTED_MAX_UPDATES",
    "StageACalibrationEvidenceError",
    "StageACalibrationMethod",
    "StageADurationDecision",
    "StageADurationDecisionStatus",
    "StageAFinalAssessment",
    "StageAFinalCriterion",
    "StageAFinalEvidence",
    "StageALearningRateDecision",
    "StageALearningRateDecisionStatus",
    "StageAPairedItemEvidence",
    "StageAScreenAssessment",
    "StageAScreenEvidence",
    "assess_stage_a_final",
    "assess_stage_a_screen",
    "decide_stage_a_duration",
    "select_stage_a_learning_rate",
]
