"""Direct-M0 Stage-A screen eligibility layered over the carried reducer."""

from __future__ import annotations

from collections.abc import Sequence
from math import fsum, isfinite

from duraseed.training.stage_a_calibration import (
    STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE,
    STAGE_A_LEARNING_RATE_GRIDS,
    StageACalibrationEvidenceError,
    StageACalibrationMethod,
    StageALearningRateDecision,
    StageALearningRateDecisionStatus,
    StageAScreenAssessment,
    StageAScreenEvidence,
    assess_stage_a_screen,
    select_stage_a_learning_rate,
)
from duraseed.training.teacher_dose import summarize_paired_control_change


def screen_mean_mixed_group_rate(evidence: StageAScreenEvidence) -> float | None:
    """Return the mean over the complete ten-update B-G screen."""

    if evidence.method != "B-G":
        return None
    by_step = {row.training_step: row for row in evidence.metrics}
    try:
        rates = tuple(
            by_step[step].metrics["mixed_group_rate"] for step in range(1, 11)
        )
    except KeyError as error:
        raise StageACalibrationEvidenceError(
            "direct-M0 B-G screen requires mixed_group_rate at steps 1 through 10"
        ) from error
    if any(not isfinite(value) or not 0 <= value <= 1 for value in rates):
        raise StageACalibrationEvidenceError(
            "direct-M0 B-G screen mixed_group_rate is not a probability"
        )
    return fsum(rates) / 10


def _within_two_paired_se(
    candidate: StageAScreenAssessment, best: StageAScreenAssessment
) -> bool:
    candidate_by_task = {
        item.task_id: item.current_success_rate
        for item in candidate.evidence.target_items
    }
    best_by_task = {
        item.task_id: item.current_success_rate for item in best.evidence.target_items
    }
    task_ids = tuple(sorted(best_by_task))
    difference = summarize_paired_control_change(
        tuple(candidate_by_task[task_id] for task_id in task_ids),
        tuple(best_by_task[task_id] for task_id in task_ids),
    )
    return difference.mean_change <= 2.0 * difference.standard_error


def select_direct_m0_learning_rate(
    method: StageACalibrationMethod,
    evidence: Sequence[StageAScreenEvidence],
) -> StageALearningRateDecision:
    """Add the prospective direct-M0 B-G rollout-health gate."""

    base = select_stage_a_learning_rate(method, evidence)
    if method != "B-G" or base.status is StageALearningRateDecisionStatus.INCOMPLETE:
        return base
    assessments = tuple(
        assess_stage_a_screen(row)
        for row in sorted(evidence, key=lambda value: value.learning_rate)
    )
    eligible = tuple(
        row
        for row in assessments
        if row.eligible
        and screen_mean_mixed_group_rate(row.evidence)
        >= STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE
    )
    if not eligible:
        return StageALearningRateDecision(
            method=method,
            status=StageALearningRateDecisionStatus.NO_ELIGIBLE_CANDIDATE,
            selected_learning_rate=None,
            tied_learning_rates=(),
            assessments=assessments,
            reason="no B-G LR passed operational and direct-M0 rollout-health gates",
        )
    best = min(
        eligible,
        key=lambda row: (
            -row.target_paired_gain.mean_change,
            row.evidence.learning_rate,
        ),
    )
    tied = tuple(
        sorted(
            row.evidence.learning_rate
            for row in eligible
            if _within_two_paired_se(row, best)
        )
    )
    if not set(tied).issubset(STAGE_A_LEARNING_RATE_GRIDS[method]):
        raise StageACalibrationEvidenceError("direct-M0 B-G selection left its LR grid")
    return StageALearningRateDecision(
        method=method,
        status=StageALearningRateDecisionStatus.SELECTED,
        selected_learning_rate=tied[0],
        tied_learning_rates=tied,
        assessments=assessments,
        reason=(
            "selected among B-G screens with mean mixed-group rate at least 0.20, "
            "then applied the carried paired two-SE tie rule"
        ),
    )


__all__ = ["screen_mean_mixed_group_rate", "select_direct_m0_learning_rate"]
