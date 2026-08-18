"""Reducers for the prospective verifier-valid Stage-A amendment."""

from __future__ import annotations

from collections.abc import Sequence
from math import fsum, isfinite

from duraseed.run_records import TrainingMetricRecord
from duraseed.training.stage_a_amended_evidence import (
    AMENDED_STAGE_A_LEARNING_RATES,
    STAGE_A_MAXIMUM_VALID_ANSWER_TAG_DROP,
    AmendedStageADurationDecision,
    AmendedStageAFinalAssessment,
    AmendedStageAFinalCriterion,
    AmendedStageAFinalEvidence,
    AmendedStageALearningRateDecision,
    AmendedStageALiveEvidence,
    AmendedStageAMethod,
    AmendedStageAPanelHealth,
    AmendedStageAScreenAssessment,
    AmendedStageAScreenEvidence,
    StageAAnswerTagPairedItemEvidence,
)
from duraseed.training.stage_a_calibration import (
    STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE,
    STAGE_A_MAXIMUM_LENGTH_STOP_INCREASE,
    STAGE_A_MAXIMUM_LENGTH_STOP_RATE,
    STAGE_A_MAXIMUM_SENTINEL_DROP,
    STAGE_A_MINIMUM_FAMILY_REACHABILITY,
    STAGE_A_SELECTED_MAX_UPDATES,
    StageACalibrationEvidenceError,
    StageADurationDecisionStatus,
    StageALearningRateDecisionStatus,
)
from duraseed.training.teacher_dose import (
    PairedControlChange,
    summarize_paired_control_change,
)


def _mean_boolean(
    items: Sequence[StageAAnswerTagPairedItemEvidence], name: str, *, current: bool
) -> float:
    values: list[bool] = []
    prefix = "current" if current else "origin"
    for item in items:
        values.extend(getattr(item, f"{prefix}_{name}"))
    return fsum(values) / len(values)


def _paired_change(
    items: Sequence[StageAAnswerTagPairedItemEvidence],
) -> PairedControlChange:
    return summarize_paired_control_change(
        tuple(item.origin_success_rate for item in items),
        tuple(item.current_success_rate for item in items),
    )


def _mixed_rate(
    method: AmendedStageAMethod,
    metrics: Sequence[TrainingMetricRecord],
    steps: Sequence[int],
) -> float | None:
    if method != "B-G":
        return None
    by_step = {row.training_step: row for row in metrics}
    try:
        values = tuple(
            float(by_step[step].metrics["mixed_group_rate"]) for step in steps
        )
    except (KeyError, TypeError, ValueError) as error:
        raise StageACalibrationEvidenceError(
            "amended B-G evidence lacks mixed-group health"
        ) from error
    if any(not isfinite(value) or not 0 <= value <= 1 for value in values):
        raise StageACalibrationEvidenceError(
            "amended B-G mixed-group health is invalid"
        )
    return fsum(values) / len(values)


def _panel_health(
    items: Sequence[StageAAnswerTagPairedItemEvidence],
) -> AmendedStageAPanelHealth:
    origin_tag = _mean_boolean(items, "valid_answer_tags", current=False)
    current_tag = _mean_boolean(items, "valid_answer_tags", current=True)
    origin_length = _mean_boolean(items, "length_stops", current=False)
    current_length = _mean_boolean(items, "length_stops", current=True)
    length_increase = current_length - origin_length
    return AmendedStageAPanelHealth(
        origin_tag,
        current_tag,
        origin_tag - current_tag,
        current_tag >= origin_tag - STAGE_A_MAXIMUM_VALID_ANSWER_TAG_DROP,
        origin_length,
        current_length,
        length_increase,
        current_length <= STAGE_A_MAXIMUM_LENGTH_STOP_RATE
        and length_increase <= STAGE_A_MAXIMUM_LENGTH_STOP_INCREASE,
    )


def assess_amended_stage_a_screen(
    evidence: AmendedStageAScreenEvidence,
) -> AmendedStageAScreenAssessment:
    """Gate target valid-tag retention; retain sentinel health descriptively."""

    target = _panel_health(evidence.target_items)
    sentinel = _panel_health(evidence.sentinel_items)
    mixed = _mixed_rate(evidence.method, evidence.metrics, range(1, 11))
    finite = all(
        isfinite(value) for row in evidence.metrics for value in row.metrics.values()
    )
    eligible = (
        target.valid_answer_tag_retention_passed
        and target.length_health_passed
        and finite
        and evidence.leakage_clean
        and (mixed is None or mixed >= STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE)
    )
    return AmendedStageAScreenAssessment(
        evidence,
        _paired_change(evidence.target_items),
        target,
        sentinel,
        mixed,
        finite,
        evidence.leakage_clean,
        eligible,
    )


def decide_amended_stage_a_screen(
    method: AmendedStageAMethod,
    evidence: Sequence[AmendedStageAScreenEvidence],
) -> AmendedStageALearningRateDecision:
    rows = tuple(evidence)
    if len(rows) != 1 or rows[0].method != method:
        raise StageACalibrationEvidenceError(
            "amended Stage-A requires one fixed screen per method"
        )
    assessment = assess_amended_stage_a_screen(rows[0])
    selected = AMENDED_STAGE_A_LEARNING_RATES[method]
    return AmendedStageALearningRateDecision(
        method,
        (
            StageALearningRateDecisionStatus.SELECTED
            if assessment.eligible
            else StageALearningRateDecisionStatus.NO_ELIGIBLE_CANDIDATE
        ),
        selected if assessment.eligible else None,
        (assessment,),
        (
            "fixed coordinate passed the prospective step-10 health gate"
            if assessment.eligible
            else "fixed coordinate failed the prospective step-10 health gate"
        ),
    )


def assess_amended_stage_a_final(
    evidence: AmendedStageAFinalEvidence,
) -> AmendedStageAFinalAssessment:
    target_change = _paired_change(evidence.target_items)
    sentinel_change = _paired_change(evidence.sentinel_items)
    target_health = _panel_health(evidence.target_items)
    sentinel_health = _panel_health(evidence.sentinel_items)
    families = {item.family_id for item in evidence.target_items}
    reached = {
        item.family_id for item in evidence.target_items if any(item.current_successes)
    }
    reachability = len(reached) / len(families)
    mixed = _mixed_rate(evidence.method, evidence.metrics, range(41, 51))
    finite = all(
        isfinite(value) for row in evidence.metrics for value in row.metrics.values()
    )
    lower = target_change.approximate_95_interval[0]
    criteria = [
        AmendedStageAFinalCriterion(
            "positive_target_paired_gain",
            target_change.mean_change > 0,
            target_change.mean_change,
            0.0,
        ),
        AmendedStageAFinalCriterion(
            "target_gain_approximate_95_lower_bound", lower > 0, lower, 0.0
        ),
        AmendedStageAFinalCriterion(
            "catastrophic_sentinel_drop",
            -sentinel_change.mean_change <= STAGE_A_MAXIMUM_SENTINEL_DROP,
            -sentinel_change.mean_change,
            STAGE_A_MAXIMUM_SENTINEL_DROP,
        ),
        AmendedStageAFinalCriterion(
            "family_reachability",
            reachability >= STAGE_A_MINIMUM_FAMILY_REACHABILITY,
            reachability,
            STAGE_A_MINIMUM_FAMILY_REACHABILITY,
        ),
        AmendedStageAFinalCriterion(
            "target_valid_answer_tag_retention",
            target_health.valid_answer_tag_retention_passed,
            target_health.current_valid_answer_tag_rate,
            target_health.origin_valid_answer_tag_rate
            - STAGE_A_MAXIMUM_VALID_ANSWER_TAG_DROP,
        ),
        AmendedStageAFinalCriterion(
            "absolute_length_stop_rate",
            target_health.current_length_stop_rate <= STAGE_A_MAXIMUM_LENGTH_STOP_RATE,
            target_health.current_length_stop_rate,
            STAGE_A_MAXIMUM_LENGTH_STOP_RATE,
        ),
        AmendedStageAFinalCriterion(
            "length_stop_increase",
            target_health.length_stop_increase <= STAGE_A_MAXIMUM_LENGTH_STOP_INCREASE,
            target_health.length_stop_increase,
            STAGE_A_MAXIMUM_LENGTH_STOP_INCREASE,
        ),
        AmendedStageAFinalCriterion("finite_metrics", finite, finite, True),
        AmendedStageAFinalCriterion(
            "leakage_clean", evidence.leakage_clean, evidence.leakage_clean, True
        ),
    ]
    if mixed is not None:
        criteria.append(
            AmendedStageAFinalCriterion(
                "final_ten_mixed_group_rate",
                mixed >= STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE,
                mixed,
                STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE,
            )
        )
    frozen = tuple(criteria)
    return AmendedStageAFinalAssessment(
        evidence,
        target_change,
        sentinel_change,
        reachability,
        target_health,
        sentinel_health,
        mixed,
        frozen,
        all(row.passed for row in frozen),
    )


def _origin_signature(
    items: Sequence[StageAAnswerTagPairedItemEvidence],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (item.coordinate, item.origin_observations)
        for item in sorted(items, key=lambda item: item.task_id)
    )


def _item_identity(
    items: Sequence[StageAAnswerTagPairedItemEvidence],
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((item.task_id, item.family_id) for item in items))


def decide_amended_stage_a_duration(
    decisions: Sequence[AmendedStageALearningRateDecision],
    final_evidence: Sequence[AmendedStageAFinalEvidence],
) -> AmendedStageADurationDecision:
    by_method = {row.method: row for row in decisions}
    if set(by_method) != set(AMENDED_STAGE_A_LEARNING_RATES) or any(
        row.status is not StageALearningRateDecisionStatus.SELECTED
        for row in by_method.values()
    ):
        return AmendedStageADurationDecision(
            StageADurationDecisionStatus.SCREENING_NOT_READY,
            None,
            (),
            False,
            "both fixed arms must pass step-10 health before continuation",
        )
    finals = tuple(final_evidence)
    final_by_method = {row.method: row for row in finals}
    if len(final_by_method) != len(finals):
        raise StageACalibrationEvidenceError("duplicate amended Stage-A final")
    assessments = tuple(
        assess_amended_stage_a_final(final_by_method[method])
        for method in ("B-S", "B-G")
        if method in final_by_method
    )
    if set(final_by_method) != set(AMENDED_STAGE_A_LEARNING_RATES):
        return AmendedStageADurationDecision(
            StageADurationDecisionStatus.FINAL_EVIDENCE_INCOMPLETE,
            None,
            assessments,
            False,
            "step-50 evidence is required for both fixed arms",
        )
    for method, final in final_by_method.items():
        if final.learning_rate != by_method[method].selected_learning_rate:
            raise StageACalibrationEvidenceError(
                "amended final changed its fixed learning rate"
            )
        screen = by_method[method].assessments[0].evidence
        if (
            screen.target_panel_id != final.target_panel_id
            or screen.sentinel_panel_id != final.sentinel_panel_id
            or screen.origin_sampler_checkpoint_path
            != final.origin_sampler_checkpoint_path
            or _item_identity(screen.target_items) != _item_identity(final.target_items)
            or _item_identity(screen.sentinel_items)
            != _item_identity(final.sentinel_items)
        ):
            raise StageACalibrationEvidenceError(
                "amended final changed its paired screen panels"
            )
    bs, bg = final_by_method["B-S"], final_by_method["B-G"]
    if (
        bs.target_panel_id != bg.target_panel_id
        or bs.sentinel_panel_id != bg.sentinel_panel_id
        or bs.origin_sampler_checkpoint_path != bg.origin_sampler_checkpoint_path
        or _origin_signature(bs.target_items) != _origin_signature(bg.target_items)
        or _origin_signature(bs.sentinel_items) != _origin_signature(bg.sentinel_items)
    ):
        raise StageACalibrationEvidenceError(
            "amended arms do not share their paired M0 origin"
        )
    passed = all(row.passed for row in assessments)
    return AmendedStageADurationDecision(
        (
            StageADurationDecisionStatus.FROZEN
            if passed
            else StageADurationDecisionStatus.NOT_FROZEN
        ),
        STAGE_A_SELECTED_MAX_UPDATES if passed else None,
        assessments,
        False,
        (
            "both fixed arms passed the step-50 gates"
            if passed
            else "at least one fixed arm failed; stop without extension"
        ),
    )


__all__ = [
    "AMENDED_STAGE_A_LEARNING_RATES",
    "AmendedStageADurationDecision",
    "AmendedStageAFinalAssessment",
    "AmendedStageAFinalEvidence",
    "AmendedStageALearningRateDecision",
    "AmendedStageALiveEvidence",
    "AmendedStageAPanelHealth",
    "AmendedStageAScreenAssessment",
    "AmendedStageAScreenEvidence",
    "STAGE_A_MAXIMUM_VALID_ANSWER_TAG_DROP",
    "StageAAnswerTagPairedItemEvidence",
    "assess_amended_stage_a_final",
    "assess_amended_stage_a_screen",
    "decide_amended_stage_a_duration",
    "decide_amended_stage_a_screen",
]
