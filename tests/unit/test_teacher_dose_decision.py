from __future__ import annotations

from dataclasses import replace

import pytest

from duraseed.config import load_pilot_config
from duraseed.training import (
    GateStatus,
    TeacherDoseDecisionStatus,
    TeacherDoseEvidenceError,
    assess_teacher_dose,
    decide_teacher_dose,
    summarize_paired_control_change,
)
from tests.unit.teacher_dose_fixtures import REPOSITORY_ROOT, _assessment, _summary


def test_paired_control_change_reports_uncertainty_without_equivalence_claim() -> None:
    summary = summarize_paired_control_change(
        [0.25, 0.50, 0.75, 0.25],
        [0.25, 0.25, 0.75, 0.50],
    )

    assert summary.item_count == 4
    assert summary.mean_change == 0.0
    assert summary.standard_error > 0
    lower, upper = summary.approximate_95_interval
    assert lower < 0 < upper


def test_declared_quantitative_gate_boundaries_are_inclusive() -> None:
    assessment = _assessment(1)

    assert assessment.status is GateStatus.PASSED
    assert all(
        criterion.status is GateStatus.PASSED for criterion in assessment.criteria
    )


def test_catastrophic_control_guard_rejects_only_drop_greater_than_ten_points() -> None:
    config = load_pilot_config(
        REPOSITORY_ROOT / "duraseed_pilot_config.yaml"
    ).teacher_dose
    at_boundary = assess_teacher_dose(
        _summary(1, sentinel_change=-0.10),
        config,
        optimization_stable=True,
        leakage_clean=True,
    )
    beyond_boundary = assess_teacher_dose(
        _summary(1, sentinel_change=-0.11),
        config,
        optimization_stable=True,
        leakage_clean=True,
    )

    assert at_boundary.status is GateStatus.PASSED
    assert beyond_boundary.status is GateStatus.FAILED


def test_missing_qualitative_evidence_keeps_dose_unresolved() -> None:
    config = load_pilot_config(
        REPOSITORY_ROOT / "duraseed_pilot_config.yaml"
    ).teacher_dose
    assessment = assess_teacher_dose(
        _summary(2),
        config,
        optimization_stable=None,
        leakage_clean=True,
    )

    assert assessment.status is GateStatus.UNRESOLVED
    assert (
        next(
            criterion
            for criterion in assessment.criteria
            if criterion.name == "optimization_stable"
        ).status
        is GateStatus.UNRESOLVED
    )


def test_smallest_passing_dose_requires_and_then_accepts_distinct_seed() -> None:
    calibration = (_assessment(1, passing=False), _assessment(2, passing=True))

    pending = decide_teacher_dose([1, 2, 4, 8, 16], calibration)
    selected = decide_teacher_dose(
        [1, 2, 4, 8, 16],
        calibration,
        verification_assessment=_assessment(2, seed=37, passing=True),
    )

    assert pending.status is TeacherDoseDecisionStatus.VERIFICATION_REQUIRED
    assert pending.candidate_dose == 2
    assert pending.selected_dose is None
    assert selected.status is TeacherDoseDecisionStatus.SELECTED
    assert selected.calibration_seed == 17
    assert selected.verification_seed == 37
    assert selected.selected_dose == 2


def test_verification_failure_does_not_escalate_to_a_higher_dose() -> None:
    decision = decide_teacher_dose(
        [1, 2, 4, 8, 16],
        (_assessment(1, passing=False), _assessment(2, passing=True)),
        verification_assessment=_assessment(2, seed=37, passing=False),
    )

    assert decision.status is TeacherDoseDecisionStatus.VERIFICATION_FAILED
    assert decision.candidate_dose == 2
    assert decision.selected_dose is None
    assert "no automatic escalation" in decision.reason


def test_unbroken_grid_prefix_is_required_before_selection() -> None:
    decision = decide_teacher_dose(
        [1, 2, 4, 8, 16],
        (_assessment(2, passing=True),),
    )

    assert decision.status is TeacherDoseDecisionStatus.CALIBRATION_INCOMPLETE
    assert decision.candidate_dose is None


def test_verification_must_use_the_candidate_dose_and_a_fresh_seed() -> None:
    calibration = (_assessment(1, passing=True),)

    with pytest.raises(TeacherDoseEvidenceError, match="must differ"):
        decide_teacher_dose(
            [1, 2, 4, 8, 16],
            calibration,
            verification_assessment=_assessment(1, seed=17),
        )
    with pytest.raises(TeacherDoseEvidenceError, match="candidate dose"):
        decide_teacher_dose(
            [1, 2, 4, 8, 16],
            calibration,
            verification_assessment=_assessment(2, seed=37),
        )


def test_summary_rejects_incoherent_reward_group_rates() -> None:
    summary = _summary(1)

    with pytest.raises(ValueError, match="must sum to one"):
        replace(summary, all_zero_group_rate=0.6)
