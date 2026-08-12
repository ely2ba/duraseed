from __future__ import annotations

from dataclasses import replace
from math import isfinite

import pytest

from duraseed.run_records import TrainingMetricRecord
from duraseed.training import (
    STAGE_A_LEARNING_RATE_GRIDS,
    StageACalibrationEvidenceError,
    StageADurationDecisionStatus,
    StageAFinalEvidence,
    StageALearningRateDecisionStatus,
    StageAPairedItemEvidence,
    StageAScreenEvidence,
    assess_stage_a_final,
    assess_stage_a_screen,
    decide_stage_a_duration,
    select_stage_a_learning_rate,
)


def _item(
    *,
    task_id: str,
    family_id: str,
    samples: int,
    current_success: bool,
    origin_success: bool = False,
    wrapper: bool = True,
    origin_length_stop: bool = False,
    current_length_stop: bool = False,
) -> StageAPairedItemEvidence:
    item_number = int(task_id.rsplit("-", 1)[1])
    seeds = tuple(
        10_000 * samples + item_number * samples + index for index in range(samples)
    )
    return StageAPairedItemEvidence(
        task_id=task_id,
        family_id=family_id,
        sampling_seeds=seeds,
        origin_successes=(origin_success,) * samples,
        current_successes=(current_success,) * samples,
        origin_wrapper_compliance=(True,) * samples,
        current_wrapper_compliance=(wrapper,) * samples,
        origin_length_stops=(origin_length_stop,) * samples,
        current_length_stops=(current_length_stop,) * samples,
    )


def _panel_items(
    prefix: str,
    *,
    samples: int,
    successful_indices: set[int],
    origin_successful_indices: set[int] | None = None,
    wrapper_failures: set[int] | None = None,
    origin_length_stops: set[int] | None = None,
    current_length_stops: set[int] | None = None,
) -> tuple[StageAPairedItemEvidence, ...]:
    origin_successful_indices = origin_successful_indices or set()
    wrapper_failures = wrapper_failures or set()
    origin_length_stops = origin_length_stops or set()
    current_length_stops = current_length_stops or set()
    return tuple(
        _item(
            task_id=f"{prefix}-{index}",
            family_id=f"family-{index // 8}",
            samples=samples,
            current_success=index in successful_indices,
            origin_success=index in origin_successful_indices,
            wrapper=index not in wrapper_failures,
            origin_length_stop=index in origin_length_stops,
            current_length_stop=index in current_length_stops,
        )
        for index in range(96)
    )


def _metric(step: int, **values: float) -> TrainingMetricRecord:
    return TrainingMetricRecord(
        phase="stage_a",
        training_step=step,
        metrics=values or {"loss": 1.0},
    )


def _screen(
    method: str,
    learning_rate: float,
    successes: set[int],
    *,
    wrapper_failures: set[int] | None = None,
    origin_length_stops: set[int] | None = None,
    current_length_stops: set[int] | None = None,
    leakage_clean: bool = True,
) -> StageAScreenEvidence:
    return StageAScreenEvidence(
        method=method,  # type: ignore[arg-type]
        learning_rate=learning_rate,
        target_panel_id="frozen-target",
        origin_sampler_checkpoint_path="tinker://origin",
        candidate_sampler_checkpoint_path=f"tinker://{method}/{learning_rate}",
        target_items=_panel_items(
            "target",
            samples=1,
            successful_indices=successes,
            wrapper_failures=wrapper_failures,
            origin_length_stops=origin_length_stops,
            current_length_stops=current_length_stops,
        ),
        metrics=(_metric(10),),
        leakage_clean=leakage_clean,
    )


def _selected_lr_decision(method: str):
    grid = STAGE_A_LEARNING_RATE_GRIDS[method]  # type: ignore[index]
    screens = tuple(
        _screen(method, learning_rate, set(range(40 - index * 10)))
        for index, learning_rate in enumerate(grid)
    )
    return select_stage_a_learning_rate(method, screens)  # type: ignore[arg-type]


def _final_metrics(method: str, *, mixed_rate: float = 0.25):
    if method == "B-S":
        return (_metric(25), _metric(50))
    return (_metric(25),) + tuple(
        _metric(step, mixed_group_rate=mixed_rate) for step in range(41, 51)
    )


def _final(
    method: str,
    learning_rate: float,
    *,
    target_successes: set[int] | None = None,
    sentinel_origin_successes: set[int] | None = None,
    sentinel_current_successes: set[int] | None = None,
    wrapper_failures: set[int] | None = None,
    origin_length_stops: set[int] | None = None,
    current_length_stops: set[int] | None = None,
    mixed_rate: float = 0.25,
    leakage_clean: bool = True,
) -> StageAFinalEvidence:
    target_successes = target_successes or set(range(0, 96, 2))
    sentinel_origin_successes = sentinel_origin_successes or set()
    sentinel_current_successes = sentinel_current_successes or set()
    return StageAFinalEvidence(
        method=method,  # type: ignore[arg-type]
        learning_rate=learning_rate,
        target_panel_id="frozen-target",
        sentinel_panel_id="frozen-sentinel",
        origin_sampler_checkpoint_path="tinker://origin",
        candidate_sampler_checkpoint_path=f"tinker://{method}/final",
        target_items=_panel_items(
            "target",
            samples=2,
            successful_indices=target_successes,
            wrapper_failures=wrapper_failures,
            origin_length_stops=origin_length_stops,
            current_length_stops=current_length_stops,
        ),
        sentinel_items=_panel_items(
            "sentinel",
            samples=1,
            successful_indices=sentinel_current_successes,
            origin_successful_indices=sentinel_origin_successes,
        ),
        metrics=_final_metrics(method, mixed_rate=mixed_rate),
        leakage_clean=leakage_clean,
    )


def test_screen_requires_the_frozen_step10_96_by_one_design() -> None:
    evidence = _screen("B-S", 1e-4, set(range(20)))
    assessment = assess_stage_a_screen(evidence)

    assert len(evidence.target_items) == 96
    assert {item.sample_count for item in evidence.target_items} == {1}
    assert assessment.target_paired_gain.item_count == 96
    assert assessment.eligible

    with pytest.raises(StageACalibrationEvidenceError, match="exactly 96"):
        replace(evidence, target_items=evidence.target_items[:-1])
    with pytest.raises(ValueError, match="step 10"):
        replace(evidence, training_step=11)  # type: ignore[arg-type]


def test_lr_selection_uses_paired_two_se_tie_then_smaller_lr() -> None:
    screens = (
        _screen("B-S", 1e-4, set(range(39))),
        _screen("B-S", 3e-4, set(range(40))),
        _screen("B-S", 1e-3, set(range(10))),
    )

    decision = select_stage_a_learning_rate("B-S", screens)

    assert decision.status is StageALearningRateDecisionStatus.SELECTED
    assert decision.selected_learning_rate == 1e-4
    assert decision.tied_learning_rates == (1e-4, 3e-4)


def test_screen_operational_gates_exclude_wrapper_and_catastrophic_length_failures() -> (
    None
):
    wrapper_failure = assess_stage_a_screen(
        _screen("B-S", 1e-4, set(range(30)), wrapper_failures={0, 1, 2})
    )
    absolute_length_failure = assess_stage_a_screen(
        _screen(
            "B-S",
            1e-4,
            set(range(30)),
            current_length_stops=set(range(49)),
        )
    )
    regression_failure = assess_stage_a_screen(
        _screen(
            "B-S",
            1e-4,
            set(range(30)),
            current_length_stops=set(range(10)),
        )
    )

    assert wrapper_failure.wrapper_compliance == 93 / 96
    assert not wrapper_failure.eligible
    assert absolute_length_failure.current_length_stop_rate > 0.50
    assert not absolute_length_failure.catastrophic_operational_stability_passed
    assert regression_failure.length_stop_increase > 0.10
    assert not regression_failure.eligible


def test_lr_grid_must_be_complete_and_use_identical_frozen_origin_evidence() -> None:
    complete = tuple(
        _screen("B-G", learning_rate, set(range(20)))
        for learning_rate in STAGE_A_LEARNING_RATE_GRIDS["B-G"]
    )
    incomplete = select_stage_a_learning_rate("B-G", complete[:-1])
    assert incomplete.status is StageALearningRateDecisionStatus.INCOMPLETE

    changed_first_item = replace(
        complete[-1].target_items[0],
        origin_successes=(True,),
    )
    changed = replace(
        complete[-1],
        target_items=(changed_first_item, *complete[-1].target_items[1:]),
    )
    with pytest.raises(StageACalibrationEvidenceError, match="frozen target/origin"):
        select_stage_a_learning_rate("B-G", (*complete[:-1], changed))


def test_lr_candidates_require_distinct_nonorigin_checkpoint_paths() -> None:
    complete = tuple(
        _screen("B-G", learning_rate, set(range(20)))
        for learning_rate in STAGE_A_LEARNING_RATE_GRIDS["B-G"]
    )
    with pytest.raises(StageACalibrationEvidenceError, match="differ from its origin"):
        replace(
            complete[0],
            candidate_sampler_checkpoint_path=complete[
                0
            ].origin_sampler_checkpoint_path,
        )

    relabeled = replace(
        complete[-1],
        candidate_sampler_checkpoint_path=complete[0].candidate_sampler_checkpoint_path,
    )
    with pytest.raises(StageACalibrationEvidenceError, match="distinct sampler"):
        select_stage_a_learning_rate("B-G", (*complete[:-1], relabeled))


def test_final_assessment_reports_paired_uncertainty_and_all_frozen_gates() -> None:
    evidence = _final("B-G", 1e-5)

    assessment = assess_stage_a_final(evidence)

    assert assessment.passed
    assert assessment.target_paired_gain.mean_change == 0.5
    assert assessment.target_paired_gain.approximate_95_interval[0] > 0
    assert assessment.sentinel_paired_change.mean_change == 0
    assert isfinite(assessment.sentinel_paired_change.standard_error)
    assert assessment.family_reachability == 1.0
    assert assessment.wrapper_compliance == 1.0
    assert assessment.final_ten_mixed_group_rate == 0.25


def test_final_gate_rejects_catastrophic_sentinel_drop_and_low_bg_mixed_rate() -> None:
    dropped = assess_stage_a_final(
        _final(
            "B-G",
            1e-5,
            sentinel_origin_successes=set(range(10)),
            sentinel_current_successes=set(),
            mixed_rate=0.19,
        )
    )

    assert dropped.sentinel_paired_change.mean_change == pytest.approx(-10 / 96)
    assert dropped.sentinel_paired_change.standard_error > 0
    criteria = {criterion.name: criterion for criterion in dropped.criteria}
    assert not criteria["catastrophic_sentinel_drop"].passed
    assert not criteria["final_ten_mixed_group_rate"].passed
    assert not dropped.passed


def test_final_gate_requires_positive_gain_with_positive_approximate_lower_bound() -> (
    None
):
    assessment = assess_stage_a_final(_final("B-S", 1e-4, target_successes={0}))
    criteria = {criterion.name: criterion for criterion in assessment.criteria}

    assert assessment.target_paired_gain.mean_change > 0
    assert assessment.target_paired_gain.approximate_95_interval[0] < 0
    assert criteria["positive_target_paired_gain"].passed
    assert not criteria["target_gain_approximate_95_lower_bound"].passed
    assert not assessment.passed


def test_duration_freezes_at_50_only_when_both_selected_arms_pass() -> None:
    decisions = (_selected_lr_decision("B-S"), _selected_lr_decision("B-G"))
    evidence = tuple(
        _final(method, decision.selected_learning_rate)
        for method, decision in zip(("B-S", "B-G"), decisions, strict=True)
    )

    result = decide_stage_a_duration(decisions, evidence)

    assert result.status is StageADurationDecisionStatus.FROZEN
    assert result.selected_max_updates == 50
    assert not result.automatic_extension_allowed
    assert {row.evidence.method for row in result.assessments} == {"B-S", "B-G"}


def test_failed_arm_stops_without_automatic_extension() -> None:
    decisions = (_selected_lr_decision("B-S"), _selected_lr_decision("B-G"))
    evidence = (
        _final("B-S", decisions[0].selected_learning_rate),
        _final("B-G", decisions[1].selected_learning_rate, leakage_clean=False),
    )

    result = decide_stage_a_duration(decisions, evidence)

    assert result.status is StageADurationDecisionStatus.NOT_FROZEN
    assert result.selected_max_updates is None
    assert not result.automatic_extension_allowed
    assert "stop without auto-extension" in result.reason


def test_final_evidence_requires_steps25_and50_and_bg_final_ten_update_rates() -> None:
    evidence = _final("B-S", 1e-4)
    with pytest.raises(StageACalibrationEvidenceError, match="required training steps"):
        replace(evidence, metrics=(_metric(50),))

    bg = _final("B-G", 1e-5)
    missing_step = replace(
        bg,
        metrics=tuple(row for row in bg.metrics if row.training_step != 41),
    )
    with pytest.raises(StageACalibrationEvidenceError, match="final ten updates"):
        assess_stage_a_final(missing_step)


def test_final_methods_must_share_frozen_paired_panels() -> None:
    decisions = (_selected_lr_decision("B-S"), _selected_lr_decision("B-G"))
    bs = _final("B-S", decisions[0].selected_learning_rate)
    bg = _final("B-G", decisions[1].selected_learning_rate)
    changed_origin = replace(
        bg.target_items[0],
        origin_successes=(True, True),
    )
    bg = replace(bg, target_items=(changed_origin, *bg.target_items[1:]))

    with pytest.raises(StageACalibrationEvidenceError, match="frozen paired"):
        decide_stage_a_duration(decisions, (bs, bg))
