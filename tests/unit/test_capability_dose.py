from duraseed import calibration_stage_a_terminal as _calibration_stage_a_terminal
from duraseed.run_records import TrainingMetricRecord
from duraseed.training.capability_dose import (
    assess_dose_evaluation,
    decide_dose,
)
from duraseed.training.capability_dose_evidence import (
    DoseEvaluationEvidence,
    DosePanelEvidence,
)
from duraseed.training.stage_a_amended_evidence import (
    StageAAnswerTagPairedItemEvidence,
)

assert _calibration_stage_a_terminal is not None


def _panel(
    samples: int,
    successes: int,
    *,
    invalid_tags: int = 0,
    length_stops: int = 0,
    loops: int = 0,
) -> DosePanelEvidence:
    items = []
    for item in range(96):
        positions = tuple(range(item * samples, (item + 1) * samples))
        items.append(
            StageAAnswerTagPairedItemEvidence(
                f"task-{item}",
                f"family-{item % 12}",
                positions,
                (False,) * samples,
                tuple(index < successes for index in positions),
                (True,) * samples,
                tuple(index >= invalid_tags for index in positions),
                (False,) * samples,
                tuple(index < length_stops for index in positions),
            )
        )
    return DosePanelEvidence(tuple(items), loops, 96, 32.0, 1.0, 8)


def _evaluation(
    update: int,
    phase: str,
    successes: int,
    **panel_kwargs,
) -> DoseEvaluationEvidence:
    samples = 1 if phase == "cadence" else 2
    return DoseEvaluationEvidence(
        update,
        phase,
        _panel(samples, successes, **panel_kwargs),
        _panel(1, 5) if phase == "cadence" else None,
        (
            TrainingMetricRecord(
                phase="stage_a", training_step=update, metrics={"loss": 1.0}
            ),
        ),
        True,
    )


def test_theta_stop_retains_failed_confirmation_and_continues() -> None:
    cadence_10 = _evaluation(10, "cadence", 19)
    assert decide_dose((cadence_10,)).action == "confirm"
    failed = _evaluation(10, "confirmation", 36)
    history = (cadence_10, failed)
    assert decide_dose(history).action == "continue"
    cadence_20 = _evaluation(20, "cadence", 20)
    assert decide_dose((*history, cadence_20)).action == "confirm"
    passed = _evaluation(20, "confirmation", 40)
    decision = decide_dose((*history, cadence_20, passed))
    assert decision.action == "proceed_to_pilot"
    assert decision.confirmation_count == 2
    assert [row.target.success_count for row in (*history, cadence_20, passed)] == [
        19,
        36,
        20,
        40,
    ]


def test_gate_timing_and_tier_classification() -> None:
    first = _evaluation(
        10,
        "cadence",
        0,
        invalid_tags=20,
        length_stops=60,
        loops=20,
    )
    assessment = assess_dose_evaluation(first)
    assert decide_dose((first,)).action == "continue"
    assert {
        row.name: (row.tier, row.decisive, row.passed)
        for row in assessment.criteria
        if row.name
        in {
            "cadence_loop_fraction",
            "cadence_capability_gain_lower_bound",
            "cadence_family_reachability",
        }
    } == {
        "cadence_loop_fraction": (3, False, None),
        "cadence_capability_gain_lower_bound": (3, False, None),
        "cadence_family_reachability": (3, False, None),
    }
    second = _evaluation(
        20,
        "cadence",
        0,
        invalid_tags=20,
        length_stops=60,
        loops=20,
    )
    assert decide_dose((first, second)).action == "tier2_degenerated"
    confirmation = _evaluation(20, "confirmation", 40, length_stops=40, loops=10)
    confirmed = decide_dose((confirmation,))
    assert confirmed.action == "tier2_degenerated"
    assert (
        next(
            row for row in confirmed.assessment.criteria if row.name == "loop_fraction"
        ).passed
        is False
    )
    cap = _evaluation(294, "epoch_cap", 20)
    assert decide_dose((cap,)).action == "dose_limited"
