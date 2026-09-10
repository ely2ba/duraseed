from copy import deepcopy

import pytest

from duraseed.endpoint_matching import (
    STUDENT_GRID,
    confirm,
    nominate,
    profile_gates,
    select,
    trajectory_analysis,
)


def items(count, draws, successes, prefix="panel"):
    return [
        {
            "task_id": f"{prefix}-{role}-{i}",
            "panel_role": role,
            "successes": successes,
            "trials": draws,
        }
        for role in ("targeted", "sentinel")
        for i in range(count)
    ]


def profile(successes=8, prefix="panel"):
    return {
        "items": items(256, 16, successes, prefix),
        "panels": {
            role: {
                "item_count": 256,
                "completion_count": 4096,
                "exact_success_rate": 0.99,
                "equal_item_posterior_mean": 0.01,
                "mean_pass_at_k": {"16": 0.123},
                "syntactically_invalid_rate": 0.0,
                "completion_token_length": {"median": 100},
                "unique_verified_strategy_family_count": 12,
            }
            for role in ("targeted", "sentinel")
        },
    }


def nomination():
    teacher = {"item_counts": items(96, 1, 0)}
    cadence = [{"update": u, **deepcopy(teacher)} for u in STUDENT_GRID]
    return nominate(cadence, teacher)


def selection():
    return select(nomination(), profile(), {u: profile() for u in (10, 20, 30)})


def test_nomination_uses_both_raw_roles_then_earlier_update():
    teacher = {"item_counts": items(96, 1, 0)}
    cadence = [{"update": u, **deepcopy(teacher)} for u in STUDENT_GRID]
    cadence[0]["item_counts"][0]["successes"] = 1
    cadence[1]["item_counts"][96]["successes"] = 1
    for row in cadence[3:]:
        for item in row["item_counts"]:
            item["successes"] = 1
    result = nominate(cadence, teacher)
    assert [row["update"] for row in result["nominees"]] == [30, 20, 10]
    assert result["nominees"][1]["nomination_distance"] == "5/24"


@pytest.mark.parametrize("change", ("missing", "duplicate", "draws"))
def test_incomplete_cadences_cannot_nominate(change):
    teacher = {"item_counts": items(96, 1, 0)}
    cadence = [{"update": u, **deepcopy(teacher)} for u in STUDENT_GRID]
    if change == "missing":
        cadence.pop()
    elif change == "duplicate":
        cadence[-1]["update"] = 290
    else:
        cadence[0]["item_counts"][0]["trials"] = 2
    with pytest.raises(ValueError):
        nominate(cadence, teacher)


@pytest.mark.parametrize("extra,passed", ((122, True), (123, False)))
def test_targeted_three_point_boundary_uses_raw_counts(extra, passed):
    student = profile()
    for item in student["items"][:extra]:
        item["successes"] += 1
    result = profile_gates(student, profile())
    assert result["passed"] is passed
    gate = next(
        g
        for g in result["gates"]
        if g["role"] == "targeted" and g["metric"] == "raw_pass_at_1"
    )
    assert gate["teacher"] == 0.5


@pytest.mark.parametrize("ratio,passed", ((150, True), (151, False)))
def test_median_length_ratio_is_inclusive(ratio, passed):
    student = profile()
    student["panels"]["targeted"]["completion_token_length"]["median"] = ratio
    assert profile_gates(student, profile())["passed"] is passed


def test_zero_teacher_strategy_count_requires_zero_student_count():
    teacher = profile(0)
    student = profile(0)
    for role in ("targeted", "sentinel"):
        teacher["panels"][role]["unique_verified_strategy_family_count"] = 0
        student["panels"][role]["unique_verified_strategy_family_count"] = 0
    assert profile_gates(student, teacher)["passed"]
    student["panels"]["sentinel"]["unique_verified_strategy_family_count"] = 1
    assert not profile_gates(student, teacher)["passed"]


def test_coverage_comes_from_at_least_one_success_not_posterior_or_summary():
    teacher = profile(1)
    student = profile(1)
    for row in student["items"][:26]:
        row["successes"] = 0
    result = profile_gates(student, teacher)
    failures = [g["metric"] for g in result["gates"] if not g["passed"]]
    assert failures == ["coverage_at_16"]


@pytest.mark.parametrize("invalid,passed", ((409, True), (410, False)))
def test_validity_gate_uses_integer_observation_counts(invalid, passed):
    student = profile()
    student["panels"]["sentinel"]["syntactically_invalid_rate"] = invalid / 4096
    assert profile_gates(student, profile())["passed"] is passed


def test_selection_prefers_per_item_agreement_over_nomination_then_dose():
    candidates = {u: profile() for u in (10, 20, 30)}
    # Same aggregate accuracy, but a worse distribution over shared problems.
    candidates[10]["items"][0]["successes"] += 1
    candidates[10]["items"][1]["successes"] -= 1
    result = select(nomination(), profile(), candidates)
    assert result["status"] == "SELECTED"
    assert result["selected_update"] == 20
    assert not result["stage_b_allowed"]


def test_no_match_cannot_start_stage_b_or_confirm():
    result = select(nomination(), profile(), {u: profile(0) for u in (10, 20, 30)})
    assert result["status"] == "NO_MATCH"
    assert not result["stage_b_allowed"]
    with pytest.raises(ValueError):
        confirm(result, profile(), profile(), profile())


def test_confirm_uses_disjoint_items_and_no_fallback_after_failure():
    selected = selection()
    with pytest.raises(ValueError, match="item-disjoint"):
        confirm(selected, profile(), profile(), profile())
    fresh = profile(prefix="confirmation")
    passed = confirm(selected, fresh, fresh, fresh)
    assert passed["status"] == "MATCHED"
    assert passed["stage_b_allowed"]
    failed = confirm(selected, fresh, fresh, profile(0, "confirmation"))
    assert failed["status"] == "CONFIRMATION_FAILED"
    assert failed["selected_update"] == selected["selected_update"]
    assert not failed["stage_b_allowed"]
    with pytest.raises(ValueError):
        confirm(failed, fresh, fresh, fresh)


def test_confirmation_item_agreement_catches_equal_aggregate_wrong_items():
    fresh = profile(prefix="confirmation")
    student = deepcopy(fresh)
    for i, row in enumerate(student["items"]):
        row["successes"] += 1 if i % 2 else -1
    result = confirm(selection(), fresh, fresh, student)
    assert all(g["passed"] for g in result["confirmation"]["gates"])
    assert not result["stage_b_allowed"]
    assert all(not g["passed"] for g in result["confirmation"]["item_agreement"])


def test_confirmation_item_gate_subtracts_teacher_repeat_discrepancy():
    fresh = profile(prefix="confirmation")
    repeat = deepcopy(fresh)
    student = deepcopy(fresh)
    for i, row in enumerate(student["items"]):
        row["successes"] += 1 if i % 2 else -1
        repeat["items"][i]["successes"] = row["successes"]
    result = confirm(selection(), fresh, repeat, student)
    assert result["stage_b_allowed"]
    assert all(
        g["excess_item_distance"] == 0 for g in result["confirmation"]["item_agreement"]
    )


def trajectories(scores):
    return {
        run: [{"update": u, "item_counts": items(192, 4, values[u])} for u in range(21)]
        for run, values in scores.items()
    }


def test_absolute_distance_does_not_cancel_crossing_curves():
    scores = {run: [0] * 21 for run in ("T1", "T2", "S1", "S2")}
    for u in range(21):
        scores["T1"][u] = scores["T2"][u] = 0 if u < 10 else 4
        scores["S1"][u] = scores["S2"][u] = 4 if u < 10 else 0
    result = trajectory_analysis(trajectories(scores))
    for panel in result["panels"].values():
        assert len(panel["distances"]) == 6
        assert panel["mean_repeat_distance"] == 0
        assert panel["excess_separation"] == 1


def test_repeat_adjusted_separation_preserves_negative_estimates():
    values = {"T1": [0] * 21, "T2": [4] * 21, "S1": [1] * 21, "S2": [3] * 21}
    result = trajectory_analysis(trajectories(values))
    assert result["panels"]["targeted"]["excess_separation"] == -0.25
    assert "verdict" not in result


def test_distance_trapezoid_weights_endpoint_half_and_missing_grid_fails():
    scores = {run: [0] * 21 for run in ("T1", "T2", "S1", "S2")}
    scores["S1"][0] = 4
    runs = trajectories(scores)
    result = trajectory_analysis(runs)
    assert result["panels"]["targeted"]["distances"]["T1:S1"] == 0.025
    runs["S1"].pop(5)
    with pytest.raises(ValueError, match="every update"):
        trajectory_analysis(runs)
