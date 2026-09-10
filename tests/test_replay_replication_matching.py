import pytest

from duraseed.replay_matching import STAGE_A_GRID
from duraseed.replay_replication_matching import match, nominate


def cadence(solver=30, policy=40):
    return {
        arm: [{"update": u, "successes": score, "trials": 96} for u in STAGE_A_GRID]
        for arm, score in (("R-S", solver), ("R-P", policy))
    }


def assessments(solver=(1200, 1200, 1200), policy=(1200, 1200, 1200)):
    return {
        arm: [
            {"update": update, "successes": score, "trials": 4096}
            for update, score in zip((10, 20, 30), scores, strict=True)
        ]
        for arm, scores in (("R-S", solver), ("R-P", policy))
    }


def test_weak_max_anchor_has_no_historical_seed_target():
    result = nominate(47, cadence())
    assert result["seed"] == 47
    assert result["anchor"] == "5/16"
    assert "target" not in result
    for rows in result["nominees"].values():
        assert [row["update"] for row in rows] == [10, 20, 30]


def test_outlier_max_is_used_mechanically_without_smoothing_or_replacement():
    rows = cadence(10, 40)
    rows["R-S"][-2]["successes"] = 60
    rows["R-P"][1]["successes"] = 50
    result = nominate(47, rows)
    assert result["anchor"] == "25/48"
    assert [row["update"] for row in result["nominees"]["R-S"]] == [290, 10, 20]
    assert [row["update"] for row in result["nominees"]["R-P"]] == [20, 10, 30]


def test_disjoint_cadence_ranges_do_not_add_a_gate_or_anchor_band():
    nomination = nominate(47, cadence(10, 70))
    assert nomination["anchor"] == "5/48"
    result = match(nomination, assessments((2000,) * 3, (2050,) * 3))
    assert result["status"] == "MATCHED"
    assert result["stage_b_allowed"]


@pytest.mark.parametrize("change", ("missing", "duplicate", "wrong_trials"))
def test_nomination_requires_complete_raw_cadences(change):
    rows = cadence()
    if change == "missing":
        rows["R-S"].pop()
    elif change == "duplicate":
        rows["R-S"][-1]["update"] = 290
    else:
        rows["R-P"][0]["trials"] = 95
    with pytest.raises(ValueError):
        nominate(47, rows)


@pytest.mark.parametrize("gap,expected", ((122, "MATCHED"), (123, "NO_MATCH")))
def test_exact_count_boundary_and_no_match_stops_stage_b(gap, expected):
    result = match(
        nominate(47, cadence()),
        assessments((1200,) * 3, (1200 + gap,) * 3),
    )
    assert result["status"] == expected
    assert result["stage_b_allowed"] == (expected == "MATCHED")
    assert (result["selected"] is None) == (expected == "NO_MATCH")
    assert len(result["combinations"]) == 9


def test_highest_weaker_score_precedes_smallest_gap_then_earlier_dose():
    result = match(
        nominate(47, cadence()),
        assessments((1000, 1200, 1200), (1000, 1290, 1290)),
    )
    assert result["selected"]["R-S"] == 20
    assert result["selected"]["R-P"] == 20
    assert result["selected"]["R-S_successes"] == 1200
    assert result["selected"]["R-P_successes"] == 1290


def test_smaller_gap_breaks_equal_minimum_score_before_dose():
    result = match(
        nominate(47, cadence()),
        assessments((1200,) * 3, (1290, 1200, 1200)),
    )
    assert result["selected"]["R-S"] == 10
    assert result["selected"]["R-P"] == 20


@pytest.mark.parametrize("change", ("replacement", "missing", "fraction", "trials"))
def test_high_draw_assessments_are_complete_frozen_nominees(change):
    rows = assessments()
    if change == "replacement":
        rows["R-P"][2]["update"] = 40
    elif change == "missing":
        rows["R-P"].pop()
    elif change == "fraction":
        rows["R-P"][0]["successes"] = 0.3
    else:
        rows["R-P"][0]["trials"] = 4095
    with pytest.raises(ValueError):
        match(nominate(47, cadence()), rows)


def test_no_new_absolute_floor_is_applied():
    result = match(nominate(47, cadence()), assessments((0,) * 3, (0,) * 3))
    assert result["status"] == "MATCHED"
