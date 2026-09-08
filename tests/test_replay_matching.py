from copy import deepcopy

import pytest

from duraseed.replay_matching import STAGE_A_GRID, match, nominate


def cadence(successes=31):
    return {
        arm: [{"update": u, "successes": successes, "trials": 96} for u in STAGE_A_GRID]
        for arm in ("R-S", "R-P")
    }


def assessments(successes=1323):
    return {
        arm: [
            {"update": u, "successes": successes, "trials": 4096} for u in (10, 20, 30)
        ]
        for arm in ("R-S", "R-P")
    }


def test_nomination_complete_and_earlier_tie():
    result = nominate(11, cadence())
    assert [row["update"] for row in result["nominees"]["R-S"]] == [10, 20, 30]
    incomplete = cadence()
    incomplete["R-S"].pop()
    with pytest.raises(ValueError, match="30 frozen"):
        nominate(11, incomplete)


def test_matching_tie_and_no_replacement():
    nomination = nominate(11, cadence())
    result = match(nomination, assessments())
    assert result["selected"]["R-S"] == result["selected"]["R-P"] == 10
    assert len(result["combinations"]) == 9
    changed = assessments()
    changed["R-P"][2]["update"] = 40
    with pytest.raises(ValueError, match="exactly three"):
        match(nomination, changed)


def test_no_match_never_allows_stage_b():
    result = match(nominate(11, cadence()), assessments(0))
    assert result["status"] == "NO_MATCH"
    assert result["selected"] is None
    assert not result["stage_b_allowed"]


def test_mutual_band_required_even_when_each_arm_near_target():
    rows = assessments(1220)
    for row in rows["R-P"]:
        row["successes"] = 1420
    result = match(nominate(11, cadence()), rows)
    assert result["status"] == "NO_MATCH"


def test_raw_not_posterior_and_all_candidates_required():
    rows = assessments()
    rows["R-S"][0]["successes"] = 0.323
    with pytest.raises(ValueError, match="raw binomial"):
        match(nominate(11, cadence()), rows)
    rows = deepcopy(assessments())
    rows["R-S"].pop()
    with pytest.raises(ValueError, match="exactly three"):
        match(nominate(11, cadence()), rows)
