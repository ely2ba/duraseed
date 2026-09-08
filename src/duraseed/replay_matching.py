"""Frozen replay-v1 nomination and high-draw matching; no outcome rescues."""

from __future__ import annotations

from fractions import Fraction
from itertools import product


ARMS = ("R-S", "R-P")
STAGE_A_GRID = (*range(10, 291, 10), 294)
STAGE_B_GRID = (0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480)
TARGETS = {11: Fraction(31, 96), 29: Fraction(17, 96)}
TOLERANCE = Fraction(3, 100)


def _score(row: dict, trials: int) -> Fraction:
    successes = row["successes"]
    if (
        type(successes) is not int
        or type(row["trials"]) is not int
        or row["trials"] != trials
        or not 0 <= successes <= trials
    ):
        raise ValueError("replay matching requires complete raw binomial counts")
    return Fraction(successes, trials)


def nominate(seed: int, cadence: dict[str, list[dict]]) -> dict:
    """Exactly three nominees per arm, only after each complete 294-update arm."""
    target = TARGETS[seed]
    if set(cadence) != set(ARMS):
        raise ValueError("both replay arms must be complete before nomination")
    nominees = {}
    for arm in ARMS:
        rows = cadence[arm]
        if sorted(row["update"] for row in rows) != list(STAGE_A_GRID):
            raise ValueError("nomination requires all 30 frozen cadence points")
        ranked = sorted(
            rows, key=lambda row: (abs(_score(row, 96) - target), row["update"])
        )
        nominees[arm] = [dict(row) for row in ranked[:3]]
    return {"seed": seed, "target": str(target), "nominees": nominees}


def match(nomination: dict, assessments: dict[str, list[dict]]) -> dict:
    """Assess only frozen nominees; exact rational arithmetic at the 0.03 edges."""
    seed = nomination["seed"]
    target = TARGETS[seed]
    if nomination["target"] != str(target) or set(assessments) != set(ARMS):
        raise ValueError("matching changed the frozen target or arm set")
    for arm in ARMS:
        expected = sorted(row["update"] for row in nomination["nominees"][arm])
        observed = sorted(row["update"] for row in assessments[arm])
        if len(expected) != 3 or len(set(expected)) != 3 or observed != expected:
            raise ValueError("candidate assessment must cover exactly three nominees")
        for row in assessments[arm]:
            _score(row, 256 * 16)
    combinations = []
    for solver, policy in product(assessments["R-S"], assessments["R-P"]):
        rs, rp = _score(solver, 4096), _score(policy, 4096)
        eligible = (
            abs(rs - target) <= TOLERANCE
            and abs(rp - target) <= TOLERANCE
            and abs(rs - rp) <= TOLERANCE
        )
        key = (
            abs(rs - rp),
            abs(rs - target) + abs(rp - target),
            solver["update"] + policy["update"],
            solver["update"],
            policy["update"],
        )
        combinations.append(
            (
                key,
                {
                    "R-S": solver["update"],
                    "R-P": policy["update"],
                    "R-S_successes": solver["successes"],
                    "R-P_successes": policy["successes"],
                    "trials_per_arm": 4096,
                    "eligible": eligible,
                    "selection_key": [str(value) for value in key],
                },
            )
        )
    eligible = sorted((key, row) for key, row in combinations if row["eligible"])
    return {
        "seed": seed,
        "status": "MATCHED" if eligible else "NO_MATCH",
        "target": str(target),
        "absolute_tolerance": "3/100",
        "combinations": [row for _, row in sorted(combinations)],
        "selected": eligible[0][1] if eligible else None,
        "stage_b_allowed": bool(eligible),
    }
