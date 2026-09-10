"""Prospective common-attainment matching for the new replay-order replication."""

from __future__ import annotations

from itertools import product

from duraseed.replay_matching import ARMS, STAGE_A_GRID, TOLERANCE, _score


def nominate(seed: int, cadence: dict[str, list[dict]]) -> dict:
    """Nominate three per arm near the weaker arm's maximum cadence score."""
    if set(cadence) != set(ARMS):
        raise ValueError("both replay arms must be complete before nomination")
    scores = {}
    for arm in ARMS:
        rows = cadence[arm]
        if sorted(row["update"] for row in rows) != list(STAGE_A_GRID):
            raise ValueError("nomination requires all 30 frozen cadence points")
        scores[arm] = [_score(row, 96) for row in rows]
    anchor = min(max(scores[arm]) for arm in ARMS)
    nominees = {
        arm: [
            dict(row)
            for row in sorted(
                cadence[arm],
                key=lambda row: (abs(_score(row, 96) - anchor), row["update"]),
            )[:3]
        ]
        for arm in ARMS
    }
    return {"seed": seed, "anchor": str(anchor), "nominees": nominees}


def match(nomination: dict, assessments: dict[str, list[dict]]) -> dict:
    """Only the high-draw between-arm gap decides eligibility; no anchor band."""
    if set(nomination["nominees"]) != set(ARMS) or set(assessments) != set(ARMS):
        raise ValueError("matching requires both nominated replay arms")
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
        gap = abs(rs - rp)
        key = (
            -min(rs, rp),
            gap,
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
                    "eligible": gap <= TOLERANCE,
                    "selection_key": [str(value) for value in key],
                },
            )
        )
    ranked = [row for _, row in sorted(combinations)]
    eligible = [row for row in ranked if row["eligible"]]
    return {
        "seed": nomination["seed"],
        "status": "MATCHED" if eligible else "NO_MATCH",
        "anchor": nomination["anchor"],
        "absolute_tolerance": "3/100",
        "combinations": ranked,
        "selected": eligible[0] if eligible else None,
        "stage_b_allowed": bool(eligible),
    }
