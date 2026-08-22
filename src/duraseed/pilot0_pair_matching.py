"""Frozen one-draw cadence matcher for a single Pilot seed pair."""

from __future__ import annotations

from math import sqrt
from typing import Iterable

from duraseed.runners import RunnerGateError


PAIR_SELECTION_SCHEMA = "duraseed-pilot0-pair-matching-v2"


def targeted_exact_success_rate(result: dict) -> float:
    """Raw exact-success fraction on the frozen targeted cadence population."""

    try:
        rows = tuple(
            row for row in result["item_counts"] if row["panel_role"] == "targeted"
        )
        successes = sum(int(row["successes"]) for row in rows)
        trials = sum(int(row["trials"]) for row in rows)
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError("Pilot cadence targeted counts are malformed") from error
    if not rows or trials != len(rows) or any(int(row["trials"]) != 1 for row in rows):
        raise RunnerGateError("Pilot matching requires one draw per targeted item")
    return successes / trials


def targeted_cadence_interval(result: dict) -> tuple[float, float]:
    """Approximate 95% binomial interval for one 96-item cadence result."""

    rate = targeted_exact_success_rate(result)
    trials = sum(
        int(row["trials"])
        for row in result["item_counts"]
        if row["panel_role"] == "targeted"
    )
    radius = 1.96 * sqrt(rate * (1.0 - rate) / trials)
    return max(0.0, rate - radius), min(1.0, rate + radius)


def select_paired_cadence(bs: Iterable[dict], bg: Iterable[dict]) -> dict:
    """Select the nearest real B-S/B-G cadence pair with overlapping intervals."""

    candidates = []
    for left in bs:
        for right in bg:
            left_interval = targeted_cadence_interval(left["evaluation"])
            right_interval = targeted_cadence_interval(right["evaluation"])
            if max(left_interval[0], right_interval[0]) > min(
                left_interval[1], right_interval[1]
            ):
                continue
            left_rate = targeted_exact_success_rate(left["evaluation"])
            right_rate = targeted_exact_success_rate(right["evaluation"])
            candidates.append(
                {
                    "B-S": {
                        **left["checkpoint"],
                        "targeted_exact_success_rate": left_rate,
                        "targeted_approximate_95_interval": left_interval,
                        "monitor_generation_sha256": left["evaluation"][
                            "generation_sha256"
                        ],
                    },
                    "B-G": {
                        **right["checkpoint"],
                        "targeted_exact_success_rate": right_rate,
                        "targeted_approximate_95_interval": right_interval,
                        "monitor_generation_sha256": right["evaluation"][
                            "generation_sha256"
                        ],
                    },
                    "absolute_targeted_difference": abs(left_rate - right_rate),
                }
            )
    if not candidates:
        return {
            "schema_version": PAIR_SELECTION_SCHEMA,
            "status": "unavailable",
            "reason": "no_targeted_approximate_95_interval_overlap",
            "seed_replacement_allowed": False,
        }
    selected = min(
        candidates,
        key=lambda row: (
            row["absolute_targeted_difference"],
            row["B-S"]["step"] + row["B-G"]["step"],
            row["B-S"]["step"],
            row["B-G"]["step"],
        ),
    )
    return {
        "schema_version": PAIR_SELECTION_SCHEMA,
        "status": "selected",
        "rule": (
            "nearest_targeted_exact_success_pair_with_overlapping_"
            "approximate_95_intervals_tie_earlier"
        ),
        "seed_replacement_allowed": False,
        **selected,
    }


__all__ = [
    "select_paired_cadence",
    "targeted_cadence_interval",
    "targeted_exact_success_rate",
]
