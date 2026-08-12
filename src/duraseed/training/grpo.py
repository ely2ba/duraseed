"""Pure grouped-reward diagnostics for future Tinker RL updates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class GroupedRewardDiagnostics:
    """Per-group means/advantages plus constant-reward group counts."""

    group_means: tuple[float, ...]
    centered_advantages: tuple[tuple[float, ...], ...]
    all_zero_group_count: int
    all_one_group_count: int
    mixed_group_count: int


def grouped_reward_diagnostics(
    rewards: Sequence[float],
    *,
    group_size: int,
) -> GroupedRewardDiagnostics:
    """Mean-center exact binary rewards independently within each prompt group."""

    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError("group_size must be an integer")
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    if isinstance(rewards, (str, bytes, bytearray)) or not isinstance(
        rewards, Sequence
    ):
        raise TypeError("rewards must be an explicit sequence")
    normalized = tuple(rewards)
    if not normalized or len(normalized) % group_size:
        raise ValueError("reward count must be a positive multiple of group_size")
    if any(
        type(reward) is not float or reward not in (0.0, 1.0) for reward in normalized
    ):
        raise ValueError("rewards must contain only exact binary floats")

    means: list[float] = []
    advantages: list[tuple[float, ...]] = []
    all_zero = 0
    all_one = 0
    mixed = 0
    for start in range(0, len(normalized), group_size):
        group = normalized[start : start + group_size]
        mean = math.fsum(group) / group_size
        means.append(mean)
        advantages.append(tuple(reward - mean for reward in group))
        if all(reward == 0.0 for reward in group):
            all_zero += 1
        elif all(reward == 1.0 for reward in group):
            all_one += 1
        else:
            mixed += 1

    return GroupedRewardDiagnostics(
        group_means=tuple(means),
        centered_advantages=tuple(advantages),
        all_zero_group_count=all_zero,
        all_one_group_count=all_one,
        mixed_group_count=mixed,
    )


__all__ = ["GroupedRewardDiagnostics", "grouped_reward_diagnostics"]
