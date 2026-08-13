"""Stage-A final evidence and common RL collapse reduction."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite

from duraseed.runners import RunnerGateError
from duraseed.runners.stage_a_updates import Branch
from duraseed.training.acquisition_freeze import CommonRLFreezeEvidence
from duraseed.training.stage_a_calibration import (
    STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE,
    StageAFinalEvidence,
)


@dataclass(frozen=True, slots=True)
class FinalArmEvidence:
    evidence: StageAFinalEvidence
    final_ten_mixed_group_rate: float | None
    mean_sampled_token_surprisal: float | None
    unique_completion_count: int
    successful_completion_count: int
    valid_family_count: int
    loss_health_passed: bool


def final_envelope(branch: Branch, evidence: StageAFinalEvidence) -> FinalArmEvidence:
    if branch.method == "B-S":
        return FinalArmEvidence(evidence, None, None, 0, 0, 0, True)
    steps = range(41, 51)
    mixed = tuple(
        branch.metrics[step - 1].metrics["mixed_group_rate"] for step in steps
    )
    unique = set().union(*(branch.unique_completions_by_step[step] for step in steps))
    successes = set().union(
        *(branch.successful_completions_by_step[step] for step in steps)
    )
    valid_families = set().union(
        *(branch.valid_families_by_step[step] for step in steps)
    )
    healthy = all(
        isfinite(value)
        for metric in branch.metrics[-10:]
        for value in metric.metrics.values()
    )
    return FinalArmEvidence(
        evidence,
        fsum(mixed) / len(mixed),
        fsum(branch.surprisal_by_step[step] for step in steps) / 10,
        len(unique),
        len(successes),
        len(valid_families),
        healthy,
    )


def common_rl_freeze(bg: FinalArmEvidence) -> CommonRLFreezeEvidence:
    mixed = bg.final_ten_mixed_group_rate
    surprisal = bg.mean_sampled_token_surprisal
    collapsed = (
        mixed is None
        or not isfinite(mixed)
        or mixed < STAGE_A_BG_MINIMUM_MIXED_GROUP_RATE
        or surprisal is None
        or not isfinite(surprisal)
        or surprisal < 0
        or bg.unique_completion_count < 1
        or bg.successful_completion_count < 1
        or bg.valid_family_count < 1
        or not bg.loss_health_passed
    )
    if collapsed:
        raise RunnerGateError(
            "common RL entropy-collapse gate failed; a protocol-wide fallback "
            "must be authorized before any multi-method run"
        )
    return CommonRLFreezeEvidence(
        configuration=(
            ("advantage_normalization", "group_mean_no_std"),
            ("constant_reward_groups", "skip_no_resample"),
            ("entropy_collapse_response", "hard_stop_protocol_wide_fallback"),
            ("kl_penalty", False),
            ("loss", "importance_sampling"),
            ("objective", "group_relative_exact_binary"),
        ),
        entropy_collapse_detected=False,
        entropy_collapse_gate_passed=True,
        final_ten_mixed_group_rate=mixed,
        mean_sampled_token_surprisal=surprisal,
        unique_completion_count=bg.unique_completion_count,
        successful_completion_count=bg.successful_completion_count,
        valid_family_count=bg.valid_family_count,
        loss_health_passed=bg.loss_health_passed,
    )


__all__ = ["FinalArmEvidence", "common_rl_freeze", "final_envelope"]
