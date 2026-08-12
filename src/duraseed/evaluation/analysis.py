"""Primary item-level metrics for the DuraSeed experiment.

The functions here operate on graded counts or linked manifest/generation/
reward evidence.  They have no dependency on a training or serving framework,
so reported analyses remain reproducible from retained local artifacts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import exp, fsum, isfinite, lgamma, log, log1p
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from duraseed.data.manifests import DatasetManifest
    from duraseed.run_records import GenerationRecord, RewardRecord


@dataclass(frozen=True, slots=True)
class BinomialObservation:
    """Exact successes among independently sampled completions for one item."""

    successes: int
    trials: int

    def __post_init__(self) -> None:
        _validate_counts(self.successes, self.trials)


class ReliabilityState(StrEnum):
    """Hard state used only for transition tables and figures."""

    RELIABLE = "reliable"
    UNRELIABLE = "unreliable"
    UNCERTAIN = "uncertain"


StageTransition = tuple[ReliabilityState, ReliabilityState, ReliabilityState]


@dataclass(frozen=True, slots=True)
class ValidationCheckpoint:
    """A concrete validation checkpoint eligible for matched-A selection."""

    checkpoint_id: str
    step: int
    mean_item_success: float

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a non-empty string")
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ValueError("checkpoint step must be a non-negative integer")
        _validate_probability(self.mean_item_success, "mean_item_success", closed=True)


@dataclass(frozen=True, slots=True)
class NormalizedAUC:
    """Absolute, raw-gain, and headroom-normalized areas on a shared grid."""

    absolute_auc: float
    raw_gain_auc: float
    headroom_normalized_gain_auc: float
    initial_score: float
    final_score: float


@dataclass(frozen=True, slots=True)
class MatchedProgressEstimate:
    """Stage-A retention when Stage B first reaches a declared score."""

    target_stage_b_score: float
    crossing_step: float
    stage_a_retention: float
    left_step: float
    right_step: float
    interpolation_fraction: float
    exact_checkpoint: bool


@dataclass(frozen=True, slots=True)
class AcquiredItemSurvival:
    """Hard-state survival among items newly made reliable by Stage A."""

    acquired_count: int
    retained_count: int
    uncertain_count: int
    lost_count: int
    survival_rate: float | None


@dataclass(frozen=True, slots=True)
class TokenSurprisalSummary:
    """Sampled-token surprisal with explicit log-probability coverage."""

    total_token_count: int
    observed_logprob_count: int
    coverage: float
    mean_surprisal: float | None


@dataclass(frozen=True, slots=True)
class VerifiedStrategyDiversitySummary:
    """Exact strategy-family counts among successfully verified completions."""

    total_completion_count: int
    valid_completion_count: int
    valid_completion_rate: float
    valid_family_ids: tuple[str, ...]
    unique_valid_family_count: int
    valid_family_counts: tuple[tuple[str, int], ...]


def jeffreys_posterior_parameters(successes: int, trials: int) -> tuple[float, float]:
    """Return the Beta posterior parameters for the Jeffreys prior."""

    _validate_counts(successes, trials)
    return successes + 0.5, trials - successes + 0.5


def jeffreys_posterior_mean(successes: int, trials: int) -> float:
    """Return the threshold-free item success estimate under Jeffreys' prior."""

    alpha, beta = jeffreys_posterior_parameters(successes, trials)
    return alpha / (alpha + beta)


def jeffreys_equal_tailed_interval(
    successes: int,
    trials: int,
    *,
    credible_mass: float,
) -> tuple[float, float]:
    """Return an explicitly sized equal-tailed Jeffreys credible interval."""

    _validate_probability(credible_mass, "credible_mass", closed=False)
    alpha, beta = jeffreys_posterior_parameters(successes, trials)
    tail = (1.0 - float(credible_mass)) / 2.0
    return (
        _beta_quantile(tail, alpha, beta),
        _beta_quantile(1.0 - tail, alpha, beta),
    )


def informative_group_probability(
    success_probability: float,
    group_size: int,
) -> float:
    """Return the probability that a binary-reward group contains both outcomes."""

    _validate_probability(success_probability, "success_probability", closed=True)
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 2
    ):
        raise ValueError("group_size must be an integer of at least two")
    probability = float(success_probability)
    result = 1.0 - (1.0 - probability) ** group_size - probability**group_size
    return min(1.0, max(0.0, result))


def equal_item_posterior_mean(
    observations: Iterable[BinomialObservation],
) -> float:
    """Average Jeffreys posterior item means with equal weight per item."""

    values = [
        jeffreys_posterior_mean(item.successes, item.trials) for item in observations
    ]
    if not values:
        raise ValueError("posterior item mean requires at least one item")
    return fsum(values) / len(values)


def posterior_mean_gain(
    origin: Iterable[BinomialObservation],
    current: Iterable[BinomialObservation],
) -> float:
    """Return equal-item posterior mean gain on one unchanged panel."""

    origin_items = tuple(origin)
    current_items = tuple(current)
    if len(origin_items) != len(current_items):
        raise ValueError("origin and current panels require the same item count")
    return equal_item_posterior_mean(current_items) - equal_item_posterior_mean(
        origin_items
    )


def targeted_minus_sentinel_specificity(
    origin_targeted: Iterable[BinomialObservation],
    current_targeted: Iterable[BinomialObservation],
    origin_sentinel: Iterable[BinomialObservation],
    current_sentinel: Iterable[BinomialObservation],
) -> float:
    """Return targeted-panel gain minus sentinel-panel gain within one run."""

    return posterior_mean_gain(
        origin_targeted,
        current_targeted,
    ) - posterior_mean_gain(origin_sentinel, current_sentinel)


def jeffreys_probability_at_least(
    successes: int,
    trials: int,
    tau: float,
) -> float:
    """Return ``Pr(p >= tau | successes, trials)`` under a Jeffreys prior."""

    _validate_probability(tau, "tau", closed=False)
    alpha, beta = jeffreys_posterior_parameters(successes, trials)
    # Evaluate the survival probability by symmetry rather than subtracting a
    # CDF close to one.  This preserves useful precision for unreliable items.
    return _regularized_beta(1.0 - tau, beta, alpha)


def posterior_soft_cover(
    observations: Iterable[BinomialObservation],
    tau: float,
) -> float:
    """Average posterior probability of item reliability at threshold ``tau``."""

    _validate_probability(tau, "tau", closed=False)
    values = [
        jeffreys_probability_at_least(item.successes, item.trials, tau)
        for item in observations
    ]
    if not values:
        raise ValueError("posterior-soft coverage requires at least one item")
    return fsum(values) / len(values)


def stage_a_to_b_cover_change(
    stage_a_observations: Iterable[BinomialObservation],
    stage_b_observations: Iterable[BinomialObservation],
    tau: float,
) -> float:
    """Return post-B posterior-soft Cover minus Stage-A Cover on the same items."""

    stage_a = tuple(stage_a_observations)
    stage_b = tuple(stage_b_observations)
    if len(stage_a) != len(stage_b):
        raise ValueError("Stage-A and Stage-B cover require the same item count")
    if tuple(item.trials for item in stage_a) != tuple(item.trials for item in stage_b):
        raise ValueError(
            "Stage-A and Stage-B cover require the same per-item trial budgets"
        )
    return posterior_soft_cover(stage_b, tau) - posterior_soft_cover(stage_a, tau)


def hard_reliability_state(
    observation: BinomialObservation,
    tau: float,
    *,
    reliable_probability: float = 0.90,
    unreliable_probability: float = 0.10,
) -> ReliabilityState:
    """Classify an item for transition reporting using posterior mass cutoffs."""

    _validate_hard_state_thresholds(reliable_probability, unreliable_probability)
    probability = jeffreys_probability_at_least(
        observation.successes, observation.trials, tau
    )
    if probability >= reliable_probability:
        return ReliabilityState.RELIABLE
    if probability <= unreliable_probability:
        return ReliabilityState.UNRELIABLE
    return ReliabilityState.UNCERTAIN


def base_stage_a_stage_b_transition_counts(
    observations: Iterable[
        tuple[BinomialObservation, BinomialObservation, BinomialObservation]
    ],
    tau: float,
    *,
    reliable_probability: float = 0.90,
    unreliable_probability: float = 0.10,
) -> dict[StageTransition, int]:
    """Count observed hard-state base -> Stage A -> Stage B transitions."""

    _validate_probability(tau, "tau", closed=False)
    _validate_hard_state_thresholds(reliable_probability, unreliable_probability)
    counts: Counter[StageTransition] = Counter()
    item_count = 0
    for triple in observations:
        if not isinstance(triple, tuple) or len(triple) != 3:
            raise ValueError("each transition record must contain exactly three stages")
        if not all(isinstance(item, BinomialObservation) for item in triple):
            raise TypeError("transition stages must be BinomialObservation values")
        base, stage_a, stage_b = triple
        transition = (
            hard_reliability_state(
                base,
                tau,
                reliable_probability=reliable_probability,
                unreliable_probability=unreliable_probability,
            ),
            hard_reliability_state(
                stage_a,
                tau,
                reliable_probability=reliable_probability,
                unreliable_probability=unreliable_probability,
            ),
            hard_reliability_state(
                stage_b,
                tau,
                reliable_probability=reliable_probability,
                unreliable_probability=unreliable_probability,
            ),
        )
        counts[transition] += 1
        item_count += 1
    if item_count == 0:
        raise ValueError("transition counts require at least one item")
    return dict(counts)


def acquired_item_survival(
    observations: Iterable[
        tuple[BinomialObservation, BinomialObservation, BinomialObservation]
    ],
    tau: float,
    *,
    reliable_probability: float = 0.90,
    unreliable_probability: float = 0.10,
) -> AcquiredItemSurvival:
    """Summarize post-B survival among Stage-A-acquired reliable items.

    An item is newly reliable when it was not reliable at base and is reliable
    at Stage A.  Stage-B uncertain items are reported separately from items
    that return to the unreliable state.
    """

    transition_counts = base_stage_a_stage_b_transition_counts(
        observations,
        tau,
        reliable_probability=reliable_probability,
        unreliable_probability=unreliable_probability,
    )
    retained_count = 0
    uncertain_count = 0
    lost_count = 0
    for (base, stage_a, stage_b), count in transition_counts.items():
        if (
            base is ReliabilityState.RELIABLE
            or stage_a is not ReliabilityState.RELIABLE
        ):
            continue
        if stage_b is ReliabilityState.RELIABLE:
            retained_count += count
        elif stage_b is ReliabilityState.UNCERTAIN:
            uncertain_count += count
        else:
            lost_count += count
    acquired_count = retained_count + uncertain_count + lost_count
    return AcquiredItemSurvival(
        acquired_count=acquired_count,
        retained_count=retained_count,
        uncertain_count=uncertain_count,
        lost_count=lost_count,
        survival_rate=(retained_count / acquired_count if acquired_count > 0 else None),
    )


def summarize_token_surprisal(
    logprobs: Iterable[float | None],
) -> TokenSurprisalSummary:
    """Summarize sampled-token surprisal without hiding missing logprobs."""

    total_token_count = 0
    observed: list[float] = []
    for logprob in logprobs:
        total_token_count += 1
        if logprob is None:
            continue
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            raise ValueError(
                "token logprobs must be finite non-positive numbers or None"
            )
        value = float(logprob)
        if not isfinite(value) or value > 0.0:
            raise ValueError(
                "token logprobs must be finite non-positive numbers or None"
            )
        observed.append(value)
    if total_token_count == 0:
        raise ValueError("token surprisal requires at least one token")
    observed_logprob_count = len(observed)
    return TokenSurprisalSummary(
        total_token_count=total_token_count,
        observed_logprob_count=observed_logprob_count,
        coverage=observed_logprob_count / total_token_count,
        mean_surprisal=(
            -fsum(observed) / observed_logprob_count
            if observed_logprob_count > 0
            else None
        ),
    )


def summarize_verified_strategy_diversity(
    manifest: DatasetManifest,
    generations: Iterable[GenerationRecord],
    rewards: Iterable[RewardRecord],
) -> VerifiedStrategyDiversitySummary:
    """Count exact strategy families for one linked, verified TCES item."""

    # Local import avoids a module cycle: run records use data package writers,
    # while boundary-data reducers depend on this pure analysis module.
    from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
    from duraseed.run_records import GenerationRecord, RewardRecord

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("strategy diversity requires a DatasetManifest")
    if manifest.task_family != "tces" or manifest.split != "a_test_multi":
        raise ValueError("strategy diversity requires an a_test_multi TCES manifest")
    generation_rows = tuple(generations)
    reward_rows = tuple(rewards)
    if not generation_rows:
        raise ValueError("strategy diversity requires at least one generation")
    if any(not isinstance(row, GenerationRecord) for row in generation_rows):
        raise TypeError("strategy diversity requires GenerationRecord instances")
    if any(not isinstance(row, RewardRecord) for row in reward_rows):
        raise TypeError("strategy diversity requires RewardRecord instances")
    if any(row.task_family != "tces" for row in generation_rows):
        raise ValueError("strategy diversity requires TCES generations")
    item_identities = {
        (row.task_manifest_id, row.task_id, row.source_split, row.item_index)
        for row in generation_rows
    }
    if len(item_identities) != 1:
        raise ValueError("strategy diversity requires one TCES item and manifest")
    manifest_id, task_id, source_split, item_index = next(iter(item_identities))
    if (
        manifest_id != manifest.manifest_id
        or source_split != manifest.split
        or item_index is None
    ):
        raise ValueError("strategy diversity rows differ from the supplied manifest")
    matching_records = tuple(
        record for record in manifest.records if record.task_id == task_id
    )
    if len(matching_records) != 1 or not isinstance(
        matching_records[0], TCESTaskManifestRecord
    ):
        raise ValueError("strategy diversity item is absent from the TCES manifest")
    manifest_record = matching_records[0]
    if item_index != manifest_record.item_index:
        raise ValueError("strategy diversity item index differs from the manifest")
    if len(manifest_record.valid_family_ids) < 2:
        raise ValueError("strategy diversity requires a multi-family TCES item")
    checkpoint_identities = {
        (
            row.run_id,
            row.sampler_checkpoint_path,
            row.checkpoint_stage,
            row.training_step,
        )
        for row in generation_rows
    }
    if len(checkpoint_identities) != 1:
        raise ValueError("strategy diversity requires one run and checkpoint")
    sample_ids = [row.sample_id for row in generation_rows]
    sample_indices = [row.sample_index for row in generation_rows]
    if len(sample_ids) != len(set(sample_ids)) or len(sample_indices) != len(
        set(sample_indices)
    ):
        raise ValueError("strategy diversity requires unique sample coordinates")
    reward_ids = [row.reward_id for row in reward_rows]
    reward_sample_ids = [row.sample_id for row in reward_rows]
    if len(reward_ids) != len(set(reward_ids)) or len(reward_sample_ids) != len(
        set(reward_sample_ids)
    ):
        raise ValueError("strategy diversity requires unique reward linkages")
    if set(sample_ids) != set(reward_sample_ids):
        raise ValueError("strategy diversity generation/reward linkages differ")

    rewards_by_sample = {row.sample_id: row for row in reward_rows}
    family_counts: Counter[str] = Counter()
    for generation in generation_rows:
        reward = rewards_by_sample[generation.sample_id]
        if reward.task_id != generation.task_id:
            raise ValueError("strategy diversity reward task identity differs")
        if generation.reward is None or generation.reward != reward.reward:
            raise ValueError("strategy diversity generation/reward values differ")
        verification = reward.exact_verification
        if generation.family_id != verification.strategy_family_id:
            raise ValueError("strategy diversity response family differs from verifier")
        if verification.reward == 1.0:
            family_id = verification.strategy_family_id
            if family_id is None:  # Guaranteed by the VerificationResult schema.
                raise ValueError("successful verification requires a strategy family")
            if family_id not in manifest_record.valid_family_ids:
                raise ValueError(
                    "verified strategy family is absent from the exact manifest"
                )
            family_counts[family_id] += 1

    sorted_family_counts = tuple(sorted(family_counts.items()))
    valid_completion_count = sum(family_counts.values())
    valid_family_ids = tuple(family_id for family_id, _ in sorted_family_counts)
    return VerifiedStrategyDiversitySummary(
        total_completion_count=len(generation_rows),
        valid_completion_count=valid_completion_count,
        valid_completion_rate=valid_completion_count / len(generation_rows),
        valid_family_ids=valid_family_ids,
        unique_valid_family_count=len(valid_family_ids),
        valid_family_counts=sorted_family_counts,
    )


def unbiased_pass_at_k(successes: int, trials: int, k: int) -> float:
    """Return the standard unbiased Pass@k estimator for ``trials >= k``."""

    _validate_counts(successes, trials)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if trials < k:
        raise ValueError("unbiased Pass@k requires trials >= k")
    failures = trials - successes
    if failures < k:
        return 1.0
    all_fail_probability = 1.0
    for offset in range(k):
        all_fail_probability *= (failures - offset) / (trials - offset)
    return 1.0 - all_fail_probability


def earliest_matching_checkpoint(
    checkpoints: Iterable[ValidationCheckpoint],
    *,
    target: float,
    tolerance: float,
) -> ValidationCheckpoint | None:
    """Select the earliest real checkpoint in the inclusive target band.

    ``None`` means the method failed to reach the band.  The function never
    interpolates or fabricates a checkpoint.
    """

    _validate_probability(target, "target", closed=True)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be a finite non-negative number")
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative number")
    candidates = list(checkpoints)
    if not candidates:
        raise ValueError("checkpoint selection requires at least one checkpoint")
    if not all(isinstance(item, ValidationCheckpoint) for item in candidates):
        raise TypeError("checkpoints must be ValidationCheckpoint values")
    steps = [item.step for item in candidates]
    if len(steps) != len(set(steps)):
        raise ValueError("validation checkpoint steps must be unique")
    return next(
        (
            item
            for item in sorted(candidates, key=lambda checkpoint: checkpoint.step)
            if abs(item.mean_item_success - target) <= tolerance
        ),
        None,
    )


def normalized_stage_b_auc(
    step_grid: Sequence[int | float],
    scores: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> NormalizedAUC:
    """Integrate one Stage-B curve over a normalized, caller-supplied grid."""

    steps = _validate_step_grid(step_grid)
    values = _validate_scores(scores, expected_length=len(steps))
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be a finite positive number")
    epsilon = float(epsilon)
    if not isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite positive number")
    span = steps[-1] - steps[0]
    absolute_auc = (
        fsum(
            (right_step - left_step) * (left_score + right_score) / 2.0
            for left_step, right_step, left_score, right_score in zip(
                steps, steps[1:], values, values[1:]
            )
        )
        / span
    )
    initial = values[0]
    raw_gain_auc = absolute_auc - initial
    headroom_normalized_gain_auc = raw_gain_auc / (1.0 - initial + epsilon)
    return NormalizedAUC(
        absolute_auc=absolute_auc,
        raw_gain_auc=raw_gain_auc,
        headroom_normalized_gain_auc=headroom_normalized_gain_auc,
        initial_score=initial,
        final_score=values[-1],
    )


def normalized_stage_b_aucs(
    step_grid: Sequence[int | float],
    curves: Mapping[str, Sequence[float]],
    *,
    epsilon: float = 1e-12,
) -> dict[str, NormalizedAUC]:
    """Compute comparable AUCs for multiple methods on one common step grid."""

    if not curves:
        raise ValueError("AUC comparison requires at least one curve")
    if any(not isinstance(name, str) or not name.strip() for name in curves):
        raise ValueError("curve names must be non-empty strings")
    # Passing the grid once makes it impossible to compare curves evaluated at
    # silently different steps; score-length validation catches missing points.
    return {
        name: normalized_stage_b_auc(step_grid, scores, epsilon=epsilon)
        for name, scores in curves.items()
    }


def retention_at_first_stage_b_crossing(
    step_grid: Sequence[int | float],
    stage_b_scores: Sequence[float],
    stage_a_retention: Sequence[float],
    *,
    target_stage_b_score: float,
) -> MatchedProgressEstimate | None:
    """Interpolate retention at the first observed upward Stage-B crossing.

    The function uses only adjacent observed checkpoints and never extrapolates.
    ``None`` means the run did not reach the predeclared Stage-B score.
    """

    steps = _validate_step_grid(step_grid)
    scores = _validate_scores(stage_b_scores, expected_length=len(steps))
    retention = _validate_scores(stage_a_retention, expected_length=len(steps))
    _validate_probability(target_stage_b_score, "target_stage_b_score", closed=True)
    target = float(target_stage_b_score)

    if scores[0] >= target:
        return MatchedProgressEstimate(
            target_stage_b_score=target,
            crossing_step=steps[0],
            stage_a_retention=retention[0],
            left_step=steps[0],
            right_step=steps[0],
            interpolation_fraction=0.0,
            exact_checkpoint=True,
        )

    for index in range(1, len(steps)):
        if scores[index] < target or scores[index - 1] >= target:
            continue
        if scores[index] == target:
            return MatchedProgressEstimate(
                target_stage_b_score=target,
                crossing_step=steps[index],
                stage_a_retention=retention[index],
                left_step=steps[index],
                right_step=steps[index],
                interpolation_fraction=0.0,
                exact_checkpoint=True,
            )
        fraction = (target - scores[index - 1]) / (scores[index] - scores[index - 1])
        return MatchedProgressEstimate(
            target_stage_b_score=target,
            crossing_step=steps[index - 1]
            + fraction * (steps[index] - steps[index - 1]),
            stage_a_retention=retention[index - 1]
            + fraction * (retention[index] - retention[index - 1]),
            left_step=steps[index - 1],
            right_step=steps[index],
            interpolation_fraction=fraction,
            exact_checkpoint=False,
        )
    return None


def _validate_counts(successes: int, trials: int) -> None:
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie between zero and trials")


def _validate_probability(value: float, name: str, *, closed: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    number = float(value)
    valid = 0.0 <= number <= 1.0 if closed else 0.0 < number < 1.0
    if not isfinite(number) or not valid:
        interval = "[0, 1]" if closed else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")


def _validate_hard_state_thresholds(reliable: float, unreliable: float) -> None:
    _validate_probability(reliable, "reliable_probability", closed=True)
    _validate_probability(unreliable, "unreliable_probability", closed=True)
    if unreliable >= reliable:
        raise ValueError("unreliable probability must be below reliable probability")


def _validate_step_grid(step_grid: Sequence[int | float]) -> tuple[float, ...]:
    if len(step_grid) < 2:
        raise ValueError("AUC requires at least two steps")
    if any(
        isinstance(step, bool) or not isinstance(step, (int, float))
        for step in step_grid
    ):
        raise ValueError("steps must be finite numbers")
    steps = tuple(float(step) for step in step_grid)
    if not all(isfinite(step) for step in steps):
        raise ValueError("steps must be finite numbers")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("steps must be strictly increasing and unique")
    return steps


def _validate_scores(
    scores: Sequence[float], *, expected_length: int
) -> tuple[float, ...]:
    if len(scores) != expected_length:
        raise ValueError("each curve must have one score for every common-grid step")
    for score in scores:
        _validate_probability(score, "score", closed=True)
    return tuple(float(score) for score in scores)


def _regularized_beta(x: float, alpha: float, beta: float) -> float:
    """Regularized incomplete beta using a stable continued fraction."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        lgamma(alpha + beta)
        - lgamma(alpha)
        - lgamma(beta)
        + alpha * log(x)
        + beta * log1p(-x)
    )
    front = exp(log_front)
    if x < (alpha + 1.0) / (alpha + beta + 2.0):
        result = front * _beta_continued_fraction(alpha, beta, x) / alpha
    else:
        result = 1.0 - (front * _beta_continued_fraction(beta, alpha, 1.0 - x) / beta)
    return min(1.0, max(0.0, result))


def _beta_quantile(probability: float, alpha: float, beta: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        if _regularized_beta(midpoint, alpha, beta) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _beta_continued_fraction(alpha: float, beta: float, x: float) -> float:
    maximum_iterations = 256
    convergence = 3e-14
    floor = 1e-300
    qab = alpha + beta
    qap = alpha + 1.0
    qam = alpha - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (floor if abs(d) < floor else d)
    result = d
    for iteration in range(1, maximum_iterations + 1):
        twice = 2 * iteration
        coefficient = (
            iteration * (beta - iteration) * x / ((qam + twice) * (alpha + twice))
        )
        d = 1.0 + coefficient * d
        d = floor if abs(d) < floor else d
        c = 1.0 + coefficient / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (alpha + iteration)
            * (qab + iteration)
            * x
            / ((alpha + twice) * (qap + twice))
        )
        d = 1.0 + coefficient * d
        d = floor if abs(d) < floor else d
        c = 1.0 + coefficient / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= convergence:
            return result
    raise ArithmeticError("regularized beta continued fraction did not converge")


__all__ = [
    "AcquiredItemSurvival",
    "BinomialObservation",
    "MatchedProgressEstimate",
    "NormalizedAUC",
    "ReliabilityState",
    "StageTransition",
    "TokenSurprisalSummary",
    "ValidationCheckpoint",
    "VerifiedStrategyDiversitySummary",
    "acquired_item_survival",
    "base_stage_a_stage_b_transition_counts",
    "earliest_matching_checkpoint",
    "hard_reliability_state",
    "informative_group_probability",
    "jeffreys_equal_tailed_interval",
    "jeffreys_posterior_parameters",
    "jeffreys_probability_at_least",
    "normalized_stage_b_auc",
    "normalized_stage_b_aucs",
    "posterior_soft_cover",
    "retention_at_first_stage_b_crossing",
    "stage_a_to_b_cover_change",
    "summarize_token_surprisal",
    "summarize_verified_strategy_diversity",
    "unbiased_pass_at_k",
]
