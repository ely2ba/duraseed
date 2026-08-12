"""Pure reducers for the post-M0 TCES boundary scan.

This module summarizes authenticated observations and evaluates caller-supplied
eligibility rules.  It never chooses thresholds, families, panels, or M0.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from statistics import median, pstdev
from typing import TYPE_CHECKING, Literal

from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
from duraseed.evaluation.analysis import (
    informative_group_probability,
    jeffreys_posterior_mean,
)

if TYPE_CHECKING:
    from duraseed.run_records import GenerationRecord, RewardRecord


InformativeProbabilityStatistic = Literal[
    "at_equal_item_mean",
    "mean_across_items",
]

BOUNDARY_MINIMUM_INFORMATIVE_PROBABILITY = 0.25
BOUNDARY_MAXIMUM_EQUAL_ITEM_SUCCESS = 0.50
BOUNDARY_MINIMUM_FORMAT_COMPLIANCE = 0.50
BOUNDARY_MAXIMUM_TRUNCATION_RATE = 0.50


@dataclass(frozen=True, slots=True)
class BoundaryItemSummary:
    """M0 evidence for one candidate item, grouped by intended family."""

    task_id: str
    item_index: int
    intended_family_id: str
    trials: int
    successes: int
    posterior_mean_success: float
    informative_group_probability: float
    format_compliance: float
    parser_valid_rate: float
    truncation_rate: float


@dataclass(frozen=True, slots=True)
class BoundaryFamilySummary:
    """Equal-item family summary used as input to later panel selection."""

    intended_family_id: str
    sampler_checkpoint_path: str
    group_size: int
    items: tuple[BoundaryItemSummary, ...]
    equal_item_posterior_mean_success: float
    pooled_posterior_mean_success: float
    median_item_posterior_mean_success: float
    item_posterior_mean_dispersion: float
    informative_probability_at_equal_item_mean: float
    mean_item_informative_probability: float
    successful_item_count: int
    total_successes: int
    maximum_item_success_share: float | None
    format_compliance: float
    parser_valid_rate: float
    truncation_rate: float
    observed_strategy_family_ids: tuple[str, ...]

    @property
    def item_count(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class BoundaryEligibilityCriteria:
    """All calibration-dependent gates required to assess one family."""

    informative_probability_statistic: InformativeProbabilityStatistic
    minimum_informative_group_probability: float
    maximum_equal_item_posterior_mean_success: float
    minimum_successful_items: int
    maximum_item_success_share: float
    minimum_format_compliance: float
    maximum_truncation_rate: float
    minimum_disjoint_instances_by_split: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.informative_probability_statistic not in {
            "at_equal_item_mean",
            "mean_across_items",
        }:
            raise ValueError("unknown informative-probability statistic")
        for name in (
            "minimum_informative_group_probability",
            "maximum_equal_item_posterior_mean_success",
            "maximum_item_success_share",
            "minimum_format_compliance",
            "maximum_truncation_rate",
        ):
            _validate_probability(getattr(self, name), name)
        if (
            isinstance(self.minimum_successful_items, bool)
            or not isinstance(self.minimum_successful_items, int)
            or self.minimum_successful_items < 2
        ):
            raise ValueError(
                "minimum_successful_items must be an integer of at least 2"
            )
        splits = tuple(
            split_name for split_name, _ in self.minimum_disjoint_instances_by_split
        )
        if (
            not splits
            or splits != tuple(sorted(set(splits)))
            or any(not split_name.strip() for split_name in splits)
        ):
            raise ValueError(
                "minimum disjoint-instance splits must be nonempty, unique, and sorted"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for _, count in self.minimum_disjoint_instances_by_split
        ):
            raise ValueError("minimum disjoint-instance counts must be positive")


@dataclass(frozen=True, slots=True)
class BoundaryEligibilityAssessment:
    """Fail-closed result for one family under explicit criteria."""

    intended_family_id: str
    eligible: bool
    exclusion_reasons: tuple[str, ...]
    informative_group_probability: float


class BoundaryEvidenceError(ValueError):
    """Raised when raw M0 evidence is missing, ambiguous, or inconsistent."""


def assess_refinement_finalist_gate(
    summary: BoundaryFamilySummary,
) -> BoundaryEligibilityAssessment:
    """Apply the frozen Stage-2 gate before held-out-item confirmation."""

    return _assess_observation_gate(
        summary,
        minimum_successful_items=2,
        maximum_item_success_share=None,
    )


def assess_confirmation_observation_gate(
    summary: BoundaryFamilySummary,
) -> BoundaryEligibilityAssessment:
    """Apply the frozen Stage-3 evidence gate before structural matching."""

    return _assess_observation_gate(
        summary,
        minimum_successful_items=4,
        maximum_item_success_share=0.25,
    )


def _assess_observation_gate(
    summary: BoundaryFamilySummary,
    *,
    minimum_successful_items: int,
    maximum_item_success_share: float | None,
) -> BoundaryEligibilityAssessment:
    if not isinstance(summary, BoundaryFamilySummary):
        raise TypeError("summary must be a BoundaryFamilySummary")
    reasons: list[str] = []
    informative = summary.informative_probability_at_equal_item_mean
    if informative < BOUNDARY_MINIMUM_INFORMATIVE_PROBABILITY:
        reasons.append("informative_group_probability_below_floor")
    if summary.equal_item_posterior_mean_success > BOUNDARY_MAXIMUM_EQUAL_ITEM_SUCCESS:
        reasons.append("posterior_mean_near_saturation")
    if summary.successful_item_count < minimum_successful_items:
        reasons.append("insufficient_successful_items")
    if maximum_item_success_share is not None:
        if summary.maximum_item_success_share is None:
            reasons.append("no_observed_successes")
        elif summary.maximum_item_success_share > maximum_item_success_share:
            reasons.append("single_item_success_domination")
    if summary.format_compliance < BOUNDARY_MINIMUM_FORMAT_COMPLIANCE:
        reasons.append("format_compliance_below_floor")
    if summary.truncation_rate > BOUNDARY_MAXIMUM_TRUNCATION_RATE:
        reasons.append("truncation_rate_above_ceiling")
    return BoundaryEligibilityAssessment(
        intended_family_id=summary.intended_family_id,
        eligible=not reasons,
        exclusion_reasons=tuple(reasons),
        informative_group_probability=informative,
    )


def summarize_m0_boundary(
    manifest: DatasetManifest,
    generations: Sequence[GenerationRecord],
    rewards: Sequence[RewardRecord],
    *,
    group_size: int,
    expected_run_ids: Sequence[str] | None = None,
    additional_manifests: Sequence[DatasetManifest] = (),
) -> tuple[BoundaryFamilySummary, ...]:
    """Reduce paired M0 generation/reward rows into deterministic families."""

    manifests = (manifest, *tuple(additional_manifests))
    if any(
        not isinstance(candidate, DatasetManifest)
        or candidate.task_family != "tces"
        or candidate.split != "a_candidate"
        for candidate in manifests
    ):
        raise BoundaryEvidenceError(
            "boundary summaries require a_candidate TCES manifests"
        )
    if any(not candidate.records for candidate in manifests):
        raise BoundaryEvidenceError("boundary manifests must contain records")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 2
    ):
        raise BoundaryEvidenceError("group_size must be an integer of at least two")
    records = tuple(record for candidate in manifests for record in candidate.records)
    if not all(isinstance(record, TCESTaskManifestRecord) for record in records):
        raise BoundaryEvidenceError("boundary manifest contains a non-TCES record")
    record_by_task = {record.task_id: record for record in records}
    if len(record_by_task) != len(records):
        raise BoundaryEvidenceError("boundary manifests contain duplicate task IDs")
    manifest_by_task = {
        record.task_id: candidate.manifest_id
        for candidate in manifests
        for record in candidate.records
    }

    generation_rows = tuple(generations)
    reward_rows = tuple(rewards)
    if not generation_rows:
        raise BoundaryEvidenceError("boundary summary requires generation rows")
    sample_ids = [row.sample_id for row in generation_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise BoundaryEvidenceError("generation rows contain duplicate sample IDs")
    coordinates = [
        (row.task_id, row.sampling_seed, row.sample_index) for row in generation_rows
    ]
    if len(coordinates) != len(set(coordinates)):
        raise BoundaryEvidenceError(
            "generation rows contain duplicate sample coordinates"
        )
    reward_ids = [row.reward_id for row in reward_rows]
    if len(reward_ids) != len(set(reward_ids)):
        raise BoundaryEvidenceError("reward rows contain duplicate reward IDs")
    rewards_by_sample: dict[str, RewardRecord] = {}
    for row in reward_rows:
        if row.sample_id in rewards_by_sample:
            raise BoundaryEvidenceError("reward rows duplicate a sample linkage")
        rewards_by_sample[row.sample_id] = row
    if set(rewards_by_sample) != set(sample_ids):
        raise BoundaryEvidenceError("generation and reward sample linkages differ")

    if any(
        row.run_id is None
        or row.seed is None
        or row.origin_sampler_checkpoint_path is None
        or row.sampling_seed is None
        or row.sampling_temperature is None
        or row.sampling_top_p is None
        or row.sampling_max_tokens is None
        for row in generation_rows
    ):
        raise BoundaryEvidenceError(
            "boundary rows require run, seed, origin, sampling-seed, and sampling-config identity"
        )
    run_ids = {row.run_id for row in generation_rows}
    if expected_run_ids is None:
        if len(run_ids) != 1:
            raise BoundaryEvidenceError(
                "boundary rows must share one run and M0 sampling configuration"
            )
    else:
        expected = tuple(expected_run_ids)
        if (
            not expected
            or len(expected) != len(set(expected))
            or any(
                not isinstance(run_id, str) or not run_id.strip() for run_id in expected
            )
        ):
            raise BoundaryEvidenceError(
                "expected_run_ids must be unique nonempty run identities"
            )
        if run_ids != set(expected):
            raise BoundaryEvidenceError(
                "boundary rows differ from the expected sequential run identities"
            )
    evidence_identities = {
        (
            row.seed,
            row.origin_sampler_checkpoint_path,
            row.sampler_checkpoint_path,
            row.training_step,
            row.sampling_temperature,
            row.sampling_top_p,
            row.sampling_max_tokens,
        )
        for row in generation_rows
    }
    if len(evidence_identities) != 1:
        raise BoundaryEvidenceError(
            "boundary rows must share one M0 sampling configuration"
        )
    checkpoint_path = generation_rows[0].sampler_checkpoint_path
    paired_by_task: dict[str, list[tuple[GenerationRecord, RewardRecord]]] = (
        defaultdict(list)
    )
    for generation in generation_rows:
        record = record_by_task.get(generation.task_id)
        if record is None:
            raise BoundaryEvidenceError("generation task is absent from the manifest")
        if (
            generation.task_manifest_id != manifest_by_task[generation.task_id]
            or generation.task_family != "tces"
            or generation.source_split != "a_candidate"
        ):
            raise BoundaryEvidenceError(
                "generation manifest/split identity is inconsistent"
            )
        if (
            generation.checkpoint_stage != "m0"
            or generation.purpose != "evaluation"
            or generation.sampler_checkpoint_path
            != generation.origin_sampler_checkpoint_path
        ):
            raise BoundaryEvidenceError(
                "boundary rows must evaluate the selected common M0"
            )
        if generation.item_index != record.item_index:
            raise BoundaryEvidenceError(
                "generation item_index differs from its manifest record"
            )
        if generation.assigned_family_id != record.intended_family:
            raise BoundaryEvidenceError(
                "generation assigned_family_id differs from manifest intended_family"
            )
        if generation.stop_reason is None:
            raise BoundaryEvidenceError("boundary generation must record stop_reason")
        reward = rewards_by_sample[generation.sample_id]
        if reward.task_id != generation.task_id:
            raise BoundaryEvidenceError("reward task identity differs from generation")
        if generation.reward is None or generation.reward != reward.reward:
            raise BoundaryEvidenceError(
                "generation reward differs from exact reward row"
            )
        if generation.family_id != reward.exact_verification.strategy_family_id:
            raise BoundaryEvidenceError(
                "generation response family differs from exact verification"
            )
        paired_by_task[generation.task_id].append((generation, reward))
    missing = set(record_by_task).difference(paired_by_task)
    if missing:
        raise BoundaryEvidenceError(
            "one or more manifest items have no M0 observations"
        )

    items_by_family: dict[str, list[BoundaryItemSummary]] = defaultdict(list)
    strategies_by_family: dict[str, set[str]] = defaultdict(set)
    for task_id in sorted(paired_by_task):
        record = record_by_task[task_id]
        pairs = paired_by_task[task_id]
        trials = len(pairs)
        successes = sum(int(reward.reward) for _, reward in pairs)
        posterior_mean = jeffreys_posterior_mean(successes, trials)
        valid_tags = sum(
            reward.exact_verification.valid_answer_tag for _, reward in pairs
        )
        valid_parses = sum(
            reward.exact_verification.valid_syntax for _, reward in pairs
        )
        truncated = sum(
            _is_truncation(generation.stop_reason) for generation, _ in pairs
        )
        for _, reward in pairs:
            family_id = reward.exact_verification.strategy_family_id
            if family_id is not None:
                strategies_by_family[record.intended_family].add(family_id)
        items_by_family[record.intended_family].append(
            BoundaryItemSummary(
                task_id=task_id,
                item_index=record.item_index,
                intended_family_id=record.intended_family,
                trials=trials,
                successes=successes,
                posterior_mean_success=posterior_mean,
                informative_group_probability=informative_group_probability(
                    posterior_mean, group_size
                ),
                format_compliance=valid_tags / trials,
                parser_valid_rate=valid_parses / trials,
                truncation_rate=truncated / trials,
            )
        )

    summaries: list[BoundaryFamilySummary] = []
    for family_id in sorted(items_by_family):
        items = tuple(sorted(items_by_family[family_id], key=lambda item: item.task_id))
        item_means = tuple(item.posterior_mean_success for item in items)
        equal_item_mean = fsum(item_means) / len(item_means)
        total_successes = sum(item.successes for item in items)
        total_trials = sum(item.trials for item in items)
        maximum_share = (
            max(item.successes for item in items) / total_successes
            if total_successes
            else None
        )
        summaries.append(
            BoundaryFamilySummary(
                intended_family_id=family_id,
                sampler_checkpoint_path=checkpoint_path,
                group_size=group_size,
                items=items,
                equal_item_posterior_mean_success=equal_item_mean,
                pooled_posterior_mean_success=jeffreys_posterior_mean(
                    total_successes, total_trials
                ),
                median_item_posterior_mean_success=median(item_means),
                item_posterior_mean_dispersion=(
                    pstdev(item_means) if len(item_means) > 1 else 0.0
                ),
                informative_probability_at_equal_item_mean=(
                    informative_group_probability(equal_item_mean, group_size)
                ),
                mean_item_informative_probability=fsum(
                    item.informative_group_probability for item in items
                )
                / len(items),
                successful_item_count=sum(item.successes > 0 for item in items),
                total_successes=total_successes,
                maximum_item_success_share=maximum_share,
                format_compliance=fsum(item.format_compliance for item in items)
                / len(items),
                parser_valid_rate=fsum(item.parser_valid_rate for item in items)
                / len(items),
                truncation_rate=fsum(item.truncation_rate for item in items)
                / len(items),
                observed_strategy_family_ids=tuple(
                    sorted(strategies_by_family[family_id])
                ),
            )
        )
    return tuple(summaries)


def assess_boundary_eligibility(
    summary: BoundaryFamilySummary,
    criteria: BoundaryEligibilityCriteria,
    *,
    available_disjoint_instances_by_split: Mapping[str, int],
    enumeration_complete: bool,
    can_be_matched: bool,
) -> BoundaryEligibilityAssessment:
    """Apply explicit calibrated gates without selecting any family or panel."""

    if not isinstance(summary, BoundaryFamilySummary):
        raise TypeError("summary must be a BoundaryFamilySummary")
    if not isinstance(criteria, BoundaryEligibilityCriteria):
        raise TypeError("criteria must be a BoundaryEligibilityCriteria")
    if not isinstance(available_disjoint_instances_by_split, Mapping):
        raise TypeError("available_disjoint_instances_by_split must be a mapping")
    if any(
        not isinstance(split_name, str)
        or not split_name.strip()
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for split_name, count in available_disjoint_instances_by_split.items()
    ):
        raise ValueError(
            "available disjoint-instance counts require named nonnegative integers"
        )
    if not isinstance(enumeration_complete, bool) or not isinstance(
        can_be_matched, bool
    ):
        raise ValueError("enumeration_complete and can_be_matched must be booleans")

    informative = (
        summary.informative_probability_at_equal_item_mean
        if criteria.informative_probability_statistic == "at_equal_item_mean"
        else summary.mean_item_informative_probability
    )
    reasons: list[str] = []
    if informative < criteria.minimum_informative_group_probability:
        reasons.append("informative_group_probability_below_floor")
    if (
        summary.equal_item_posterior_mean_success
        > criteria.maximum_equal_item_posterior_mean_success
    ):
        reasons.append("posterior_mean_near_saturation")
    if summary.successful_item_count < criteria.minimum_successful_items:
        reasons.append("insufficient_successful_items")
    if summary.maximum_item_success_share is None:
        reasons.append("no_observed_successes")
    elif summary.maximum_item_success_share > criteria.maximum_item_success_share:
        reasons.append("single_item_success_domination")
    if summary.format_compliance < criteria.minimum_format_compliance:
        reasons.append("format_compliance_below_floor")
    if summary.truncation_rate > criteria.maximum_truncation_rate:
        reasons.append("truncation_rate_above_ceiling")
    for split_name, minimum_count in criteria.minimum_disjoint_instances_by_split:
        if available_disjoint_instances_by_split.get(split_name, 0) < minimum_count:
            reasons.append(f"insufficient_disjoint_instances:{split_name}")
    if not enumeration_complete:
        reasons.append("incomplete_exact_enumeration")
    if not can_be_matched:
        reasons.append("not_matchable_into_panel_design")
    return BoundaryEligibilityAssessment(
        intended_family_id=summary.intended_family_id,
        eligible=not reasons,
        exclusion_reasons=tuple(reasons),
        informative_group_probability=informative,
    )


def _is_truncation(stop_reason: str) -> bool:
    normalized = stop_reason.strip().lower().rsplit(".", 1)[-1]
    return normalized in {"length", "max_tokens", "max_tokens_reached"}


def _validate_probability(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite probability")


__all__ = [
    "BOUNDARY_MAXIMUM_EQUAL_ITEM_SUCCESS",
    "BOUNDARY_MAXIMUM_TRUNCATION_RATE",
    "BOUNDARY_MINIMUM_FORMAT_COMPLIANCE",
    "BOUNDARY_MINIMUM_INFORMATIVE_PROBABILITY",
    "BoundaryEligibilityAssessment",
    "BoundaryEligibilityCriteria",
    "BoundaryEvidenceError",
    "BoundaryFamilySummary",
    "BoundaryItemSummary",
    "InformativeProbabilityStatistic",
    "assess_boundary_eligibility",
    "assess_confirmation_observation_gate",
    "assess_refinement_finalist_gate",
    "summarize_m0_boundary",
]
