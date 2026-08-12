from __future__ import annotations

from functools import lru_cache
from math import isclose

import pytest

from duraseed.evaluation.analysis import (
    AcquiredItemSurvival,
    BinomialObservation,
    ReliabilityState,
    ValidationCheckpoint,
    VerifiedStrategyDiversitySummary,
    acquired_item_survival,
    base_stage_a_stage_b_transition_counts,
    earliest_matching_checkpoint,
    equal_item_posterior_mean,
    hard_reliability_state,
    informative_group_probability,
    jeffreys_equal_tailed_interval,
    jeffreys_posterior_mean,
    jeffreys_posterior_parameters,
    jeffreys_probability_at_least,
    normalized_stage_b_auc,
    normalized_stage_b_aucs,
    posterior_mean_gain,
    posterior_soft_cover,
    retention_at_first_stage_b_crossing,
    stage_a_to_b_cover_change,
    summarize_token_surprisal,
    summarize_verified_strategy_diversity,
    targeted_minus_sentinel_specificity,
    unbiased_pass_at_k,
)
from duraseed.data.manifests import (
    DatasetManifest,
    TCESTaskManifestRecord,
    build_manifest,
    build_tces_record,
)
from duraseed.provenance import derive_namespaced_seed
from duraseed.run_records import GenerationRecord, RewardRecord
from duraseed.schemas import VerificationFailure, VerificationResult
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig


@lru_cache(maxsize=None)
def _multi_family_manifest(seed: int = 404) -> DatasetManifest:
    split = "a_test_multi"
    generator = TCESGenerator(
        derive_namespaced_seed(seed, "dataset.tces.split", split),
        TCESGeneratorConfig(
            n_operands=3,
            max_tree_depth=3,
            max_ast_nodes=5,
            max_valid_expressions=1_000,
            split=split,
        ),
    )
    records: list[TCESTaskManifestRecord] = []
    for item_index in range(32):
        record = build_tces_record(generator.generate(item_index))
        records.append(record)
        if len(record.valid_family_ids) >= 2:
            return build_manifest(
                name=f"strategy-diversity-{seed}",
                task_family="tces",
                split=split,
                generator_version="1.0.0",
                root_seed=seed,
                records=records,
            )
    raise AssertionError("test generator produced no multi-family TCES item")


def _successful_verification(family_id: str) -> VerificationResult:
    return VerificationResult(
        valid_answer_tag=True,
        valid_lexing=True,
        valid_syntax=True,
        valid_operand_multiset=True,
        valid_operations=True,
        valid_intermediates=True,
        correct_target=True,
        reward=1.0,
        value_num=1,
        value_den=1,
        canonical_expression="1",
        strategy_family_id=family_id,
        ast_json={"type": "integer", "value": 1},
    )


def _strategy_rows(
    verifications: list[VerificationResult],
    *,
    manifest: DatasetManifest | None = None,
    task_family: str = "tces",
) -> tuple[list[GenerationRecord], list[RewardRecord]]:
    manifest = manifest or _multi_family_manifest()
    record = manifest.records[-1]
    assert isinstance(record, TCESTaskManifestRecord)
    task_id = record.task_id
    generations: list[GenerationRecord] = []
    rewards: list[RewardRecord] = []
    source_split = "a_test_multi" if task_family == "tces" else "b_test"
    for sample_index, verification in enumerate(verifications):
        sample_id = f"{task_id}:sample:{sample_index}"
        generations.append(
            GenerationRecord(
                sample_id=sample_id,
                sample_index=sample_index,
                sampling_seed=100 + sample_index,
                purpose="evaluation",
                checkpoint_stage="stage_a",
                training_step=10,
                sampler_checkpoint_path="tinker://sampler/stage-a",
                task_manifest_id=manifest.manifest_id,
                task_id=task_id,
                task_family=task_family,  # type: ignore[arg-type]
                source_split=source_split,
                prompt_text="prompt",
                completion_text="completion",
                prompt_tokens=1,
                sampled_tokens=1,
                run_id="strategy-diversity-run",
                item_index=record.item_index,
                family_id=verification.strategy_family_id,
                reward=verification.reward,
            )
        )
        rewards.append(
            RewardRecord(
                reward_id=f"{task_id}:reward:{sample_index}",
                sample_id=sample_id,
                task_id=task_id,
                reward=verification.reward,
                exact_verification=verification,
            )
        )
    return generations, rewards


def test_jeffreys_posterior_and_symmetric_tail_known_values() -> None:
    assert jeffreys_posterior_parameters(3, 8) == (3.5, 5.5)
    assert jeffreys_posterior_mean(3, 8) == pytest.approx(3.5 / 9.0)
    assert jeffreys_probability_at_least(1, 2, 0.5) == pytest.approx(0.5)
    # Beta(1/2, 3/2) has a closed-form tail at 1/2.
    assert jeffreys_probability_at_least(0, 1, 0.5) == pytest.approx(0.1816901138162093)

    lower, upper = jeffreys_equal_tailed_interval(1, 2, credible_mass=0.90)
    assert lower == pytest.approx(1.0 - upper)
    assert lower < 0.5 < upper


def test_informative_group_probability_matches_binary_group_formula() -> None:
    assert informative_group_probability(0.0, 8) == 0.0
    assert informative_group_probability(1.0, 8) == 0.0
    assert informative_group_probability(0.5, 8) == pytest.approx(127 / 128)
    assert informative_group_probability(0.2, 2) == pytest.approx(0.32)

    with pytest.raises(ValueError, match="at least two"):
        informative_group_probability(0.5, 1)


def test_posterior_soft_cover_averages_item_posterior_mass() -> None:
    observations = [BinomialObservation(0, 1), BinomialObservation(1, 1)]

    assert posterior_soft_cover(observations, 0.5) == pytest.approx(0.5)


def test_equal_item_mean_and_targeted_minus_sentinel_specificity() -> None:
    origin_targeted = [BinomialObservation(0, 4), BinomialObservation(1, 4)]
    current_targeted = [BinomialObservation(2, 4), BinomialObservation(3, 4)]
    origin_sentinel = [BinomialObservation(0, 4), BinomialObservation(2, 4)]
    current_sentinel = [BinomialObservation(1, 4), BinomialObservation(2, 4)]

    assert equal_item_posterior_mean(origin_targeted) == pytest.approx(0.2)
    assert posterior_mean_gain(origin_targeted, current_targeted) == pytest.approx(0.4)
    assert targeted_minus_sentinel_specificity(
        origin_targeted,
        current_targeted,
        origin_sentinel,
        current_sentinel,
    ) == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("stage_a", "stage_b", "expected_sign"),
    [
        (BinomialObservation(0, 1), BinomialObservation(1, 1), 1),
        (BinomialObservation(1, 1), BinomialObservation(1, 1), 0),
        (BinomialObservation(1, 1), BinomialObservation(0, 1), -1),
    ],
)
def test_stage_a_to_b_cover_change_can_be_positive_zero_or_negative(
    stage_a: BinomialObservation,
    stage_b: BinomialObservation,
    expected_sign: int,
) -> None:
    absolute_post_b = posterior_soft_cover((stage_b,), 0.5)
    change = stage_a_to_b_cover_change((stage_a,), (stage_b,), 0.5)

    assert change == pytest.approx(
        absolute_post_b - posterior_soft_cover((stage_a,), 0.5)
    )
    assert (change > 0) - (change < 0) == expected_sign


def test_stage_a_to_b_cover_change_rejects_different_sampling_budgets() -> None:
    with pytest.raises(ValueError, match="same per-item trial budgets"):
        stage_a_to_b_cover_change(
            (BinomialObservation(1, 8),),
            (BinomialObservation(2, 16),),
            0.25,
        )


def test_hard_states_and_base_stage_a_stage_b_transition_counts() -> None:
    unreliable = BinomialObservation(0, 64)
    uncertain = BinomialObservation(16, 64)
    reliable = BinomialObservation(64, 64)

    assert hard_reliability_state(unreliable, 0.25) is ReliabilityState.UNRELIABLE
    assert hard_reliability_state(uncertain, 0.25) is ReliabilityState.UNCERTAIN
    assert hard_reliability_state(reliable, 0.25) is ReliabilityState.RELIABLE
    assert base_stage_a_stage_b_transition_counts(
        [
            (unreliable, reliable, reliable),
            (unreliable, reliable, uncertain),
            (reliable, reliable, reliable),
        ],
        0.25,
    ) == {
        (
            ReliabilityState.UNRELIABLE,
            ReliabilityState.RELIABLE,
            ReliabilityState.RELIABLE,
        ): 1,
        (
            ReliabilityState.UNRELIABLE,
            ReliabilityState.RELIABLE,
            ReliabilityState.UNCERTAIN,
        ): 1,
        (
            ReliabilityState.RELIABLE,
            ReliabilityState.RELIABLE,
            ReliabilityState.RELIABLE,
        ): 1,
    }


def test_acquired_item_survival_distinguishes_retained_uncertain_and_lost() -> None:
    unreliable = BinomialObservation(0, 64)
    uncertain = BinomialObservation(16, 64)
    reliable = BinomialObservation(64, 64)

    assert acquired_item_survival(
        [
            (unreliable, reliable, reliable),
            (unreliable, reliable, uncertain),
            (unreliable, reliable, unreliable),
            (uncertain, reliable, reliable),
            (reliable, reliable, reliable),
        ],
        0.25,
    ) == AcquiredItemSurvival(
        acquired_count=4,
        retained_count=2,
        uncertain_count=1,
        lost_count=1,
        survival_rate=pytest.approx(0.5),
    )


def test_acquired_item_survival_is_undefined_when_stage_a_acquires_no_items() -> None:
    reliable = BinomialObservation(64, 64)

    summary = acquired_item_survival([(reliable, reliable, reliable)], 0.25)

    assert summary.acquired_count == 0
    assert summary.retained_count == 0
    assert summary.uncertain_count == 0
    assert summary.lost_count == 0
    assert summary.survival_rate is None


def test_acquired_item_survival_preserves_transition_validation() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        acquired_item_survival([], 0.25)
    with pytest.raises(ValueError, match="exactly three stages"):
        acquired_item_survival([(BinomialObservation(0, 1),)], 0.25)  # type: ignore[list-item]


def test_token_surprisal_reports_observation_coverage() -> None:
    summary = summarize_token_surprisal([-1.0, None, -3.0, 0.0])

    assert summary.total_token_count == 4
    assert summary.observed_logprob_count == 3
    assert summary.coverage == pytest.approx(0.75)
    assert summary.mean_surprisal == pytest.approx(4 / 3)


def test_token_surprisal_allows_explicitly_missing_logprobs() -> None:
    summary = summarize_token_surprisal([None, None])

    assert summary.total_token_count == 2
    assert summary.observed_logprob_count == 0
    assert summary.coverage == 0.0
    assert summary.mean_surprisal is None


@pytest.mark.parametrize(
    "invalid", [True, float("nan"), float("inf"), float("-inf"), 0.1]
)
def test_token_surprisal_rejects_invalid_observed_logprobs(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite non-positive"):
        summarize_token_surprisal([invalid])


def test_token_surprisal_requires_at_least_one_token() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        summarize_token_surprisal([])


def test_verified_strategy_diversity_counts_only_exact_successes() -> None:
    manifest = _multi_family_manifest()
    record = manifest.records[-1]
    assert isinstance(record, TCESTaskManifestRecord)
    first_family, second_family = record.valid_family_ids[:2]
    generations, rewards = _strategy_rows(
        [
            _successful_verification(second_family),
            VerificationResult(failure_code=VerificationFailure.INVALID_SYNTAX),
            _successful_verification(first_family),
            _successful_verification(second_family),
        ],
        manifest=manifest,
    )
    summary = summarize_verified_strategy_diversity(manifest, generations, rewards)

    assert summary == VerifiedStrategyDiversitySummary(
        total_completion_count=4,
        valid_completion_count=3,
        valid_completion_rate=0.75,
        valid_family_ids=tuple(sorted((first_family, second_family))),
        unique_valid_family_count=2,
        valid_family_counts=tuple(sorted(((first_family, 1), (second_family, 2)))),
    )


def test_verified_strategy_diversity_reports_no_valid_families() -> None:
    manifest = _multi_family_manifest()
    generations, rewards = _strategy_rows(
        [VerificationResult(failure_code=VerificationFailure.WRONG_TARGET)],
        manifest=manifest,
    )
    summary = summarize_verified_strategy_diversity(manifest, generations, rewards)

    assert summary.total_completion_count == 1
    assert summary.valid_completion_count == 0
    assert summary.valid_completion_rate == 0.0
    assert summary.valid_family_ids == ()
    assert summary.unique_valid_family_count == 0
    assert summary.valid_family_counts == ()


def test_verified_strategy_diversity_requires_typed_nonempty_input() -> None:
    manifest = _multi_family_manifest()
    with pytest.raises(ValueError, match="at least one generation"):
        summarize_verified_strategy_diversity(manifest, [], [])
    with pytest.raises(TypeError, match="GenerationRecord instances"):
        summarize_verified_strategy_diversity(
            manifest,
            [object()],  # type: ignore[list-item]
            [],
        )


def test_verified_strategy_diversity_rejects_maps_mixed_items_and_broken_links() -> (
    None
):
    manifest = _multi_family_manifest()
    record = manifest.records[-1]
    assert isinstance(record, TCESTaskManifestRecord)
    family_id = record.valid_family_ids[0]
    maps_generations, maps_rewards = _strategy_rows(
        [_successful_verification(family_id)],
        manifest=manifest,
        task_family="maps",
    )
    with pytest.raises(ValueError, match="TCES generations"):
        summarize_verified_strategy_diversity(
            manifest,
            maps_generations,
            maps_rewards,
        )

    first_generations, first_rewards = _strategy_rows(
        [_successful_verification(family_id)],
        manifest=manifest,
    )
    second_manifest = _multi_family_manifest(405)
    second_record = second_manifest.records[-1]
    assert isinstance(second_record, TCESTaskManifestRecord)
    second_generations, second_rewards = _strategy_rows(
        [_successful_verification(second_record.valid_family_ids[0])],
        manifest=second_manifest,
    )
    with pytest.raises(ValueError, match="one TCES item"):
        summarize_verified_strategy_diversity(
            manifest,
            first_generations + second_generations,
            first_rewards + second_rewards,
        )
    with pytest.raises(ValueError, match="linkages differ"):
        summarize_verified_strategy_diversity(manifest, first_generations, [])


def test_unbiased_pass_at_k_known_value_and_edges() -> None:
    assert unbiased_pass_at_k(2, 10, 3) == pytest.approx(8 / 15)
    assert unbiased_pass_at_k(0, 10, 4) == 0.0
    assert unbiased_pass_at_k(8, 10, 3) == 1.0
    with pytest.raises(ValueError, match="trials >= k"):
        unbiased_pass_at_k(1, 2, 3)


def test_checkpoint_selection_uses_earliest_real_checkpoint_without_interpolation() -> (
    None
):
    checkpoints = [
        ValidationCheckpoint("step-100", 100, 0.251),
        ValidationCheckpoint("step-50", 50, 0.265),
        ValidationCheckpoint("step-25", 25, 0.239),
    ]

    selected = earliest_matching_checkpoint(
        checkpoints,
        target=0.25,
        tolerance=0.015,
    )
    assert selected == checkpoints[2]
    assert earliest_matching_checkpoint(checkpoints, target=0.9, tolerance=0.01) is None


def test_normalized_auc_uses_actual_common_grid_spacing() -> None:
    epsilon = 1e-12
    result = normalized_stage_b_auc([0, 1, 3], [0.2, 0.4, 0.8], epsilon=epsilon)

    assert result.absolute_auc == pytest.approx(0.5)
    assert result.raw_gain_auc == result.absolute_auc - result.initial_score
    assert result.headroom_normalized_gain_auc == (
        result.raw_gain_auc / (1.0 - result.initial_score + epsilon)
    )
    assert result.initial_score == 0.2
    assert result.final_score == 0.8
    assert isclose(
        normalized_stage_b_aucs(
            [0, 1, 3],
            {"B-G": [0.2, 0.4, 0.8]},
        )["B-G"].absolute_auc,
        0.5,
    )


def test_retention_at_exact_and_interpolated_stage_b_progress() -> None:
    exact = retention_at_first_stage_b_crossing(
        [0, 10, 40],
        [0.1, 0.4, 0.7],
        [0.9, 0.8, 0.5],
        target_stage_b_score=0.4,
    )
    assert exact is not None
    assert exact.crossing_step == 10
    assert exact.stage_a_retention == 0.8
    assert exact.left_step == exact.right_step == 10
    assert exact.exact_checkpoint is True

    interpolated = retention_at_first_stage_b_crossing(
        [0, 10, 40],
        [0.1, 0.3, 0.9],
        [0.9, 0.7, 0.4],
        target_stage_b_score=0.5,
    )
    assert interpolated is not None
    assert interpolated.interpolation_fraction == pytest.approx(1 / 3)
    assert interpolated.crossing_step == pytest.approx(20.0)
    assert interpolated.stage_a_retention == pytest.approx(0.6)
    assert interpolated.left_step == 10
    assert interpolated.right_step == 40
    assert interpolated.exact_checkpoint is False


def test_matched_progress_uses_first_crossing_and_never_extrapolates() -> None:
    already_reached = retention_at_first_stage_b_crossing(
        [0, 5, 10],
        [0.6, 0.3, 0.8],
        [0.95, 0.9, 0.7],
        target_stage_b_score=0.5,
    )
    assert already_reached is not None
    assert already_reached.crossing_step == 0
    assert already_reached.stage_a_retention == 0.95

    first_upward = retention_at_first_stage_b_crossing(
        [0, 5, 10, 20],
        [0.1, 0.7, 0.2, 0.8],
        [0.95, 0.85, 0.8, 0.6],
        target_stage_b_score=0.5,
    )
    assert first_upward is not None
    assert first_upward.crossing_step == pytest.approx(10 / 3)
    assert first_upward.stage_a_retention == pytest.approx(53 / 60)

    assert (
        retention_at_first_stage_b_crossing(
            [0, 5, 10],
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],
            target_stage_b_score=0.5,
        )
        is None
    )


@pytest.mark.parametrize(
    ("steps", "scores", "retention", "message"),
    [
        ([0, 0], [0.1, 0.2], [0.9, 0.8], "strictly increasing"),
        ([0, 1], [0.1], [0.9, 0.8], "every common-grid step"),
        ([0, 1], [0.1, 0.2], [0.9], "every common-grid step"),
        ([0, 1], [0.1, 1.1], [0.9, 0.8], "score"),
    ],
)
def test_matched_progress_rejects_incomparable_curves(
    steps: list[int],
    scores: list[float],
    retention: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        retention_at_first_stage_b_crossing(
            steps,
            scores,
            retention,
            target_stage_b_score=0.5,
        )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: BinomialObservation(-1, 16), "successes"),
        (lambda: BinomialObservation(1, 0), "trials"),
        (lambda: posterior_soft_cover([], 0.25), "at least one"),
        (lambda: equal_item_posterior_mean([]), "at least one"),
        (
            lambda: posterior_mean_gain(
                [BinomialObservation(0, 1)],
                [BinomialObservation(0, 1), BinomialObservation(1, 1)],
            ),
            "same item count",
        ),
        (lambda: jeffreys_probability_at_least(1, 2, 1.0), "tau"),
        (
            lambda: normalized_stage_b_auc([0, 0], [0.2, 0.3]),
            "strictly increasing",
        ),
        (
            lambda: normalized_stage_b_aucs(
                [0, 1, 2],
                {"B-G": [0.2, 0.3]},
            ),
            "every common-grid step",
        ),
    ],
)
def test_analysis_rejects_inputs_that_would_change_scientific_meaning(
    call: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()  # type: ignore[operator]
