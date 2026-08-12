from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import pytest

from duraseed.data.boundary import (
    BoundaryEligibilityCriteria,
    BoundaryEvidenceError,
    assess_boundary_eligibility,
    assess_confirmation_observation_gate,
    assess_refinement_finalist_gate,
    summarize_m0_boundary,
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


ROOT_SEED = 313
M0_PATH = "tinker://sampler/m0-boundary"


@lru_cache(maxsize=None)
def _shared_family_manifest(item_count: int) -> DatasetManifest:
    split = "a_candidate"
    config = TCESGeneratorConfig(
        n_operands=3,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_valid_expressions=1_000,
        split=split,
    )
    generator = TCESGenerator(
        derive_namespaced_seed(ROOT_SEED, "dataset.tces.split", split),
        config,
    )
    records = [build_tces_record(generator.generate(index)) for index in range(32)]
    by_family: dict[str, list[TCESTaskManifestRecord]] = defaultdict(list)
    for record in records:
        for family_id in record.valid_family_ids:
            by_family[family_id].append(record)
    family_id, candidates = next(
        (family_id, candidates)
        for family_id, candidates in sorted(by_family.items())
        if len(candidates) >= item_count
    )
    selected: list[TCESTaskManifestRecord] = []
    for record in candidates[:item_count]:
        payload = record.model_dump(mode="python")
        payload["intended_family"] = family_id
        selected.append(TCESTaskManifestRecord.model_validate(payload))
    return build_manifest(
        name=f"boundary-test-{item_count}",
        task_family="tces",
        split=split,
        generator_version="1.0.0",
        root_seed=ROOT_SEED,
        records=selected,
    )


def _verification(success: bool) -> VerificationResult:
    if success:
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
            strategy_family_id="observed-output-family",
            ast_json={"type": "integer", "value": 1},
        )
    return VerificationResult(
        valid_answer_tag=True,
        valid_lexing=True,
        valid_syntax=True,
        valid_operand_multiset=True,
        valid_operations=True,
        valid_intermediates=True,
        correct_target=False,
        reward=0.0,
        failure_code=VerificationFailure.WRONG_TARGET,
    )


def _rows(
    manifest: DatasetManifest,
    trials: list[int],
    successes: list[int],
) -> tuple[list[GenerationRecord], list[RewardRecord]]:
    generations: list[GenerationRecord] = []
    rewards: list[RewardRecord] = []
    assert len(manifest.records) == len(trials) == len(successes)
    for record, item_trials, item_successes in zip(manifest.records, trials, successes):
        assert isinstance(record, TCESTaskManifestRecord)
        for sample_index in range(item_trials):
            verification = _verification(sample_index < item_successes)
            sample_id = f"sample:{record.item_index}:{sample_index}"
            generation = GenerationRecord(
                sample_id=sample_id,
                sample_index=sample_index,
                sampling_seed=10_000 + record.item_index,
                purpose="evaluation",
                checkpoint_stage="m0",
                training_step=2,
                sampler_checkpoint_path=M0_PATH,
                task_manifest_id=manifest.manifest_id,
                task_id=record.task_id,
                task_family="tces",
                source_split="a_candidate",
                prompt_text="boundary prompt",
                completion_text="<answer>candidate</answer>",
                prompt_tokens=10,
                sampled_tokens=2,
                sampling_temperature=1.0,
                sampling_top_p=0.95,
                sampling_max_tokens=256,
                run_id="boundary-run",
                seed=5,
                origin_sampler_checkpoint_path=M0_PATH,
                item_index=record.item_index,
                assigned_family_id=record.intended_family,
                family_id=verification.strategy_family_id,
                stop_reason="stop",
                reward=verification.reward,
            )
            generations.append(generation)
            rewards.append(
                RewardRecord(
                    reward_id=f"reward:{record.item_index}:{sample_index}",
                    sample_id=sample_id,
                    task_id=record.task_id,
                    reward=verification.reward,
                    exact_verification=verification,
                )
            )
    return generations, rewards


def test_boundary_summary_uses_equal_item_weighting_not_pooled_completions() -> None:
    manifest = _shared_family_manifest(2)
    generations, rewards = _rows(manifest, [1, 9], [1, 0])
    generations[0] = generations[0].model_copy(update={"stop_reason": "length"})

    summary = summarize_m0_boundary(manifest, generations, rewards, group_size=8)[0]

    assert summary.item_count == 2
    assert summary.equal_item_posterior_mean_success == pytest.approx(0.4)
    assert summary.pooled_posterior_mean_success == pytest.approx(1.5 / 11)
    assert summary.equal_item_posterior_mean_success != pytest.approx(
        summary.pooled_posterior_mean_success
    )
    assert summary.successful_item_count == 1
    assert summary.maximum_item_success_share == 1.0
    assert summary.format_compliance == 1.0
    assert summary.truncation_rate == 0.5

    reversed_summary = summarize_m0_boundary(
        manifest,
        list(reversed(generations)),
        list(reversed(rewards)),
        group_size=8,
    )
    assert reversed_summary == (summary,)


def test_boundary_summary_combines_separate_held_out_item_manifests() -> None:
    source = _shared_family_manifest(4)
    first = build_manifest(
        name="boundary-first-items",
        task_family="tces",
        split="a_candidate",
        generator_version="1.0.0",
        root_seed=ROOT_SEED,
        records=source.records[:2],
    )
    held_out = build_manifest(
        name="boundary-held-out-items",
        task_family="tces",
        split="a_candidate",
        generator_version="1.0.0",
        root_seed=ROOT_SEED,
        records=source.records[2:],
    )
    first_generations, first_rewards = _rows(first, [2, 2], [1, 0])
    held_generations, held_rewards = _rows(held_out, [2, 2], [1, 1])

    summary = summarize_m0_boundary(
        first,
        first_generations + held_generations,
        first_rewards + held_rewards,
        group_size=8,
        additional_manifests=(held_out,),
    )[0]

    assert summary.item_count == 4
    assert summary.successful_item_count == 3
    assert summary.total_successes == 3
    with pytest.raises(BoundaryEvidenceError, match="duplicate task IDs"):
        summarize_m0_boundary(
            first,
            first_generations,
            first_rewards,
            group_size=8,
            additional_manifests=(first,),
        )


def test_boundary_eligibility_uses_explicit_gates_and_multi_item_success() -> None:
    manifest = _shared_family_manifest(4)
    generations, rewards = _rows(manifest, [1, 1, 1, 1], [1, 1, 1, 1])
    summary = summarize_m0_boundary(manifest, generations, rewards, group_size=8)[0]
    assert summary.successful_item_count == 4
    assert summary.maximum_item_success_share == 0.25

    criteria = BoundaryEligibilityCriteria(
        informative_probability_statistic="at_equal_item_mean",
        minimum_informative_group_probability=0.0,
        maximum_equal_item_posterior_mean_success=1.0,
        minimum_successful_items=2,
        maximum_item_success_share=0.25,
        minimum_format_compliance=1.0,
        maximum_truncation_rate=0.0,
        minimum_disjoint_instances_by_split=(
            ("a_rl_train", 8),
            ("a_validation", 8),
        ),
    )
    assessment = assess_boundary_eligibility(
        summary,
        criteria,
        available_disjoint_instances_by_split={
            "a_rl_train": 8,
            "a_validation": 8,
        },
        enumeration_complete=True,
        can_be_matched=True,
    )
    assert assessment.eligible is True
    assert assessment.exclusion_reasons == ()

    rejected = assess_boundary_eligibility(
        summary,
        BoundaryEligibilityCriteria(
            informative_probability_statistic="mean_across_items",
            minimum_informative_group_probability=1.0,
            maximum_equal_item_posterior_mean_success=0.5,
            minimum_successful_items=2,
            maximum_item_success_share=0.20,
            minimum_format_compliance=1.0,
            maximum_truncation_rate=0.0,
            minimum_disjoint_instances_by_split=(
                ("a_rl_train", 9),
                ("a_validation", 8),
            ),
        ),
        available_disjoint_instances_by_split={
            "a_rl_train": 8,
            "a_validation": 8,
        },
        enumeration_complete=False,
        can_be_matched=False,
    )
    assert rejected.eligible is False
    assert rejected.exclusion_reasons == (
        "informative_group_probability_below_floor",
        "posterior_mean_near_saturation",
        "single_item_success_domination",
        "insufficient_disjoint_instances:a_rl_train",
        "incomplete_exact_enumeration",
        "not_matchable_into_panel_design",
    )


def test_frozen_refinement_and_confirmation_observation_gates() -> None:
    manifest = _shared_family_manifest(4)
    refinement_generations, refinement_rewards = _rows(
        manifest, [16, 16, 16, 16], [1, 1, 0, 0]
    )
    refinement = summarize_m0_boundary(
        manifest, refinement_generations, refinement_rewards, group_size=8
    )[0]

    assert assess_refinement_finalist_gate(refinement).eligible is True
    confirmation_rejection = assess_confirmation_observation_gate(refinement)
    assert confirmation_rejection.eligible is False
    assert confirmation_rejection.exclusion_reasons == (
        "insufficient_successful_items",
        "single_item_success_domination",
    )

    confirmed_generations, confirmed_rewards = _rows(
        manifest, [16, 16, 16, 16], [1, 1, 1, 1]
    )
    confirmed = summarize_m0_boundary(
        manifest, confirmed_generations, confirmed_rewards, group_size=8
    )[0]
    assert assess_confirmation_observation_gate(confirmed).eligible is True


def test_boundary_summary_fails_closed_on_missing_or_ambiguous_linkage() -> None:
    manifest = _shared_family_manifest(2)
    generations, rewards = _rows(manifest, [1, 1], [1, 0])

    with pytest.raises(BoundaryEvidenceError, match="linkages differ"):
        summarize_m0_boundary(manifest, generations, rewards[:-1], group_size=8)

    with pytest.raises(BoundaryEvidenceError, match="duplicate sample IDs"):
        summarize_m0_boundary(
            manifest, generations + [generations[0]], rewards, group_size=8
        )

    mismatched = list(generations)
    mismatched[0] = mismatched[0].model_copy(
        update={"assigned_family_id": "wrong-family"}
    )
    with pytest.raises(BoundaryEvidenceError, match="assigned_family_id"):
        summarize_m0_boundary(manifest, mismatched, rewards, group_size=8)

    mixed_runs = list(generations)
    mixed_runs[0] = mixed_runs[0].model_copy(update={"run_id": "other-run"})
    with pytest.raises(BoundaryEvidenceError, match="one run"):
        summarize_m0_boundary(manifest, mixed_runs, rewards, group_size=8)

    sequential = summarize_m0_boundary(
        manifest,
        mixed_runs,
        rewards,
        group_size=8,
        expected_run_ids=("boundary-run", "other-run"),
    )
    assert len(sequential) == 1
    with pytest.raises(BoundaryEvidenceError, match="expected sequential"):
        summarize_m0_boundary(
            manifest,
            mixed_runs,
            rewards,
            group_size=8,
            expected_run_ids=("boundary-run", "missing-run"),
        )

    with pytest.raises(TypeError, match="criteria"):
        assess_boundary_eligibility(
            summarize_m0_boundary(manifest, generations, rewards, group_size=8)[0],
            None,  # type: ignore[arg-type]
            available_disjoint_instances_by_split={
                "a_rl_train": 8,
                "a_validation": 8,
            },
            enumeration_complete=True,
            can_be_matched=True,
        )
