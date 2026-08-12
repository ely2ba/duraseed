"""Shared construction for the split, assertion-identical v0 dose tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from duraseed.config import load_pilot_config
from duraseed.data.boundary_confirmation import regenerate_family_templates
from duraseed.data.boundary_protocol import build_broad_manifest
from duraseed.data.manifests import build_manifest, build_tces_record
from duraseed.data.panels import (
    FamilyPanelArtifact,
    PanelLabel,
    SeedBlockPanelAssignment,
)
from duraseed.data.splits import derive_tces_split_seed
from duraseed.run_records import GenerationRecord, RewardRecord
from duraseed.tasks.tces import (
    TCESFamilyGenerator,
    TCESGeneratorConfig,
    enumerate_task,
    generate_teacher_trace,
    render_prompt,
)
from duraseed.training import TeacherDoseGateSummary, assess_teacher_dose
from duraseed.training import summarize_paired_control_change, verify_task_completion


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _summary(
    dose: int,
    *,
    seed: int = 17,
    mixed: float = 0.30,
    reachability: float = 0.60,
    saturated: float = 0.20,
    format_compliance: float = 0.97,
    sentinel_change: float = -0.10,
) -> TeacherDoseGateSummary:
    return TeacherDoseGateSummary(
        demonstrations_per_family=dose,
        training_seed=seed,
        run_id=f"dose-{dose}-seed-{seed}",
        m0_sampler_checkpoint_path="tinker://m0/sampler",
        seeded_sampler_checkpoint_path=f"tinker://dose-{dose}/sampler",
        gate_manifest_id=SHA_A,
        panel_artifact_id=SHA_B,
        targeted_family_count=12,
        targeted_item_count=96,
        equal_item_posterior_success=0.25,
        mixed_group_rate=mixed,
        all_zero_group_rate=1.0 - mixed - saturated,
        all_one_group_rate=saturated,
        family_reachability=reachability,
        format_compliance=format_compliance,
        sentinel_control_change=summarize_paired_control_change(
            [0.5, 0.5, 0.5, 0.5],
            [0.5 + sentinel_change] * 4,
        ),
        sampled_token_mean_surprisal=1.5,
        sampled_token_logprob_coverage=1.0,
    )


def _assessment(
    dose: int,
    *,
    seed: int = 17,
    passing: bool = True,
):
    config = load_pilot_config(
        REPOSITORY_ROOT / "duraseed_pilot_config.yaml"
    ).teacher_dose
    summary = _summary(
        dose,
        seed=seed,
        mixed=0.30 if passing else 0.29,
    )
    return assess_teacher_dose(
        summary,
        config,
        optimization_stable=True,
        leakage_clean=True,
    )


def _raw_gate_fixture():
    root_seed = 5
    generator_config = TCESGeneratorConfig(
        n_operands=3,
        operand_min=2,
        operand_max=20,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_attempts=512,
    )
    broad = build_broad_manifest(
        generator_config,
        family_count=3,
        items_per_family=2,
        template_scan_ceiling=32,
        variant_index_offset=64,
    )
    families = tuple(sorted({record.intended_family for record in broad.records})[:2])
    templates = regenerate_family_templates(generator_config, broad, families)
    split = "a_seed_gate"
    split_seed = derive_tces_split_seed(root_seed, split)
    records = []
    for ordinal, family in enumerate(families):
        variants = TCESFamilyGenerator(
            split_seed,
            templates[family],
            replace(generator_config, split=split),
        ).generate_many(8, start_index=1_000 * ordinal)
        records.extend(build_tces_record(variant) for variant in variants)
    gate = build_manifest(
        name="teacher-dose-gate-fixture",
        split=split,
        generator_version="1.0.0",
        root_seed=root_seed,
        records=records,
        metadata={"scope": "unit-test"},
    )
    panel = FamilyPanelArtifact(
        m0_checkpoint_path="tinker://m0/sampler",
        candidate_family_table_manifest_id=SHA_A,
        panel_matching_report_id=SHA_B,
        allocation_seed=3,
        panel_a_family_ids=(families[0],),
        panel_b_family_ids=(families[1],),
        seed_block_assignments=(
            SeedBlockPanelAssignment(
                training_seed=17,
                targeted_panel=PanelLabel.A,
                sentinel_panel=PanelLabel.B,
            ),
            SeedBlockPanelAssignment(
                training_seed=37,
                targeted_panel=PanelLabel.B,
                sentinel_panel=PanelLabel.A,
            ),
        ),
    )

    def rows_for(
        family: str,
        *,
        checkpoint: str,
        stage: str,
        samples: int,
        panel_role: str,
        success_count: int,
    ):
        generations = []
        rewards = []
        selected = [
            record for record in gate.records if record.intended_family == family
        ]
        for record in selected:
            enumeration = enumerate_task(record.to_task())
            success = generate_teacher_trace(
                enumeration.family_representatives[record.intended_family]
            )
            for sample_index in range(samples):
                completion = (
                    success if sample_index < success_count else "<answer>0</answer>"
                )
                verification = verify_task_completion(completion, record.to_task())
                sample_id = f"dose-fixture:{stage}:{record.task_id}:{sample_index}"
                reward = RewardRecord(
                    reward_id=f"reward:{sample_id}",
                    sample_id=sample_id,
                    task_id=record.task_id,
                    reward=verification.reward,
                    exact_verification=verification,
                )
                rewards.append(reward)
                generations.append(
                    GenerationRecord(
                        sample_id=sample_id,
                        sample_index=sample_index,
                        sampling_seed=10_000 + record.item_index * 10 + sample_index,
                        purpose="evaluation",
                        checkpoint_stage=stage,
                        training_step=2 if stage == "m0" else 16,
                        sampler_checkpoint_path=checkpoint,
                        task_manifest_id=gate.manifest_id,
                        task_id=record.task_id,
                        task_family="tces",
                        source_split=split,
                        prompt_text=render_prompt(record.to_task()),
                        completion_text=completion,
                        prompt_tokens=1,
                        sampled_tokens=1,
                        sampling_temperature=1.0,
                        sampling_top_p=0.95,
                        sampling_max_tokens=4096,
                        run_id="teacher-dose-fixture",
                        method=None,
                        seed=17,
                        origin_sampler_checkpoint_path="tinker://m0/sampler",
                        item_index=record.item_index,
                        assigned_family_id=record.intended_family,
                        family_id=verification.strategy_family_id,
                        panel_role=panel_role,
                        completion_token_ids=(1,),
                        completion_logprobs=(-0.5,),
                        stop_reason="stop",
                        renderer_termination="stop_sequence",
                        reward=verification.reward,
                    )
                )
        return tuple(generations), tuple(rewards)

    target_generations, target_rewards = rows_for(
        families[0],
        checkpoint="tinker://dose/sampler",
        stage="stage_a",
        samples=8,
        panel_role="targeted",
        success_count=4,
    )
    sentinel_m0_generations, sentinel_m0_rewards = rows_for(
        families[1],
        checkpoint="tinker://m0/sampler",
        stage="m0",
        samples=1,
        panel_role="sentinel",
        success_count=0,
    )
    sentinel_seeded_generations, sentinel_seeded_rewards = rows_for(
        families[1],
        checkpoint="tinker://dose/sampler",
        stage="stage_a",
        samples=1,
        panel_role="sentinel",
        success_count=0,
    )
    return {
        "demonstrations_per_family": 2,
        "training_seed": 17,
        "panel_artifact": panel,
        "gate_manifest": gate,
        "m0_sampler_checkpoint_path": "tinker://m0/sampler",
        "seeded_sampler_checkpoint_path": "tinker://dose/sampler",
        "targeted_generations": target_generations,
        "targeted_rewards": target_rewards,
        "sentinel_m0_generations": sentinel_m0_generations,
        "sentinel_m0_rewards": sentinel_m0_rewards,
        "sentinel_seeded_generations": sentinel_seeded_generations,
        "sentinel_seeded_rewards": sentinel_seeded_rewards,
    }
