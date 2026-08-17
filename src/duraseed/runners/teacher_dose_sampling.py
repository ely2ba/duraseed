"""Exact archived-coordinate teacher-dose gate sampling."""

from __future__ import annotations

from pathlib import Path

from duraseed.data.manifests import TCESTaskManifestRecord
from duraseed.run_records import GenerationRecord, RewardRecord, append_jsonl
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import (
    SamplingCoordinates,
    SamplingTask,
    sample_seeded,
)
from duraseed.tasks.tces import render_prompt


async def sample_gate(
    inputs: CalibrationLiveInputs,
    sampler: object,
    records: tuple[TCESTaskManifestRecord, ...],
    *,
    seed: int,
    group_size: int,
    sampler_path: str,
    checkpoint_stage: str,
    training_step: int | None = None,
    role: str,
    output_directory: Path,
    journal: RemoteJournal,
) -> tuple[tuple[GenerationRecord, ...], tuple[RewardRecord, ...]]:
    generations: list[GenerationRecord] = []
    rewards: list[RewardRecord] = []
    for record in records:
        prompt_text = render_prompt(record.to_task())
        prompt = inputs.runtime.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}], role="assistant"
        )
        journal.begin(
            "sample-group",
            {"seed": seed, "role": role, "task_id": record.task_id},
            {
                "prefill_tokens": int(prompt.length) * group_size,
                "sample_tokens": inputs.max_tokens.selected_max_tokens * group_size,
                "train_tokens": 0,
            },
        )
        observations = await sample_seeded(
            inputs.runtime,
            sampler,
            SamplingTask(
                inputs.teacher_sources.gate_manifest.manifest_id,
                record.task_id,
                "tces",
                "a_seed_gate",
                prompt_text,
                record.to_task(),
                record.item_index,
                record.intended_family,
                role,
            ),
            SamplingCoordinates(
                inputs.run_id,
                f"teacher-{seed}-{role}-{sampler_path}",
                "evaluation",
                checkpoint_stage,  # type: ignore[arg-type]
                (
                    inputs.m0_training_step
                    if checkpoint_stage == "m0"
                    else (
                        inputs.config.teacher_dose.calibration_updates
                        if training_step is None
                        else training_step
                    )
                ),
                sampler_path,
                inputs.m0_sampler_path,
                seed,
                f"tinker.teacher_dose.seed-{seed}.gate",
            ),
            group_size=group_size,
            max_tokens=inputs.max_tokens.selected_max_tokens,
            temperature=float(inputs.config.evaluation["temperature"]),
            top_p=float(inputs.config.evaluation["top_p"]),
            ledger=inputs.teacher_ledger,
        )
        generations.extend(row.generation for row in observations)
        rewards.extend(row.reward for row in observations)
        for row in observations:
            append_jsonl(output_directory / "generations.jsonl", row.generation)
            append_jsonl(output_directory / "rewards.jsonl", row.reward)
        journal.complete({"operation": "sample-group", "row_count": len(observations)})
    return tuple(generations), tuple(rewards)


__all__ = ["sample_gate"]
