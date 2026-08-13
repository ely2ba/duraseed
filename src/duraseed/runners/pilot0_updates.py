"""Frozen B-S and B-G Stage-A update paths for Pilot 0."""

from __future__ import annotations

from math import fsum
from pathlib import Path

from duraseed.data.stage_a_prompt_pools import PromptPoolStratum
from duraseed.pilot0_contract import Pilot0Inputs, PilotSeedSources
from duraseed.pilot0_data import scheduled_stage_a_records
from duraseed.provenance import derive_namespaced_seed
from duraseed.run_records import TrainingMetricRecord, append_jsonl
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_remote import ephemeral_sampler
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import (
    RuntimeBundle,
    SampleObservation,
    SamplingCoordinates,
    SamplingTask,
    apply_update,
    rl_datums,
    sample_seeded,
    sft_datum,
)
from duraseed.tasks.tces import render_prompt
from duraseed.training.grpo import grouped_reward_diagnostics
from duraseed.training.sft import VerifiedSourceRecord


GROUP_SIZE = 8


def _metric(
    output: Path,
    *,
    method: str,
    step: int,
    values: dict[str, float],
) -> TrainingMetricRecord:
    row = TrainingMetricRecord(phase="stage_a", training_step=step, metrics=values)
    append_jsonl(
        output / "metrics.jsonl", {**row.model_dump(mode="json"), "method": method}
    )
    return row


async def supervised_update(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    runtime: RuntimeBundle,
    *,
    step: int,
    learning_rate: float,
    pools: dict[PromptPoolStratum, tuple],
    sources: dict[str, VerifiedSourceRecord],
    output: Path,
    journal: RemoteJournal,
) -> TrainingMetricRecord:
    records = scheduled_stage_a_records(
        pools, source.prompt_pools.artifact.bs_slot_order, step
    )
    datums = [sft_datum(runtime, sources[row.task_id]) for row in records]
    journal.begin(
        "pilot0-stage-a-sft-update",
        {"seed": source.seed, "method": "B-S", "step": step},
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": sum(int(row.model_input.length) for row in datums),
        },
    )
    values = await apply_update(
        runtime,
        datums,
        loss_fn="cross_entropy",
        learning_rate=learning_rate,
        ledger=inputs.ledger,
    )
    row = _metric(output, method="B-S", step=step, values=values)
    journal.complete({"operation": "pilot0-stage-a-sft-update", "step": step})
    return row


def _group_seeds(seed: int, step: int, group: int, task_id: str) -> tuple[int, ...]:
    root = derive_namespaced_seed(
        seed, "pilot0.stage_a.bg_rollout", step, group, task_id
    )
    return tuple(
        derive_namespaced_seed(root, "pilot0.stage_a.group_sample", index)
        for index in range(GROUP_SIZE)
    )


async def grouped_rl_update(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    runtime: RuntimeBundle,
    *,
    step: int,
    learning_rate: float,
    pools: dict[PromptPoolStratum, tuple],
    boundary_sampler_path: str,
    output: Path,
    journal: RemoteJournal,
) -> TrainingMetricRecord:
    sampler, sampler_path = await ephemeral_sampler(
        inputs,
        runtime,
        journal,
        coordinate={"seed": source.seed, "method": "B-G", "step": step},
    )
    records = scheduled_stage_a_records(
        pools, source.prompt_pools.artifact.bg_group_order, step
    )
    mixed_rows: list[SampleObservation] = []
    advantages: list[float] = []
    observed_logprobs: list[float] = []
    all_zero = all_one = mixed = 0
    for group_index, record in enumerate(records):
        prompt_text = render_prompt(record.to_task())
        prompt = runtime.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}], role="assistant"
        )
        journal.begin(
            "pilot0-stage-a-rl-group",
            {
                "seed": source.seed,
                "step": step,
                "group": group_index,
                "task_id": record.task_id,
            },
            {
                "prefill_tokens": int(prompt.length) * GROUP_SIZE,
                "sample_tokens": inputs.acquisition.selected_max_tokens * GROUP_SIZE,
                "train_tokens": 0,
            },
        )
        rows = await sample_seeded(
            runtime,
            sampler,
            SamplingTask(
                source.prompt_pools.a_rl_train_manifest.manifest_id,
                record.task_id,
                "tces",
                "a_rl_train",
                prompt_text,
                record.to_task(),
                record.item_index,
                record.intended_family,
                "training",
            ),
            SamplingCoordinates(
                inputs.run_id,
                f"seed-{source.seed}-B-G-step-{step}-group-{group_index}",
                "training",
                "stage_a",
                step,
                sampler_path,
                boundary_sampler_path,
                source.seed,
                "pilot0.stage_a.bg_rollout",
                "B-G",
            ),
            group_size=GROUP_SIZE,
            max_tokens=inputs.acquisition.selected_max_tokens,
            temperature=float(inputs.config.evaluation["temperature"]),
            top_p=float(inputs.config.evaluation["top_p"]),
            ledger=inputs.ledger,
            explicit_seeds=_group_seeds(source.seed, step, group_index, record.task_id),
        )
        diagnostics = grouped_reward_diagnostics(
            [float(row.reward.reward) for row in rows], group_size=GROUP_SIZE
        )
        all_zero += diagnostics.all_zero_group_count
        all_one += diagnostics.all_one_group_count
        mixed += diagnostics.mixed_group_count
        observed_logprobs.extend(value for row in rows for value in row.logprobs)
        group_advantages = (
            diagnostics.centered_advantages[0]
            if diagnostics.mixed_group_count
            else (0.0,) * GROUP_SIZE
        )
        for row, advantage in zip(rows, group_advantages, strict=True):
            generation = row.generation
            if diagnostics.mixed_group_count:
                generation = generation.model_copy(update={"advantage": advantage})
                mixed_rows.append(
                    SampleObservation(
                        generation, row.reward, row.prompt, row.tokens, row.logprobs
                    )
                )
                advantages.append(advantage)
            append_jsonl(output / "generations.jsonl", generation)
            append_jsonl(output / "rewards.jsonl", row.reward)
        journal.complete(
            {"operation": "pilot0-stage-a-rl-group", "row_count": len(rows)}
        )
    if mixed == 0:
        raise RunnerGateError(f"Pilot-0 B-G step {step} has no mixed reward group")
    datums = rl_datums(runtime, mixed_rows, advantages)
    journal.begin(
        "pilot0-stage-a-rl-update",
        {"seed": source.seed, "method": "B-G", "step": step},
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": sum(int(row.model_input.length) for row in datums),
        },
    )
    values = await apply_update(
        runtime,
        datums,
        loss_fn="importance_sampling",
        learning_rate=learning_rate,
        ledger=inputs.ledger,
    )
    values.update(
        mixed_group_rate=mixed / len(records),
        mixed_group_count=float(mixed),
        all_zero_group_count=float(all_zero),
        all_one_group_count=float(all_one),
        mean_sampled_token_surprisal=-fsum(observed_logprobs) / len(observed_logprobs),
    )
    row = _metric(output, method="B-G", step=step, values=values)
    journal.complete({"operation": "pilot0-stage-a-rl-update", "step": step})
    return row


__all__ = ["grouped_rl_update", "supervised_update"]
