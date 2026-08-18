"""Concrete supervised and grouped-RL Stage-A update paths."""

from __future__ import annotations

from math import fsum
from pathlib import Path
from typing import Literal

from duraseed.data.stage_a_prompt_pools import PromptPoolStratum
from duraseed.calibration_seeds import (
    ephemeral_sampler_path,
    stage_a_group_seeds as _explicit_group_seeds,
)
from duraseed.run_records import TrainingMetricRecord, append_jsonl
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners.stage_a_branch import Branch
from duraseed.runners.stage_a_evidence import scheduled_records
from duraseed.runners.stage_a_update_failure import apply_grouped_update_or_fail
from duraseed.runners.stage_a_update_failure import update_health_failure
from duraseed.runtime import (
    SampleObservation,
    SamplingCoordinates,
    SamplingTask,
    TokenBudget,
    apply_update,
    bind_model,
    restore_checkpoint,
    rl_datums,
    sample_seeded,
    sft_datum,
)
from duraseed.tasks.tces import render_prompt
from duraseed.training.grpo import grouped_reward_diagnostics
from duraseed.training.sft import VerifiedSourceRecord

CALIBRATION_SEED = 17
GROUP_SIZE = 8


async def restore_branch(
    inputs: CalibrationLiveInputs,
    state_path: str,
    method: Literal["B-S", "B-G"],
    learning_rate: float,
    output: Path,
    journal: RemoteJournal,
) -> Branch:
    journal.begin(
        "restore-stage-a-branch",
        {"method": method, "learning_rate": learning_rate},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    client = await restore_checkpoint(
        inputs.runtime,
        state_path,
        full_state=False,
        ledger=inputs.stage_a_ledger,
        user_metadata={"gate": "stage-a", "method": method},
    )
    append_jsonl(
        output / "branch-events.jsonl",
        {"event": "restored", "method": method, "learning_rate": learning_rate},
    )
    journal.complete({"operation": "restore-stage-a-branch"})
    return Branch(
        method,
        learning_rate,
        bind_model(inputs.runtime.sdk, inputs.runtime.service, client),
    )


def _record_metric(
    output: Path, branch: Branch, step: int, values: dict[str, float]
) -> TrainingMetricRecord:
    metric = TrainingMetricRecord(phase="stage_a", training_step=step, metrics=values)
    branch.metrics.append(metric)
    append_jsonl(
        output / "metrics.jsonl",
        {
            **metric.model_dump(mode="json"),
            "method": branch.method,
            "learning_rate": branch.learning_rate,
        },
    )
    return metric


async def supervised_update(
    inputs: CalibrationLiveInputs,
    branch: Branch,
    step: int,
    pools: dict[PromptPoolStratum, tuple],
    source_by_task: dict[str, VerifiedSourceRecord],
    output: Path,
    journal: RemoteJournal,
) -> TrainingMetricRecord:
    records = scheduled_records(pools, inputs.prompt_pools.artifact.bs_slot_order, step)
    datums = [sft_datum(branch.runtime, source_by_task[row.task_id]) for row in records]
    journal.begin(
        "stage-a-sft-update",
        {"method": branch.method, "learning_rate": branch.learning_rate, "step": step},
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": sum(int(row.model_input.length) for row in datums),
        },
    )
    values = await apply_update(
        branch.runtime,
        datums,
        loss_fn="cross_entropy",
        learning_rate=branch.learning_rate,
        ledger=inputs.stage_a_ledger,
    )
    metric = _record_metric(output, branch, step, values)
    journal.complete({"operation": "stage-a-sft-update", "step": step})
    return metric


async def grouped_rl_update(
    inputs: CalibrationLiveInputs,
    branch: Branch,
    step: int,
    pools: dict[PromptPoolStratum, tuple],
    boundary_sampler_path: str,
    output: Path,
    journal: RemoteJournal,
) -> TrainingMetricRecord:
    journal.begin(
        "ephemeral-stage-a-sampler",
        {"method": "B-G", "learning_rate": branch.learning_rate, "step": step},
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": 0,
            "fixed_usd": 0.05,
        },
    )
    inputs.stage_a_ledger.reserve_call(TokenBudget(0, 0, 0), fixed_usd=0.05)
    try:
        sampler = (
            await branch.runtime.model.save_weights_and_get_sampling_client_async()
        )
    except Exception:
        inputs.stage_a_ledger.abort_call()
        raise
    inputs.stage_a_ledger.settle_call(TokenBudget(0, 0, 0))
    sampler_path = ephemeral_sampler_path(
        sampler, inputs.run_id, output.name, branch.learning_rate, step
    )
    append_jsonl(
        output / "branch-events.jsonl",
        {"event": "ephemeral-sampler", "step": step, "path": sampler_path},
    )
    journal.complete({"operation": "ephemeral-stage-a-sampler"})
    records = scheduled_records(
        pools, inputs.prompt_pools.artifact.bg_group_order, step
    )
    mixed_rows: list[SampleObservation] = []
    advantages: list[float] = []
    all_zero = all_one = mixed = 0
    observed_logprobs: list[float] = []
    for group_index, record in enumerate(records):
        prompt_text = render_prompt(record.to_task())
        prompt = inputs.runtime.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}], role="assistant"
        )
        journal.begin(
            "stage-a-rl-group",
            {"step": step, "group_index": group_index, "task_id": record.task_id},
            {
                "prefill_tokens": int(prompt.length) * GROUP_SIZE,
                "sample_tokens": inputs.max_tokens.selected_max_tokens * GROUP_SIZE,
                "train_tokens": 0,
            },
        )
        rows = await sample_seeded(
            inputs.runtime,
            sampler,
            SamplingTask(
                inputs.prompt_pools.a_rl_train_manifest.manifest_id,
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
                f"B-G-{branch.learning_rate:.0e}-{step}-{group_index}",
                "training",
                "stage_a",
                step,
                sampler_path,
                boundary_sampler_path,
                CALIBRATION_SEED,
                "tinker.stage_a.bg_rollout",
                "B-G",
            ),
            group_size=GROUP_SIZE,
            max_tokens=inputs.max_tokens.selected_max_tokens,
            temperature=float(inputs.config.evaluation["temperature"]),
            top_p=float(inputs.config.evaluation["top_p"]),
            ledger=inputs.stage_a_ledger,
            explicit_seeds=_explicit_group_seeds(step, group_index, record.task_id),
        )
        branch.unique_completions_by_step.setdefault(step, set()).update(
            row.generation.completion_text for row in rows
        )
        branch.valid_families_by_step.setdefault(step, set()).update(
            row.reward.exact_verification.strategy_family_id
            for row in rows
            if row.reward.exact_verification.strategy_family_id is not None
        )
        successes = tuple(row for row in rows if row.reward.reward == 1.0)
        branch.successful_completions_by_step.setdefault(step, set()).update(
            row.generation.completion_text for row in successes
        )
        diagnostics = grouped_reward_diagnostics(
            [float(row.reward.reward) for row in rows], group_size=GROUP_SIZE
        )
        all_zero += diagnostics.all_zero_group_count
        all_one += diagnostics.all_one_group_count
        mixed += diagnostics.mixed_group_count
        observed_logprobs.extend(value for row in rows for value in row.logprobs)
        if diagnostics.mixed_group_count:
            group_advantages = diagnostics.centered_advantages[0]
            for row, advantage in zip(rows, group_advantages, strict=True):
                updated = SampleObservation(
                    row.generation.model_copy(update={"advantage": advantage}),
                    row.reward,
                    row.prompt,
                    row.tokens,
                    row.logprobs,
                )
                mixed_rows.append(updated)
                advantages.append(advantage)
                append_jsonl(output / "generations.jsonl", updated.generation)
        else:
            for row in rows:
                append_jsonl(output / "generations.jsonl", row.generation)
        for row in rows:
            append_jsonl(output / "rewards.jsonl", row.reward)
        journal.complete({"operation": "stage-a-rl-group", "row_count": len(rows)})
    if mixed == 0:
        raise update_health_failure(
            output,
            branch,
            step,
            reason="zero_mixed_group",
            mixed=mixed,
            all_zero=all_zero,
            all_one=all_one,
            optimizer_update_completed=False,
        )
    datums = rl_datums(branch.runtime, mixed_rows, advantages)
    values = await apply_grouped_update_or_fail(
        inputs,
        branch,
        step,
        output,
        journal,
        datums,
        apply_update,
        mixed=mixed,
        all_zero=all_zero,
        all_one=all_one,
    )
    values.update(
        mixed_group_rate=mixed / len(records),
        mixed_group_count=float(mixed),
        all_zero_group_count=float(all_zero),
        all_one_group_count=float(all_one),
    )
    branch.surprisal_by_step[step] = -fsum(observed_logprobs) / len(observed_logprobs)
    metric = _record_metric(output, branch, step, values)
    append_jsonl(
        output / "group-diagnostics.jsonl",
        {
            "step": step,
            "mixed_group_rate": mixed / len(records),
            "mean_sampled_token_surprisal": branch.surprisal_by_step[step],
            "constant_groups_skipped": all_zero + all_one,
            "resampled_group_count": 0,
        },
    )
    journal.complete({"operation": "stage-a-rl-update", "step": step})
    return metric
