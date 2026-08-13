"""Concrete sampling mechanics for the fixed boundary live gate."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from duraseed.boundary_live_artifacts import BoundaryLiveArtifacts
from duraseed.boundary_live_sources import BoundaryLiveSource
from duraseed.config import PilotConfig
from duraseed.data.boundary import summarize_m0_boundary
from duraseed.data.boundary_protocol import BOUNDARY_ENGINEERING_SEED
from duraseed.data.manifests import DatasetManifest
from duraseed.runners import RunnerGateError
from duraseed.runtime import (
    RuntimeBundle,
    SamplingCoordinates,
    SamplingTask,
    TokenBudget,
    TokenLedger,
    UsageQuantities,
    sample_seeded,
)
from duraseed.tasks.tces import render_prompt


ACTION_CAPS = {
    "extension1-confirm": Decimal("40"),
    "extension2-broad": Decimal("10"),
    "extension2-refine": Decimal("30"),
    "extension2-confirm": Decimal("40"),
}
PREFILL_LIMITS = {
    "extension1-confirm": 1_500_000,
    "extension2-broad": 500_000,
    "extension2-refine": 1_000_000,
    "extension2-confirm": 1_500_000,
}


def action_limits(
    action: str,
    manifest: DatasetManifest,
    samples: int,
    max_tokens: int,
    *,
    task_count: int | None = None,
) -> TokenBudget:
    count = manifest.record_count if task_count is None else task_count
    if count < 0 or count > manifest.record_count:
        raise ValueError("task_count must be within the supplied manifest")
    return TokenBudget(PREFILL_LIMITS[action], count * samples * max_tokens, 0)


def summarize(
    manifest: DatasetManifest,
    generations: tuple[Any, ...],
    rewards: tuple[Any, ...],
    config: PilotConfig,
    *,
    expected: tuple[str, ...] | None = None,
):
    return summarize_m0_boundary(
        manifest,
        generations,
        rewards,
        group_size=config.tinker.group_size,
        expected_run_ids=expected,
    )


async def collect_groups(
    artifacts: BoundaryLiveArtifacts,
    runtime: RuntimeBundle,
    sampler: Any,
    manifest: DatasetManifest,
    *,
    action: str,
    run_id: str,
    source: BoundaryLiveSource,
    samples: int,
    sample_start: int,
    config: PilotConfig,
    ledger: TokenLedger,
    task_ids: frozenset[str] | None = None,
    allow_empty: bool = False,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Collect or resume whole task groups with the archived seed contract."""

    generations, rewards = [], []
    records = tuple(
        row for row in manifest.records if task_ids is None or row.task_id in task_ids
    )
    if not records:
        if allow_empty and manifest.record_count == 0 and task_ids is None:
            return (), ()
        raise RunnerGateError("planned boundary sampling grid is empty or incomplete")
    if task_ids is not None and {row.task_id for row in records} != task_ids:
        raise RunnerGateError("planned boundary sampling grid is empty or incomplete")
    planned = TokenBudget(
        sum(
            int(
                runtime.renderer.build_generation_prompt(
                    [{"role": "user", "content": render_prompt(record.to_task())}],
                    role="assistant",
                ).length
            )
            * samples
            for record in records
        ),
        len(records) * samples * config.tinker.max_sampled_tokens,
        0,
    )
    planned_cost = ledger.prices.cost(
        UsageQuantities(
            prefill_tokens=planned.prefill,
            sample_tokens=planned.sample,
        )
    )
    if (
        planned.prefill > ledger.limits.prefill
        or planned.sample > ledger.limits.sample
        or planned_cost > ledger.authorized_usd
    ):
        raise RunnerGateError("complete action grid exceeds its token or dollar cap")
    coordinates = SamplingCoordinates(
        run_id=f"{run_id}:{action}",
        label=action,
        purpose="evaluation",
        checkpoint_stage="m0",
        training_step=source.contract.training_step,
        sampler_checkpoint_path=source.contract.sampler_checkpoint_path,
        origin_sampler_checkpoint_path=source.contract.sampler_checkpoint_path,
        experiment_seed=BOUNDARY_ENGINEERING_SEED,
        seed_namespace="tinker.tces_boundary_broad",
    )
    sample_indices = tuple(range(sample_start, sample_start + samples))
    for record in records:
        prompt_text = render_prompt(record.to_task())
        prior = artifacts.completed_group(
            action,
            record.task_id,
            manifest_id=manifest.manifest_id,
            run_id=coordinates.run_id,
            sample_indices=sample_indices,
        )
        if prior is None:
            prompt = runtime.renderer.build_generation_prompt(
                [{"role": "user", "content": prompt_text}], role="assistant"
            )
            artifacts.begin_group(
                action,
                record.task_id,
                manifest_id=manifest.manifest_id,
                run_id=coordinates.run_id,
                sample_indices=sample_indices,
                reservation=TokenBudget(
                    int(prompt.length) * samples,
                    config.tinker.max_sampled_tokens * samples,
                    0,
                ),
            )
            observations = await sample_seeded(
                runtime,
                sampler,
                SamplingTask(
                    manifest.manifest_id,
                    record.task_id,
                    "tces",
                    record.split,
                    prompt_text,
                    record.to_task(),
                    record.item_index,
                    record.intended_family,
                    "calibration-only",
                ),
                coordinates,
                group_size=samples,
                max_tokens=config.tinker.max_sampled_tokens,
                temperature=float(config.evaluation["temperature"]),
                top_p=float(config.evaluation["top_p"]),
                ledger=ledger,
                sample_index_start=sample_start,
            )
            artifacts.append_group(action, record.task_id, observations)
            artifacts.write_billing({action: ledger})
            prior = (
                tuple(row.generation for row in observations),
                tuple(row.reward for row in observations),
            )
        generations.extend(prior[0])
        rewards.extend(prior[1])
    return tuple(generations), tuple(rewards)


__all__ = ["ACTION_CAPS", "action_limits", "collect_groups", "summarize"]
