"""Checkpoint and update mechanics for the live Stage-A calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from duraseed.run_records import (
    GenerationRecord,
    RewardRecord,
    TrainingMetricRecord,
    append_jsonl,
)
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners.stage_a_evidence import evaluate_panel
from duraseed.runners.stage_a_updates import (
    Branch,
    grouped_rl_update,
    supervised_update,
)
from duraseed.runtime import create_sampler, save_sampler_checkpoint


CANDIDATE_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class StageAOriginEvidence:
    boundary_sampler_path: str
    boundary_state_path: str
    target_generations: tuple[GenerationRecord, ...]
    target_rewards: tuple[RewardRecord, ...]
    sentinel_generations: tuple[GenerationRecord, ...]
    sentinel_rewards: tuple[RewardRecord, ...]


async def create_recorded_sampler(
    inputs: CalibrationLiveInputs,
    path: str,
    output: Path,
    journal: RemoteJournal,
) -> object:
    journal.begin(
        "create-stage-a-sampler",
        {"path": path},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    sampler = await create_sampler(
        inputs.runtime, ledger=inputs.stage_a_ledger, checkpoint_path=path
    )
    append_jsonl(output / "branch-events.jsonl", {"event": "sampler", "path": path})
    journal.complete({"operation": "create-stage-a-sampler"})
    return sampler


async def save_candidate(
    inputs: CalibrationLiveInputs,
    branch: Branch,
    output: Path,
    journal: RemoteJournal,
    step: int,
    checkpoint_suffix: str = "",
) -> str:
    journal.begin(
        "save-stage-a-candidate",
        {"method": branch.method, "learning_rate": branch.learning_rate, "step": step},
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": 0,
            "fixed_usd": 0.05,
        },
    )
    path = await save_sampler_checkpoint(
        branch.runtime,
        name=(
            f"{inputs.run_id}-{branch.method}-{branch.learning_rate:.0e}"
            f"-step-{step}{checkpoint_suffix}"
        ),
        ttl_seconds=CANDIDATE_TTL_SECONDS,
        ledger=inputs.stage_a_ledger,
        reserved_storage_usd=0.05,
    )
    append_jsonl(
        output / "checkpoints.jsonl",
        {
            "method": branch.method,
            "learning_rate": branch.learning_rate,
            "step": step,
            "sampler": path,
        },
    )
    journal.complete({"operation": "save-stage-a-candidate", "path": path})
    return path


async def run_update(
    inputs: CalibrationLiveInputs,
    branch: Branch,
    step: int,
    pools: dict,
    sources: dict,
    boundary_sampler: str,
    output: Path,
    journal: RemoteJournal,
) -> TrainingMetricRecord:
    if branch.method == "B-S":
        return await supervised_update(
            inputs, branch, step, pools, sources, output, journal
        )
    return await grouped_rl_update(
        inputs, branch, step, pools, boundary_sampler, output, journal
    )


async def build_origin(
    inputs: CalibrationLiveInputs,
    output: Path,
    journal: RemoteJournal,
) -> StageAOriginEvidence:
    """Evaluate the frozen M0 once as the common direct Stage-A origin."""

    sampler_path, state_path = inputs.m0_sampler_path, inputs.m0_state_path
    sampler = await create_recorded_sampler(inputs, sampler_path, output, journal)
    target = await evaluate_panel(
        inputs,
        sampler,
        output,
        role="targeted",
        samples_per_item=2,
        sampler_path=sampler_path,
        training_step=inputs.m0_training_step,
        label="origin-target",
        origin_sampler_path=sampler_path,
        journal=journal,
        checkpoint_stage="m0",
    )
    sentinel = await evaluate_panel(
        inputs,
        sampler,
        output,
        role="sentinel",
        samples_per_item=1,
        sampler_path=sampler_path,
        training_step=inputs.m0_training_step,
        label="origin-sentinel",
        origin_sampler_path=sampler_path,
        journal=journal,
        checkpoint_stage="m0",
    )
    return StageAOriginEvidence(
        sampler_path,
        state_path,
        tuple(row.generation for row in target),
        tuple(row.reward for row in target),
        tuple(row.generation for row in sentinel),
        tuple(row.reward for row in sentinel),
    )


__all__ = [
    "build_origin",
    "create_recorded_sampler",
    "run_update",
    "save_candidate",
    "StageAOriginEvidence",
]
