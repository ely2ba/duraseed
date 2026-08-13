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
from duraseed.runners.stage_a_evidence import boundary_sources, evaluate_panel
from duraseed.runners.stage_a_updates import (
    Branch,
    grouped_rl_update,
    supervised_update,
)
from duraseed.runtime import (
    apply_update,
    bind_model,
    create_sampler,
    restore_checkpoint,
    save_checkpoint,
    save_sampler_checkpoint,
    sft_datum,
)


CANDIDATE_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class StageAOriginEvidence:
    boundary_sampler_path: str
    boundary_state_path: str
    target_generations: tuple[GenerationRecord, ...]
    target_rewards: tuple[RewardRecord, ...]
    sentinel_generations: tuple[GenerationRecord, ...]
    sentinel_rewards: tuple[RewardRecord, ...]


def _batch(values: list[object], step: int, size: int = 32) -> list[object]:
    start = (step - 1) * size
    return [values[(start + offset) % len(values)] for offset in range(size)]


async def build_boundary_seed(
    inputs: CalibrationLiveInputs,
    output: Path,
    dose: int,
    teacher_learning_rate: float,
    journal: RemoteJournal,
    checkpoint_suffix: str = "",
) -> tuple[str, str]:
    journal.begin(
        "restore-boundary-origin",
        {"dose": dose, "learning_rate": teacher_learning_rate},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    client = await restore_checkpoint(
        inputs.runtime,
        inputs.m0_state_path,
        full_state=False,
        ledger=inputs.stage_a_ledger,
        user_metadata={"gate": "stage-a-boundary-seed", "seed": "17"},
    )
    append_jsonl(output / "branch-events.jsonl", {"event": "boundary-restored"})
    journal.complete({"operation": "restore-boundary-origin"})
    runtime = bind_model(inputs.runtime.sdk, inputs.runtime.service, client)
    datums = [sft_datum(runtime, row) for row in boundary_sources(inputs, dose)]
    for step in range(1, inputs.config.teacher_dose.calibration_updates + 1):
        batch = _batch(datums, step)
        journal.begin(
            "boundary-seed-update",
            {"step": step},
            {
                "prefill_tokens": 0,
                "sample_tokens": 0,
                "train_tokens": sum(int(row.model_input.length) for row in batch),
            },
        )
        values = await apply_update(
            runtime,
            batch,
            loss_fn="cross_entropy",
            learning_rate=teacher_learning_rate,
            ledger=inputs.stage_a_ledger,
        )
        append_jsonl(
            output / "metrics.jsonl",
            {"subphase": "boundary-seed", "step": step, "metrics": values},
        )
        journal.complete({"operation": "boundary-seed-update", "step": step})
    journal.begin(
        "save-boundary-pair",
        {"step": inputs.config.teacher_dose.calibration_updates},
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": 0,
            "fixed_usd": 1.0,
        },
    )
    pair = await save_checkpoint(
        runtime,
        name=f"{inputs.run_id}-boundary-seed-17{checkpoint_suffix}",
        ttl_seconds=None,
        ledger=inputs.stage_a_ledger,
        reserved_storage_usd=1.0,
    )
    append_jsonl(
        output / "checkpoints.jsonl",
        {
            "kind": "boundary-seed",
            "sampler": pair.sampler_path,
            "state": pair.state_path,
        },
    )
    journal.complete({"operation": "save-boundary-pair"})
    return pair.sampler_path, pair.state_path


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
    *,
    selected_dose: int,
    teacher_learning_rate: float,
    checkpoint_suffix: str,
) -> StageAOriginEvidence:
    """Build origin evidence inside one indivisible Stage-A attempt."""

    sampler_path, state_path = await build_boundary_seed(
        inputs,
        output,
        selected_dose,
        teacher_learning_rate,
        journal,
        checkpoint_suffix,
    )
    sampler = await create_recorded_sampler(inputs, sampler_path, output, journal)
    target = await evaluate_panel(
        inputs,
        sampler,
        output,
        role="targeted",
        samples_per_item=2,
        sampler_path=sampler_path,
        training_step=0,
        label="origin-target",
        origin_sampler_path=sampler_path,
        journal=journal,
    )
    sentinel = await evaluate_panel(
        inputs,
        sampler,
        output,
        role="sentinel",
        samples_per_item=1,
        sampler_path=sampler_path,
        training_step=0,
        label="origin-sentinel",
        origin_sampler_path=sampler_path,
        journal=journal,
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
    "build_boundary_seed",
    "build_origin",
    "create_recorded_sampler",
    "run_update",
    "save_candidate",
    "StageAOriginEvidence",
]
