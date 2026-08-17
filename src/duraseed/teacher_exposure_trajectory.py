"""Memory-bounded execution of one progressive teacher-exposure trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any

from duraseed.calibration_attempts import ArmAttempt
from duraseed.run_records import TrainingMetricRecord, append_jsonl
from duraseed.runners.teacher_dose_evidence import (
    cyclic_batch,
    gate_records,
    teacher_families,
    teacher_records,
)
from duraseed.runners.teacher_dose_sampling import sample_gate
from duraseed.runtime import (
    apply_update,
    bind_model,
    create_sampler,
    restore_checkpoint,
    save_sampler_checkpoint,
    sft_datum,
)
from duraseed.training.teacher_exposure import (
    ExposurePointEvidence,
    REPAIR_DOSE,
    REPAIR_LEARNING_RATE,
    assess_exposure_point,
)


SAMPLER_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass(slots=True)
class ActiveExposureTrajectory:
    seed: int
    attempt: ArmAttempt
    runtime: Any
    targeted: tuple[str, ...]
    sentinel: tuple[str, ...]
    datums: list[Any]
    metrics: list[TrainingMetricRecord]
    points: list[ExposurePointEvidence]


async def start_trajectory(
    inputs: Any, attempt: ArmAttempt, seed: int
) -> ActiveExposureTrajectory:
    journal = attempt.journal
    assert journal is not None
    targeted, sentinel = teacher_families(inputs, seed)
    journal.begin(
        "restore-exposure-trajectory",
        {"seed": seed, "dose": REPAIR_DOSE, "learning_rate": REPAIR_LEARNING_RATE},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    client = await restore_checkpoint(
        inputs.runtime,
        inputs.m0_state_path,
        full_state=False,
        ledger=inputs.teacher_ledger,
        user_metadata={"gate": "teacher-exposure-repair", "seed": str(seed)},
    )
    journal.complete({"operation": "restore-exposure-trajectory"})
    runtime = bind_model(inputs.runtime.sdk, inputs.runtime.service, client)
    datums = [
        sft_datum(runtime, row)
        for row in teacher_records(inputs, targeted, REPAIR_DOSE)
    ]
    return ActiveExposureTrajectory(
        seed, attempt, runtime, targeted, sentinel, datums, [], []
    )


async def update_trajectory(
    inputs: Any, active: ActiveExposureTrajectory, stop: int
) -> None:
    journal = active.attempt.journal
    assert journal is not None
    for step in range(len(active.metrics) + 1, stop + 1):
        batch = cyclic_batch(active.datums, step)
        journal.begin(
            "teacher-exposure-update",
            {"seed": active.seed, "step": step},
            {
                "prefill_tokens": 0,
                "sample_tokens": 0,
                "train_tokens": sum(int(row.model_input.length) for row in batch),
            },
        )
        metric = TrainingMetricRecord(
            phase="stage_a",
            training_step=step,
            metrics=await apply_update(
                active.runtime,
                batch,
                loss_fn="cross_entropy",
                learning_rate=REPAIR_LEARNING_RATE,
                ledger=inputs.teacher_ledger,
            ),
        )
        active.metrics.append(metric)
        append_jsonl(active.attempt.directory / "metrics.jsonl", metric)
        journal.complete({"operation": "teacher-exposure-update", "step": step})


async def evaluate_checkpoint(
    inputs: Any, active: ActiveExposureTrajectory, updates: int
) -> ExposurePointEvidence:
    journal = active.attempt.journal
    assert journal is not None
    journal.begin(
        "save-exposure-checkpoint",
        {"seed": active.seed, "updates": updates},
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": 0,
            "fixed_usd": 0.05,
        },
    )
    sampler_path = await save_sampler_checkpoint(
        active.runtime,
        name=(
            f"{inputs.run_id}-teacher-exposure-seed-{active.seed}-step-{updates}"
            f"-{active.attempt.directory.name}"
        ),
        ttl_seconds=SAMPLER_TTL_SECONDS,
        ledger=inputs.teacher_ledger,
        reserved_storage_usd=0.05,
    )
    journal.complete({"operation": "save-exposure-checkpoint", "path": sampler_path})
    journal.begin(
        "create-exposure-sampler",
        {"seed": active.seed, "updates": updates, "sampler_path": sampler_path},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    sampler = await create_sampler(
        inputs.runtime, ledger=inputs.teacher_ledger, checkpoint_path=sampler_path
    )
    journal.complete({"operation": "create-exposure-sampler"})
    checkpoint_output = active.attempt.directory / f"checkpoint-{updates}"
    target_g, target_r = await sample_gate(
        inputs,
        sampler,
        gate_records(inputs, active.targeted),
        seed=active.seed,
        group_size=inputs.config.teacher_dose.gate_samples_per_item,
        sampler_path=sampler_path,
        checkpoint_stage="stage_a",
        training_step=updates,
        role="targeted",
        output_directory=checkpoint_output,
        journal=journal,
    )
    sentinel_g, sentinel_r = await sample_gate(
        inputs,
        sampler,
        gate_records(inputs, active.sentinel),
        seed=active.seed,
        group_size=1,
        sampler_path=sampler_path,
        checkpoint_stage="stage_a",
        training_step=updates,
        role="sentinel",
        output_directory=checkpoint_output,
        journal=journal,
    )
    baseline = inputs.parent_teacher_evidence.baseline(active.seed)
    point = assess_exposure_point(
        config=inputs.config.teacher_dose,
        panel=inputs.teacher_sources.panel,
        gate_manifest=inputs.teacher_sources.gate_manifest,
        m0_sampler_path=inputs.m0_sampler_path,
        seed=active.seed,
        updates=updates,
        sampler_path=sampler_path,
        target_generations=target_g,
        target_rewards=target_r,
        baseline_generations=baseline.generations,
        baseline_rewards=baseline.rewards,
        sentinel_generations=sentinel_g,
        sentinel_rewards=sentinel_r,
        metrics=tuple(active.metrics),
    )
    del target_g, target_r, sentinel_g, sentinel_r
    gc.collect()
    return point


__all__ = [
    "ActiveExposureTrajectory",
    "evaluate_checkpoint",
    "start_trajectory",
    "update_trajectory",
]
