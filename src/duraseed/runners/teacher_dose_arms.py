"""One bounded teacher-dose baseline or training arm attempt."""

from __future__ import annotations

from duraseed.calibration_attempts import ArmAttempt, ArmAttempts
from duraseed.run_records import TrainingMetricRecord, append_jsonl
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.teacher_dose_evidence import (
    RAW_TEACHER_ARM,
    TEACHER_BASELINE,
    RawTeacherArm,
    TeacherBaseline,
    assess_arm,
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
from duraseed.training.acquisition_freeze import TeacherDoseArmEvidence


SAMPLER_TTL_SECONDS = 7 * 24 * 60 * 60


def _arm_id(seed: int, dose: int, learning_rate: float) -> str:
    return f"seed-{seed}-dose-{dose}-lr-{learning_rate:.0e}".replace(
        "+", "plus"
    ).replace(".", "p")


async def baseline_attempt(
    inputs: CalibrationLiveInputs,
    attempts: ArmAttempts,
    *,
    seed: int,
) -> TeacherBaseline:
    arm = attempts.open(f"baseline-seed-{seed}")
    if arm.completed:
        return TEACHER_BASELINE.validate_python(arm.completed_payload)
    assert arm.journal is not None
    _, sentinel = teacher_families(inputs, seed)
    arm.journal.begin(
        "create-m0-sampler",
        {"seed": seed, "sampler_path": inputs.m0_sampler_path},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    sampler = await create_sampler(
        inputs.runtime,
        ledger=inputs.teacher_ledger,
        checkpoint_path=inputs.m0_sampler_path,
    )
    arm.journal.complete({"operation": "create-m0-sampler"})
    generations, rewards = await sample_gate(
        inputs,
        sampler,
        gate_records(inputs, sentinel),
        seed=seed,
        group_size=1,
        sampler_path=inputs.m0_sampler_path,
        checkpoint_stage="m0",
        role="sentinel",
        output_directory=arm.directory,
        journal=arm.journal,
    )
    result = TeacherBaseline(generations, rewards)
    attempts.complete(arm, result)
    return result


async def teacher_arm_attempt(
    inputs: CalibrationLiveInputs,
    attempts: ArmAttempts,
    baseline: TeacherBaseline,
    *,
    seed: int,
    dose: int,
    learning_rate: float,
) -> TeacherDoseArmEvidence:
    arm = attempts.open(_arm_id(seed, dose, learning_rate))
    if arm.completed:
        raw = RAW_TEACHER_ARM.validate_python(arm.completed_payload)
        return TeacherDoseArmEvidence(learning_rate, assess_arm(inputs, raw))
    raw = await _execute_arm(
        inputs,
        arm,
        baseline,
        seed=seed,
        dose=dose,
        learning_rate=learning_rate,
    )
    attempts.complete(arm, raw)
    return TeacherDoseArmEvidence(learning_rate, assess_arm(inputs, raw))


async def _execute_arm(
    inputs: CalibrationLiveInputs,
    arm: ArmAttempt,
    baseline: TeacherBaseline,
    *,
    seed: int,
    dose: int,
    learning_rate: float,
) -> RawTeacherArm:
    journal = arm.journal
    assert journal is not None
    targeted, sentinel = teacher_families(inputs, seed)
    journal.begin(
        "restore-arm",
        {"seed": seed, "dose": dose, "learning_rate": learning_rate},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    client = await restore_checkpoint(
        inputs.runtime,
        inputs.m0_state_path,
        full_state=False,
        ledger=inputs.teacher_ledger,
        user_metadata={"gate": "teacher-dose", "seed": str(seed)},
    )
    journal.complete({"operation": "restore-arm"})
    branch = bind_model(inputs.runtime.sdk, inputs.runtime.service, client)
    datums = [sft_datum(branch, row) for row in teacher_records(inputs, targeted, dose)]
    metrics = []
    for step in range(1, inputs.config.teacher_dose.calibration_updates + 1):
        batch = cyclic_batch(datums, step)
        journal.begin(
            "teacher-update",
            {"seed": seed, "dose": dose, "learning_rate": learning_rate, "step": step},
            {
                "prefill_tokens": 0,
                "sample_tokens": 0,
                "train_tokens": sum(int(row.model_input.length) for row in batch),
            },
        )
        metric = await apply_update(
            branch,
            batch,
            loss_fn="cross_entropy",
            learning_rate=learning_rate,
            ledger=inputs.teacher_ledger,
        )
        metric_record = TrainingMetricRecord(
            phase="stage_a", training_step=step, metrics=metric
        )
        metrics.append(metric_record)
        append_jsonl(
            arm.directory / "metrics.jsonl",
            metric_record,
        )
        journal.complete({"operation": "teacher-update", "step": step})
    journal.begin(
        "save-arm-sampler",
        {"seed": seed, "dose": dose, "learning_rate": learning_rate},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0, "fixed_usd": 0.05},
    )
    sampler_path = await save_sampler_checkpoint(
        branch,
        name=(f"{inputs.run_id}-{arm.directory.parent.name}-{arm.directory.name}"),
        ttl_seconds=SAMPLER_TTL_SECONDS,
        ledger=inputs.teacher_ledger,
        reserved_storage_usd=0.05,
    )
    journal.complete({"operation": "save-arm-sampler", "path": sampler_path})
    journal.begin(
        "create-arm-sampler",
        {"sampler_path": sampler_path},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    sampler = await create_sampler(
        inputs.runtime, ledger=inputs.teacher_ledger, checkpoint_path=sampler_path
    )
    journal.complete({"operation": "create-arm-sampler"})
    target_g, target_r = await sample_gate(
        inputs,
        sampler,
        gate_records(inputs, targeted),
        seed=seed,
        group_size=inputs.config.teacher_dose.gate_samples_per_item,
        sampler_path=sampler_path,
        checkpoint_stage="stage_a",
        role="targeted",
        output_directory=arm.directory,
        journal=journal,
    )
    sentinel_g, sentinel_r = await sample_gate(
        inputs,
        sampler,
        gate_records(inputs, sentinel),
        seed=seed,
        group_size=1,
        sampler_path=sampler_path,
        checkpoint_stage="stage_a",
        role="sentinel",
        output_directory=arm.directory,
        journal=journal,
    )
    return RawTeacherArm(
        seed,
        dose,
        learning_rate,
        sampler_path,
        target_g,
        target_r,
        baseline.generations,
        baseline.rewards,
        sentinel_g,
        sentinel_r,
        tuple(metrics),
    )


__all__ = ["baseline_attempt", "teacher_arm_attempt"]
