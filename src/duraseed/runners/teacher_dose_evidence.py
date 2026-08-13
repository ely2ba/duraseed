"""Authenticated local inputs and reductions for the live teacher-dose grid."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from pydantic import TypeAdapter

from duraseed.data.manifests import TCESTaskManifestRecord
from duraseed.run_records import GenerationRecord, RewardRecord, TrainingMetricRecord
from duraseed.runners import RunnerGateError
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.tasks.tces import enumerate_task, generate_teacher_trace
from duraseed.training.sft import build_teacher_dose_records
from duraseed.training.teacher_dose import (
    TeacherDoseAssessment,
    assess_teacher_dose,
    summarize_teacher_dose_gate,
)


@dataclass(frozen=True, slots=True)
class RawTeacherArm:
    seed: int
    dose: int
    learning_rate: float
    sampler_path: str
    target_generations: tuple[GenerationRecord, ...]
    target_rewards: tuple[RewardRecord, ...]
    baseline_generations: tuple[GenerationRecord, ...]
    baseline_rewards: tuple[RewardRecord, ...]
    sentinel_generations: tuple[GenerationRecord, ...]
    sentinel_rewards: tuple[RewardRecord, ...]
    metrics: tuple[TrainingMetricRecord, ...]


RAW_TEACHER_ARM = TypeAdapter(RawTeacherArm)


@dataclass(frozen=True, slots=True)
class TeacherBaseline:
    generations: tuple[GenerationRecord, ...]
    rewards: tuple[RewardRecord, ...]


TEACHER_BASELINE = TypeAdapter(TeacherBaseline)


def teacher_families(
    inputs: CalibrationLiveInputs, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    panel = inputs.teacher_sources.panel
    assignment = next(
        (row for row in panel.seed_block_assignments if row.training_seed == seed), None
    )
    if assignment is None:
        raise RunnerGateError("teacher-dose seed is absent from the crossed panels")
    by_label = {"A": panel.panel_a_family_ids, "B": panel.panel_b_family_ids}
    return (
        by_label[assignment.targeted_panel.value],
        by_label[assignment.sentinel_panel.value],
    )


def gate_records(
    inputs: CalibrationLiveInputs, families: tuple[str, ...]
) -> tuple[TCESTaskManifestRecord, ...]:
    rows = tuple(
        row
        for row in inputs.teacher_sources.gate_manifest.records
        if isinstance(row, TCESTaskManifestRecord) and row.intended_family in families
    )
    if len(rows) != 8 * len(families):
        raise RunnerGateError("teacher-dose gate panel is incomplete")
    return rows


def teacher_records(
    inputs: CalibrationLiveInputs, families: tuple[str, ...], dose: int
):
    manifest = inputs.teacher_sources.target_train_manifest
    completions = []
    for record in manifest.records:
        if not isinstance(record, TCESTaskManifestRecord):
            raise RunnerGateError("teacher training manifest is not TCES")
        if record.intended_family not in families:
            continue
        enumeration = enumerate_task(record.to_task())
        expression = enumeration.family_representatives.get(record.intended_family)
        if not enumeration.complete or expression is None:
            raise RunnerGateError("teacher task lacks its intended-family solution")
        completions.append((record, generate_teacher_trace(expression)))
    return build_teacher_dose_records(
        source_manifest=manifest,
        solver_completions=completions,
        selected_families=families,
        demonstrations_per_family=dose,
    )


def cyclic_batch(values: list[object], step: int, size: int = 32) -> list[object]:
    start = (step - 1) * size
    return [values[(start + offset) % len(values)] for offset in range(size)]


def assess_arm(
    inputs: CalibrationLiveInputs, raw: RawTeacherArm
) -> TeacherDoseAssessment:
    expected_steps = tuple(range(1, inputs.config.teacher_dose.calibration_updates + 1))
    actual_steps = tuple(row.training_step for row in raw.metrics)
    optimization_stable = (
        inputs.config.teacher_dose.calibration_updates == 16
        and actual_steps == expected_steps
        and all(
            row.phase == "stage_a"
            and row.metrics
            and all(isfinite(value) for value in row.metrics.values())
            for row in raw.metrics
        )
    )
    summary = summarize_teacher_dose_gate(
        demonstrations_per_family=raw.dose,
        training_seed=raw.seed,
        panel_artifact=inputs.teacher_sources.panel,
        gate_manifest=inputs.teacher_sources.gate_manifest,
        m0_sampler_checkpoint_path=inputs.m0_sampler_path,
        seeded_sampler_checkpoint_path=raw.sampler_path,
        targeted_generations=raw.target_generations,
        targeted_rewards=raw.target_rewards,
        sentinel_m0_generations=raw.baseline_generations,
        sentinel_m0_rewards=raw.baseline_rewards,
        sentinel_seeded_generations=raw.sentinel_generations,
        sentinel_seeded_rewards=raw.sentinel_rewards,
        group_size=inputs.config.teacher_dose.gate_samples_per_item,
    )
    return assess_teacher_dose(
        summary,
        inputs.config.teacher_dose,
        optimization_stable=optimization_stable,
        leakage_clean=True,
    )


__all__ = [
    "RAW_TEACHER_ARM",
    "TEACHER_BASELINE",
    "RawTeacherArm",
    "TeacherBaseline",
    "assess_arm",
    "cyclic_batch",
    "gate_records",
    "teacher_families",
    "teacher_records",
]
