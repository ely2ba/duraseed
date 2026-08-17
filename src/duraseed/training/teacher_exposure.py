"""Frozen progressive repair for the failed 16-update teacher exposure."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from duraseed.runners import RunnerGateError
from duraseed.teacher_exposure_spec import (
    REPAIR_CHECKPOINT_UPDATES,
    REPAIR_DOSE,
    REPAIR_LEARNING_RATE,
    REPAIR_SEEDS,
    REPAIR_SPEC,
)
from duraseed.training.acquisition_freeze import TeacherDoseRecipe
from duraseed.training.teacher_dose import (
    GateStatus,
    TeacherDoseAssessment,
    TeacherDoseDecision,
    TeacherDoseDecisionStatus,
    assess_teacher_dose,
    summarize_teacher_dose_gate,
)

if TYPE_CHECKING:
    from duraseed.run_records import (
        GenerationRecord,
        RewardRecord,
        TrainingMetricRecord,
    )


@dataclass(frozen=True, slots=True)
class ExposurePointEvidence:
    updates: int
    sampler_path: str
    assessment: TeacherDoseAssessment


@dataclass(frozen=True, slots=True)
class ExposureTrajectoryEvidence:
    seed: int
    points: tuple[ExposurePointEvidence, ...]
    metrics: tuple[TrainingMetricRecord, ...]


@dataclass(frozen=True, slots=True)
class TeacherExposureEvidence:
    repair_spec: dict[str, object]
    parent_lineage: dict[str, object]
    trajectories: tuple[ExposureTrajectoryEvidence, ...]


@dataclass(frozen=True, slots=True)
class TeacherExposureSelection:
    status: str
    selected_updates: int | None
    reason: str
    recipe: TeacherDoseRecipe | None


def stable_metric_prefix(
    metrics: tuple[TrainingMetricRecord, ...], updates: int
) -> bool:
    prefix = tuple(row for row in metrics if row.training_step <= updates)
    return tuple(row.training_step for row in prefix) == tuple(
        range(1, updates + 1)
    ) and all(
        row.phase == "stage_a"
        and row.metrics
        and all(isfinite(value) for value in row.metrics.values())
        for row in prefix
    )


def assess_exposure_point(
    *,
    config: object,
    panel: object,
    gate_manifest: object,
    m0_sampler_path: str,
    seed: int,
    updates: int,
    sampler_path: str,
    target_generations: tuple[GenerationRecord, ...],
    target_rewards: tuple[RewardRecord, ...],
    baseline_generations: tuple[GenerationRecord, ...],
    baseline_rewards: tuple[RewardRecord, ...],
    sentinel_generations: tuple[GenerationRecord, ...],
    sentinel_rewards: tuple[RewardRecord, ...],
    metrics: tuple[TrainingMetricRecord, ...],
) -> ExposurePointEvidence:
    """Reduce one large raw checkpoint immediately to its unchanged gate result."""

    if any(
        row.training_step != updates
        for row in (*target_generations, *sentinel_generations)
    ):
        raise RunnerGateError("teacher exposure generation step differs")
    summary = summarize_teacher_dose_gate(
        demonstrations_per_family=REPAIR_DOSE,
        training_seed=seed,
        panel_artifact=panel,
        gate_manifest=gate_manifest,
        m0_sampler_checkpoint_path=m0_sampler_path,
        seeded_sampler_checkpoint_path=sampler_path,
        targeted_generations=target_generations,
        targeted_rewards=target_rewards,
        sentinel_m0_generations=baseline_generations,
        sentinel_m0_rewards=baseline_rewards,
        sentinel_seeded_generations=sentinel_generations,
        sentinel_seeded_rewards=sentinel_rewards,
        group_size=config.gate_samples_per_item,  # type: ignore[attr-defined]
    )
    return ExposurePointEvidence(
        updates,
        sampler_path,
        assess_teacher_dose(
            summary,
            config,  # type: ignore[arg-type]
            optimization_stable=stable_metric_prefix(metrics, updates),
            leakage_clean=True,
        ),
    )


def select_teacher_exposure(
    evidence: TeacherExposureEvidence,
) -> TeacherExposureSelection:
    """Freeze the earliest update that passes every crossed orientation."""

    if evidence.repair_spec != REPAIR_SPEC:
        raise RunnerGateError("teacher exposure repair specification differs")
    trajectories = {row.seed: row for row in evidence.trajectories}
    if len(trajectories) != len(evidence.trajectories):
        raise RunnerGateError("teacher exposure contains duplicate trajectories")
    if tuple(sorted(trajectories)) != REPAIR_SEEDS:
        raise RunnerGateError("teacher exposure omitted a crossed orientation")
    by_seed = {
        seed: {point.updates: point for point in trajectory.points}
        for seed, trajectory in trajectories.items()
    }
    ladders = {tuple(sorted(points)) for points in by_seed.values()}
    if (
        any(len(by_seed[row.seed]) != len(row.points) for row in trajectories.values())
        or len(ladders) != 1
        or not ladders
        or next(iter(ladders))
        not in tuple(REPAIR_CHECKPOINT_UPDATES[:stop] for stop in range(1, 4))
    ):
        raise RunnerGateError("teacher exposure checkpoint ladder differs")
    if any(
        point.assessment.summary.training_seed != seed
        or point.assessment.summary.demonstrations_per_family != REPAIR_DOSE
        or point.assessment.summary.seeded_sampler_checkpoint_path != point.sampler_path
        for seed, points in by_seed.items()
        for point in points.values()
    ):
        raise RunnerGateError("teacher exposure point identity differs")
    if any(
        tuple(row.training_step for row in trajectory.metrics)
        != tuple(range(1, trajectory.points[-1].updates + 1))
        for trajectory in trajectories.values()
    ):
        raise RunnerGateError("teacher exposure trajectory metrics differ")
    completed_ladder = next(iter(ladders))
    selected = next(
        (
            updates
            for updates in completed_ladder
            if all(
                by_seed[seed][updates].assessment.status is GateStatus.PASSED
                for seed in REPAIR_SEEDS
            )
        ),
        None,
    )
    if selected is None:
        if completed_ladder != REPAIR_CHECKPOINT_UPDATES:
            return TeacherExposureSelection(
                "in_progress",
                None,
                "no joint pass yet; continue to the next frozen checkpoint",
                None,
            )
        return TeacherExposureSelection(
            "no_stable_checkpoint",
            None,
            "no checkpoint through update 12 passed all existing gates in both orientations",
            None,
        )
    decision = TeacherDoseDecision(
        TeacherDoseDecisionStatus.SELECTED,
        REPAIR_SEEDS[0],
        REPAIR_SEEDS[1],
        REPAIR_DOSE,
        REPAIR_DOSE,
        f"update {selected} is the earliest checkpoint passing both orientations",
    )
    recipe = TeacherDoseRecipe(
        decision,
        REPAIR_LEARNING_RATE,
        evidence,  # type: ignore[arg-type]
        selected,
    )
    return TeacherExposureSelection("selected", selected, decision.reason, recipe)


__all__ = [
    "ExposurePointEvidence",
    "ExposureTrajectoryEvidence",
    "REPAIR_CHECKPOINT_UPDATES",
    "REPAIR_DOSE",
    "REPAIR_LEARNING_RATE",
    "REPAIR_SEEDS",
    "REPAIR_SPEC",
    "TeacherExposureEvidence",
    "TeacherExposureSelection",
    "assess_exposure_point",
    "select_teacher_exposure",
    "stable_metric_prefix",
]
