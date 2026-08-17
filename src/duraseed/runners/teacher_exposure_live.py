"""Two-orientation progressive teacher-exposure repair."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from duraseed.calibration_attempts import ArmAttempts
from duraseed.calibration_budget import (
    persist_budget_preflight,
    require_remaining_budget,
    teacher_exposure_budget,
)
from duraseed.provenance import canonical_json_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.teacher_exposure_trajectory import (
    ActiveExposureTrajectory,
    evaluate_checkpoint,
    start_trajectory,
    update_trajectory,
)
from duraseed.training.teacher_allocation_sources import (
    validate_teacher_allocation_base_sources,
)
from duraseed.training.teacher_exposure import (
    ExposureTrajectoryEvidence,
    REPAIR_CHECKPOINT_UPDATES,
    REPAIR_SEEDS,
    REPAIR_SPEC,
    TeacherExposureEvidence,
    TeacherExposureSelection,
    select_teacher_exposure,
)


_TRAJECTORY = TypeAdapter(ExposureTrajectoryEvidence)


def _arm_id(seed: int) -> str:
    return f"trajectory-seed-{seed}"


def _evidence(
    inputs: CalibrationLiveInputs,
    trajectories: tuple[ExposureTrajectoryEvidence, ...],
) -> TeacherExposureEvidence:
    return TeacherExposureEvidence(
        REPAIR_SPEC,
        inputs.parent_teacher_evidence.lineage,
        trajectories,
    )


def _prefix(
    trajectory: ExposureTrajectoryEvidence, updates: int
) -> ExposureTrajectoryEvidence:
    return ExposureTrajectoryEvidence(
        trajectory.seed,
        tuple(point for point in trajectory.points if point.updates <= updates),
        tuple(row for row in trajectory.metrics if row.training_step <= updates),
    )


async def collect_teacher_exposure(
    inputs: CalibrationLiveInputs, output: Path, *, preflight_sha256: str
) -> tuple[TeacherExposureEvidence, TeacherExposureSelection]:
    """Run both continuous trajectories in lockstep and stop at the first joint pass."""

    validate_teacher_allocation_base_sources(inputs.teacher_sources)
    attempts = ArmAttempts(
        output,
        inputs.teacher_ledger,
        run_id=inputs.run_id,
        action="teacher-dose",
        project_id=inputs.project_id,
        preflight_sha256=preflight_sha256,
        reconciliations=tuple(
            row for row in inputs.reconciled_restarts if row.action == "teacher-dose"
        ),
    )
    completed_ids = attempts.completed_arm_ids
    if completed_ids not in (
        frozenset(),
        frozenset({_arm_id(17)}),
        frozenset(_arm_id(seed) for seed in REPAIR_SEEDS),
    ):
        raise RunnerGateError(
            "completed teacher-exposure trajectories are not a prefix"
        )
    budget = teacher_exposure_budget(inputs, completed_ids)
    persist_budget_preflight(
        output,
        {
            **require_remaining_budget(
                budget,
                inputs.teacher_ledger,
                prior_billed_usd=attempts.prior_billed_usd,
            ),
            "run_id": inputs.run_id,
            "action": "teacher-dose",
            "preflight_sha256": preflight_sha256,
        },
    )
    trajectories: dict[int, ExposureTrajectoryEvidence] = {}
    active: dict[int, ActiveExposureTrajectory] = {}
    for seed in REPAIR_SEEDS:
        attempt = attempts.open(_arm_id(seed))
        if attempt.completed:
            trajectory = _TRAJECTORY.validate_json(
                canonical_json_bytes(attempt.completed_payload)
            )
            ladder = tuple(point.updates for point in trajectory.points)
            if trajectory.seed != seed or ladder not in tuple(
                REPAIR_CHECKPOINT_UPDATES[:stop] for stop in range(1, 4)
            ):
                raise RunnerGateError("completed teacher-exposure evidence differs")
            trajectories[seed] = trajectory
        else:
            active[seed] = await start_trajectory(inputs, attempt, seed)
    target_ladder = (
        tuple(point.updates for point in trajectories[17].points)
        if 17 in trajectories
        else REPAIR_CHECKPOINT_UPDATES
    )
    for updates in target_ladder:
        for trajectory in active.values():
            await update_trajectory(inputs, trajectory, updates)
            trajectory.points.append(
                await evaluate_checkpoint(inputs, trajectory, updates)
            )
        current = {
            **{
                seed: _prefix(trajectory, updates)
                for seed, trajectory in trajectories.items()
            },
            **{
                seed: ExposureTrajectoryEvidence(
                    seed, tuple(row.points), tuple(row.metrics)
                )
                for seed, row in active.items()
            },
        }
        if len(current) == 2:
            selection = select_teacher_exposure(
                _evidence(inputs, tuple(current[seed] for seed in REPAIR_SEEDS))
            )
            if selection.recipe is not None and (
                not trajectories or updates == target_ladder[-1]
            ):
                break
    for seed in REPAIR_SEEDS:
        if seed in active:
            trajectory = ExposureTrajectoryEvidence(
                seed, tuple(active[seed].points), tuple(active[seed].metrics)
            )
            attempts.complete(active[seed].attempt, trajectory)
            trajectories[seed] = trajectory
    attempts.assert_no_unused_reconciliations()
    evidence = _evidence(inputs, tuple(trajectories[seed] for seed in REPAIR_SEEDS))
    return evidence, select_teacher_exposure(evidence)


__all__ = ["collect_teacher_exposure"]
