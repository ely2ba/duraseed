"""Teacher-dose collector under its deterministic launch allocation."""

from __future__ import annotations

from pathlib import Path

from duraseed.calibration_attempts import ArmAttempts
from duraseed.calibration_budget import (
    persist_budget_preflight,
    require_remaining_budget,
    teacher_dose_budget,
)
from duraseed.runners import RunnerGateError
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.teacher_dose_arms import (
    baseline_attempt,
    teacher_arm_attempt,
)
from duraseed.training.acquisition_freeze import (
    TeacherDoseArmEvidence,
    TeacherDoseLiveEvidence,
)
from duraseed.training.teacher_allocation_sources import (
    validate_teacher_allocation_base_sources,
)
from duraseed.training.teacher_dose import GateStatus


CALIBRATION_SEED = 17
VERIFICATION_SEED = 37


def _safe_arm(value: str) -> str:
    return value.replace("+", "plus").replace(".", "p")


def _validate_completed_prefix(
    inputs: CalibrationLiveInputs, completed: frozenset[str]
) -> None:
    if not completed:
        return
    rates = sorted(inputs.config.tinker.learning_rates.teacher_seed_sft.grid)
    training = [
        _safe_arm(f"seed-17-dose-{dose}-lr-{rate:.0e}")
        for dose in inputs.config.teacher_dose.demonstrations_per_family
        for rate in rates
    ]
    present = [arm for arm in training if arm in completed]
    if "baseline-seed-17" not in completed or present != training[: len(present)]:
        raise RunnerGateError("completed teacher-dose arms are not a canonical prefix")
    allowed = {"baseline-seed-17", *present}
    verification = set(completed.difference(allowed))
    has_verification_baseline = "baseline-seed-37" in verification
    if has_verification_baseline:
        verification.remove("baseline-seed-37")
    if has_verification_baseline and (not present or len(present) % len(rates)):
        raise RunnerGateError("completed teacher verification lacks a selected dose")
    if verification:
        if len(verification) != 1 or "baseline-seed-37" not in completed:
            raise RunnerGateError("completed teacher verification arms differ")
        selected_dose = inputs.config.teacher_dose.demonstrations_per_family[
            len(present) // len(rates) - 1
        ]
        expected = {
            _safe_arm(f"seed-37-dose-{selected_dose}-lr-{rate:.0e}") for rate in rates
        }
        if not verification.issubset(expected):
            raise RunnerGateError("completed teacher verification coordinate differs")


async def collect_teacher_dose(
    inputs: CalibrationLiveInputs,
    output_directory: Path,
    *,
    preflight_sha256: str,
) -> TeacherDoseLiveEvidence:
    """Resume complete arms exactly; rerun only a reconciled incomplete arm."""

    if inputs.teacher_ledger.authorized_usd <= 0:
        raise RunnerGateError("teacher-dose collector requires its preflight ledger")
    validate_teacher_allocation_base_sources(inputs.teacher_sources)
    attempts = ArmAttempts(
        output_directory,
        inputs.teacher_ledger,
        run_id=inputs.run_id,
        action="teacher-dose",
        project_id=inputs.project_id,
        preflight_sha256=preflight_sha256,
        reconciliations=tuple(
            row for row in inputs.reconciled_restarts if row.action == "teacher-dose"
        ),
    )
    _validate_completed_prefix(inputs, attempts.completed_arm_ids)
    budget = teacher_dose_budget(inputs, attempts.completed_arm_ids)
    budget_preflight = require_remaining_budget(
        budget,
        inputs.teacher_ledger,
        prior_billed_usd=attempts.prior_billed_usd,
    )
    persist_budget_preflight(
        output_directory,
        {
            **budget_preflight,
            "run_id": inputs.run_id,
            "action": "teacher-dose",
            "preflight_sha256": preflight_sha256,
        },
    )
    arms: list[TeacherDoseArmEvidence] = []
    selected: TeacherDoseArmEvidence | None = None
    selected_dose: int | None = None
    baseline = await baseline_attempt(inputs, attempts, seed=CALIBRATION_SEED)
    for dose in inputs.config.teacher_dose.demonstrations_per_family:
        current = [
            await teacher_arm_attempt(
                inputs,
                attempts,
                baseline,
                seed=CALIBRATION_SEED,
                dose=dose,
                learning_rate=learning_rate,
            )
            for learning_rate in sorted(
                inputs.config.tinker.learning_rates.teacher_seed_sft.grid
            )
        ]
        arms.extend(current)
        passing = [row for row in current if row.assessment.status is GateStatus.PASSED]
        if passing:
            selected = min(passing, key=lambda row: row.learning_rate)
            selected_dose = dose
            break
    if selected is None or selected_dose is None:
        raise RunnerGateError("every configured teacher dose failed")
    expected_verification = _safe_arm(
        f"seed-37-dose-{selected_dose}-lr-{selected.learning_rate:.0e}"
    )
    completed_verifications = {
        arm for arm in attempts.completed_arm_ids if arm.startswith("seed-37-dose-")
    }
    if completed_verifications.difference({expected_verification}):
        raise RunnerGateError("completed teacher verification selection differs")
    verification_baseline = await baseline_attempt(
        inputs, attempts, seed=VERIFICATION_SEED
    )
    verification = await teacher_arm_attempt(
        inputs,
        attempts,
        verification_baseline,
        seed=VERIFICATION_SEED,
        dose=selected_dose,
        learning_rate=selected.learning_rate,
    )
    attempts.assert_no_unused_reconciliations()
    return TeacherDoseLiveEvidence(tuple(arms), verification)


__all__ = ["collect_teacher_dose"]
