"""Concrete reduction for teacher dose/allocation and Stage-A calibration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from duraseed.config import PilotConfig, load_pilot_config
from duraseed.runners import (
    Action,
    RunPlan,
    RunnerGateError,
    authorize_launch,
    validate_mock_output_root,
)
from duraseed.runtime import (
    RuntimeBundle,
    SampleObservation,
    TokenLedger,
    apply_update,
    rl_datums,
    sft_datum,
)
from duraseed.training.stage_a_calibration import (
    StageADurationDecision,
    StageAFinalEvidence,
    StageALearningRateDecision,
    StageAScreenEvidence,
    decide_stage_a_duration,
    select_stage_a_learning_rate,
)
from duraseed.training.teacher_allocation import TeacherTokenMeasurer
from duraseed.training.teacher_allocation_freeze import (
    TeacherAllocationFreezeResult,
    build_teacher_allocation_freeze,
)
from duraseed.training.teacher_allocation_sources import TeacherAllocationSources
from duraseed.training.teacher_dose import (
    TeacherDoseAssessment,
    TeacherDoseDecision,
    TeacherDoseDecisionStatus,
    decide_teacher_dose,
)
from duraseed.training.sft import VerifiedSourceRecord


FROZEN_MAPS_PROFILE = "shortest2_cap2"
FROZEN_MAPS_LEARNING_RATE = 3e-4
FROZEN_MAPS_UPDATES = 480
CALIBRATION_REMOTE_COST_CAP_USD = Decimal("300")


@dataclass(frozen=True, slots=True)
class CalibrationInputs:
    """Authenticated fake or remote outputs consumed by the carried reducers."""

    calibration_doses: tuple[TeacherDoseAssessment, ...]
    verification_dose: TeacherDoseAssessment
    allocation_sources: TeacherAllocationSources
    token_measurer: TeacherTokenMeasurer
    bs_screens: tuple[StageAScreenEvidence, ...]
    bg_screens: tuple[StageAScreenEvidence, ...]
    final_evidence: tuple[StageAFinalEvidence, ...]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    dose: TeacherDoseDecision
    allocation: TeacherAllocationFreezeResult
    learning_rates: tuple[StageALearningRateDecision, StageALearningRateDecision]
    duration: StageADurationDecision


async def apply_supervised_batch(
    runtime: RuntimeBundle,
    sources: Sequence[VerifiedSourceRecord],
    *,
    learning_rate: float,
    ledger: TokenLedger,
) -> dict[str, float]:
    """Use the shared runtime for the SFT updates calibrated here."""

    datums = [sft_datum(runtime, source) for source in sources]
    return await apply_update(
        runtime,
        datums,
        loss_fn="cross_entropy",
        learning_rate=learning_rate,
        ledger=ledger,
    )


async def apply_group_batch(
    runtime: RuntimeBundle,
    observations: Sequence[SampleObservation],
    advantages: Sequence[float],
    *,
    learning_rate: float,
    ledger: TokenLedger,
) -> dict[str, float]:
    datums = rl_datums(runtime, observations, advantages)
    return await apply_update(
        runtime,
        datums,
        loss_fn="importance_sampling",
        learning_rate=learning_rate,
        ledger=ledger,
    )


def validate_frozen_maps(config: PilotConfig) -> None:
    observed = (
        config.stage_b.selected_profile,
        config.tinker.learning_rates.stage_b_sft.selected,
        config.stage_b.selected_max_updates,
    )
    expected = (
        FROZEN_MAPS_PROFILE,
        FROZEN_MAPS_LEARNING_RATE,
        FROZEN_MAPS_UPDATES,
    )
    if observed != expected:
        raise RunnerGateError(
            "MAPS Stage-B must remain shortest2_cap2 / 3e-4 / step 480"
        )


def build_plan(config: PilotConfig) -> RunPlan:
    validate_frozen_maps(config)
    command = "uv run duraseed calibration --config duraseed_pilot_config.yaml"
    gates = (
        "--confirm-panel-frozen --confirm-live-smoke "
        "--confirm-human-approval --confirm-remaining-balance"
    )
    return RunPlan(
        name="calibration",
        actions=(Action("acquisition-calibration", Decimal("300")),),
        launch_preconditions=(
            "panel_frozen",
            "live_smoke_passed",
            "human_approval",
            "remaining_balance_verified",
        ),
        dry_run_command=f"{command} --dry-run",
        mock_command=("uv run pytest tests/unit/test_acquisition_calibration_flow.py"),
        authorization_command=(
            f"{command} --authorize --authorized-cost-usd 300 {gates}"
        ),
    )


def load_plan(config_path: str | Path) -> RunPlan:
    return build_plan(load_pilot_config(config_path))


def reduce_calibration(
    config: PilotConfig,
    inputs: CalibrationInputs,
    *,
    output_root: str | Path | None = None,
) -> CalibrationResult:
    """Run every real local decision in dependency order without writing output."""

    validate_mock_output_root(output_root)
    validate_frozen_maps(config)
    dose = decide_teacher_dose(
        config.teacher_dose.demonstrations_per_family,
        inputs.calibration_doses,
        verification_assessment=inputs.verification_dose,
    )
    if (
        dose.status is not TeacherDoseDecisionStatus.SELECTED
        or dose.selected_dose is None
    ):
        raise RunnerGateError("teacher dose did not select; allocation cannot run")
    sources = inputs.allocation_sources
    if sources.selected_dose != dose.selected_dose:
        raise RunnerGateError("allocation sources do not use the selected dose")
    allocation = build_teacher_allocation_freeze(
        sources=sources,
        token_measurer=inputs.token_measurer,
    )
    if not allocation.selected:
        raise RunnerGateError("teacher allocation did not select; Stage-A cannot run")
    decisions = (
        select_stage_a_learning_rate("B-S", inputs.bs_screens),
        select_stage_a_learning_rate("B-G", inputs.bg_screens),
    )
    duration = decide_stage_a_duration(decisions, inputs.final_evidence)
    return CalibrationResult(dose, allocation, decisions, duration)


def authorize_calibration(
    config: PilotConfig,
    *,
    execute: bool,
    authorized_cost_usd: str | float | Decimal | None,
    panel_frozen: bool,
    live_smoke_passed: bool,
    human_approval: bool,
    remaining_balance_verified: bool,
):
    plan = build_plan(config)
    return authorize_launch(
        plan,
        execute=execute,
        authorized_cost_usd=authorized_cost_usd,
        preconditions={
            "panel_frozen": panel_frozen,
            "live_smoke_passed": live_smoke_passed,
            "human_approval": human_approval,
            "remaining_balance_verified": remaining_balance_verified,
        },
    )


def preflight_text(config: PilotConfig) -> str:
    plan = build_plan(config)
    return "\n".join(
        (
            "Calibration plan (no Tinker calls, no writes):",
            *(f"- {a.name}: ${a.cost_cap_usd}" for a in plan.actions),
            f"Remote cost cap: ${plan.remote_cost_cap_usd}",
            f"Dry-run: {plan.dry_run_command}",
            f"Mock: {plan.mock_command}",
            f"Authorization only (does not execute): {plan.authorization_command}",
            (
                "Required freezes: teacher dose/allocation; Stage-A LR/duration; "
                "one common RL configuration after the entropy-collapse gate; "
                "one shared max-tokens value after the acquisition truncation gate"
            ),
            "MAPS frozen (not an action): shortest2_cap2 / 3e-4 / step 480",
        )
    )


__all__: Sequence[str] = (
    "CALIBRATION_REMOTE_COST_CAP_USD",
    "FROZEN_MAPS_LEARNING_RATE",
    "FROZEN_MAPS_PROFILE",
    "FROZEN_MAPS_UPDATES",
    "CalibrationInputs",
    "CalibrationResult",
    "authorize_calibration",
    "apply_supervised_batch",
    "apply_group_batch",
    "build_plan",
    "load_plan",
    "preflight_text",
    "reduce_calibration",
    "validate_frozen_maps",
)
