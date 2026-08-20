"""Indivisible live execution of the frozen B-S capability dose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duraseed.calibration_attempts import ArmAttempts
from duraseed.calibration_budget import (
    persist_budget_preflight,
    require_remaining_budget,
)
from duraseed.calibration_build import write_calibration_sources
from duraseed.calibration_provenance import start_calibration_run
from duraseed.calibration_sources import CalibrationSourceEvidence
from duraseed.capability_dose_billing import CapabilityDoseBilling
from duraseed.capability_dose_budget import (
    CapabilityDoseBudget,
    DOSE_CAP_USD,
    DOSE_TOKEN_CEILING,
)
from duraseed.capability_dose_preflight import capability_dose_preflight
from duraseed.capability_dose_provenance import (
    existing_capability_dose_terminal,
    finish_capability_dose,
    interrupt_capability_dose,
)
from duraseed.config import PilotConfig
from duraseed.data.io import atomic_write_bytes
from duraseed.data.stage_a_prompt_pools import StageAPromptPoolBundle
from duraseed.provenance import canonical_json_bytes, sha256_bytes, validate_sha256_id
from duraseed.runners import RunnerGateError
from duraseed.runners.capability_dose_arms import run_capability_dose_arm
from duraseed.runners.stage_a_setup import build_origin
from duraseed.runtime import RuntimeBundle, TokenLedger
from duraseed.training.acquisition_freeze import MaxTokenFreezeEvidence
from duraseed.training.sft import VerifiedSourceRecord
from duraseed.training.teacher_allocation_sources import TeacherAllocationSources


@dataclass(frozen=True, slots=True)
class CapabilityDoseInputs:
    config: PilotConfig
    runtime: RuntimeBundle
    stage_a_ledger: TokenLedger
    dose_budget: CapabilityDoseBudget
    actual_lifetime_billing: CapabilityDoseBilling
    output_root: Path
    run_id: str
    project_id: str
    tinker_session_id: str
    git_commit: str
    rest_client: Any
    teacher_sources: TeacherAllocationSources
    prompt_pools: StageAPromptPoolBundle
    sources: CalibrationSourceEvidence
    max_tokens: MaxTokenFreezeEvidence
    dose_sources: dict[str, VerifiedSourceRecord]
    charter_sha256: str
    panel_split_authorization_sha256: str
    panel_split_equivalence_sha256: str

    def __post_init__(self) -> None:
        for value in (
            self.charter_sha256,
            self.panel_split_authorization_sha256,
            self.panel_split_equivalence_sha256,
        ):
            validate_sha256_id(value)
        if (
            not self.run_id.strip()
            or any(value in self.run_id for value in "/\\")
            or not self.project_id.strip()
            or not self.tinker_session_id.strip()
            or not self.git_commit.strip()
            or self.stage_a_ledger.limits != DOSE_TOKEN_CEILING
            or self.stage_a_ledger.authorized_usd != float(DOSE_CAP_USD)
            or self.max_tokens.selected_max_tokens != 4096
            or self.actual_lifetime_billing.actual_lifetime_spend_usd
            + float(DOSE_CAP_USD)
            > 300
        ):
            raise ValueError("capability-dose launch differs from the frozen contract")

    @property
    def smoke(self):
        return self.sources.smoke

    @property
    def m0_sampler_path(self) -> str:
        return self.sources.m0_sampler_path

    @property
    def m0_state_path(self) -> str:
        return self.sources.m0_state_path

    @property
    def m0_training_step(self) -> int:
        return self.sources.m0_training_step


async def run_live_capability_dose(inputs: CapabilityDoseInputs) -> dict[str, Any]:
    root = inputs.output_root / inputs.run_id
    root.mkdir(parents=True, exist_ok=True)
    preflight = capability_dose_preflight(inputs)
    preflight_path = root / "preflight.json"
    payload = canonical_json_bytes(preflight)
    if preflight_path.exists() and preflight_path.read_bytes() != payload:
        raise RunnerGateError("capability-dose restart preflight changed")
    atomic_write_bytes(preflight_path, payload)
    preflight_sha256 = sha256_bytes(payload)
    terminal = existing_capability_dose_terminal(root, preflight_sha256)
    if terminal is not None:
        return terminal
    write_calibration_sources(
        root / "calibration-inputs", inputs.teacher_sources, inputs.prompt_pools
    )
    start_calibration_run(inputs, root)
    attempts = ArmAttempts(
        root / "capability-dose-arms",
        inputs.stage_a_ledger,
        run_id=inputs.run_id,
        action="capability-dose",
        project_id=inputs.project_id,
        preflight_sha256=preflight_sha256,
    )
    budget = require_remaining_budget(
        inputs.dose_budget,
        inputs.stage_a_ledger,
        prior_billed_usd=attempts.prior_billed_usd,
    )
    persist_budget_preflight(
        root / "capability-dose-arms",
        {
            **budget,
            "run_id": inputs.run_id,
            "action": "capability-dose",
            "preflight_sha256": preflight_sha256,
        },
    )
    attempt = attempts.open("b-s-capability-dose")
    try:
        if not attempt.completed:
            assert attempt.journal is not None
            origin = await build_origin(inputs, attempt.directory, attempt.journal)
            evidence = await run_capability_dose_arm(
                inputs, origin, attempt.directory, attempt.journal
            )
            attempts.complete(attempt, evidence)
        attempts.assert_no_unused_reconciliations()
        return finish_capability_dose(inputs, root, preflight_sha256=preflight_sha256)
    except Exception as error:
        interrupt_capability_dose(inputs, root, error)
        raise


__all__ = ["CapabilityDoseInputs", "run_live_capability_dose"]
