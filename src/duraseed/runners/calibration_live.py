from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duraseed.calibration_parent import CalibrationParentEvidence
from duraseed.calibration_attempts import ReconciledRestart, hydrate_attempt_ledger
from duraseed.calibration_build import write_calibration_sources
from duraseed.calibration_integrity import (
    seal_calibration_action,
    validate_committed_action,
)
from duraseed.calibration_preflight import (
    calibration_preflight,
    validate_repair_allocation,
    validate_restart_reconciliations,
)
from duraseed.calibration_provenance import (
    finish_calibration_run,
    read_calibration_session_ids,
    start_calibration_run,
    validate_action_ttl_audit,
    verify_action_ttls,
)
from duraseed.calibration_sources import CalibrationSourceEvidence
from duraseed.calibration_stage_a_terminal import (
    StageAScientificFailure,
    existing_stage_a_terminal,
    finish_stage_a_terminal,
)
from duraseed.calibration_state import (
    SCHEMA_VERSION,
    artifact as _artifact,
    checkpoint as _checkpoint,
    commit_action as _commit_action,
    existing as _existing,
    read as _read,
    usage as _usage,
    write as _write,
)
from duraseed.config import PilotConfig
from duraseed.data.stage_a_prompt_pools import StageAPromptPoolBundle
from duraseed.provenance import sha256_bytes, validate_sha256_id
from duraseed.run_records import RunStatus
from duraseed.runners import RunnerGateError
from duraseed.runtime import RuntimeBundle, TokenLedger
from duraseed.training.acquisition_freeze import (
    MaxTokenFreezeEvidence,
    freeze_acquisition,
)
from duraseed.training.teacher_allocation_sources import TeacherAllocationSources


@dataclass(frozen=True, slots=True)
class CalibrationLiveInputs:
    config: PilotConfig
    runtime: RuntimeBundle
    teacher_ledger: TokenLedger
    stage_a_ledger: TokenLedger
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
    parent_teacher_evidence: CalibrationParentEvidence
    panel_split_authorization_sha256: str
    panel_split_equivalence_sha256: str
    precalibration_billing_sha256: str
    precalibration_raw_billing_sha256: str
    reconciled_restarts: tuple[ReconciledRestart, ...] = ()

    def __post_init__(self) -> None:
        validate_sha256_id(self.panel_split_authorization_sha256)
        validate_sha256_id(self.panel_split_equivalence_sha256)
        validate_sha256_id(self.precalibration_billing_sha256)
        validate_sha256_id(self.precalibration_raw_billing_sha256)
        validate_repair_allocation(self)

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


async def run_live_calibration(inputs: CalibrationLiveInputs) -> dict[str, Any]:
    from duraseed.runners.stage_a_live import collect_stage_a

    if (
        inputs.smoke.protocol_max_tokens != inputs.config.tinker.max_sampled_tokens
        or inputs.config.stage_a.provisional_max_tokens != 256
        or inputs.max_tokens.selected_max_tokens
        > inputs.config.tinker.max_sampled_tokens
    ):
        raise RunnerGateError(
            "runtime or prospective acquisition max-token contract differs"
        )
    if (
        not inputs.run_id.strip()
        or any(value in inputs.run_id for value in "/\\")
        or not inputs.project_id.strip()
        or not inputs.tinker_session_id.strip()
        or not inputs.git_commit.strip()
    ):
        raise ValueError("calibration launch identity must be nonempty and path-safe")
    root = inputs.output_root / inputs.run_id
    root.mkdir(parents=True, exist_ok=True)
    preflight = calibration_preflight(inputs, SCHEMA_VERSION)
    preflight_path = root / "preflight.json"
    if preflight_path.exists() and _read(preflight_path) != preflight:
        raise RunnerGateError("calibration restart preflight changed")
    _write(preflight_path, preflight)
    preflight_sha256 = sha256_bytes(preflight_path.read_bytes())
    terminal = existing_stage_a_terminal(root, preflight_sha256, inputs)
    if terminal is not None:
        return terminal
    validate_restart_reconciliations(inputs, preflight_sha256)
    prior_sessions = read_calibration_session_ids(root)
    if any(
        row.failed_tinker_session_id not in prior_sessions
        for row in inputs.reconciled_restarts
    ):
        raise RunnerGateError("restart failed session is absent from run lineage")
    write_calibration_sources(
        root / "calibration-inputs", inputs.teacher_sources, inputs.prompt_pools
    )
    run = start_calibration_run(inputs, root)
    state, artifacts = _checkpoint(root, preflight_sha256)
    if "stage-a" in artifacts:
        validate_action_ttl_audit("stage-a", root / "stage-a-arms", inputs)
        validate_committed_action(
            "stage-a",
            root / "stage-a-arms",
            artifacts["stage-a"],
            teacher_updates=0,
        )
        hydrate_attempt_ledger(root / "stage-a-arms", inputs.stage_a_ledger)
    if state["status"] == "completed":
        if run.status is not RunStatus.COMPLETED:
            finish_calibration_run(inputs, root, RunStatus.COMPLETED)
        return {"state": state, "artifacts": artifacts}
    try:
        if "stage-a" not in artifacts:
            evidence, common_rl = await collect_stage_a(
                inputs,
                root / "stage-a-arms",
                preflight_sha256=preflight_sha256,
            )
            integrity = seal_calibration_action(
                "stage-a",
                root / "stage-a-arms",
                teacher_updates=0,
            )
            await verify_action_ttls("stage-a", root / "stage-a-arms", inputs)
            freeze = freeze_acquisition(
                evidence, inputs.smoke, inputs.max_tokens, common_rl
            )
            artifacts["stage-a"] = _artifact(
                "stage-a",
                inputs.stage_a_ledger.authorized_usd,
                preflight_sha256=preflight_sha256,
                origin="direct-m0",
                freeze=freeze,
                integrity=integrity,
                checkpoint_ttl_audit_sha256=sha256_bytes(
                    (root / "stage-a-arms/checkpoint-ttl-audit.json").read_bytes()
                ),
                usage=_usage(inputs.stage_a_ledger),
            )
            state, artifacts = _commit_action(
                root,
                "stage-a",
                artifacts["stage-a"],
                preflight_sha256,
            )
        state, artifacts = _checkpoint(root, preflight_sha256)
        finish_calibration_run(inputs, root, RunStatus.COMPLETED)
        return {"state": state, "artifacts": artifacts}
    except StageAScientificFailure as failure:
        try:
            integrity = seal_calibration_action(
                "stage-a",
                root / "stage-a-arms",
                teacher_updates=0,
                stage_a_screen_only=failure.screen_only,
            )
            await verify_action_ttls(
                "stage-a",
                root / "stage-a-arms",
                inputs,
                stage_a_screen_only=failure.screen_only,
            )
            return finish_stage_a_terminal(
                inputs,
                root,
                preflight_sha256=preflight_sha256,
                failure=failure,
                integrity=integrity,
                ttl_audit_sha256=sha256_bytes(
                    (root / "stage-a-arms/checkpoint-ttl-audit.json").read_bytes()
                ),
            )
        except BaseException as error:
            _record_interruption(inputs, root, preflight_sha256, error)
            raise
    except BaseException as error:
        _record_interruption(inputs, root, preflight_sha256, error)
        raise


def _record_interruption(
    inputs: CalibrationLiveInputs,
    root: Path,
    preflight_sha256: str,
    error: BaseException,
) -> None:
    state, _ = _existing(root, preflight_sha256)
    state.update(status="interrupted", error=f"{type(error).__name__}: {error}")
    _write(root / "state.json", state)
    finish_calibration_run(
        inputs,
        root,
        RunStatus.INTERRUPTED,
        error=f"{type(error).__name__}: {error}",
    )
