"""Fail-closed concrete launch boundary for live acquisition calibration."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

from duraseed.calibration_attempts import load_reconciled_restart
from duraseed.calibration_budget import calibration_allocation
from duraseed.calibration_completion import completed_calibration
from duraseed.calibration_input_loader import (
    ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256,
    load_calibration_source_objects,
)
from duraseed.calibration_launch_auth import authenticate_precalibration_billing
from duraseed.calibration_preflight import (
    calibration_preflight,
    validate_restart_reconciliations,
)
from duraseed.calibration_provenance import read_calibration_session_ids
from duraseed.calibration_state import SCHEMA_VERSION
from duraseed.calibration_sources import (
    authenticate_calibration_sources,
    load_max_token_evidence,
)
from duraseed.config import load_pilot_config
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.runners import RunnerGateError, authorize_launch
from duraseed.runners.calibration import build_plan
from duraseed.runners.calibration_live import (
    CalibrationLiveInputs,
    run_live_calibration,
)
from duraseed.runtime import (
    MODEL_ID,
    RENDERER_NAME,
    RuntimeBundle,
    TokenLedger,
    create_service,
    load_sdk,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _git_commit() -> str:
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise RunnerGateError("calibration launch requires a clean git worktree")
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RunnerGateError("cannot bind calibration to a git commit") from error


async def run_remote_calibration(
    *,
    run_id: str,
    output_root: str | Path,
    config_path: str | Path,
    boundary_directory: str | Path,
    source_directory: str | Path,
    smoke_acceptance_path: str | Path,
    m0_selection_path: str | Path,
    m0_ttl_path: str | Path,
    panel_split_authorization_path: str | Path,
    panel_split_equivalence_path: str | Path,
    max_token_specification_path: str | Path,
    max_token_authorization_path: str | Path,
    max_token_evidence_path: str | Path,
    billing_reconciliation_path: str | Path,
    raw_billing_path: str | Path,
    project_id: str,
    authorized_cost_usd: str | None,
    human_approval: bool,
    restart_evidence: tuple[tuple[str | Path, str | Path], ...] = (),
) -> Path:
    """Authenticate every byte locally, then construct and run the real SDK path."""

    if not run_id.strip() or any(value in run_id for value in "/\\"):
        raise RunnerGateError("run_id must be one nonempty filename token")
    if not project_id.strip():
        raise RunnerGateError("an explicit Tinker project_id is required")
    config = load_pilot_config(config_path)
    billing = authenticate_precalibration_billing(
        billing_reconciliation_path,
        raw_billing_path,
        boundary_directory=boundary_directory,
        project_id=project_id,
    )
    loaded = load_calibration_source_objects(
        config=config,
        boundary_directory=boundary_directory,
        source_directory=source_directory,
        panel_split_authorization_path=panel_split_authorization_path,
        panel_split_equivalence_path=panel_split_equivalence_path,
    )
    source_evidence = authenticate_calibration_sources(
        config=config,
        project_id=project_id,
        smoke_acceptance_path=smoke_acceptance_path,
        m0_selection_path=m0_selection_path,
        m0_ttl_path=m0_ttl_path,
        boundary_directory=boundary_directory,
        teacher_sources=loaded.teacher,
        prompt_pools=loaded.prompts,
    )
    max_tokens = load_max_token_evidence(
        max_token_specification_path,
        max_token_authorization_path,
        max_token_evidence_path,
    )
    plan = build_plan(config)
    authorization = authorize_launch(
        plan,
        execute=True,
        authorized_cost_usd=authorized_cost_usd,
        preconditions={
            "panel_frozen": (
                ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256 is not None
                and loaded.authorization_sha256
                == ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256
            ),
            "live_smoke_passed": source_evidence.smoke.runtime_diagnostic_passed,
            "human_approval": human_approval,
            "remaining_balance_verified": (
                billing.remaining_balance_usd - plan.remote_cost_cap_usd
                >= billing.protected_reserve_usd
            ),
        },
    )
    if authorization.authorized_cost_usd != plan.remote_cost_cap_usd:
        raise RunnerGateError("calibration authorization is not exactly $300")
    restarts = tuple(
        load_reconciled_restart(artifact, raw) for artifact, raw in restart_evidence
    )
    tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
    renderers = importlib.import_module("tinker_cookbook.renderers")
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_ID)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer, model_name=MODEL_ID)
    budget_inputs = SimpleNamespace(
        config=config,
        runtime=SimpleNamespace(
            renderer=renderer,
            sdk=SimpleNamespace(train_on_what=renderers.TrainOnWhat),
        ),
        teacher_sources=loaded.teacher,
        prompt_pools=loaded.prompts,
        max_tokens=max_tokens,
    )
    allocation = calibration_allocation(budget_inputs)
    git_commit = _git_commit()
    teacher_ledger = TokenLedger(allocation.teacher_tokens, allocation.teacher_cap_usd)
    stage_a_ledger = TokenLedger(allocation.stage_a_tokens, allocation.stage_a_cap_usd)
    root = Path(output_root) / run_id
    local_inputs = SimpleNamespace(
        config=config,
        teacher_ledger=teacher_ledger,
        stage_a_ledger=stage_a_ledger,
        run_id=run_id,
        project_id=project_id,
        git_commit=git_commit,
        teacher_sources=loaded.teacher,
        prompt_pools=loaded.prompts,
        sources=source_evidence,
        smoke=source_evidence.smoke,
        max_tokens=max_tokens,
        m0_sampler_path=source_evidence.m0_sampler_path,
        m0_state_path=source_evidence.m0_state_path,
        panel_split_authorization_sha256=loaded.authorization_sha256,
        panel_split_equivalence_sha256=loaded.equivalence_sha256,
        precalibration_billing_sha256=billing.artifact_sha256,
        precalibration_raw_billing_sha256=billing.raw_billing_sha256,
        reconciled_restarts=restarts,
    )
    preflight_sha256 = sha256_bytes(
        canonical_json_bytes(calibration_preflight(local_inputs, SCHEMA_VERSION))
    )
    validate_restart_reconciliations(local_inputs, preflight_sha256)
    prior_sessions = read_calibration_session_ids(root)
    if any(row.failed_tinker_session_id not in prior_sessions for row in restarts):
        raise RunnerGateError("restart failed session is absent from run lineage")
    if completed_calibration(local_inputs, root):
        return root
    sdk = load_sdk()
    service = create_service(
        sdk,
        project_id=project_id,
        user_metadata={"gate_name": "acquisition-calibration", "run_id": run_id},
    )
    runtime = RuntimeBundle(sdk, service, None, renderer, tokenizer)
    holder = service._get_session_holder()
    session_id = str(holder.get_session_id())
    if not session_id.strip():
        raise RunnerGateError("Tinker service returned no session identity")
    inputs = CalibrationLiveInputs(
        config=config,
        runtime=runtime,
        teacher_ledger=teacher_ledger,
        stage_a_ledger=stage_a_ledger,
        output_root=Path(output_root),
        run_id=run_id,
        project_id=project_id,
        tinker_session_id=session_id,
        git_commit=git_commit,
        rest_client=service.create_rest_client(),
        teacher_sources=loaded.teacher,
        prompt_pools=loaded.prompts,
        sources=source_evidence,
        max_tokens=max_tokens,
        panel_split_authorization_sha256=loaded.authorization_sha256,
        panel_split_equivalence_sha256=loaded.equivalence_sha256,
        precalibration_billing_sha256=billing.artifact_sha256,
        precalibration_raw_billing_sha256=billing.raw_billing_sha256,
        reconciled_restarts=restarts,
    )
    await run_live_calibration(inputs)
    return root


__all__ = ["run_remote_calibration"]
