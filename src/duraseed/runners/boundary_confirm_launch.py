"""Fresh-client launch for the authenticated confirmation-only continuation."""

from __future__ import annotations

import importlib
from pathlib import Path

from duraseed.boundary_confirmation_preparation import prepare_boundary_confirmation
from duraseed.boundary_confirmation_resume import validate_confirmation_resume
from duraseed.boundary_live_sources import load_boundary_live_source
from duraseed.config import load_pilot_config
from duraseed.git_guard import require_clean_worktree
from duraseed.runners import RunnerGateError
from duraseed.runners.boundary_confirm_resume import (
    execute_boundary_confirmation_resume,
)
from duraseed.runners.boundary_launch import _git_commit
from duraseed.runtime import (
    MODEL_ID,
    RENDERER_NAME,
    RuntimeBundle,
    TokenBudget,
    TokenLedger,
    create_sampler,
    create_service,
    load_sdk,
)


async def run_remote_boundary_confirmation_resume(
    *,
    project_id: str,
    run_id: str,
    source_root: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    human_approval: bool,
) -> Path:
    """Use the existing $120 authorization for confirmation-only continuation."""

    if human_approval is not True:
        raise RunnerGateError(
            "confirmation continuation requires explicit human approval"
        )
    require_clean_worktree(gate_name="boundary confirmation continuation")
    config = load_pilot_config(config_path)
    source = load_boundary_live_source(config, source_root)
    if project_id != source.contract.project_id:
        raise RunnerGateError("project ID differs from the authenticated M0 source")
    directory = Path(output_root) / run_id
    snapshot = validate_confirmation_resume(directory, config=config, source=source)
    git_commit = _git_commit()
    prepared = prepare_boundary_confirmation(snapshot, config, git_commit=git_commit)

    # Capacity may take hours, so create the remote client only after it finishes.
    sdk = load_sdk()
    tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_ID)
    renderer = sdk.get_renderer(RENDERER_NAME, tokenizer, model_name=MODEL_ID)
    service = create_service(
        sdk,
        project_id=project_id,
        user_metadata={
            "gate_name": "boundary-extension-confirmation-resume",
            "run_id": run_id,
        },
    )
    runtime = RuntimeBundle(sdk, service, None, renderer, tokenizer)
    sampler = await create_sampler(
        runtime,
        ledger=TokenLedger(TokenBudget(0, 0, 0), 0),
        checkpoint_path=source.contract.sampler_checkpoint_path,
    )
    await execute_boundary_confirmation_resume(
        runtime,
        sampler,
        source=source,
        config=config,
        prepared=prepared,
        git_commit=git_commit,
    )
    return directory


__all__ = ["run_remote_boundary_confirmation_resume"]
