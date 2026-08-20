"""Credential-safe launch boundary for the frozen capability-dose run."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from duraseed.calibration_input_loader import (
    ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256,
    load_calibration_source_objects,
)
from duraseed.calibration_sources import (
    authenticate_calibration_sources,
    load_max_token_evidence,
)
from duraseed.capability_dose_billing import (
    CORRECTION_RUN_ID,
    load_capability_dose_billing,
)
from duraseed.capability_dose_budget import (
    DOSE_CAP_USD,
    build_capability_dose_plan,
    capability_dose_budget,
)
from duraseed.capability_dose_preflight import capability_dose_preflight
from duraseed.capability_dose_provenance import existing_capability_dose_terminal
from duraseed.capability_dose_sources import prepare_capability_dose_sources
from duraseed.config import load_pilot_config
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.runners import RunnerGateError, authorize_launch
from duraseed.runners.calibration_launch import _git_commit
from duraseed.runners.capability_dose_live import (
    CapabilityDoseInputs,
    run_live_capability_dose,
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
CHARTER_PATH = REPOSITORY_ROOT / "docs/amendment-capability-targeted-acquisition.md"


async def run_remote_capability_dose(
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
    lifetime_console_evidence_path: str | Path,
    project_id: str,
    authorized_cost_usd: str | None,
    human_approval: bool,
) -> Path:
    if (
        not run_id.strip()
        or any(value in run_id for value in "/\\")
        or run_id == CORRECTION_RUN_ID
        or not project_id.strip()
    ):
        raise RunnerGateError("capability-dose launch identity is invalid")
    output = Path(output_root)
    config = load_pilot_config(config_path)
    loaded = load_calibration_source_objects(
        config=config,
        boundary_directory=boundary_directory,
        source_directory=source_directory,
        panel_split_authorization_path=panel_split_authorization_path,
        panel_split_equivalence_path=panel_split_equivalence_path,
    )
    sources = authenticate_calibration_sources(
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
    billing = load_capability_dose_billing(
        output / CORRECTION_RUN_ID,
        Path(lifetime_console_evidence_path),
        project_id=project_id,
    )
    plan = build_capability_dose_plan(config)
    authorization = authorize_launch(
        plan,
        execute=True,
        authorized_cost_usd=authorized_cost_usd,
        preconditions={
            "actual_lifetime_reconciled": (
                billing.actual_lifetime_spend_usd + float(DOSE_CAP_USD) <= 300
            ),
            "human_approval": human_approval,
        },
    )
    if (
        authorization.authorized_cost_usd != DOSE_CAP_USD
        or loaded.authorization_sha256 != ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256
    ):
        raise RunnerGateError("capability-dose authorization or panel freeze differs")
    sdk = load_sdk()
    tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_ID)
    renderer = sdk.get_renderer(RENDERER_NAME, tokenizer, model_name=MODEL_ID)
    local_runtime = RuntimeBundle(sdk, None, None, renderer, tokenizer)
    local = SimpleNamespace(
        config=config,
        runtime=local_runtime,
        prompt_pools=loaded.prompts,
        max_tokens=max_tokens,
    )
    budget = capability_dose_budget(local)
    git_commit = _git_commit()
    charter_sha256 = sha256_bytes(CHARTER_PATH.read_bytes())
    ledger = TokenLedger(budget.tokens, float(budget.cent_ceiling_usd))
    preflight_inputs = SimpleNamespace(
        config=config,
        run_id=run_id,
        project_id=project_id,
        git_commit=git_commit,
        charter_sha256=charter_sha256,
        sources=sources,
        prompt_pools=loaded.prompts,
        panel_split_authorization_sha256=loaded.authorization_sha256,
        panel_split_equivalence_sha256=loaded.equivalence_sha256,
        dose_budget=budget,
        actual_lifetime_billing=billing,
    )
    preflight_sha256 = sha256_bytes(
        canonical_json_bytes(capability_dose_preflight(preflight_inputs))
    )
    root = output / run_id
    if root.exists() and existing_capability_dose_terminal(root, preflight_sha256):
        return root
    dose_sources = prepare_capability_dose_sources(local)
    service = create_service(
        sdk,
        project_id=project_id,
        user_metadata={"gate_name": "capability-dose", "run_id": run_id},
    )
    runtime = RuntimeBundle(sdk, service, None, renderer, tokenizer)
    session_id = str(service._get_session_holder().get_session_id())
    if not session_id.strip():
        raise RunnerGateError("Tinker service returned no session identity")
    inputs = CapabilityDoseInputs(
        config,
        runtime,
        ledger,
        budget,
        billing,
        output,
        run_id,
        project_id,
        session_id,
        git_commit,
        service.create_rest_client(),
        loaded.teacher,
        loaded.prompts,
        sources,
        max_tokens,
        dose_sources,
        charter_sha256,
        loaded.authorization_sha256,
        loaded.equivalence_sha256,
    )
    await run_live_capability_dose(inputs)
    return root


__all__ = ["run_remote_capability_dose"]
