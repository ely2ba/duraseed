"""Credential-safe launch boundary for one frozen Pilot-0 seed pair."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from duraseed.calibration_input_loader import (
    ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256,
    load_calibration_source_objects,
)
from duraseed.calibration_sources import authenticate_calibration_sources
from duraseed.config import load_pilot_config
from duraseed.pilot0_budget import build_pilot0_pair_plan, calculate_pilot0_budget
from duraseed.pilot0_contract import (
    PILOT_PAIR_PLANNING_CAP_USD,
    PILOT_TWO_PAIR_PLANNING_CAP_USD,
    Pilot0Inputs,
    PilotStageARecipe,
    validate_pilot0_inputs,
)
from duraseed.pilot0_pair_auth import (
    authenticate_pilot_sources,
    load_pilot_pair_billing,
)
from duraseed.pilot0_recovery import prepare_pilot0_recovery
from duraseed.pilot0_source_build import read_pilot_seed_sources
from duraseed.runners import RunnerGateError, authorize_launch
from duraseed.runners.calibration_launch import _git_commit
from duraseed.runners.pilot0_live import run_pilot0
from duraseed.runtime import (
    MODEL_ID,
    RENDERER_NAME,
    RuntimeBundle,
    TokenBudget,
    TokenLedger,
    create_service,
    load_sdk,
)


async def run_remote_pilot0_pair(
    *,
    run_id: str,
    pair_index: int,
    prepared_sources: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    boundary_directory: str | Path,
    source_directory: str | Path,
    smoke_acceptance_path: str | Path,
    m0_selection_path: str | Path,
    m0_ttl_path: str | Path,
    panel_split_authorization_path: str | Path,
    panel_split_equivalence_path: str | Path,
    dose_root: str | Path,
    stage_b_recipe_path: str | Path,
    lifetime_console_evidence_path: str | Path,
    prior_pair_root: str | Path | None,
    project_id: str,
    authorized_cost_usd: str | None,
    human_approval: bool,
    resume_interrupted: bool = False,
) -> Path:
    """Authenticate and launch one pair; never creates a service before approval."""

    if (
        not run_id.strip()
        or any(value in run_id for value in "/\\")
        or pair_index not in (1, 2)
        or not project_id.strip()
    ):
        raise RunnerGateError("Pilot-0 pair launch identity is invalid")
    output = Path(output_root)
    root = output / run_id
    if root.exists() and not resume_interrupted:
        raise RunnerGateError("Pilot-0 run ID already exists; no reroll is authorized")
    if resume_interrupted and not root.exists():
        raise RunnerGateError("Pilot-0 resume requires the interrupted run root")
    config = load_pilot_config(config_path)
    loaded = load_calibration_source_objects(
        config=config,
        boundary_directory=boundary_directory,
        source_directory=source_directory,
        panel_split_authorization_path=panel_split_authorization_path,
        panel_split_equivalence_path=panel_split_equivalence_path,
    )
    base = authenticate_calibration_sources(
        config=config,
        project_id=project_id,
        smoke_acceptance_path=smoke_acceptance_path,
        m0_selection_path=m0_selection_path,
        m0_ttl_path=m0_ttl_path,
        boundary_directory=boundary_directory,
        teacher_sources=loaded.teacher,
        prompt_pools=loaded.prompts,
    )
    sources = read_pilot_seed_sources(prepared_sources)
    source = sources[pair_index - 1]
    authentication = authenticate_pilot_sources(
        base=base,
        source=source,
        dose_root=Path(dose_root),
        stage_b_recipe_path=Path(stage_b_recipe_path),
    )
    billing = load_pilot_pair_billing(
        Path(dose_root),
        Path(lifetime_console_evidence_path),
        project_id=project_id,
        pair_index=pair_index,
        prior_pair_root=None if prior_pair_root is None else Path(prior_pair_root),
    )
    if loaded.authorization_sha256 != ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256:
        raise RunnerGateError("Pilot panel-split authorization differs")

    sdk = load_sdk()
    tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_ID)
    renderer = sdk.get_renderer(RENDERER_NAME, tokenizer, model_name=MODEL_ID)
    local_runtime = RuntimeBundle(sdk, None, None, renderer, tokenizer)
    planning = SimpleNamespace(
        runtime=local_runtime,
        source=source,
        acquisition=PilotStageARecipe(),
        ledger=TokenLedger(
            TokenBudget(10**15, 10**15, 10**15), PILOT_PAIR_PLANNING_CAP_USD
        ),
    )
    budget = calculate_pilot0_budget(planning)  # type: ignore[arg-type]
    plan = build_pilot0_pair_plan(budget.cent_ceiling_usd)
    authorization = authorize_launch(
        plan,
        execute=True,
        authorized_cost_usd=authorized_cost_usd,
        preconditions={
            "actual_spend_reconciled": (
                billing.remaining_balance_usd >= float(budget.cent_ceiling_usd)
                and Decimal(str(billing.lineage["actual_lifetime_spend_usd"]))
                + budget.cent_ceiling_usd
                <= Decimal(str(PILOT_TWO_PAIR_PLANNING_CAP_USD))
            ),
            "pair_order_valid": (
                (pair_index == 1 and billing.prior_pair_result_sha256 is None)
                or (pair_index == 2 and billing.prior_pair_result_sha256 is not None)
            ),
            "human_approval": human_approval,
        },
    )
    if authorization.authorized_cost_usd != budget.cent_ceiling_usd:
        raise RunnerGateError("Pilot pair authorization differs from exact preflight")
    active_git_commit = _git_commit()
    if resume_interrupted:
        try:
            stored_preflight = json.loads((root / "preflight.json").read_bytes())
            stored_lineage = stored_preflight["lineage"]
            git_commit = str(stored_lineage["git_commit"])
            primary_session_id = str(stored_lineage["session_id"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RunnerGateError("Pilot-0 resume preflight is invalid") from error
    else:
        git_commit = active_git_commit
        primary_session_id = "local-preflight-no-service"
    ledger = TokenLedger(budget.tokens, float(budget.cent_ceiling_usd))
    local_inputs = Pilot0Inputs(
        config=config,
        runtime=local_runtime,
        ledger=ledger,
        output_root=output,
        run_id=run_id,
        git_commit=git_commit,
        project_id=project_id,
        pair_index=pair_index,
        m0_sampler_path=base.m0_sampler_path,
        m0_state_path=base.m0_state_path,
        panel=loaded.teacher.panel,
        source=source,
        source_authentication=authentication,
        billing=billing,
        session_id=primary_session_id,
        dose_terminal_sha256=authentication.lineage["dose_terminal_sha256"],
        stage_b_recipe_artifact_sha256=(
            authentication.lineage["stage_b_recipe_sha256"]
        ),
        prior_pair_result_sha256=billing.prior_pair_result_sha256,
    )
    validate_pilot0_inputs(local_inputs)
    if not calculate_pilot0_budget(local_inputs).passed:
        raise RunnerGateError("Pilot pair ledger differs from exact preflight")
    service = create_service(
        sdk,
        project_id=project_id,
        user_metadata={
            "gate_name": "pilot0-paired-seed",
            "run_id": run_id,
            "pair_index": str(pair_index),
            "infrastructure_recovery": str(resume_interrupted).lower(),
        },
    )
    session_id = str(service._get_session_holder().get_session_id())
    if not session_id.strip():
        raise RunnerGateError("Tinker service returned no session identity")
    runtime = RuntimeBundle(sdk, service, None, renderer, tokenizer)
    if resume_interrupted:
        prepare_pilot0_recovery(
            root,
            recovery_session_id=session_id,
            recovery_git_commit=active_git_commit,
        )
        inputs = replace(local_inputs, runtime=runtime)
    else:
        inputs = replace(local_inputs, session_id=session_id, runtime=runtime)
    await run_pilot0(inputs)
    return root


__all__ = ["run_remote_pilot0_pair"]
