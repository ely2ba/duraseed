"""Authenticated launch boundary for the paid boundary-extension runner."""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import subprocess
from typing import Any

from duraseed.boundary_live_sources import load_boundary_live_source
from duraseed.config import load_pilot_config
from duraseed.git_guard import require_clean_worktree
from duraseed.live_smoke_gate import PHASE_LABEL, TOTAL_CAP_USD, TTL_SECONDS
from duraseed.post_smoke_billing import SmokeBillingTotals
from duraseed.run_records import RunRecord, RunStatus, read_run_record
from duraseed.runners import (
    LaunchAuthorization,
    RunnerGateError,
    authorize_launch,
)
from duraseed.runners.boundary_extension import build_plan
from duraseed.runners.boundary_billing import authenticate_post_smoke_billing
from duraseed.runners.boundary_live import execute_boundary_live
from duraseed.runtime import (
    MODEL_ID,
    RENDERER_NAME,
    LORA_RANK,
    RuntimeBundle,
    TokenBudget,
    TokenLedger,
    create_sampler,
    create_service,
    load_sdk,
)


def _json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid {label} artifact") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"invalid {label} artifact")
    return resolved, value


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise RunnerGateError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RunnerGateError(f"{label} must be a UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunnerGateError(f"{label} must be a UTC timestamp")
    return parsed.astimezone(UTC)


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise RunnerGateError(f"{label} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise RunnerGateError(f"{label} must be a finite decimal") from error
    if not result.is_finite():
        raise RunnerGateError(f"{label} must be a finite decimal")
    return result


def authenticate_live_smoke(
    path: str | Path, *, project_id: str
) -> tuple[str, str, datetime]:
    """Authenticate the completed real smoke and return its run ID and digest."""

    resolved, value = _json(path, "live-smoke acceptance")
    max_tokens = value.get("max_tokens")
    lineage = value.get("checkpoint_lineage")
    required = (
        value.get("phase_label") == PHASE_LABEL,
        value.get("status") == "passed",
        value.get("real_data") is True,
        value.get("online_offline_reward_parity") is True,
        value.get("stop_contract_verified") is True,
        value.get("full_state_resume") is True,
        value.get("weights_only_branch") is True,
        isinstance(max_tokens, dict),
        isinstance(lineage, dict),
    )
    if not all(required):
        raise RunnerGateError("live-smoke acceptance gates did not all pass")
    assert isinstance(max_tokens, dict) and isinstance(lineage, dict)
    if (
        max_tokens.get("protocol_value") != 4096
        or max_tokens.get("runtime_diagnostic_passed") is not True
        or not isinstance(max_tokens.get("sample_count"), int)
        or max_tokens["sample_count"] < 1
        or not all(
            isinstance(lineage.get(name), str) and lineage[name].strip()
            for name in (
                "stage_a_state_path",
                "resumed_roundtrip_state_path",
                "stage_b_sampler_path",
            )
        )
    ):
        raise RunnerGateError("live-smoke runtime evidence is incomplete")
    try:
        run = RunRecord.model_validate_json((resolved.parent / "run.json").read_bytes())
    except (OSError, ValueError) as error:
        raise RunnerGateError("live-smoke run record is invalid") from error
    finished = run.finished_at
    paths = tuple(
        lineage[name]
        for name in (
            "stage_a_state_path",
            "resumed_roundtrip_state_path",
            "stage_b_sampler_path",
        )
    )
    if (
        run.status is not RunStatus.COMPLETED
        or run.run_kind != "engineering_smoke"
        or run.project_id != project_id
        or run.model_id != MODEL_ID
        or run.renderer != RENDERER_NAME
        or run.lora_rank != LORA_RANK
        or run.authorized_cost_usd != float(TOTAL_CAP_USD)
        or run.reserved_cost_usd != float(TOTAL_CAP_USD)
        or finished is None
        or run.final_sampler_checkpoint_path != paths[-1]
        or len(set(paths)) != len(paths)
        or _decimal(value.get("observed_cost_usd"), "smoke observed cost")
        != _decimal(run.cost_usd, "smoke run cost")
        or _decimal(run.cost_usd, "smoke run cost") > TOTAL_CAP_USD
    ):
        raise RunnerGateError("live-smoke run identity is incomplete or inconsistent")
    try:
        ttl = json.loads(
            (resolved.parent / "checkpoint_ttl_audit.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("smoke TTL audit is invalid") from error
    if not isinstance(ttl, list):
        raise RunnerGateError("smoke TTL audit is invalid")
    by_path = {
        row.get("path"): row
        for row in ttl
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    if (
        len(by_path) != len(ttl)
        or set(by_path) != set(paths)
        or any(
            row.get("ttl_seconds") != TTL_SECONDS
            or not isinstance(row.get("training_run_id"), str)
            or not row["training_run_id"].strip()
            or _utc(row.get("expires_at"), "smoke checkpoint expiry") <= finished
            for row in by_path.values()
        )
    ):
        raise RunnerGateError("smoke TTL audit does not cover the retained checkpoints")
    run_id = resolved.parent.name
    digest = "sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest()
    return run_id, digest, finished


def authorize_boundary(
    *,
    authorized_cost_usd: str | None,
    smoke_acceptance: str | Path,
    billing_reconciliation: str | Path,
    human_approval: bool,
    project_id: str,
) -> tuple[LaunchAuthorization, str]:
    smoke_run_id, smoke_sha256, finished_at = authenticate_live_smoke(
        smoke_acceptance, project_id=project_id
    )
    smoke_run = read_run_record(Path(smoke_acceptance).resolve().parent)
    authenticate_post_smoke_billing(
        billing_reconciliation,
        smoke_run_id=smoke_run_id,
        smoke_sha256=smoke_sha256,
        smoke_finished_at=finished_at,
        smoke_tokens=SmokeBillingTotals(
            smoke_run.prompt_tokens, smoke_run.sampled_tokens, smoke_run.train_tokens
        ),
        project_id=project_id,
    )
    authorization = authorize_launch(
        build_plan(),
        execute=True,
        authorized_cost_usd=authorized_cost_usd,
        preconditions={
            "live_smoke_passed": True,
            "boundary_extension_human_approval": human_approval,
            "extension1_source_authenticated": True,
            "remaining_balance_verified": True,
        },
    )
    return authorization, smoke_sha256


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


async def run_remote_boundary(
    *,
    authorization: LaunchAuthorization,
    project_id: str,
    run_id: str,
    source_root: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    extension1_confirmation_path: str | Path,
) -> Path:
    """Bind the selected M0 sampler and execute the fixed four-action chain."""

    if (
        authorization.plan_name != "boundary-extension"
        or authorization.authorized_cost_usd != build_plan().remote_cost_cap_usd
    ):
        raise RunnerGateError("the exact $120 boundary authorization is required")
    require_clean_worktree(gate_name="boundary extension")
    config = load_pilot_config(config_path)
    source = load_boundary_live_source(config, source_root)
    if project_id != source.contract.project_id:
        raise RunnerGateError("project ID differs from the authenticated M0 source")
    sdk = load_sdk()
    tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_ID)
    renderer = sdk.get_renderer(RENDERER_NAME, tokenizer, model_name=MODEL_ID)
    service = create_service(
        sdk,
        project_id=project_id,
        user_metadata={"gate_name": "boundary-extension", "run_id": run_id},
    )
    runtime = RuntimeBundle(sdk, service, None, renderer, tokenizer)
    bootstrap = TokenLedger(TokenBudget(0, 0, 0), 0)
    sampler = await create_sampler(
        runtime,
        ledger=bootstrap,
        checkpoint_path=source.contract.sampler_checkpoint_path,
    )
    await execute_boundary_live(
        runtime,
        sampler,
        source=source,
        config=config,
        output_root=output_root,
        run_id=run_id,
        git_commit=_git_commit(),
        extension1_confirmation_path=extension1_confirmation_path,
    )
    return Path(output_root) / run_id
