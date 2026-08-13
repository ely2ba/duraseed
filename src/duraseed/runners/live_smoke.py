"""Bounded real-runtime vertical slice for the live smoke gate."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Literal

from duraseed.config import PilotConfig, load_pilot_config
from duraseed.live_smoke_gate import (
    PHASE_LABEL,
    PROTOCOL_MAX_TOKENS,
    TOTAL_CAP_USD,
    SmokeRun,
    SmokeSettings,
    authorize,
    build_plan,
    preflight_text,
    write_json,
)
from duraseed.live_smoke_sampling import SmokeSampler
from duraseed.live_smoke_resume import run_resume_branch
from duraseed.live_smoke_updates import run_stage_a
from duraseed.run_records import RunStatus
from duraseed.runners import (
    LaunchAuthorization,
    RunnerGateError,
    validate_mock_output_root,
)
from duraseed.runners.live_smoke_data import SmokeInputs, build_inputs
from duraseed.runtime import (
    RuntimeBundle,
    TokenBudget,
    TokenLedger,
    create_sampler,
    create_service,
    load_sdk,
    resolve_model,
    set_and_verify_ttl,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def require_clean_worktree() -> None:
    """Refuse paid execution when recorded HEAD would omit local changes."""

    try:
        status = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RunnerGateError(
            "live smoke cannot authenticate the git worktree"
        ) from error
    if status:
        raise RunnerGateError("live smoke requires a clean git worktree")


async def execute_smoke(
    runtime: RuntimeBundle,
    ledger: TokenLedger,
    settings: SmokeSettings,
    config: PilotConfig,
    inputs: SmokeInputs,
    *,
    evidence_origin: Literal["remote", "mock"],
    project_id: str,
    existing_run: SmokeRun | None = None,
) -> Path:
    """Use the identical orchestration with either Tinker or behavioral fakes."""

    if evidence_origin == "mock":
        validate_mock_output_root(settings.output_root)
    if (
        settings.group_size != config.tinker.group_size
        or settings.max_tokens != config.tinker.max_sampled_tokens
        or settings.max_tokens != PROTOCOL_MAX_TOKENS
    ):
        raise ValueError("smoke sampling values violate configured ceilings")
    run = existing_run or SmokeRun.start(
        settings,
        config,
        inputs.manifests,
        evidence_origin=evidence_origin,
        project_id=project_id,
    )
    samples = SmokeSampler(settings.run_id, settings.seed, run.directory, ledger, run)
    try:
        base_sampler = await run.paid(
            "client:base-sampler",
            ledger,
            lambda: create_sampler(runtime, ledger=ledger),
            reservation={"token_budget": 0},
            persist=lambda _: {"base_model": runtime.model is not None},
        )
        probe_rows = await samples.probe_transport(
            runtime,
            base_sampler,
            inputs.tces_task,
            inputs.maps_task,
            settings.group_size,
        )
        diagnostics = await run_stage_a(
            runtime, ledger, run, samples, settings, inputs, settings.max_tokens
        )
        resume = await run_resume_branch(
            runtime, ledger, run, samples, settings, inputs, settings.max_tokens
        )
        acceptance = samples.acceptance(
            runtime,
            (*inputs.rl_tasks, inputs.maps_task),
            probe_rows=probe_rows,
        )
        acceptance.update(
            phase_label=PHASE_LABEL,
            status="passed" if evidence_origin == "remote" else "mock_passed",
            real_data=evidence_origin == "remote",
            full_state_resume=True,
            resumed_roundtrip_state_path=resume.resumed_roundtrip_state_path,
            weights_only_branch=True,
            checkpoint_lineage={
                "stage_a_state_path": resume.stage_a_state_path,
                "resumed_roundtrip_state_path": resume.resumed_roundtrip_state_path,
                "stage_b_sampler_path": resume.stage_b_sampler_path,
                "stage_b_state_path": None,
            },
            updates={
                "tces_sft": True,
                "tces_group_relative_rl": True,
                "maps_sft": True,
            },
            maps_before_after_reward=[
                resume.maps_before.reward.reward,
                resume.maps_after.reward.reward,
            ],
            group_relative_rl={
                "mean": diagnostics.group_means[0],
                "advantages": diagnostics.centered_advantages[0],
            },
            observed_cost_usd=ledger.observed_cost_usd,
        )
        paths = (
            resume.stage_a_state_path,
            resume.resumed_roundtrip_state_path,
            resume.stage_b_sampler_path,
        )

        def persist_ttl(rows):  # type: ignore[no-untyped-def]
            evidence = [
                {
                    "path": row.path,
                    "training_run_id": row.training_run_id,
                    "expires_at": row.expires_at.isoformat(),
                    "checkpoint_type": str(row.checkpoint_type),
                    "ttl_seconds": row.ttl_seconds,
                }
                for row in rows
            ]
            write_json(run.directory / "checkpoint_ttl_audit.json", evidence)
            return {"verified_paths": [row["path"] for row in evidence]}

        await run.paid(
            "checkpoint:ttl-audit",
            ledger,
            lambda: set_and_verify_ttl(
                runtime,
                runtime.service.create_rest_client(),
                paths,
                ttl_seconds=7 * 24 * 60 * 60,
                ledger=ledger,
            ),
            reservation={"token_budget": 0, "path_count": len(paths)},
            persist=persist_ttl,
        )
        write_json(
            run.directory / "sampling_diagnostics.json",
            {
                "stop_contract": acceptance["stop_contract"],
                "max_tokens": acceptance["max_tokens"],
            },
        )
        if (
            not acceptance["online_offline_reward_parity"]
            or not acceptance["stop_contract_verified"]
        ):
            raise RuntimeError("live smoke acceptance criteria failed")
        write_json(run.directory / "acceptance.json", acceptance)
        run.finish(
            RunStatus.COMPLETED,
            ledger,
            sampler_path=resume.stage_b_sampler_path,
        )
        return run.directory
    except Exception as error:
        write_json(
            run.directory / "failure.json",
            {
                "phase_label": PHASE_LABEL,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        run.finish(RunStatus.FAILED, ledger, deviations=[str(error)])
        raise


async def run_remote(
    settings: SmokeSettings, authorization: LaunchAuthorization, *, project_id: str
) -> Path:
    if (
        authorization.plan_name != PHASE_LABEL
        or authorization.authorized_cost_usd != TOTAL_CAP_USD
    ):
        raise ValueError("the exact $25 live-smoke authorization is required")
    if not project_id.strip():
        raise ValueError("an explicit Tinker project ID is required")
    require_clean_worktree()
    config = load_pilot_config(settings.config_path)
    inputs = build_inputs(settings.seed)
    sample_limit = settings.max_tokens * (
        settings.group_size * (2 + len(inputs.rl_tasks)) + 2
    )
    ledger = TokenLedger(TokenBudget(200_000, sample_limit, 100_000), 25.0)
    run = SmokeRun.start(
        settings,
        config,
        inputs.manifests,
        evidence_origin="remote",
        project_id=project_id,
    )
    try:
        sdk = load_sdk()

        async def construct_service():
            return create_service(
                sdk,
                project_id=project_id,
                user_metadata={"phase_label": PHASE_LABEL, "run_id": settings.run_id},
            )

        service = await run.paid(
            "client:service",
            ledger,
            construct_service,
            reservation={"token_budget": 0},
            persist=lambda _: {"project_id": project_id},
        )
        runtime = await run.paid(
            "client:create-model",
            ledger,
            lambda: resolve_model(sdk, service, seed=settings.seed, ledger=ledger),
            reservation={"model": "Qwen/Qwen3.5-9B-Base", "lora_rank": 32},
            persist=lambda value: {
                "sdk_version": value.sdk.sdk_version,
                "cookbook_version": value.sdk.cookbook_version,
            },
        )
        return await execute_smoke(
            runtime,
            ledger,
            settings,
            config,
            inputs,
            evidence_origin="remote",
            project_id=project_id,
            existing_run=run,
        )
    except Exception as error:
        if not (run.directory / "failure.json").exists():
            write_json(
                run.directory / "failure.json",
                {
                    "phase_label": PHASE_LABEL,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            run.finish(RunStatus.FAILED, ledger, deviations=[str(error)])
        raise


__all__ = [
    "PHASE_LABEL",
    "TOTAL_CAP_USD",
    "SmokeSettings",
    "authorize",
    "build_plan",
    "execute_smoke",
    "preflight_text",
    "run_remote",
]
