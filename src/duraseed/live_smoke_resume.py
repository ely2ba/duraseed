"""Checkpoint, resume, branch, and MAPS proof for the live smoke."""

from __future__ import annotations

from dataclasses import dataclass

from duraseed.live_smoke_gate import PHASE_LABEL, TTL_SECONDS, SmokeRun, SmokeSettings
from duraseed.live_smoke_sampling import SmokeSampler
from duraseed.run_records import append_jsonl
from duraseed.runners.live_smoke_data import SmokeInputs
from duraseed.runtime import (
    RuntimeBundle,
    SampleObservation,
    TokenLedger,
    apply_update,
    bind_model,
    create_sampler,
    restore_checkpoint,
    save_checkpoint,
    save_sampler_checkpoint,
    sft_datum,
)


@dataclass(frozen=True, slots=True)
class ResumeBranchResult:
    stage_a_state_path: str
    resumed_roundtrip_state_path: str
    stage_b_sampler_path: str
    maps_before: SampleObservation
    maps_after: SampleObservation


def _metric(run: SmokeRun, action: str):
    def persist(metrics: dict[str, float]) -> dict[str, object]:
        append_jsonl(
            run.directory / "metrics.jsonl",
            {"phase_label": PHASE_LABEL, "action": action, "metrics": metrics},
        )
        return {"metrics": metrics}

    return persist


async def run_resume_branch(
    runtime: RuntimeBundle,
    ledger: TokenLedger,
    run: SmokeRun,
    samples: SmokeSampler,
    settings: SmokeSettings,
    inputs: SmokeInputs,
    max_tokens: int,
) -> ResumeBranchResult:
    def checkpoint_evidence(pair):  # type: ignore[no-untyped-def]
        evidence = {
            "phase_label": PHASE_LABEL,
            "sampler_path": pair.sampler_path,
            "state_path": pair.state_path,
            "ttl_seconds": TTL_SECONDS,
        }
        append_jsonl(run.directory / "checkpoints.jsonl", evidence)
        return evidence

    checkpoint = await run.paid(
        "checkpoint:stage-a-pair",
        ledger,
        lambda: save_checkpoint(
            runtime,
            name=f"{settings.run_id}-stage-a",
            ttl_seconds=TTL_SECONDS,
            ledger=ledger,
            reserved_storage_usd=1.0,
        ),
        reservation={"fixed_usd": 1.0, "ttl_seconds": TTL_SECONDS},
        persist=checkpoint_evidence,
    )
    stage_a_sampler = await run.paid(
        "client:stage-a-sampler",
        ledger,
        lambda: create_sampler(
            runtime, ledger=ledger, checkpoint_path=checkpoint.sampler_path
        ),
        reservation={"token_budget": 0},
        persist=lambda _: {"source_sampler_path": checkpoint.sampler_path},
    )
    before = await samples.collect(
        runtime,
        stage_a_sampler,
        inputs.maps_task,
        label="maps-before-eval",
        stage="stage_a",
        path=checkpoint.sampler_path,
        count=1,
        max_tokens=max_tokens,
    )

    async def restore(full_state: bool):
        model = await restore_checkpoint(
            runtime, checkpoint.state_path, full_state=full_state, ledger=ledger
        )
        info = await model.get_info_async()
        if not getattr(info, "model_id", None):
            raise RuntimeError("restored model omitted its training identity")
        return model, info

    full_model, full_info = await run.paid(
        "restore:full-state",
        ledger,
        lambda: restore(True),
        reservation={"token_budget": 0, "source_state_path": checkpoint.state_path},
        persist=lambda value: {"training_run_id": str(value[1].model_id)},
    )
    branch_model, branch_info = await run.paid(
        "restore:weights-only-branch",
        ledger,
        lambda: restore(False),
        reservation={"token_budget": 0, "source_state_path": checkpoint.state_path},
        persist=lambda value: {"training_run_id": str(value[1].model_id)},
    )
    if not getattr(full_info, "model_id", None) or full_info.model_id == getattr(
        branch_info, "model_id", None
    ):
        raise RuntimeError("resume and weights-only branch are not distinct")
    full_runtime = bind_model(runtime.sdk, runtime.service, full_model)
    branch_runtime = bind_model(runtime.sdk, runtime.service, branch_model)
    resume_datum = sft_datum(full_runtime, inputs.tces_source)
    await run.paid(
        "update:resumed-full-state",
        ledger,
        lambda: apply_update(
            full_runtime,
            [resume_datum],
            loss_fn="cross_entropy",
            learning_rate=1e-4,
            ledger=ledger,
        ),
        reservation={
            "loss_fn": "cross_entropy",
            "train_tokens": resume_datum.model_input.length,
        },
        persist=_metric(run, "resumed_tces_sft"),
    )
    roundtrip_state = await run.paid(
        "checkpoint:resumed-full-state",
        ledger,
        lambda: save_checkpoint(
            full_runtime,
            name=f"{settings.run_id}-resumed",
            ttl_seconds=TTL_SECONDS,
            ledger=ledger,
            reserved_storage_usd=1.0,
        ),
        reservation={"fixed_usd": 1.0, "ttl_seconds": TTL_SECONDS},
        persist=checkpoint_evidence,
    )

    async def roundtrip_restore():
        model = await restore_checkpoint(
            full_runtime,
            roundtrip_state.state_path,
            full_state=True,
            ledger=ledger,
        )
        info = await model.get_info_async()
        if not getattr(info, "model_id", None):
            raise RuntimeError("round-trip restore omitted its training identity")
        return model, info

    await run.paid(
        "restore:resumed-full-state-roundtrip",
        ledger,
        roundtrip_restore,
        reservation={
            "token_budget": 0,
            "source_state_path": roundtrip_state.state_path,
        },
        persist=lambda value: {"training_run_id": str(value[1].model_id)},
    )
    maps_datum = sft_datum(branch_runtime, inputs.maps_source)
    await run.paid(
        "update:maps-sft",
        ledger,
        lambda: apply_update(
            branch_runtime,
            [maps_datum],
            loss_fn="cross_entropy",
            learning_rate=1e-4,
            ledger=ledger,
        ),
        reservation={
            "loss_fn": "cross_entropy",
            "train_tokens": maps_datum.model_input.length,
        },
        persist=_metric(run, "maps_sft"),
    )
    stage_b_path = await run.paid(
        "checkpoint:stage-b-sampler",
        ledger,
        lambda: save_sampler_checkpoint(
            branch_runtime,
            name=f"{settings.run_id}-stage-b",
            ttl_seconds=TTL_SECONDS,
            ledger=ledger,
            reserved_storage_usd=1.0,
        ),
        reservation={"fixed_usd": 1.0, "ttl_seconds": TTL_SECONDS},
        persist=lambda value: {"sampler_path": value},
    )
    stage_b_sampler = await run.paid(
        "client:stage-b-sampler",
        ledger,
        lambda: create_sampler(
            branch_runtime, ledger=ledger, checkpoint_path=stage_b_path
        ),
        reservation={"token_budget": 0},
        persist=lambda _: {"source_sampler_path": stage_b_path},
    )
    after = await samples.collect(
        branch_runtime,
        stage_b_sampler,
        inputs.maps_task,
        label="maps-after-eval",
        stage="stage_b",
        path=stage_b_path,
        count=1,
        max_tokens=max_tokens,
    )
    return ResumeBranchResult(
        checkpoint.state_path,
        roundtrip_state.state_path,
        stage_b_path,
        before[0],
        after[0],
    )


__all__ = ["ResumeBranchResult", "run_resume_branch"]
