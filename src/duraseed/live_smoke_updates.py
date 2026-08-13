"""TCES SFT and group-relative update mechanics for the live smoke."""

from __future__ import annotations

from duraseed.live_smoke_gate import PHASE_LABEL, TTL_SECONDS, SmokeRun, SmokeSettings
from duraseed.live_smoke_sampling import SmokeSampler
from duraseed.run_records import append_jsonl
from duraseed.runners.live_smoke_data import SmokeInputs
from duraseed.runtime import (
    RuntimeBundle,
    TokenLedger,
    apply_update,
    create_sampler,
    rl_datums,
    save_sampler_checkpoint,
    sft_datum,
)
from duraseed.training.grpo import GroupedRewardDiagnostics, grouped_reward_diagnostics


def _metric_persist(run: SmokeRun, action: str):
    def persist(metrics: dict[str, float]) -> dict[str, object]:
        append_jsonl(
            run.directory / "metrics.jsonl",
            {"phase_label": PHASE_LABEL, "action": action, "metrics": metrics},
        )
        return {"metrics": metrics}

    return persist


async def run_stage_a(
    runtime: RuntimeBundle,
    ledger: TokenLedger,
    run: SmokeRun,
    samples: SmokeSampler,
    settings: SmokeSettings,
    inputs: SmokeInputs,
    max_tokens: int,
) -> GroupedRewardDiagnostics:
    datum = sft_datum(runtime, inputs.tces_source)
    await run.paid(
        "update:tces-sft",
        ledger,
        lambda: apply_update(
            runtime,
            [datum],
            loss_fn="cross_entropy",
            learning_rate=1e-4,
            ledger=ledger,
        ),
        reservation={
            "loss_fn": "cross_entropy",
            "train_tokens": datum.model_input.length,
        },
        persist=_metric_persist(run, "tces_sft"),
    )
    path = await run.paid(
        "checkpoint:after-sft-sampler",
        ledger,
        lambda: save_sampler_checkpoint(
            runtime,
            name=f"{settings.run_id}-after-sft",
            ttl_seconds=TTL_SECONDS,
            ledger=ledger,
            reserved_storage_usd=1.0,
        ),
        reservation={"fixed_usd": 1.0, "ttl_seconds": TTL_SECONDS},
        persist=lambda value: {"sampler_path": value},
    )
    client = await run.paid(
        "client:after-sft-sampler",
        ledger,
        lambda: create_sampler(runtime, ledger=ledger, checkpoint_path=path),
        reservation={"token_budget": 0},
        persist=lambda _: {"source_sampler_path": path},
    )
    selected = None
    diagnostics = None
    for attempt, task in enumerate(inputs.rl_tasks):
        rows = await samples.collect(
            runtime,
            client,
            task,
            label=f"rl-group-{attempt}",
            stage="stage_a",
            path=path,
            count=settings.group_size,
            max_tokens=max_tokens,
        )
        candidate = grouped_reward_diagnostics(
            [float(row.reward.reward) for row in rows], group_size=settings.group_size
        )
        if candidate.mixed_group_count == 1:
            selected, diagnostics = rows, candidate
            break
    if selected is None or diagnostics is None:
        raise RuntimeError("no mixed exact-reward group inside the smoke attempt cap")
    advantages = diagnostics.centered_advantages[0]
    datums = rl_datums(runtime, selected, advantages)
    await run.paid(
        "update:group-relative-rl",
        ledger,
        lambda: apply_update(
            runtime,
            datums,
            loss_fn="importance_sampling",
            learning_rate=1e-5,
            ledger=ledger,
        ),
        reservation={
            "loss_fn": "importance_sampling",
            "train_tokens": sum(datum.model_input.length for datum in datums),
        },
        persist=_metric_persist(run, "group_relative_rl"),
    )
    return diagnostics


__all__ = ["run_stage_a"]
