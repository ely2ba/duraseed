"""Direct-M0 checkpointed Stage-A branches for one paired Pilot seed."""

from __future__ import annotations

from pathlib import Path

from duraseed.pilot0_contract import (
    CADENCE_CHECKPOINT_TTL_SECONDS,
    Pilot0Inputs,
    PilotSeedSources,
    stage_a_grid,
)
from duraseed.pilot0_data import ordered_stage_a_pools, stage_a_solver_sources
from duraseed.pilot0_integrity import segment_coordinates
from duraseed.pilot0_recovery import recovery_segment
from duraseed.runners.pilot0_remote import (
    read_segment,
    restore_runtime,
    sampler_for_path,
    save_pair,
    write_segment,
)
from duraseed.runners.pilot0_sampling import evaluate_manifest
from duraseed.runners.pilot0_updates import grouped_rl_update, supervised_update
from duraseed.runners.remote_journal import RemoteJournal


async def _branch_segment(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    origin: dict,
    previous: dict,
    *,
    method: str,
    start: int,
    stop: int,
    pools: dict,
    sources: dict,
    output: Path,
    preflight_sha256: str,
) -> dict:
    learning_rate = inputs.acquisition.learning_rates[
        "static_sft" if method == "B-S" else "group_relative_rl"
    ]
    expected = segment_coordinates(
        inputs,
        source,
        preflight_sha256,
        kind="stage-a",
        method=method,
        start=start,
        stop=stop,
        origin_sampler_path=origin["sampler_path"],
        origin_state_path=origin["state_path"],
        parent_sampler_path=previous["sampler_path"],
        parent_state_path=previous["state_path"],
        learning_rate=learning_rate,
    )
    recovery = recovery_segment(inputs, output)
    completed = read_segment(output, expected, reconciled_resume=recovery is not None)
    if completed is not None:
        return completed
    output.mkdir(parents=True, exist_ok=True)
    journal = RemoteJournal(output, reconciled_resume=recovery is not None)
    if recovery is None:
        runtime = await restore_runtime(
            inputs,
            journal,
            path=previous["state_path"],
            full_state=start > 0,
            coordinate=expected,
        )
        for step in range(start + 1, stop + 1):
            if method == "B-S":
                await supervised_update(
                    inputs,
                    source,
                    runtime,
                    step=step,
                    learning_rate=learning_rate,
                    pools=pools,
                    sources=sources,
                    output=output,
                    journal=journal,
                )
            else:
                await grouped_rl_update(
                    inputs,
                    source,
                    runtime,
                    step=step,
                    learning_rate=learning_rate,
                    pools=pools,
                    boundary_sampler_path=origin["sampler_path"],
                    output=output,
                    journal=journal,
                )
        if stop % 10:
            return write_segment(
                output,
                {
                    **expected,
                    "step": stop,
                    "learning_rate": learning_rate,
                    "cadence_evaluated": False,
                    "checkpoint_retained": False,
                },
                ledger=inputs.ledger,
            )
        pair = await save_pair(
            inputs,
            runtime,
            journal,
            name=f"{inputs.run_id}-seed-{source.seed}-{method}-step-{stop}",
            ttl_seconds=CADENCE_CHECKPOINT_TTL_SECONDS,
            coordinate=expected,
        )
        sampler_path, state_path = pair.sampler_path, pair.state_path
    else:
        if (
            method != recovery.get("method")
            or start != recovery.get("start")
            or stop != recovery.get("stop")
        ):
            raise RuntimeError("Pilot recovery changed its interrupted segment")
        sampler_path = str(recovery["sampler_path"])
        state_path = str(recovery["state_path"])
    result = {
        **expected,
        "step": stop,
        "sampler_path": sampler_path,
        "state_path": state_path,
        "learning_rate": learning_rate,
        "cadence_evaluated": True,
        "checkpoint_retained": True,
    }
    sampler = await sampler_for_path(
        inputs, journal, path=sampler_path, coordinate=expected
    )
    monitor = await evaluate_manifest(
        inputs,
        source,
        manifest=source.a_cadence,
        sampler=sampler,
        sampler_path=sampler_path,
        origin_sampler_path=origin["sampler_path"],
        method=method,  # type: ignore[arg-type]
        checkpoint_stage="stage_a",
        training_step=stop,
        label=f"seed-{source.seed}-{method}-stage-a-monitor-step-{stop}",
        samples_per_item=1,
        max_tokens=inputs.acquisition.selected_max_tokens,
        seed_namespace="pilot0.a_monitor",
        output=output / "a-monitor",
    )
    result["monitor_generation_sha256"] = monitor["generation_sha256"]
    return write_segment(output, result, ledger=inputs.ledger)


async def _branch(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    origin: dict,
    *,
    method: str,
    output: Path,
    preflight_sha256: str,
) -> dict:
    pools = ordered_stage_a_pools(source)
    sources = stage_a_solver_sources(source) if method == "B-S" else {}
    previous = {
        "sampler_path": origin["sampler_path"],
        "state_path": origin["state_path"],
    }
    segments = {}
    grid = stage_a_grid(method)
    for start, stop in zip(grid[:-1], grid[1:], strict=True):
        segment = await _branch_segment(
            inputs,
            source,
            origin,
            previous,
            method=method,
            start=start,
            stop=stop,
            pools=pools,
            sources=sources,
            output=output / f"steps-{start}-{stop}",
            preflight_sha256=preflight_sha256,
        )
        segments[str(stop)] = segment
        previous = segment
    return {
        "kind": "stage-a-full-frozen-duration",
        "seed": source.seed,
        "method": method,
        "origin_sampler_path": origin["sampler_path"],
        "origin_state_path": origin["state_path"],
        "optimizer_inheritance": "weights_only_fresh_then_full_state_resume",
        "full_duration_updates": grid[-1],
        "segments": segments,
    }


async def run_stage_a_seed(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    output: Path,
    *,
    preflight_sha256: str,
) -> tuple[dict, dict]:
    m0 = {
        "sampler_path": inputs.m0_sampler_path,
        "state_path": inputs.m0_state_path,
    }
    bs = await _branch(
        inputs,
        source,
        m0,
        method="B-S",
        output=output / "B-S",
        preflight_sha256=preflight_sha256,
    )
    bg = await _branch(
        inputs,
        source,
        m0,
        method="B-G",
        output=output / "B-G",
        preflight_sha256=preflight_sha256,
    )
    return bs, bg


__all__ = ["run_stage_a_seed"]
