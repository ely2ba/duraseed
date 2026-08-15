"""Shared boundary origin and checkpointed Stage-A branches for Pilot 0."""

from __future__ import annotations

from pathlib import Path

from duraseed.pilot0_contract import Pilot0Inputs, PilotSeedSources, STAGE_A_GRID
from duraseed.pilot0_data import ordered_stage_a_pools, stage_a_solver_sources
from duraseed.pilot0_integrity import segment_coordinates
from duraseed.runners.pilot0_origins import boundary_origin, m0_evidence
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
    completed = read_segment(output, expected)
    if completed is not None:
        return completed
    output.mkdir(parents=True, exist_ok=True)
    journal = RemoteJournal(output)
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
    pair = await save_pair(
        inputs,
        runtime,
        journal,
        name=f"{inputs.run_id}-seed-{source.seed}-{method}-step-{stop}",
        ttl_seconds=None if stop == STAGE_A_GRID[-1] else 7 * 24 * 60 * 60,
        coordinate=expected,
    )
    sampler = await sampler_for_path(
        inputs, journal, path=pair.sampler_path, coordinate=expected
    )
    monitor = await evaluate_manifest(
        inputs,
        source,
        manifest=source.prompt_pools.a_monitor_manifest,
        sampler=sampler,
        sampler_path=pair.sampler_path,
        origin_sampler_path=origin["sampler_path"],
        method=method,  # type: ignore[arg-type]
        checkpoint_stage="stage_a",
        training_step=stop,
        label=f"seed-{source.seed}-{method}-stage-a-monitor-step-{stop}",
        samples_per_item=int(inputs.config.stage_a.monitor_samples_per_item),
        max_tokens=inputs.acquisition.selected_max_tokens,
        seed_namespace="pilot0.a_monitor",
        output=output / "a-monitor",
    )
    result = {
        **expected,
        "sampler_path": pair.sampler_path,
        "state_path": pair.state_path,
        "learning_rate": learning_rate,
        "monitor_generation_sha256": monitor["generation_sha256"],
    }
    if stop == STAGE_A_GRID[-1]:
        validation = await evaluate_manifest(
            inputs,
            source,
            manifest=source.a_validation,
            sampler=sampler,
            sampler_path=pair.sampler_path,
            origin_sampler_path=origin["sampler_path"],
            method=method,  # type: ignore[arg-type]
            checkpoint_stage="stage_a",
            training_step=stop,
            label=f"seed-{source.seed}-{method}-fixed-budget-a-validation",
            samples_per_item=int(inputs.config.evaluation["pilot_samples_per_item"]),
            max_tokens=inputs.acquisition.selected_max_tokens,
            seed_namespace="pilot0.a_validation",
            output=output / "a-validation",
        )
        result["fixed_budget_a_validation_sha256"] = validation["generation_sha256"]
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
    for start, stop in zip(STAGE_A_GRID[:-1], STAGE_A_GRID[1:], strict=True):
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
        "kind": "stage-a-fixed-budget",
        "seed": source.seed,
        "method": method,
        "origin_sampler_path": origin["sampler_path"],
        "origin_state_path": origin["state_path"],
        "optimizer_inheritance": "weights_only_fresh_then_full_state_resume",
        "selected_sampler_path": previous["sampler_path"],
        "selected_state_path": previous["state_path"],
        "segments": segments,
        "matched_a_selection": "pending_post_pilot_target_freeze",
    }


async def run_stage_a_seed(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    output: Path,
    *,
    preflight_sha256: str,
) -> tuple[dict, dict, dict, dict]:
    m0 = await m0_evidence(
        inputs, source, output / "m0", preflight_sha256=preflight_sha256
    )
    origin = await boundary_origin(
        inputs,
        source,
        output / "boundary-origin",
        preflight_sha256=preflight_sha256,
    )
    bs = await _branch(
        inputs,
        source,
        origin,
        method="B-S",
        output=output / "B-S",
        preflight_sha256=preflight_sha256,
    )
    bg = await _branch(
        inputs,
        source,
        origin,
        method="B-G",
        output=output / "B-G",
        preflight_sha256=preflight_sha256,
    )
    return m0, origin, bs, bg


__all__ = ["run_stage_a_seed"]
