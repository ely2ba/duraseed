"""Checkpointed MAPS supervised probe and retained-TCES frontier for Pilot 0."""

from __future__ import annotations

from pathlib import Path

from duraseed.pilot0_contract import (
    Pilot0Inputs,
    PilotSeedSources,
    STAGE_B_GRID,
    STAGE_B_LEARNING_RATE,
)
from duraseed.pilot0_data import stage_b_sources
from duraseed.pilot0_integrity import stage_b_segment_coordinates
from duraseed.run_records import TrainingMetricRecord, append_jsonl
from duraseed.runners.pilot0_remote import (
    read_segment,
    restore_runtime,
    save_pair,
    write_segment,
)
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners.pilot0_stage_b_evaluation import evaluate_stage_b_step
from duraseed.runtime import apply_update, sft_datum


async def _stage_zero(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    stage_a: dict,
    *,
    method: str,
    output: Path,
    preflight_sha256: str,
    a_validation_seed_namespace: str,
    a_validation_samples_per_item: int | None,
) -> dict:
    parent = {
        "sampler_path": stage_a["selected_sampler_path"],
        "state_path": stage_a["selected_state_path"],
    }
    expected = stage_b_segment_coordinates(
        inputs,
        source,
        preflight_sha256,
        stage_a,
        parent,
        method=method,
        start=0,
        stop=0,
        learning_rate=STAGE_B_LEARNING_RATE,
    )
    completed = read_segment(output, expected)
    if completed is not None:
        return completed
    output.mkdir(parents=True, exist_ok=True)
    journal = RemoteJournal(output)
    evidence = await evaluate_stage_b_step(
        inputs,
        source,
        stage_a,
        method=method,
        step=0,
        sampler_path=stage_a["selected_sampler_path"],
        journal=journal,
        output=output,
        a_validation_seed_namespace=a_validation_seed_namespace,
        a_validation_samples_per_item=a_validation_samples_per_item,
    )
    return write_segment(
        output,
        {
            **expected,
            "sampler_path": stage_a["selected_sampler_path"],
            "state_path": stage_a["selected_state_path"],
            **evidence,
        },
        ledger=inputs.ledger,
    )


async def _train_segment(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    stage_a: dict,
    previous: dict,
    *,
    method: str,
    start: int,
    stop: int,
    datums: list,
    output: Path,
    preflight_sha256: str,
    a_validation_seed_namespace: str,
    a_validation_samples_per_item: int | None,
) -> dict:
    expected = stage_b_segment_coordinates(
        inputs,
        source,
        preflight_sha256,
        stage_a,
        previous,
        method=method,
        start=start,
        stop=stop,
        learning_rate=STAGE_B_LEARNING_RATE,
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
        batch = [
            datums[((step - 1) * 32 + offset) % len(datums)] for offset in range(32)
        ]
        journal.begin(
            "pilot0-stage-b-sft-update",
            {"seed": source.seed, "method": method, "step": step},
            {
                "prefill_tokens": 0,
                "sample_tokens": 0,
                "train_tokens": sum(int(row.model_input.length) for row in batch),
            },
        )
        values = await apply_update(
            runtime,
            batch,
            loss_fn="cross_entropy",
            learning_rate=STAGE_B_LEARNING_RATE,
            ledger=inputs.ledger,
        )
        metric = TrainingMetricRecord(
            phase="stage_b", training_step=step, metrics=values
        )
        append_jsonl(
            output / "metrics.jsonl",
            {**metric.model_dump(mode="json"), "method": method},
        )
        journal.complete({"operation": "pilot0-stage-b-sft-update", "step": step})
    pair = await save_pair(
        inputs,
        runtime,
        journal,
        name=f"{inputs.run_id}-seed-{source.seed}-{method}-stage-b-step-{stop}",
        ttl_seconds=None if stop == STAGE_B_GRID[-1] else 7 * 24 * 60 * 60,
        coordinate=expected,
    )
    evidence = await evaluate_stage_b_step(
        inputs,
        source,
        stage_a,
        method=method,
        step=stop,
        sampler_path=pair.sampler_path,
        journal=journal,
        output=output,
        a_validation_seed_namespace=a_validation_seed_namespace,
        a_validation_samples_per_item=a_validation_samples_per_item,
    )
    return write_segment(
        output,
        {
            **expected,
            "sampler_path": pair.sampler_path,
            "state_path": pair.state_path,
            **evidence,
        },
        ledger=inputs.ledger,
    )


async def run_stage_b(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    stage_a: dict,
    *,
    method: str,
    output: Path,
    preflight_sha256: str,
    a_validation_seed_namespace: str = "pilot0.a_validation",
    a_validation_samples_per_item: int | None = None,
) -> dict:
    datums = [sft_datum(inputs.runtime, row) for row in stage_b_sources(source)]
    segments = {
        "0": await _stage_zero(
            inputs,
            source,
            stage_a,
            method=method,
            output=output / "step-0",
            preflight_sha256=preflight_sha256,
            a_validation_seed_namespace=a_validation_seed_namespace,
            a_validation_samples_per_item=a_validation_samples_per_item,
        )
    }
    previous = segments["0"]
    for start, stop in zip(STAGE_B_GRID, STAGE_B_GRID[1:], strict=True):
        segment = await _train_segment(
            inputs,
            source,
            stage_a,
            previous,
            method=method,
            start=start,
            stop=stop,
            datums=datums,
            output=output / f"steps-{start}-{stop}",
            preflight_sha256=preflight_sha256,
            a_validation_seed_namespace=a_validation_seed_namespace,
            a_validation_samples_per_item=a_validation_samples_per_item,
        )
        segments[str(stop)] = segment
        previous = segment
    return {
        "kind": "stage-b-fixed-probe",
        "seed": source.seed,
        "method": method,
        "origin_stage_a_sampler_path": stage_a["selected_sampler_path"],
        "origin_stage_a_state_path": stage_a["selected_state_path"],
        "optimizer_inheritance": "weights_only_fresh_then_full_state_resume",
        "profile": inputs.config.stage_b.selected_profile,
        "learning_rate": STAGE_B_LEARNING_RATE,
        "selected_max_updates": STAGE_B_GRID[-1],
        "evaluation_grid": list(STAGE_B_GRID),
        "segments": segments,
        "matched_a_selection": "pending_post_pilot_target_freeze",
    }


__all__ = ["run_stage_b"]
