"""Shared M0 evidence and boundary-teacher origins for Pilot 0."""

from __future__ import annotations

from pathlib import Path

from duraseed.pilot0_contract import Pilot0Inputs, PilotSeedSources
from duraseed.pilot0_data import boundary_teacher_sources
from duraseed.pilot0_integrity import segment_coordinates
from duraseed.run_records import append_jsonl
from duraseed.runners.pilot0_remote import (
    read_segment,
    restore_runtime,
    sampler_for_path,
    save_pair,
    write_segment,
)
from duraseed.runners.pilot0_sampling import evaluate_manifest
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import apply_update, sft_datum


async def m0_evidence(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    output: Path,
    *,
    preflight_sha256: str,
) -> dict:
    expected = segment_coordinates(
        inputs,
        source,
        preflight_sha256,
        kind="m0-evidence",
        method=None,
        start=0,
        stop=0,
        origin_sampler_path=inputs.m0_sampler_path,
        origin_state_path=inputs.m0_state_path,
    )
    completed = read_segment(output, expected)
    if completed is not None:
        return completed
    output.mkdir(parents=True, exist_ok=True)
    journal = RemoteJournal(output)
    sampler = await sampler_for_path(
        inputs, journal, path=inputs.m0_sampler_path, coordinate=expected
    )
    evaluations = {}
    for label, manifest, samples in (
        (
            "monitor",
            source.prompt_pools.a_monitor_manifest,
            int(inputs.config.stage_a.monitor_samples_per_item),
        ),
        (
            "validation",
            source.a_validation,
            int(inputs.config.evaluation["pilot_samples_per_item"]),
        ),
    ):
        evaluations[label] = await evaluate_manifest(
            inputs,
            source,
            manifest=manifest,
            sampler=sampler,
            sampler_path=inputs.m0_sampler_path,
            origin_sampler_path=inputs.m0_sampler_path,
            method=None,
            checkpoint_stage="m0",
            training_step=0,
            label=f"seed-{source.seed}-m0-a-{label}",
            samples_per_item=samples,
            max_tokens=inputs.acquisition.selected_max_tokens,
            seed_namespace=f"pilot0.a_{label}",
            output=output / f"a-{label}",
        )
    return write_segment(
        output,
        {
            **expected,
            "sampler_path": inputs.m0_sampler_path,
            "monitor_sha256": evaluations["monitor"]["generation_sha256"],
            "a_validation_sha256": evaluations["validation"]["generation_sha256"],
        },
        ledger=inputs.ledger,
    )


async def boundary_origin(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    output: Path,
    *,
    preflight_sha256: str,
) -> dict:
    expected = segment_coordinates(
        inputs,
        source,
        preflight_sha256,
        kind="boundary-origin",
        method=None,
        start=0,
        stop=inputs.config.teacher_dose.calibration_updates,
        origin_sampler_path=inputs.m0_sampler_path,
        origin_state_path=inputs.m0_state_path,
        teacher_recipe_artifact_sha256=inputs.teacher_recipe_artifact_sha256,
    )
    completed = read_segment(output, expected)
    if completed is not None:
        return completed
    output.mkdir(parents=True, exist_ok=True)
    journal = RemoteJournal(output)
    runtime = await restore_runtime(
        inputs,
        journal,
        path=inputs.m0_state_path,
        full_state=False,
        coordinate=expected,
    )
    datums = [
        sft_datum(runtime, row) for row in boundary_teacher_sources(inputs, source)
    ]
    learning_rate = inputs.teacher_recipe.selected_learning_rate
    updates = inputs.config.teacher_dose.calibration_updates
    for step in range(1, updates + 1):
        batch = [
            datums[((step - 1) * 32 + offset) % len(datums)] for offset in range(32)
        ]
        journal.begin(
            "pilot0-boundary-seed-update",
            {"seed": source.seed, "step": step},
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
            learning_rate=learning_rate,
            ledger=inputs.ledger,
        )
        append_jsonl(
            output / "metrics.jsonl",
            {"phase": "boundary_seed", "step": step, "metrics": values},
        )
        journal.complete({"operation": "pilot0-boundary-seed-update", "step": step})
    pair = await save_pair(
        inputs,
        runtime,
        journal,
        name=f"{inputs.run_id}-seed-{source.seed}-boundary-origin",
        ttl_seconds=None,
        coordinate={**expected, "step": updates},
    )
    sampler = await sampler_for_path(
        inputs, journal, path=pair.sampler_path, coordinate=expected
    )
    evaluations = {}
    for label, manifest, samples, namespace in (
        (
            "monitor",
            source.prompt_pools.a_monitor_manifest,
            int(inputs.config.stage_a.monitor_samples_per_item),
            "pilot0.a_monitor",
        ),
        (
            "validation",
            source.a_validation,
            int(inputs.config.evaluation["pilot_samples_per_item"]),
            "pilot0.a_validation",
        ),
    ):
        evaluations[label] = await evaluate_manifest(
            inputs,
            source,
            manifest=manifest,
            sampler=sampler,
            sampler_path=pair.sampler_path,
            origin_sampler_path=inputs.m0_sampler_path,
            method=None,
            checkpoint_stage="stage_a",
            training_step=0,
            label=f"seed-{source.seed}-shared-boundary-origin-a-{label}",
            samples_per_item=samples,
            max_tokens=inputs.acquisition.selected_max_tokens,
            seed_namespace=namespace,
            output=output / f"a-{label}",
        )
    return write_segment(
        output,
        {
            **expected,
            "sampler_path": pair.sampler_path,
            "state_path": pair.state_path,
            "optimizer_inheritance": "weights_only_fresh_from_m0",
            "teacher_dose": inputs.teacher_recipe.decision.selected_dose,
            "teacher_learning_rate": learning_rate,
            "updates": updates,
            "monitor_sha256": evaluations["monitor"]["generation_sha256"],
            "a_validation_sha256": evaluations["validation"]["generation_sha256"],
        },
        ledger=inputs.ledger,
    )


__all__ = ["boundary_origin", "m0_evidence"]
