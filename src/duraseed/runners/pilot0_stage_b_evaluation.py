"""MAPS and retained-TCES evaluations at one Pilot-0 Stage-B checkpoint."""

from __future__ import annotations

from pathlib import Path

from duraseed.pilot0_contract import (
    Pilot0Inputs,
    PilotSeedSources,
    STAGE_B_GRID,
    STAGE_B_MAX_TOKENS,
)
from duraseed.runners.pilot0_remote import sampler_for_path
from duraseed.runners.pilot0_sampling import evaluate_manifest
from duraseed.runners.remote_journal import RemoteJournal


async def evaluate_stage_b_step(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    stage_a: dict,
    *,
    method: str,
    step: int,
    sampler_path: str,
    journal: RemoteJournal,
    output: Path,
) -> dict:
    sampler = await sampler_for_path(
        inputs,
        journal,
        path=sampler_path,
        coordinate={"seed": source.seed, "method": method, "step": step},
    )
    maps = await evaluate_manifest(
        inputs,
        source,
        manifest=source.b_validation,
        sampler=sampler,
        sampler_path=sampler_path,
        origin_sampler_path=stage_a["selected_sampler_path"],
        method=method,  # type: ignore[arg-type]
        checkpoint_stage="stage_b",
        training_step=step,
        label=f"seed-{source.seed}-{method}-stage-b-maps-step-{step}",
        samples_per_item=int(inputs.config.evaluation["pilot_samples_per_item"]),
        max_tokens=STAGE_B_MAX_TOKENS,
        seed_namespace="pilot0.stage_b.maps.validation",
        output=output / "b-validation",
    )
    if step == 0:
        stage_a_final = stage_a["segments"]["50"]
        return {
            "maps_generation_sha256": maps["generation_sha256"],
            "retention_generation_sha256": stage_a_final["monitor_generation_sha256"],
            "fixed_budget_a_validation_sha256": stage_a_final[
                "fixed_budget_a_validation_sha256"
            ],
            "retention_manifest_id": source.prompt_pools.a_monitor_manifest.manifest_id,
        }
    retention = await evaluate_manifest(
        inputs,
        source,
        manifest=source.prompt_pools.a_monitor_manifest,
        sampler=sampler,
        sampler_path=sampler_path,
        origin_sampler_path=stage_a["selected_sampler_path"],
        method=method,  # type: ignore[arg-type]
        checkpoint_stage="stage_b",
        training_step=step,
        label=f"seed-{source.seed}-{method}-stage-b-retention-step-{step}",
        samples_per_item=int(inputs.config.stage_a.monitor_samples_per_item),
        max_tokens=inputs.acquisition.selected_max_tokens,
        seed_namespace="pilot0.a_monitor",
        output=output / "a-retention",
    )
    result = {
        "maps_generation_sha256": maps["generation_sha256"],
        "retention_generation_sha256": retention["generation_sha256"],
        "retention_manifest_id": retention["manifest_id"],
    }
    if step == STAGE_B_GRID[-1]:
        validation = await evaluate_manifest(
            inputs,
            source,
            manifest=source.a_validation,
            sampler=sampler,
            sampler_path=sampler_path,
            origin_sampler_path=stage_a["selected_sampler_path"],
            method=method,  # type: ignore[arg-type]
            checkpoint_stage="stage_b",
            training_step=step,
            label=f"seed-{source.seed}-{method}-stage-b-fixed-budget-a-validation",
            samples_per_item=int(inputs.config.evaluation["pilot_samples_per_item"]),
            max_tokens=inputs.acquisition.selected_max_tokens,
            seed_namespace="pilot0.a_validation",
            output=output / "a-validation",
        )
        result["fixed_budget_a_validation_sha256"] = validation["generation_sha256"]
    return result


__all__ = ["evaluate_stage_b_step"]
