"""Frozen post-hoc cadence matching and selected pre-B profiles."""

from __future__ import annotations

from pathlib import Path

from duraseed.data.io import atomic_write_bytes
from duraseed.pilot0_contract import Pilot0Inputs, PilotSeedSources, stage_a_grid
from duraseed.pilot0_evidence import read_evaluation
from duraseed.pilot0_pair_matching import select_paired_cadence
from duraseed.pilot0_profiles import pre_b_capability_profile
from duraseed.provenance import canonical_json_bytes, canonical_json_hash
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_remote import sampler_for_path
from duraseed.runners.pilot0_sampling import evaluate_manifest
from duraseed.runners.remote_journal import RemoteJournal


class PilotMatchingUnavailable(RunnerGateError):
    """The predeclared paired checkpoints have no overlapping interval."""


def _cadence_rows(root: Path, method: str, branch: dict) -> tuple[dict, ...]:
    rows = []
    grid = stage_a_grid(method)
    for start, stop in zip(grid[:-1], grid[1:], strict=True):
        if stop % 10:
            continue
        segment = branch["segments"].get(str(stop))
        result = read_evaluation(root / method / f"steps-{start}-{stop}" / "a-monitor")
        if not isinstance(segment, dict) or result is None:
            raise RunnerGateError("Pilot cadence evidence is incomplete")
        rows.append(
            {
                "checkpoint": {
                    "step": stop,
                    "sampler_path": segment["sampler_path"],
                    "state_path": segment["state_path"],
                },
                "evaluation": result,
            }
        )
    return tuple(rows)


async def select_and_profile(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    root: Path,
    bs: dict,
    bg: dict,
) -> tuple[dict, dict, tuple[dict, dict]]:
    matching = select_paired_cadence(
        _cadence_rows(root, "B-S", bs), _cadence_rows(root, "B-G", bg)
    )
    atomic_write_bytes(root / "matching.json", canonical_json_bytes(matching))
    if matching["status"] != "selected":
        raise PilotMatchingUnavailable(
            "Pilot pair has no overlap-matched Stage-A checkpoints"
        )
    selected_branches, profiles = {}, []
    cover = tuple(
        float(value) for value in inputs.config.evaluation["reliability_tau_report"]
    )
    for method in ("B-S", "B-G"):
        selected = matching[method]
        output = root / method / "selected-pre-b"
        output.mkdir(parents=True, exist_ok=True)
        journal = RemoteJournal(output)
        sampler = await sampler_for_path(
            inputs,
            journal,
            path=selected["sampler_path"],
            coordinate={
                "seed": source.seed,
                "method": method,
                "step": selected["step"],
            },
        )
        monitor = await evaluate_manifest(
            inputs,
            source,
            manifest=source.prompt_pools.a_monitor_manifest,
            sampler=sampler,
            sampler_path=selected["sampler_path"],
            origin_sampler_path=inputs.m0_sampler_path,
            method=method,  # type: ignore[arg-type]
            checkpoint_stage="stage_a",
            training_step=selected["step"],
            label=f"seed-{source.seed}-{method}-selected-pre-b-monitor",
            samples_per_item=4,
            max_tokens=inputs.acquisition.selected_max_tokens,
            seed_namespace="pilot0.a_monitor",
            output=output / "a-monitor",
        )
        validation = await evaluate_manifest(
            inputs,
            source,
            manifest=source.a_validation,
            sampler=sampler,
            sampler_path=selected["sampler_path"],
            origin_sampler_path=inputs.m0_sampler_path,
            method=method,  # type: ignore[arg-type]
            checkpoint_stage="stage_a",
            training_step=selected["step"],
            label=f"seed-{source.seed}-{method}-selected-pre-b",
            samples_per_item=16,
            max_tokens=inputs.acquisition.selected_max_tokens,
            seed_namespace="pilot0.a_validation.pre_b",
            output=output / "a-validation",
        )
        branch = {
            "kind": "stage-a-post-hoc-matched",
            "seed": source.seed,
            "method": method,
            "selected_step": selected["step"],
            "selected_sampler_path": selected["sampler_path"],
            "selected_state_path": selected["state_path"],
            "matched_selection_sha256": canonical_json_hash(matching),
            "optimizer_inheritance": "weights_only_fresh_for_stage_b",
            "selected_evidence": {
                "cadence_generation_sha256": selected["monitor_generation_sha256"],
                "monitor_generation_sha256": monitor["generation_sha256"],
                "stage_a_validation_sha256": validation["generation_sha256"],
            },
        }
        selected_branches[method] = branch
        profiles.append(
            pre_b_capability_profile(
                origin_kind="post_hoc_overlap_matched",
                seed=source.seed,
                method=method,
                manifest=source.a_validation,
                evaluation_directories=(output / "a-validation",),
                cover_thresholds=cover,
                expected_sampler_path=selected["sampler_path"],
            )
        )
    atomic_write_bytes(root / "pre-b-profiles.json", canonical_json_bytes(profiles))
    return selected_branches["B-S"], selected_branches["B-G"], tuple(profiles)  # type: ignore[return-value]


__all__ = ["PilotMatchingUnavailable", "select_and_profile"]
