"""Fresh full-validation selection of matched Stage-A Pilot checkpoints."""

from __future__ import annotations

from pathlib import Path

from duraseed.data.io import atomic_write_bytes
from duraseed.pilot0_contract import Pilot0Inputs
from duraseed.pilot0_evidence import read_evaluation
from duraseed.pilot0_matching import (
    SELECTION_SCHEMA,
    combine_evaluations,
    select_checkpoint,
    targeted_posterior_mean,
    targeted_sampling_se,
)
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_remote import (
    read_segment,
    sampler_for_path,
    write_segment,
)
from duraseed.runners.pilot0_sampling import evaluate_manifest
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.pilot0_integrity import segment_coordinates


def _write_once(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RunnerGateError(f"matched-A resume changed {path.name}")
    else:
        atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def _source(inputs: Pilot0Inputs, seed: int):  # type: ignore[no-untyped-def]
    return next(source for source in inputs.seed_sources if source.seed == seed)


def _origin(pilot_root: Path, seed: int) -> dict:
    value = read_segment(pilot_root / f"seed-{seed}" / "boundary-origin", {})
    if value is None:
        raise RunnerGateError("matched-A shared boundary origin is missing")
    return value


async def _evaluate_candidate(
    inputs: Pilot0Inputs,
    source: object,
    candidate: dict,
    *,
    method: str,
    origin_sampler_path: str,
    output: Path,
    preflight_sha256: str,
    target_sha256: str,
    samples: int,
    sample_index_start: int,
) -> dict:
    step = int(candidate["step"])
    expected = segment_coordinates(
        inputs,
        source,
        preflight_sha256,
        kind="stage-a-matched-selection",
        method=method,
        start=step,
        stop=step,
        sampler_path=candidate["sampler_path"],
        state_path=candidate["state_path"],
        origin_sampler_path=origin_sampler_path,
        matched_target_sha256=target_sha256,
        samples_per_item=samples,
        sample_index_start=sample_index_start,
    )
    completed = read_segment(output, expected)
    if completed is not None:
        result = read_evaluation(output / "a-validation")
        if result is None:
            raise RunnerGateError("matched-A segment omitted validation evidence")
        return result
    output.mkdir(parents=True, exist_ok=True)
    journal = RemoteJournal(output)
    sampler = await sampler_for_path(
        inputs,
        journal,
        path=candidate["sampler_path"],
        coordinate={"seed": source.seed, "method": method, "step": step},
    )
    result = await evaluate_manifest(
        inputs,
        source,
        manifest=source.a_validation,
        sampler=sampler,
        sampler_path=candidate["sampler_path"],
        origin_sampler_path=origin_sampler_path,
        method=method,  # type: ignore[arg-type]
        checkpoint_stage="stage_a",
        training_step=step,
        label=f"seed-{source.seed}-{method}-matched-a-selection-step-{step}",
        samples_per_item=samples,
        sample_index_start=sample_index_start,
        max_tokens=inputs.acquisition.selected_max_tokens,
        seed_namespace=(
            inputs.config.statistics.checkpoint_selection_fresh_seed_namespace
        ),
        output=output / "a-validation",
    )
    write_segment(
        output,
        {
            **expected,
            "generation_sha256": result["generation_sha256"],
            "reward_sha256": result["reward_sha256"],
        },
        ledger=inputs.ledger,
    )
    return result


async def run_matched_selection(
    inputs: Pilot0Inputs,
    pilot_root: Path,
    output: Path,
    *,
    plan: dict,
    target_artifact: dict,
    preflight_sha256: str,
    target_sha256: str,
) -> tuple[dict, str]:
    """Evaluate candidates and freeze one real in-band checkpoint per cell."""

    initial_samples = int(
        inputs.config.statistics.checkpoint_selection_initial_samples_per_item
    )
    maximum_samples = int(
        inputs.config.statistics.checkpoint_selection_max_samples_per_item
    )
    maximum_se = float(inputs.config.statistics.checkpoint_selection_max_panel_mean_se)
    initial: dict[tuple[int, str, int], dict] = {}
    for cell in plan["cells"]:
        seed, method = int(cell["seed"]), str(cell["method"])
        source = _source(inputs, seed)
        origin = _origin(pilot_root, seed)
        for candidate in cell["candidates"]:
            step = int(candidate["step"])
            initial[seed, method, step] = await _evaluate_candidate(
                inputs,
                source,
                candidate,
                method=method,
                origin_sampler_path=origin["sampler_path"],
                output=output
                / "candidates"
                / f"seed-{seed}"
                / method
                / f"step-{step}"
                / "initial",
                preflight_sha256=preflight_sha256,
                target_sha256=target_sha256,
                samples=initial_samples,
                sample_index_start=0,
            )
    extend_all = any(
        targeted_sampling_se(result) > maximum_se for result in initial.values()
    )
    extension: dict[tuple[int, str, int], dict] = {}
    if extend_all:
        extra = maximum_samples - initial_samples
        if extra < 1:
            raise RunnerGateError("matched-A SE gate has no extension budget")
        for cell in plan["cells"]:
            seed, method = int(cell["seed"]), str(cell["method"])
            source = _source(inputs, seed)
            origin = _origin(pilot_root, seed)
            for candidate in cell["candidates"]:
                step = int(candidate["step"])
                extension[seed, method, step] = await _evaluate_candidate(
                    inputs,
                    source,
                    candidate,
                    method=method,
                    origin_sampler_path=origin["sampler_path"],
                    output=output
                    / "candidates"
                    / f"seed-{seed}"
                    / method
                    / f"step-{step}"
                    / "extension",
                    preflight_sha256=preflight_sha256,
                    target_sha256=target_sha256,
                    samples=extra,
                    sample_index_start=initial_samples,
                )
    selections = []
    for cell in plan["cells"]:
        seed, method = int(cell["seed"]), str(cell["method"])
        candidates = []
        for candidate in cell["candidates"]:
            step = int(candidate["step"])
            evidence = [initial[seed, method, step]]
            if extend_all:
                evidence.append(extension[seed, method, step])
            combined = combine_evaluations(evidence)
            combined.update({"seed": seed, "method": method, "step": step})
            combined_path = (
                output
                / "candidates"
                / f"seed-{seed}"
                / method
                / f"step-{step}"
                / "combined.json"
            )
            combined_sha256 = _write_once(combined_path, combined)
            candidates.append(
                {
                    **candidate,
                    "targeted_posterior_mean": targeted_posterior_mean(combined),
                    "samples_per_item": combined["samples_per_item"],
                    "initial_sampling_se": targeted_sampling_se(
                        initial[seed, method, step]
                    ),
                    "combined_sha256": combined_sha256,
                }
            )
        selected = select_checkpoint(
            candidates,
            target=float(target_artifact["target"]),
            tolerance=float(target_artifact["tolerance"]),
        )
        selections.append(
            {
                "seed": seed,
                "method": method,
                "candidates": candidates,
                "selected": selected,
            }
        )
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "status": (
            "selected"
            if all(row["selected"] is not None for row in selections)
            else "unavailable"
        ),
        "target_artifact_sha256": target_sha256,
        "global_extension_to_32": extend_all,
        "cells": selections,
    }
    selection_sha256 = _write_once(output / "selection.json", selection)
    return selection, selection_sha256


__all__ = ["run_matched_selection"]
