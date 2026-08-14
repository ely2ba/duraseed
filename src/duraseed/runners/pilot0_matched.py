"""Guarded post-Pilot matched-Stage-A selection and Stage-B follow-up."""

from __future__ import annotations

import json
from pathlib import Path

from duraseed.data.io import atomic_write_bytes
from duraseed.pilot0_contract import METHODS, PILOT_SEEDS, STAGE_B_GRID, Pilot0Inputs
from duraseed.pilot0_evidence import read_evaluation
from duraseed.pilot0_matched_budget import (
    calculate_matched_pilot_budget,
    validate_matched_authorization,
)
from duraseed.pilot0_matching import (
    build_candidate_plan,
    freeze_target,
    paired_matched_aggregate,
    summarize_matched_cell,
)
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.pilot0_reporting import evidence_index
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_matched_selection import run_matched_selection
from duraseed.runners.pilot0_remote import hydrate_ledger, read_segment
from duraseed.runners.pilot0_stage_b import run_stage_b


SCHEMA_VERSION = "duraseed-pilot0-matched-live-v1"


def _write_once(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RunnerGateError(f"matched follow-up resume changed {path.name}")
    else:
        atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(
            f"matched follow-up artifact is unreadable: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise RunnerGateError(
            f"matched follow-up artifact is not an object: {path.name}"
        )
    return value


def prepare_matched_followup(
    inputs: Pilot0Inputs,
    pilot_root: Path,
    output: Path,
    *,
    matched_run_id: str,
) -> dict:
    """Freeze target, candidate window, and exact pessimistic cap without access."""

    if not matched_run_id.strip() or Path(matched_run_id).name != matched_run_id:
        raise RunnerGateError("matched follow-up run ID is invalid")
    if not inputs.git_commit.strip():
        raise RunnerGateError("matched follow-up git identity is missing")
    if matched_run_id == pilot_root.name:
        raise RunnerGateError("matched follow-up requires a distinct run ID")
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "matched-target.json"
    if not target_path.exists() and any(output.iterdir()):
        raise RunnerGateError(
            "matched target must be frozen before follow-up artifacts"
        )
    tolerance = float(inputs.config.statistics.checkpoint_match_tolerance)
    target = freeze_target(pilot_root, tolerance=tolerance)
    if (
        target["source_bundle_sha256"] != inputs.source_authentication.bundle_sha256
        or target["resolved_config_hash"] != inputs.config.resolved_config_hash()
    ):
        raise RunnerGateError("matched follow-up inputs differ from completed Pilot 0")
    target_sha256 = _write_once(target_path, target)
    plan = build_candidate_plan(pilot_root, target)
    plan_sha256 = _write_once(output / "candidate-plan.json", plan)
    budget = calculate_matched_pilot_budget(inputs, plan)
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_exact_cap_authorization",
        "matched_run_id": matched_run_id,
        "pilot0_run_id": pilot_root.name,
        "pilot0_result_sha256": target["pilot0_result_sha256"],
        "pilot0_preflight_sha256": target["pilot0_preflight_sha256"],
        "git_commit": inputs.git_commit,
        "project_id": inputs.project_id,
        "source_bundle_sha256": inputs.source_authentication.bundle_sha256,
        "resolved_config_hash": inputs.config.resolved_config_hash(),
        "matched_target_sha256": target_sha256,
        "candidate_plan_sha256": plan_sha256,
        "candidate_checkpoint_count": budget.candidate_checkpoint_count,
        "selection_max_samples_per_item": int(
            inputs.config.statistics.checkpoint_selection_max_samples_per_item
        ),
        "matched_stage_b_cell_count": len(PILOT_SEEDS) * len(METHODS),
        "token_budget": {
            "prefill": budget.tokens.prefill,
            "sample": budget.tokens.sample,
            "train": budget.tokens.train,
        },
        "fixed_storage_usd": budget.fixed_storage_usd,
        "required_upper_bound_usd": budget.upper_bound_usd,
        "no_rerun_without_new_authorization": True,
    }
    _write_once(output / "preflight.json", preflight)
    return preflight


def _evaluation(path: Path) -> dict:
    value = read_evaluation(path)
    if value is None:
        raise RunnerGateError(f"matched Stage-B evidence is missing: {path}")
    return value


def _stage_b_evaluation(root: Path, step: int, name: str) -> dict:
    if step == 0:
        path = root / "step-0"
    else:
        previous = STAGE_B_GRID[STAGE_B_GRID.index(step) - 1]
        path = root / f"steps-{previous}-{step}"
    return _evaluation(path / name)


def _ledger(inputs: Pilot0Inputs) -> dict:
    committed, observed = inputs.ledger.committed, inputs.ledger.observed
    return {
        "committed_tokens": {
            "prefill": committed.prefill,
            "sample": committed.sample,
            "train": committed.train,
        },
        "observed_tokens": {
            "prefill": observed.prefill,
            "sample": observed.sample,
            "train": observed.train,
        },
        "committed_cost_usd": inputs.ledger.committed_cost_usd,
        "observed_cost_usd": inputs.ledger.observed_cost_usd,
    }


async def run_matched_followup(
    inputs: Pilot0Inputs,
    pilot_root: Path,
    output: Path,
    *,
    authorization: dict,
) -> dict:
    """Run only after the deterministic post-Pilot cap has been authorized."""

    preflight = prepare_matched_followup(
        inputs, pilot_root, output, matched_run_id=inputs.run_id
    )
    preflight_raw = (output / "preflight.json").read_bytes()
    authorization_sha256 = validate_matched_authorization(
        inputs, authorization, preflight_raw, preflight
    )
    if (
        _write_once(output / "authorization.json", authorization)
        != authorization_sha256
    ):
        raise RunnerGateError("matched authorization serialization changed")
    tokens = preflight["token_budget"]
    limits = inputs.ledger.limits
    if (limits.prefill, limits.sample, limits.train) != (
        tokens["prefill"],
        tokens["sample"],
        tokens["train"],
    ):
        raise RunnerGateError("matched follow-up ledger differs from its preflight")
    hydrate_ledger(inputs, output)
    target = _read_json(output / "matched-target.json")
    plan = _read_json(output / "candidate-plan.json")
    preflight_sha256 = sha256_bytes(preflight_raw)
    target_sha256 = sha256_bytes((output / "matched-target.json").read_bytes())
    selection, selection_sha256 = await run_matched_selection(
        inputs,
        pilot_root,
        output,
        plan=plan,
        target_artifact=target,
        preflight_sha256=preflight_sha256,
        target_sha256=target_sha256,
    )
    if selection["status"] != "selected":
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "matched_checkpoint_unavailable",
            "matched_target_sha256": target_sha256,
            "matched_selection_sha256": selection_sha256,
            "authorization_sha256": authorization_sha256,
            "ledger": _ledger(inputs),
        }
        _write_once(output / "result.json", result)
        return result
    source_by_seed = {source.seed: source for source in inputs.seed_sources}
    selected_by_cell = {
        (row["seed"], row["method"]): row["selected"] for row in selection["cells"]
    }
    sample_counts = {row["selected"]["samples_per_item"] for row in selection["cells"]}
    if len(sample_counts) != 1:
        raise RunnerGateError("matched selections changed their sampling budget")
    samples = sample_counts.pop()
    for seed in PILOT_SEEDS:
        origin = read_segment(pilot_root / f"seed-{seed}" / "boundary-origin", {})
        if origin is None:
            raise RunnerGateError("matched Stage B lost its shared origin")
        for method in METHODS:
            selected = selected_by_cell[seed, method]
            stage_a = {
                "kind": "stage-a-matched",
                "seed": seed,
                "method": method,
                "origin_sampler_path": origin["sampler_path"],
                "origin_state_path": origin["state_path"],
                "selected_sampler_path": selected["sampler_path"],
                "selected_state_path": selected["state_path"],
                "selected_step": selected["step"],
                "matched_target_sha256": target_sha256,
                "matched_selection_sha256": selection_sha256,
                "selected_evidence": {
                    "monitor_generation_sha256": selected["monitor_generation_sha256"],
                    "stage_a_validation_sha256": selected["combined_sha256"],
                },
            }
            await run_stage_b(
                inputs,
                source_by_seed[seed],
                stage_a,
                method=method,
                output=output / f"seed-{seed}" / method / "stage-b",
                preflight_sha256=preflight_sha256,
                a_validation_seed_namespace=(
                    inputs.config.statistics.checkpoint_selection_fresh_seed_namespace
                ),
                a_validation_samples_per_item=samples,
            )
    cells = []
    for seed in PILOT_SEEDS:
        for method in METHODS:
            selected = selected_by_cell[seed, method]
            combined = _read_json(
                output
                / "candidates"
                / f"seed-{seed}"
                / method
                / f"step-{selected['step']}"
                / "combined.json"
            )
            stage_b = output / f"seed-{seed}" / method / "stage-b"
            cells.append(
                summarize_matched_cell(
                    seed=seed,
                    method=method,
                    target=float(target["target"]),
                    stage_a=combined,
                    stage_b_maps=tuple(
                        _stage_b_evaluation(stage_b, step, "b-validation")
                        for step in STAGE_B_GRID
                    ),
                    stage_b_final_retention=_stage_b_evaluation(
                        stage_b, STAGE_B_GRID[-1], "a-validation"
                    ),
                )
            )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "evidence_collected",
        "matched_target_sha256": target_sha256,
        "matched_selection_sha256": selection_sha256,
        "authorization_sha256": authorization_sha256,
        "cells": cells,
        "paired_primary_contrast": paired_matched_aggregate(cells),
        "evidence_index_sha256": _write_once(
            output / "evidence-index.json", evidence_index(output)
        ),
        "ledger": _ledger(inputs),
    }
    _write_once(output / "result.json", result)
    return result


__all__ = ["prepare_matched_followup", "run_matched_followup"]
