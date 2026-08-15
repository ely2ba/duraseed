"""Prospective matched-Stage-A target and checkpoint selection for Pilot 0."""

from __future__ import annotations

import json
from math import fsum, sqrt
from pathlib import Path
from typing import Iterable

from duraseed.evaluation.analysis import (
    BinomialObservation,
    equal_item_posterior_mean,
    normalized_stage_b_auc,
)
from duraseed.pilot0_contract import (
    METHODS,
    PILOT_SEEDS,
    STAGE_A_GRID,
    STAGE_B_GRID,
)
from duraseed.pilot0_evidence import read_evaluation
from duraseed.pilot0_analysis import monitor_retention_summary
from duraseed.provenance import canonical_json_hash, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_remote import read_segment


TARGET_SCHEMA = "duraseed-pilot0-matched-target-v1"
TARGET_RULE = "minimum_four_seed_method_fixed_budget_targeted_posterior_means"
PLAN_SCHEMA = "duraseed-pilot0-matched-plan-v1"
SELECTION_SCHEMA = "duraseed-pilot0-matched-selection-v1"


def _evaluation(path: Path) -> dict:
    result = read_evaluation(path)
    if result is None:
        raise RunnerGateError(f"matched-A source evaluation is missing: {path}")
    return result


def panel_posterior_mean(result: dict, role: str) -> float:
    """Return the equal-item Jeffreys posterior mean for one panel role."""

    try:
        observations = tuple(
            BinomialObservation(int(row["successes"]), int(row["trials"]))
            for row in result["item_counts"]
            if row["panel_role"] == role
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError("matched-A targeted counts are malformed") from error
    if not observations:
        raise RunnerGateError(f"matched-A evaluation has no {role} items")
    return equal_item_posterior_mean(observations)


def targeted_posterior_mean(result: dict) -> float:
    return panel_posterior_mean(result, "targeted")


def targeted_sampling_se(result: dict) -> float:
    """Plug-in sampling SE of the equally weighted targeted-panel mean."""

    try:
        rows = tuple(
            row for row in result["item_counts"] if row["panel_role"] == "targeted"
        )
        variances = tuple(
            (float(row["successes"]) / int(row["trials"]))
            * (1.0 - float(row["successes"]) / int(row["trials"]))
            / int(row["trials"])
            for row in rows
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise RunnerGateError("matched-A sampling-SE counts are malformed") from error
    if not rows:
        raise RunnerGateError("matched-A sampling SE has no targeted items")
    return sqrt(fsum(variances)) / len(rows)


def combine_evaluations(results: Iterable[dict]) -> dict:
    """Combine disjoint seeded draws from one unchanged validation population."""

    values = tuple(results)
    if not values:
        raise RunnerGateError("matched-A combination requires evaluation evidence")
    combined: dict[tuple[str, str], list[int]] = {}
    for result in values:
        try:
            rows = result["item_counts"]
        except (KeyError, TypeError) as error:
            raise RunnerGateError(
                "matched-A combination evidence is malformed"
            ) from error
        for row in rows:
            try:
                key = (str(row["task_id"]), str(row["panel_role"]))
                counts = (int(row["successes"]), int(row["trials"]))
            except (KeyError, TypeError, ValueError) as error:
                raise RunnerGateError("matched-A item counts are malformed") from error
            if counts[0] < 0 or counts[1] < counts[0] or counts[1] < 1:
                raise RunnerGateError("matched-A item counts are invalid")
            target = combined.setdefault(key, [0, 0])
            target[0] += counts[0]
            target[1] += counts[1]
    expected = {(row["task_id"], row["panel_role"]) for row in values[0]["item_counts"]}
    if set(combined) != expected or any(
        {(row["task_id"], row["panel_role"]) for row in value["item_counts"]}
        != expected
        for value in values
    ):
        raise RunnerGateError("matched-A draws changed the validation population")
    return {
        "samples_per_item": sum(int(value["samples_per_item"]) for value in values),
        "item_counts": [
            {
                "task_id": task_id,
                "panel_role": role,
                "successes": counts[0],
                "trials": counts[1],
            }
            for (task_id, role), counts in sorted(combined.items())
        ],
        "source_generation_sha256s": [value["generation_sha256"] for value in values],
    }


def _endpoint_path(root: Path, seed: int, method: str) -> Path:
    return root / f"seed-{seed}" / method / "steps-25-50" / "a-validation"


def freeze_target(pilot_root: Path, *, tolerance: float) -> dict:
    """Freeze the maximum-common-support target from four fixed endpoints."""

    try:
        pilot_raw = (pilot_root / "result.json").read_bytes()
        pilot_result = json.loads(pilot_raw)
        preflight_raw = (pilot_root / "preflight.json").read_bytes()
        preflight = json.loads(preflight_raw)
    except OSError as error:
        raise RunnerGateError("matched-A target requires completed Pilot 0") from error
    except json.JSONDecodeError as error:
        raise RunnerGateError("matched-A Pilot-0 result is unreadable") from error
    if (
        not isinstance(pilot_result, dict)
        or not isinstance(preflight, dict)
        or pilot_result.get("run_id") != pilot_root.name
        or pilot_result.get("status") != "evidence_collected"
        or pilot_result.get("evidence_status") != "complete"
        or pilot_result.get("primary_analysis_status") != "complete"
        or preflight.get("run_id") != pilot_root.name
        or tuple(preflight.get("methods", ())) != METHODS
        or tuple(preflight.get("seeds", ())) != PILOT_SEEDS
        or tuple(preflight.get("stage_a_grid", ())) != STAGE_A_GRID
    ):
        raise RunnerGateError("matched-A target requires complete Pilot-0 evidence")
    endpoints = []
    for seed in PILOT_SEEDS:
        for method in METHODS:
            path = _endpoint_path(pilot_root, seed, method)
            result = _evaluation(path)
            endpoints.append(
                {
                    "seed": seed,
                    "method": method,
                    "step": STAGE_A_GRID[-1],
                    "targeted_posterior_mean": targeted_posterior_mean(result),
                    "generation_sha256": result["generation_sha256"],
                    "reward_sha256": result["reward_sha256"],
                }
            )
    target = min(row["targeted_posterior_mean"] for row in endpoints)
    return {
        "schema_version": TARGET_SCHEMA,
        "status": "frozen_before_candidate_reevaluation",
        "pilot0_run_id": pilot_root.name,
        "pilot0_result_sha256": sha256_bytes(pilot_raw),
        "pilot0_preflight_sha256": sha256_bytes(preflight_raw),
        "source_bundle_sha256": preflight.get("lineage", {}).get(
            "source_bundle_sha256"
        ),
        "resolved_config_hash": preflight.get("lineage", {}).get(
            "resolved_config_hash"
        ),
        "metric": "equal_item_targeted_jeffreys_posterior_mean",
        "rule": TARGET_RULE,
        "target": target,
        "tolerance": tolerance,
        "endpoints": endpoints,
    }


def _monitor(root: Path, seed: int, method: str, step: int) -> tuple[dict, dict]:
    if step == 0:
        segment_path = root / f"seed-{seed}" / "boundary-origin"
        evaluation_path = segment_path / "a-monitor"
    else:
        previous = STAGE_A_GRID[STAGE_A_GRID.index(step) - 1]
        segment_path = root / f"seed-{seed}" / method / f"steps-{previous}-{step}"
        evaluation_path = segment_path / "a-monitor"
    segment = read_segment(segment_path, {})
    if segment is None:
        raise RunnerGateError("matched-A candidate checkpoint segment is missing")
    return segment, _evaluation(evaluation_path)


def build_candidate_plan(pilot_root: Path, target_artifact: dict) -> dict:
    """Locate the first apparent crossing and its real adjacent checkpoints."""

    target = float(target_artifact["target"])
    cells = []
    for seed in PILOT_SEEDS:
        for method in METHODS:
            endpoint = next(
                row
                for row in target_artifact["endpoints"]
                if row["seed"] == seed and row["method"] == method
            )
            target_defining = endpoint["targeted_posterior_mean"] == target
            checkpoints = []
            for step in STAGE_A_GRID:
                segment, evaluation = _monitor(pilot_root, seed, method, step)
                checkpoints.append(
                    {
                        "step": step,
                        "monitor_targeted_posterior_mean": targeted_posterior_mean(
                            evaluation
                        ),
                        "monitor_generation_sha256": evaluation["generation_sha256"],
                        "sampler_path": segment["sampler_path"],
                        "state_path": segment["state_path"],
                    }
                )
            crossing_index = (
                len(checkpoints) - 1
                if target_defining
                else next(
                    (
                        index
                        for index, row in enumerate(checkpoints)
                        if row["monitor_targeted_posterior_mean"] >= target
                    ),
                    len(checkpoints) - 1,
                )
            )
            lower = max(0, crossing_index - 1)
            upper = min(len(checkpoints), crossing_index + 2)
            cells.append(
                {
                    "seed": seed,
                    "method": method,
                    "crossing_step": checkpoints[crossing_index]["step"],
                    "crossing_source": (
                        "fixed_endpoint_full_validation"
                        if target_defining
                        or checkpoints[crossing_index][
                            "monitor_targeted_posterior_mean"
                        ]
                        < target
                        else "cheap_monitor"
                    ),
                    "target_defining_cell": target_defining,
                    "structurally_missing_predecessor": crossing_index == 0,
                    "structurally_missing_successor": crossing_index
                    == len(checkpoints) - 1,
                    "candidates": checkpoints[lower:upper],
                }
            )
    return {
        "schema_version": PLAN_SCHEMA,
        "status": "planned_before_candidate_reevaluation",
        "target_artifact_sha256": canonical_json_hash(target_artifact),
        "candidate_rule": "real_predecessor_crossing_successor_where_available",
        "cells": cells,
    }


def select_checkpoint(
    candidates: Iterable[dict], *, target: float, tolerance: float
) -> dict | None:
    """Select the closest in-band real checkpoint, breaking ties earlier."""

    eligible = tuple(
        row
        for row in candidates
        if abs(float(row["targeted_posterior_mean"]) - target) <= tolerance
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            abs(float(row["targeted_posterior_mean"]) - target),
            row["step"],
        ),
    )


def summarize_matched_cell(
    *,
    seed: int,
    method: str,
    target: float,
    stage_a: dict,
    stage_b_maps: tuple[dict, ...],
    stage_b_retention: tuple[dict, ...],
    stage_b_final_retention: dict,
) -> dict:
    """Reduce the matched-origin endpoints without reusing fixed-budget claims."""

    if len(stage_b_maps) != len(STAGE_B_GRID):
        raise RunnerGateError("matched Stage B is missing a MAPS checkpoint")
    maps_scores = tuple(panel_posterior_mean(row, "stage-b") for row in stage_b_maps)
    pre_target = targeted_posterior_mean(stage_a)
    post_target = targeted_posterior_mean(stage_b_final_retention)
    pre_sentinel = panel_posterior_mean(stage_a, "sentinel")
    post_sentinel = panel_posterior_mean(stage_b_final_retention, "sentinel")
    auc = normalized_stage_b_auc(STAGE_B_GRID, maps_scores)
    retention = monitor_retention_summary(stage_b_retention)
    return {
        "seed": seed,
        "method": method,
        "matched_target": target,
        "pre_b_targeted_score": pre_target,
        "pre_b_target_deviation": pre_target - target,
        "fixed_budget_targeted_retention": post_target,
        "fixed_budget_targeted_change": post_target - pre_target,
        "pre_b_sentinel_score": pre_sentinel,
        "fixed_budget_sentinel_retention": post_sentinel,
        "fixed_budget_sentinel_change": post_sentinel - pre_sentinel,
        "maps_scores": maps_scores,
        "maps_absolute_auc": auc.absolute_auc,
        "raw_gain_stage_b_auc": auc.raw_gain_auc,
        "headroom_normalized_gain_stage_b_auc": auc.headroom_normalized_gain_auc,
        **retention,
    }


def paired_matched_aggregate(cells: Iterable[dict]) -> dict:
    """Return paired B-S minus B-G matched-origin Pilot contrasts."""

    values = tuple(cells)
    by_cell = {(row["seed"], row["method"]): row for row in values}
    differences = []
    for seed in PILOT_SEEDS:
        if any((seed, method) not in by_cell for method in METHODS):
            raise RunnerGateError("matched aggregate requires all four cells")
        left, right = by_cell[seed, "B-S"], by_cell[seed, "B-G"]
        differences.append(
            {
                "seed": seed,
                "raw_gain_stage_b_auc": left["raw_gain_stage_b_auc"]
                - right["raw_gain_stage_b_auc"],
                "fixed_budget_targeted_retention": left[
                    "fixed_budget_targeted_retention"
                ]
                - right["fixed_budget_targeted_retention"],
                "targeted_monitor_retention_absolute_auc": left[
                    "targeted_monitor_retention_absolute_auc"
                ]
                - right["targeted_monitor_retention_absolute_auc"],
            }
        )
    return {
        "contrast": "B-S_minus_B-G",
        "paired_seed_differences": differences,
        "mean_raw_gain_stage_b_auc_difference": fsum(
            row["raw_gain_stage_b_auc"] for row in differences
        )
        / len(differences),
        "mean_fixed_budget_targeted_retention_difference": fsum(
            row["fixed_budget_targeted_retention"] for row in differences
        )
        / len(differences),
        "mean_targeted_monitor_retention_absolute_auc_difference": fsum(
            row["targeted_monitor_retention_absolute_auc"] for row in differences
        )
        / len(differences),
        "paired_seed_count": len(differences),
        "degrees_of_freedom": len(differences) - 1,
    }


__all__ = [
    "PLAN_SCHEMA",
    "SELECTION_SCHEMA",
    "TARGET_SCHEMA",
    "TARGET_RULE",
    "build_candidate_plan",
    "combine_evaluations",
    "freeze_target",
    "paired_matched_aggregate",
    "panel_posterior_mean",
    "select_checkpoint",
    "targeted_posterior_mean",
    "targeted_sampling_se",
    "summarize_matched_cell",
]
