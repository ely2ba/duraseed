"""Fail-closed two-seed fixed-budget B-S/B-G Pilot-0 orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from duraseed.data.io import atomic_write_bytes
from duraseed.pilot0_analysis import (
    Pilot0MethodSummary,
    paired_primary_aggregate,
    summarize_method,
)
from duraseed.pilot0_budget import calculate_pilot0_budget
from duraseed.pilot0_contract import (
    METHODS,
    STAGE_A_GRID,
    STAGE_B_GRID,
    Pilot0Inputs,
    validate_pilot0_inputs,
)
from duraseed.pilot0_evidence import read_evaluation
from duraseed.pilot0_reporting import evidence_index, reward_group_health, usage_summary
from duraseed.provenance import canonical_json_bytes, canonical_json_value, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_remote import hydrate_ledger
from duraseed.runners.pilot0_stage_a import run_stage_a_seed
from duraseed.runners.pilot0_stage_b import run_stage_b


SCHEMA_VERSION = "duraseed-pilot0-live-v1"


def _write(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def _evaluation(path: Path) -> dict:
    value = read_evaluation(path.parent)
    if value is None:
        raise RunnerGateError(f"Pilot-0 evaluation is missing: {path.parent.name}")
    return value


def _preflight(inputs: Pilot0Inputs, root: Path) -> dict:
    budget = calculate_pilot0_budget(inputs)
    artifact = canonical_json_value(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_label": "pilot-0",
            "run_id": inputs.run_id,
            "authorized_usd": budget.authorized_usd,
            "upper_bound_usd": budget.upper_bound_usd,
            "fixed_storage_usd": budget.fixed_storage_usd,
            "ephemeral_sampler_count": len(inputs.seed_sources) * STAGE_A_GRID[-1],
            "rerun_reservation_usd": budget.rerun_reservation_usd,
            "rerun_policy": budget.rerun_policy,
            "token_budget": budget.tokens,
            "budget_gate_passed": budget.passed,
            "methods": list(METHODS),
            "seeds": [source.seed for source in inputs.seed_sources],
            "stage_a_grid": list(STAGE_A_GRID),
            "stage_b_grid": list(STAGE_B_GRID),
            "sealed_test_access": False,
            "matched_a_selection": "pending_post_pilot_target_freeze",
            "lineage": {
                "git_commit": inputs.git_commit,
                "project_id": inputs.project_id,
                "resolved_config_hash": inputs.config.resolved_config_hash(),
                "source_bundle_sha256": (inputs.source_authentication.bundle_sha256),
                "launch_authorization_sha256": (
                    inputs.source_authentication.authorization_sha256
                ),
                "post_calibration_billing_sha256": (
                    inputs.source_authentication.bundle.post_calibration_billing_sha256
                ),
                "sealed_b_test_envelope_sha256": (
                    inputs.source_authentication.sealed_envelope_sha256
                ),
                "m0_sampler_path": inputs.m0_sampler_path,
                "m0_state_path": inputs.m0_state_path,
                "panel_artifact_sha256": inputs.panel_artifact_sha256,
                "teacher_recipe_artifact_sha256": inputs.teacher_recipe_artifact_sha256,
                "acquisition_artifact_sha256": inputs.acquisition_artifact_sha256,
                "stage_b_recipe_artifact_sha256": inputs.stage_b_recipe_artifact_sha256,
                "manifest_ids": {
                    str(source.seed): {
                        "a_rl_train": source.prompt_pools.a_rl_train_manifest.manifest_id,
                        "a_monitor": source.prompt_pools.a_monitor_manifest.manifest_id,
                        "a_seed_train": source.teacher_train.manifest_id,
                        "a_validation": source.a_validation.manifest_id,
                        "b_train": source.b_train.manifest_id,
                        "b_validation": source.b_validation.manifest_id,
                    }
                    for source in inputs.seed_sources
                },
            },
        }
    )
    path = root / "preflight.json"
    if path.exists():
        if path.read_bytes() != canonical_json_bytes(artifact):
            raise RunnerGateError("Pilot-0 resume changed its frozen preflight")
    else:
        _write(path, artifact)
    return artifact


def _state(root: Path, status: str, **values: object) -> None:
    _write(
        root / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            **values,
        },
    )


def _ledger(inputs: Pilot0Inputs) -> dict:
    return {
        "committed_tokens": canonical_json_value(inputs.ledger.committed),
        "observed_tokens": canonical_json_value(inputs.ledger.observed),
        "committed_cost_usd": inputs.ledger.committed_cost_usd,
        "observed_cost_usd": inputs.ledger.observed_cost_usd,
    }


def _stage_b_result(root: Path, step: int) -> dict:
    if step == 0:
        return _evaluation(root / "step-0" / "b-validation" / "result.json")
    prior = STAGE_B_GRID[STAGE_B_GRID.index(step) - 1]
    return _evaluation(root / f"steps-{prior}-{step}" / "b-validation" / "result.json")


def _stage_b_retention(root: Path, step: int) -> dict:
    if step == 0:
        return _evaluation(root.parent / "steps-25-50" / "a-monitor" / "result.json")
    prior = STAGE_B_GRID[STAGE_B_GRID.index(step) - 1]
    return _evaluation(root / f"steps-{prior}-{step}" / "a-retention" / "result.json")


def _summary(
    root: Path,
    source_seed: int,
    method: str,
    cover_thresholds: tuple[float, ...],
) -> Pilot0MethodSummary:
    seed_root = root / f"seed-{source_seed}"
    stage_a = seed_root / method / "steps-25-50" / "a-validation" / "result.json"
    stage_b = seed_root / method / "stage-b"
    return summarize_method(
        seed=source_seed,
        method=method,
        m0_validation=_evaluation(seed_root / "m0" / "a-validation" / "result.json"),
        m0_monitor=_evaluation(seed_root / "m0" / "a-monitor" / "result.json"),
        stage_a_final=_evaluation(stage_a),
        stage_b_maps=tuple(_stage_b_result(stage_b, step) for step in STAGE_B_GRID),
        stage_b_retention=tuple(
            _stage_b_retention(stage_b, step) for step in STAGE_B_GRID
        ),
        stage_b_final_retention=_evaluation(
            stage_b / "steps-320-480" / "a-validation" / "result.json"
        ),
        cover_thresholds=cover_thresholds,
    )


async def run_pilot0(inputs: Pilot0Inputs) -> dict:
    """Run only after the complete pessimistic `$600` preflight passes."""

    validate_pilot0_inputs(inputs)
    root = inputs.output_root / inputs.run_id
    root.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(inputs, root)
    if preflight["budget_gate_passed"] is not True:
        _state(
            root,
            "blocked_budget",
            authorized_usd=preflight["authorized_usd"],
            upper_bound_usd=preflight["upper_bound_usd"],
        )
        raise RunnerGateError(
            "Pilot-0 pessimistic upper bound exceeds the exact $600 authorization"
        )
    token_budget = preflight["token_budget"]
    if (
        inputs.ledger.limits.prefill != token_budget["prefill"]
        or inputs.ledger.limits.sample != token_budget["sample"]
        or inputs.ledger.limits.train != token_budget["train"]
    ):
        raise RunnerGateError("Pilot-0 ledger must equal the full-path preflight")
    preflight_sha256 = sha256_bytes((root / "preflight.json").read_bytes())
    hydrate_ledger(inputs, root)
    _state(root, "running", completed_cells=[], ledger=_ledger(inputs))
    completed: list[str] = []
    try:
        for source in sorted(inputs.seed_sources, key=lambda value: value.seed):
            seed_root = root / f"seed-{source.seed}"
            _, _, bs, bg = await run_stage_a_seed(
                inputs,
                source,
                seed_root,
                preflight_sha256=preflight_sha256,
            )
            for method, stage_a in (("B-S", bs), ("B-G", bg)):
                await run_stage_b(
                    inputs,
                    source,
                    stage_a,
                    method=method,
                    output=seed_root / method / "stage-b",
                    preflight_sha256=preflight_sha256,
                )
                completed.append(f"seed-{source.seed}:{method}")
                _state(
                    root, "running", completed_cells=completed, ledger=_ledger(inputs)
                )
        summaries = tuple(
            _summary(
                root,
                source.seed,
                method,
                tuple(
                    float(value)
                    for value in inputs.config.evaluation["reliability_tau_report"]
                ),
            )
            for source in inputs.seed_sources
            for method in METHODS
        )
        budget = calculate_pilot0_budget(inputs)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "evidence_collected",
            "evidence_status": "complete",
            "primary_analysis_status": "complete",
            "analysis_status": "matched_a_sensitivity_pending",
            "run_id": inputs.run_id,
            "cells": canonical_json_value(summaries),
            "paired_primary_contrast": canonical_json_value(
                paired_primary_aggregate(summaries)
            ),
            "primary_future_learning_endpoint": "raw_gain_stage_b_auc",
            "primary_stability_endpoint": "fixed_budget_targeted_retention",
            "matched_a_selection": "pending_post_pilot_target_freeze",
            "reward_group_health": reward_group_health(
                root, tuple(source.seed for source in inputs.seed_sources)
            ),
            "token_and_cost_summary": usage_summary(inputs.ledger, budget),
            "sealed_test_access": False,
            "preflight_sha256": preflight_sha256,
            "evidence_index_sha256": _write(
                root / "evidence-index.json", evidence_index(root)
            ),
            "ledger": _ledger(inputs),
        }
        result_hash = _write(root / "result.json", result)
        _state(
            root,
            "evidence_collected",
            completed_cells=completed,
            result_sha256=result_hash,
            ledger=_ledger(inputs),
        )
        return result
    except BaseException as error:
        _state(
            root,
            "interrupted",
            completed_cells=completed,
            error=f"{type(error).__name__}: {error}",
            ledger=_ledger(inputs),
        )
        raise


__all__ = ["run_pilot0"]
