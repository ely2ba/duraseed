"""One separately authorized paired-seed Pilot-0 orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from duraseed.data.io import atomic_write_bytes
from duraseed.pilot0_analysis import summarize_selected_method
from duraseed.pilot0_budget import calculate_pilot0_budget
from duraseed.pilot0_contract import (
    METHODS,
    STAGE_B_GRID,
    Pilot0Inputs,
    validate_pilot0_inputs,
)
from duraseed.pilot0_evidence import read_evaluation
from duraseed.pilot0_recovery import pilot0_session_ids
from duraseed.pilot0_reporting import evidence_index, reward_group_health, usage_summary
from duraseed.pilot0_source_build import write_pilot_seed_sources
from duraseed.provenance import canonical_json_bytes, canonical_json_value, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_remote import hydrate_ledger
from duraseed.runners.pilot0_selection import (
    PilotMatchingUnavailable,
    select_and_profile,
)
from duraseed.runners.pilot0_stage_a import run_stage_a_seed
from duraseed.runners.pilot0_stage_b import run_stage_b
from duraseed.training.stage_a_update_health import StageAUpdateHealthFailure


SCHEMA_VERSION = "duraseed-pilot0-pair-live-v2"


def _write(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def _state(root: Path, status: str, **values: object) -> None:
    _write(
        root / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": root.name,
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


def _preflight(inputs: Pilot0Inputs, root: Path) -> dict:
    budget = calculate_pilot0_budget(inputs)
    source = inputs.source
    artifact = canonical_json_value(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_label": "pilot-0-paired-seed",
            "run_id": inputs.run_id,
            "pair_index": inputs.pair_index,
            "seed": source.seed,
            "methods": METHODS,
            "authorized_usd": float(budget.cent_ceiling_usd),
            "upper_bound_usd": budget.upper_bound_usd,
            "fixed_storage_usd": budget.fixed_storage_usd,
            "token_budget": budget.tokens,
            "budget_gate_passed": budget.passed,
            "stage_a_updates": {"B-S": 294, "B-G": 50},
            "stage_b_grid": STAGE_B_GRID,
            "sealed_test_access": False,
            "lineage": {
                "git_commit": inputs.git_commit,
                "project_id": inputs.project_id,
                "session_id": inputs.session_id,
                "source_bundle_sha256": inputs.source_authentication.bundle_sha256,
                "dose_terminal_sha256": inputs.dose_terminal_sha256,
                "stage_b_recipe_artifact_sha256": (
                    inputs.stage_b_recipe_artifact_sha256
                ),
                "prior_pair_result_sha256": inputs.prior_pair_result_sha256,
                "m0_sampler_path": inputs.m0_sampler_path,
                "m0_state_path": inputs.m0_state_path,
                "manifest_ids": {
                    "a_rl_train": source.prompt_pools.a_rl_train_manifest.manifest_id,
                    "a_monitor": source.prompt_pools.a_monitor_manifest.manifest_id,
                    "a_cadence": source.a_cadence.manifest_id,
                    "a_validation": source.a_validation.manifest_id,
                    "b_train": source.b_train.manifest_id,
                    "b_validation": source.b_validation.manifest_id,
                },
                "billing": inputs.billing.lineage,
            },
        }
    )
    path = root / "preflight.json"
    if path.exists() and path.read_bytes() != canonical_json_bytes(artifact):
        raise RunnerGateError("Pilot pair resume changed its frozen preflight")
    _write(path, artifact)
    return artifact


def _evaluation(directory: Path) -> dict:
    result = read_evaluation(directory)
    if result is None:
        raise RunnerGateError(f"Pilot evaluation is missing: {directory}")
    return result


def _stage_b_result(root: Path, step: int) -> dict:
    if step == 0:
        return _evaluation(root / "step-0" / "b-validation")
    prior = STAGE_B_GRID[STAGE_B_GRID.index(step) - 1]
    return _evaluation(root / f"steps-{prior}-{step}" / "b-validation")


def _stage_b_retention(seed_root: Path, method: str, step: int) -> dict:
    if step == 0:
        return _evaluation(seed_root / method / "selected-pre-b" / "a-monitor")
    prior = STAGE_B_GRID[STAGE_B_GRID.index(step) - 1]
    return _evaluation(
        seed_root / method / "stage-b" / f"steps-{prior}-{step}" / "a-retention"
    )


def _method_summary(seed_root: Path, seed: int, method: str) -> dict:
    stage_b = seed_root / method / "stage-b"
    return summarize_selected_method(
        seed=seed,
        method=method,
        stage_a_selected=_evaluation(
            seed_root / method / "selected-pre-b" / "a-validation"
        ),
        stage_b_maps=tuple(_stage_b_result(stage_b, step) for step in STAGE_B_GRID),
        stage_b_retention=tuple(
            _stage_b_retention(seed_root, method, step) for step in STAGE_B_GRID
        ),
        stage_b_final_retention=_evaluation(stage_b / "steps-320-480" / "a-validation"),
    )


def _paired_contrast(cells: tuple[dict, dict]) -> dict:
    left, right = cells
    return {
        "contrast": "B-S_minus_B-G",
        "paired_seed": left["seed"],
        "raw_gain_stage_b_auc_difference": (
            left["F2_stage_b_learning"]["maps_auc"].raw_gain_auc
            - right["F2_stage_b_learning"]["maps_auc"].raw_gain_auc
        ),
        "fixed_budget_targeted_retention_difference": (
            left["F1_retention"]["fixed_budget_targeted_retention"]
            - right["F1_retention"]["fixed_budget_targeted_retention"]
        ),
        "interpretation": "single paired seed; pair 2 is a separate launch",
    }


def _billing_handoff(inputs: Pilot0Inputs, root: Path, status: str) -> None:
    _write(
        root / "billing-reconciliation-required.json",
        {
            "schema_version": "duraseed-pilot0-pair-billing-required-v1",
            "status": "pending",
            "run_id": inputs.run_id,
            "run_status": status,
            "pair_index": inputs.pair_index,
            "project_id": inputs.project_id,
            "session_ids": pilot0_session_ids(root, inputs.session_id),
            "prelaunch_actual_lifetime_spend_usd": (
                inputs.billing.actual_lifetime_spend_usd
            ),
            "authorized_usd": inputs.ledger.authorized_usd,
            "local_observed_cost_usd": inputs.ledger.observed_cost_usd,
            "prelaunch_billing_lineage": inputs.billing.lineage,
        },
    )


async def run_pilot0(inputs: Pilot0Inputs) -> dict:
    """Run one pair only; pair 2 cannot enter until pair 1 is durable."""

    validate_pilot0_inputs(inputs)
    root = inputs.output_root / inputs.run_id
    root.mkdir(parents=True, exist_ok=True)
    write_pilot_seed_sources(root / "pilot-inputs", inputs.source)
    preflight = _preflight(inputs, root)
    if preflight["budget_gate_passed"] is not True:
        _state(root, "blocked_budget")
        raise RunnerGateError("Pilot pair exceeds its exact authorization")
    hydrate_ledger(inputs, root)
    preflight_sha256 = sha256_bytes((root / "preflight.json").read_bytes())
    _state(root, "running", ledger=_ledger(inputs))
    seed_root = root / f"seed-{inputs.source.seed}"
    try:
        bs, bg = await run_stage_a_seed(
            inputs, inputs.source, seed_root, preflight_sha256=preflight_sha256
        )
        selected_bs, selected_bg, profiles = await select_and_profile(
            inputs, inputs.source, seed_root, bs, bg
        )
        for method, selected in (("B-S", selected_bs), ("B-G", selected_bg)):
            await run_stage_b(
                inputs,
                inputs.source,
                selected,
                method=method,
                output=seed_root / method / "stage-b",
                preflight_sha256=preflight_sha256,
            )
        cells = tuple(
            _method_summary(seed_root, inputs.source.seed, method) for method in METHODS
        )
        budget = calculate_pilot0_budget(inputs)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "evidence_collected",
            "run_id": inputs.run_id,
            "pair_index": inputs.pair_index,
            "seed": inputs.source.seed,
            "F1_F2_cells": cells,
            "paired_primary_contrast": _paired_contrast(cells),
            "F3_pre_b_profiles": profiles,
            "matching": canonical_json_value(
                json.loads((seed_root / "matching.json").read_bytes())
            ),
            "reward_group_health": reward_group_health(root, (inputs.source.seed,)),
            "token_and_cost_summary": usage_summary(inputs.ledger, budget),
            "sealed_test_access": False,
            "preflight_sha256": preflight_sha256,
            "evidence_index_sha256": _write(
                root / "evidence-index.json", evidence_index(root)
            ),
            "ledger": _ledger(inputs),
        }
        result_sha256 = _write(root / "result.json", result)
        _billing_handoff(inputs, root, "evidence_collected")
        _state(
            root,
            "evidence_collected",
            result_sha256=result_sha256,
            F1_F2_F3_complete=True,
            ledger=_ledger(inputs),
        )
        return result
    except (StageAUpdateHealthFailure, PilotMatchingUnavailable) as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "scientific_failure",
            "run_id": inputs.run_id,
            "pair_index": inputs.pair_index,
            "seed": inputs.source.seed,
            "reason": str(error),
            "seed_replacement_allowed": False,
            "update_health_evidence": getattr(error, "evidence", None),
            "ledger": _ledger(inputs),
        }
        result_sha256 = _write(root / "result.json", failure)
        _billing_handoff(inputs, root, "scientific_failure")
        _state(
            root,
            "scientific_failure",
            result_sha256=result_sha256,
            ledger=_ledger(inputs),
        )
        return failure
    except BaseException as error:
        _state(
            root,
            "interrupted",
            error=f"{type(error).__name__}: {error}",
            ledger=_ledger(inputs),
        )
        raise


__all__ = ["run_pilot0"]
