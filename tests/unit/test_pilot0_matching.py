from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed import pilot0_matching
from duraseed import pilot0_matched_budget
from duraseed.pilot0_matched_budget import (
    calculate_matched_pilot_budget,
    validate_matched_authorization,
)
from duraseed.pilot0_matching import (
    build_candidate_plan,
    combine_evaluations,
    freeze_target,
    paired_matched_aggregate,
    select_checkpoint,
    summarize_matched_cell,
    targeted_posterior_mean,
    targeted_sampling_se,
)
from duraseed.pilot0_pair_matching import select_paired_cadence
from duraseed.pilot0_contract import STAGE_B_GRID
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners import pilot0_matched_selection
from duraseed.runners.pilot0_matched_selection import run_matched_selection
from duraseed.runtime import TokenBudget, TokenLedger


def _result(successes: int, trials: int = 16) -> dict:
    return {
        "samples_per_item": trials,
        "generation_sha256": "sha256:" + f"{successes:064x}",
        "reward_sha256": "sha256:" + f"{successes + 100:064x}",
        "item_counts": [
            {
                "task_id": "target-1",
                "panel_role": "targeted",
                "successes": successes,
                "trials": trials,
            },
            {
                "task_id": "target-2",
                "panel_role": "targeted",
                "successes": successes,
                "trials": trials,
            },
            {
                "task_id": "sentinel",
                "panel_role": "sentinel",
                "successes": 1,
                "trials": trials,
            },
        ],
    }


def _cadence_result(successes: int) -> dict:
    return {
        "generation_sha256": "sha256:" + f"{successes:064x}",
        "item_counts": [
            {
                "task_id": f"target-{index}",
                "panel_role": "targeted",
                "successes": int(index < successes),
                "trials": 1,
            }
            for index in range(96)
        ],
    }


def _cadence_row(step: int, successes: int) -> dict:
    return {
        "checkpoint": {
            "step": step,
            "sampler_path": f"sampler-{step}",
            "state_path": f"state-{step}",
        },
        "evaluation": _cadence_result(successes),
    }


def test_frozen_pair_match_uses_one_draw_success_intervals_and_earlier_tie() -> None:
    selected = select_paired_cadence(
        (_cadence_row(10, 10), _cadence_row(20, 20)),
        (_cadence_row(10, 19), _cadence_row(20, 30)),
    )
    assert selected["status"] == "selected"
    assert selected["B-S"]["step"] == 20
    assert selected["B-G"]["step"] == 10
    assert selected["B-S"]["targeted_exact_success_rate"] == pytest.approx(20 / 96)
    unavailable = select_paired_cadence((_cadence_row(10, 1),), (_cadence_row(10, 90),))
    assert unavailable["status"] == "unavailable"
    assert unavailable["seed_replacement_allowed"] is False


def test_target_is_minimum_of_four_full_fixed_budget_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "run_id": tmp_path.name,
                "status": "evidence_collected",
                "evidence_status": "complete",
                "primary_analysis_status": "complete",
            }
        )
    )
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {
                "run_id": tmp_path.name,
                "methods": ["B-S", "B-G"],
                "seeds": [11, 29],
                "stage_a_grid": [0, 10, 25, 50],
                "lineage": {
                    "source_bundle_sha256": "sha256:" + "1" * 64,
                    "resolved_config_hash": "sha256:" + "2" * 64,
                },
            }
        )
    )
    scores = {(11, "B-S"): 8, (11, "B-G"): 6, (29, "B-S"): 7, (29, "B-G"): 5}

    def evaluation(path: Path) -> dict:
        seed = int(
            next(part for part in path.parts if part.startswith("seed-")).split("-")[1]
        )
        method = next(part for part in path.parts if part in {"B-S", "B-G"})
        return _result(scores[seed, method])

    monkeypatch.setattr(pilot0_matching, "_evaluation", evaluation)
    artifact = freeze_target(tmp_path, tolerance=0.015)
    assert artifact["target"] == pytest.approx((5.5 / 17))
    assert (
        artifact["rule"]
        == "minimum_four_seed_method_fixed_budget_targeted_posterior_means"
    )
    assert len(artifact["endpoints"]) == 4


def test_candidate_plan_uses_endpoint_for_weakest_cell_and_real_neighbors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = targeted_posterior_mean(_result(8))
    endpoints = [
        {
            "seed": seed,
            "method": method,
            "targeted_posterior_mean": target if (seed, method) == (11, "B-S") else 0.9,
        }
        for seed in (11, 29)
        for method in ("B-S", "B-G")
    ]
    monitor_successes = {
        (11, "B-S"): (0, 12, 12, 12),
        (11, "B-G"): (0, 12, 12, 12),
        (29, "B-S"): (0, 0, 12, 12),
        (29, "B-G"): (0, 0, 0, 0),
    }

    def monitor(root: Path, seed: int, method: str, step: int) -> tuple[dict, dict]:
        index = (0, 10, 25, 50).index(step)
        value = monitor_successes[seed, method][index]
        segment = {
            "sampler_path": f"sampler-{seed}-{method}-{step}",
            "state_path": f"state-{step}",
        }
        return segment, _result(value)

    monkeypatch.setattr(pilot0_matching, "_monitor", monitor)
    plan = build_candidate_plan(
        tmp_path,
        {"target": target, "endpoints": endpoints, "schema_version": "target"},
    )
    by_cell = {(row["seed"], row["method"]): row for row in plan["cells"]}
    weakest = by_cell[11, "B-S"]
    assert weakest["target_defining_cell"] is True
    assert weakest["crossing_step"] == 50
    assert [row["step"] for row in weakest["candidates"]] == [25, 50]
    assert [row["step"] for row in by_cell[11, "B-G"]["candidates"]] == [0, 10, 25]
    assert [row["step"] for row in by_cell[29, "B-S"]["candidates"]] == [10, 25, 50]
    assert by_cell[29, "B-G"]["crossing_source"] == "fixed_endpoint_full_validation"


def test_selection_is_closest_in_band_with_earlier_tie_break() -> None:
    candidates = (
        {"step": 25, "targeted_posterior_mean": 0.24},
        {"step": 50, "targeted_posterior_mean": 0.26},
        {"step": 100, "targeted_posterior_mean": 0.251},
    )
    assert select_checkpoint(candidates, target=0.25, tolerance=0.02)["step"] == 100
    tied = candidates[:2]
    assert select_checkpoint(tied, target=0.25, tolerance=0.02)["step"] == 25
    assert select_checkpoint(tied, target=0.5, tolerance=0.01) is None


def test_combination_preserves_items_and_sampling_se() -> None:
    first, second = _result(4), _result(6)
    combined = combine_evaluations((first, second))
    assert combined["samples_per_item"] == 32
    assert combined["item_counts"][1]["successes"] == 10
    assert targeted_posterior_mean(combined) == pytest.approx(10.5 / 33)
    assert targeted_sampling_se(_result(8)) == pytest.approx((0.25 / 16 / 2) ** 0.5)
    changed = _result(6)
    changed["item_counts"][0]["task_id"] = "other"
    with pytest.raises(RunnerGateError, match="changed the validation population"):
        combine_evaluations((first, changed))


def test_matched_summary_includes_monitor_retention_auc() -> None:
    maps = tuple(
        {
            "item_counts": [
                {
                    "task_id": "maps",
                    "panel_role": "stage-b",
                    "successes": index,
                    "trials": 16,
                }
            ]
        }
        for index in range(len(STAGE_B_GRID))
    )
    cell = summarize_matched_cell(
        seed=11,
        method="B-S",
        target=0.5,
        stage_a=_result(8),
        stage_b_maps=maps,
        stage_b_retention=tuple(_result(8) for _ in STAGE_B_GRID),
        stage_b_final_retention=_result(6),
    )

    assert cell["targeted_monitor_retention_absolute_auc"] == pytest.approx(0.5)
    assert len(cell["targeted_monitor_retention_curve"]) == len(STAGE_B_GRID)
    other = {**cell, "method": "B-G"}
    cells = (cell, other, {**cell, "seed": 29}, {**other, "seed": 29})
    aggregate = paired_matched_aggregate(cells)
    assert aggregate["mean_targeted_monitor_retention_absolute_auc_difference"] == 0


def test_authorization_binds_exact_preflight_and_ledger_cap() -> None:
    raw = canonical_json_bytes({"preflight": True})
    ledger = TokenLedger(TokenBudget(1, 2, 3), 12.5)
    inputs = SimpleNamespace(run_id="matched-run", ledger=ledger)
    authorization = {
        "schema_version": "duraseed-pilot0-matched-authorization-v1",
        "status": "accepted",
        "preflight_sha256": sha256_bytes(raw),
        "matched_run_id": "matched-run",
        "authorizer": "Ely",
        "authorized_at_utc": "2026-08-14T22:00:00+00:00",
        "authorized_usd": 12.5,
        "no_rerun_authorized": True,
    }
    assert validate_matched_authorization(
        inputs, authorization, raw, {"required_upper_bound_usd": 12.4}
    ) == sha256_bytes(canonical_json_bytes(authorization))
    authorization["authorized_usd"] = 12.4
    with pytest.raises(RunnerGateError, match="exact bound authorization"):
        validate_matched_authorization(
            inputs, authorization, raw, {"required_upper_bound_usd": 12.4}
        )


def test_budget_reserves_max_selection_and_four_complete_stage_b_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = SimpleNamespace(task_family="tces")
    manifest = SimpleNamespace(records=(record,))
    source = SimpleNamespace(
        seed=11,
        a_validation=manifest,
        b_validation=manifest,
        b_train=manifest,
        prompt_pools=SimpleNamespace(a_monitor_manifest=manifest),
    )
    inputs = SimpleNamespace(
        runtime=object(),
        acquisition=SimpleNamespace(selected_max_tokens=256),
        config=SimpleNamespace(
            statistics=SimpleNamespace(checkpoint_selection_max_samples_per_item=32),
            evaluation={"pilot_samples_per_item": 16},
            stage_a=SimpleNamespace(monitor_samples_per_item=4),
        ),
        seed_sources=(source, SimpleNamespace(**{**source.__dict__, "seed": 29})),
    )
    datum = SimpleNamespace(model_input=SimpleNamespace(length=100))
    monkeypatch.setattr(pilot0_matched_budget, "_prompt_length", lambda *args: 100)
    monkeypatch.setattr(pilot0_matched_budget, "sft_datum", lambda *args: datum)
    monkeypatch.setattr(
        pilot0_matched_budget, "stage_b_sources", lambda source: (record,)
    )
    plan = {
        "cells": [
            {"seed": seed, "method": method, "candidates": [{}, {}]}
            for seed in (11, 29)
            for method in ("B-S", "B-G")
        ]
    }
    budget = calculate_matched_pilot_budget(inputs, plan)
    assert budget.candidate_checkpoint_count == 8
    assert budget.fixed_storage_usd == pytest.approx(7.6)
    assert budget.tokens == TokenBudget(124_800, 229_376, 6_144_000)


def test_selection_extends_every_candidate_with_disjoint_sample_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    async def evaluate(*args, samples: int, sample_index_start: int, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((samples, sample_index_start))
        result = _result(samples // 2, samples)
        result["generation_sha256"] = "sha256:" + f"{sample_index_start + 1:064x}"
        return result

    monkeypatch.setattr(pilot0_matched_selection, "_evaluate_candidate", evaluate)
    monkeypatch.setattr(
        pilot0_matched_selection,
        "_origin",
        lambda *args: {"sampler_path": "origin"},
    )
    config = SimpleNamespace(
        statistics=SimpleNamespace(
            checkpoint_selection_initial_samples_per_item=16,
            checkpoint_selection_max_samples_per_item=32,
            checkpoint_selection_max_panel_mean_se=0.01,
        )
    )
    inputs = SimpleNamespace(
        config=config,
        seed_sources=(SimpleNamespace(seed=11),),
    )
    candidate = {
        "step": 50,
        "sampler_path": "sampler",
        "state_path": "state",
        "monitor_generation_sha256": "sha256:" + "1" * 64,
    }
    selection, _ = asyncio.run(
        run_matched_selection(
            inputs,
            tmp_path,
            tmp_path / "matched",
            plan={"cells": [{"seed": 11, "method": "B-S", "candidates": [candidate]}]},
            target_artifact={"target": 0.5, "tolerance": 0.015},
            preflight_sha256="sha256:" + "2" * 64,
            target_sha256="sha256:" + "3" * 64,
        )
    )
    assert calls == [(16, 0), (16, 16)]
    assert selection["global_extension_to_32"] is True
    assert selection["cells"][0]["selected"]["samples_per_item"] == 32
