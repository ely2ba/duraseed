from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.data.manifests import build_manifest, build_tces_record
from duraseed.pilot0_evidence import finish_evaluation
from duraseed.pilot0_exposure import stage_a_exposure_report
from duraseed.pilot0_profiles import (
    matched_pre_b_profiles,
    matched_retention_grid,
    pre_b_capability_profile,
)
from duraseed.provenance import (
    canonical_json_bytes,
    derive_namespaced_seed,
    sha256_bytes,
)
from duraseed.runners import RunnerGateError
from duraseed.run_records import GenerationRecord, RewardRecord, append_jsonl
from duraseed.runtime.billing import PRICE_SNAPSHOT, UsageQuantities
from duraseed.schemas import VerificationFailure, VerificationResult
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig


def _manifest():  # type: ignore[no-untyped-def]
    split, seed = "a_validation", 91
    generator = TCESGenerator(
        derive_namespaced_seed(seed, "dataset.tces.split", split),
        TCESGeneratorConfig(
            n_operands=3,
            max_tree_depth=3,
            max_ast_nodes=5,
            max_valid_expressions=1_000,
            split=split,
        ),
    )
    records = [build_tces_record(generator.generate(index)) for index in range(2)]
    return build_manifest(
        name="pre-b-profile-test",
        split=split,
        generator_version="1.0.0",
        root_seed=seed,
        records=records,
    )


def _success(family_id: str) -> VerificationResult:
    return VerificationResult(
        valid_answer_tag=True,
        valid_lexing=True,
        valid_syntax=True,
        valid_operand_multiset=True,
        valid_operations=True,
        valid_intermediates=True,
        correct_target=True,
        reward=1.0,
        value_num=1,
        value_den=1,
        canonical_expression="1",
        strategy_family_id=family_id,
        ast_json={"type": "integer", "value": 1},
    )


def _evaluation(directory: Path):  # type: ignore[no-untyped-def]
    manifest = _manifest()
    sample_ids = set()
    counts = {}
    for record, role, success_count in zip(
        manifest.records, ("targeted", "sentinel"), (2, 1), strict=True
    ):
        counts[record.task_id] = {
            "panel_role": role,
            "successes": success_count,
            "trials": 4,
        }
        for sample_index in range(4):
            verification = (
                _success(record.valid_family_ids[0])
                if sample_index < success_count
                else VerificationResult(failure_code=VerificationFailure.INVALID_SYNTAX)
            )
            sample_id = f"{record.task_id}:sample-{sample_index}"
            sample_ids.add(sample_id)
            append_jsonl(
                directory / "generations.jsonl",
                GenerationRecord(
                    sample_id=sample_id,
                    sample_index=sample_index,
                    sampling_seed=sample_index,
                    purpose="evaluation",
                    checkpoint_stage="stage_a",
                    training_step=50,
                    sampler_checkpoint_path="tinker://test/stage-a",
                    task_manifest_id=manifest.manifest_id,
                    task_id=record.task_id,
                    task_family="tces",
                    source_split="a_validation",
                    prompt_text="prompt",
                    completion_text="completion",
                    prompt_tokens=3,
                    sampled_tokens=2,
                    run_id="profile-test",
                    method="B-S",
                    seed=91,
                    item_index=record.item_index,
                    assigned_family_id=record.intended_family,
                    family_id=verification.strategy_family_id,
                    panel_role=role,
                    completion_token_ids=(1, 2),
                    completion_logprobs=(-0.25, -0.75),
                    reward=verification.reward,
                ),
            )
            append_jsonl(
                directory / "rewards.jsonl",
                RewardRecord(
                    reward_id=f"reward:{sample_id}",
                    sample_id=sample_id,
                    task_id=record.task_id,
                    reward=verification.reward,
                    exact_verification=verification,
                ),
            )
    finish_evaluation(
        directory,
        manifest=manifest,
        label="profile-test",
        sampler_path="tinker://test/stage-a",
        samples_per_item=4,
        counts=counts,
        sample_ids=sample_ids,
        reward_ids=set(sample_ids),
        coordinates={},
    )
    return manifest


def test_pre_b_profile_is_descriptive_and_bounds_pass_at_k(tmp_path: Path) -> None:
    manifest = _evaluation(tmp_path)
    profile = pre_b_capability_profile(
        origin_kind="fixed_budget_stage_a",
        seed=91,
        method="B-S",
        manifest=manifest,
        evaluation_directories=(tmp_path,),
        cover_thresholds=(0.25, 0.5),
    )

    assert (
        profile["analysis_role"]
        == "descriptive_only_not_checkpoint_matching_or_selection"
    )
    assert profile["panels"]["targeted"]["exact_success_rate"] == 0.5
    assert profile["panels"]["sentinel"]["syntactically_invalid_rate"] == 0.75
    assert profile["panels"]["targeted"]["length_stop_rate"] == 0.0
    assert profile["panels"]["targeted"]["loop_fraction_among_length_stops"] == 0.0
    assert profile["panels"]["targeted"]["unique_completion_count"] == 1
    assert set(profile["panels"]["targeted"]["mean_pass_at_k"]) == {"1", "4"}
    assert all(set(row["pass_at_k"]) == {"1", "4"} for row in profile["items"])
    assert profile["panels"]["targeted"]["sampled_token_surprisal"][
        "mean_surprisal"
    ] == pytest.approx(0.5)


def test_matched_profile_is_bound_to_selected_combined_artifact(tmp_path: Path) -> None:
    directory = tmp_path / "candidates" / "seed-91" / "B-S" / "step-25"
    evaluation = directory / "initial" / "a-validation"
    manifest = _evaluation(evaluation)
    result = json.loads((evaluation / "result.json").read_bytes())
    combined = {
        "source_generation_sha256s": [result["generation_sha256"]],
        "item_counts": result["item_counts"],
    }
    combined_bytes = canonical_json_bytes(combined)
    (directory / "combined.json").write_bytes(combined_bytes)
    selection = {
        "global_extension_to_32": False,
        "cells": [
            {
                "seed": 91,
                "method": "B-S",
                "selected": {
                    "step": 25,
                    "sampler_path": "tinker://test/stage-a",
                    "combined_sha256": sha256_bytes(combined_bytes),
                },
            }
        ],
    }
    inputs = SimpleNamespace(
        seed_sources=(SimpleNamespace(seed=91, a_validation=manifest),),
        config=SimpleNamespace(evaluation={"reliability_tau_report": [0.25, 0.5]}),
    )

    assert len(matched_pre_b_profiles(tmp_path, selection, inputs)) == 1
    selection["cells"][0]["selected"]["combined_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(RunnerGateError, match="differs from selection"):
        matched_pre_b_profiles(tmp_path, selection, inputs)


def test_matched_retention_grid_starts_at_selected_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def evaluation(path: Path) -> dict:
        calls.append(path)
        return {
            "generation_sha256": "sha256:" + "1" * 64,
            "sampler_path": "tinker://test/selected",
        }

    monkeypatch.setattr("duraseed.pilot0_profiles.read_evaluation", evaluation)
    selected = {
        "step": 25,
        "monitor_generation_sha256": "sha256:" + "1" * 64,
        "sampler_path": "tinker://test/selected",
    }
    grid = matched_retention_grid(
        tmp_path, tmp_path / "matched-stage-b", 91, "B-S", selected
    )

    assert len(grid) == 11
    assert calls[0] == tmp_path / "seed-91" / "B-S" / "steps-10-25" / "a-monitor"
    assert calls[-1] == tmp_path / "matched-stage-b" / "steps-320-480" / "a-retention"


def _exposure_segments(root: Path, seed: int) -> None:
    for method in ("B-S", "B-G"):
        for start, stop in zip((0, 10, 25), (10, 25, 50), strict=True):
            directory = root / f"seed-{seed}" / method / f"steps-{start}-{stop}"
            directory.mkdir(parents=True)
            for step in range(start + 1, stop + 1):
                append_jsonl(
                    directory / "metrics.jsonl",
                    {
                        "phase": "stage_a",
                        "training_step": step,
                        "method": method,
                        "metrics": {"local.train_tokens": 2.0},
                    },
                )
            if method == "B-G":
                append_jsonl(
                    directory / "generations.jsonl",
                    GenerationRecord(
                        sample_id=f"{start}-{stop}",
                        sample_index=0,
                        purpose="training",
                        checkpoint_stage="stage_a",
                        training_step=stop,
                        sampler_checkpoint_path="tinker://test/rollout",
                        task_manifest_id="sha256:" + "1" * 64,
                        task_id=f"task-{stop}",
                        task_family="tces",
                        source_split="a_rl_train",
                        prompt_text="prompt",
                        completion_text="completion",
                        prompt_tokens=3,
                        sampled_tokens=4,
                        run_id="exposure-test",
                        method="B-G",
                        seed=seed,
                    ),
                )


def test_stage_a_exposure_uses_exact_update_and_rollout_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _exposure_segments(tmp_path, 11)
    monkeypatch.setattr("duraseed.pilot0_exposure.read_segment", lambda *a, **k: {})
    monkeypatch.setattr(
        "duraseed.pilot0_exposure.read_evaluation",
        lambda *a, **k: {
            "item_counts": [
                {
                    "panel_role": "targeted",
                    "successes": 3,
                    "trials": 4,
                },
                {
                    "panel_role": "sentinel",
                    "successes": 1,
                    "trials": 4,
                },
            ]
        },
    )
    report = stage_a_exposure_report(tmp_path, (11,), PRICE_SNAPSHOT)
    cells = {(row["seed"], row["method"]): row for row in report["cells"]}
    assert cells[11, "B-S"]["points"][0] == {
        "stage_a_step": 0,
        "incremental_optimizer_updates": 0,
        "cumulative_optimizer_updates": 0,
        "incremental_tokens": {"prefill": 0, "sample": 0, "train": 0},
        "cumulative_tokens": {"prefill": 0, "sample": 0, "train": 0},
        "cumulative_rollout_completions": 0,
        "cumulative_token_cost_usd": 0.0,
        "targeted_monitor_posterior_mean": pytest.approx(0.7),
        "sentinel_monitor_posterior_mean": pytest.approx(0.3),
    }
    bs = cells[11, "B-S"]["points"][-1]
    bg = cells[11, "B-G"]["points"][-1]

    assert bs["cumulative_tokens"] == {"prefill": 0, "sample": 0, "train": 100}
    assert bg["cumulative_tokens"] == {"prefill": 9, "sample": 12, "train": 100}
    assert bg["cumulative_rollout_completions"] == 3
    assert bg["cumulative_optimizer_updates"] == 50
    assert bg["targeted_monitor_posterior_mean"] == pytest.approx(0.7)
    assert bg["sentinel_monitor_posterior_mean"] == pytest.approx(0.3)
    assert bg["cumulative_token_cost_usd"] == pytest.approx(
        PRICE_SNAPSHOT.cost(
            UsageQuantities(prefill_tokens=9, sample_tokens=12, train_tokens=100)
        )
    )
