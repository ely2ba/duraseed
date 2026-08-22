"""Descriptive pre-Stage-B profiles from authenticated Pilot-0 evaluations."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from math import fsum
from pathlib import Path
from statistics import median
from typing import Any

from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
from duraseed.evaluation.analysis import (
    BinomialObservation,
    equal_item_posterior_mean,
    jeffreys_posterior_mean,
    posterior_soft_cover,
    summarize_token_surprisal,
    unbiased_pass_at_k,
)
from duraseed.pilot0_evidence import read_evaluation
from duraseed.pilot0_contract import STAGE_B_GRID
from duraseed.provenance import sha256_bytes
from duraseed.run_records import GenerationRecord, RewardRecord
from duraseed.runners import RunnerGateError
from duraseed.runners.capability_dose_evidence import detected_loop


_PASS_K = (1, 4, 16)
STAGE_A_GRID = (0, 10, 25, 50)  # legacy matched reducer only


def _evaluation_rows(
    manifest: DatasetManifest,
    directories: tuple[Path, ...],
    expected_sampler_path: str | None,
) -> tuple[tuple[GenerationRecord, ...], tuple[RewardRecord, ...], tuple[str, ...]]:
    if not directories:
        raise RunnerGateError("pre-B profile requires evaluation evidence")
    generations, rewards, hashes = [], [], []
    for directory in directories:
        result = read_evaluation(directory)
        if result is None or result.get("manifest_id") != manifest.manifest_id:
            raise RunnerGateError("pre-B profile changed its validation manifest")
        try:
            generation_rows = tuple(
                GenerationRecord.model_validate_json(line)
                for line in (directory / "generations.jsonl").read_text().splitlines()
            )
            reward_rows = tuple(
                RewardRecord.model_validate_json(line)
                for line in (directory / "rewards.jsonl").read_text().splitlines()
            )
        except (OSError, ValueError) as error:
            raise RunnerGateError("pre-B profile evidence is unreadable") from error
        generations.extend(generation_rows)
        rewards.extend(reward_rows)
        hashes.append(str(result["generation_sha256"]))
    sample_ids = [row.sample_id for row in generations]
    reward_ids = [row.sample_id for row in rewards]
    if (
        len(sample_ids) != len(set(sample_ids))
        or len(reward_ids) != len(set(reward_ids))
        or set(sample_ids) != set(reward_ids)
        or len({row.sampler_checkpoint_path for row in generations}) != 1
        or (
            expected_sampler_path is not None
            and any(
                row.sampler_checkpoint_path != expected_sampler_path
                for row in generations
            )
        )
    ):
        raise RunnerGateError("pre-B profile draws changed identity or checkpoint")
    return tuple(generations), tuple(rewards), tuple(hashes)


def _length_summary(rows: list[GenerationRecord]) -> dict:
    values = sorted(row.sampled_tokens for row in rows)
    return {
        "completion_count": len(values),
        "minimum": values[0],
        "mean": fsum(values) / len(values),
        "median": median(values),
        "maximum": values[-1],
    }


def pre_b_capability_profile(
    *,
    origin_kind: str,
    seed: int,
    method: str,
    manifest: DatasetManifest,
    evaluation_directories: tuple[Path, ...],
    cover_thresholds: tuple[float, ...],
    expected_sampler_path: str | None = None,
) -> dict:
    """Describe a selected Stage-A origin without adding another matching gate."""

    records = {row.task_id: row for row in manifest.records}
    if (
        manifest.task_family != "tces"
        or manifest.split != "a_validation"
        or any(not isinstance(row, TCESTaskManifestRecord) for row in records.values())
    ):
        raise RunnerGateError("pre-B profile requires the TCES validation manifest")
    generations, rewards, source_hashes = _evaluation_rows(
        manifest, evaluation_directories, expected_sampler_path
    )
    reward_by_sample = {row.sample_id: row for row in rewards}
    by_task: dict[str, list[tuple[GenerationRecord, RewardRecord]]] = defaultdict(list)
    for generation in generations:
        reward = reward_by_sample[generation.sample_id]
        record = records.get(generation.task_id)
        verification = reward.exact_verification
        if (
            not isinstance(record, TCESTaskManifestRecord)
            or generation.task_manifest_id != manifest.manifest_id
            or generation.source_split != manifest.split
            or generation.seed != seed
            or generation.method != method
            or generation.assigned_family_id != record.intended_family
            or generation.family_id != verification.strategy_family_id
            or (
                verification.reward == 1.0
                and generation.family_id not in record.valid_family_ids
            )
        ):
            raise RunnerGateError("pre-B profile rows do not join their manifest")
        by_task[generation.task_id].append((generation, reward))
    if set(by_task) != set(records):
        raise RunnerGateError("pre-B profile does not cover every validation item")

    items = []
    for task_id, rows in sorted(by_task.items()):
        roles = {row[0].panel_role for row in rows}
        if len(roles) != 1 or not roles.issubset({"targeted", "sentinel"}):
            raise RunnerGateError("pre-B profile changed one item's panel role")
        role = roles.pop()
        record = records[task_id]
        successes, trials = sum(int(row[1].reward) for row in rows), len(rows)
        items.append(
            {
                "task_id": task_id,
                "panel_role": role,
                "assigned_family_id": record.intended_family,
                "successes": successes,
                "trials": trials,
                "exact_success_rate": successes / trials,
                "posterior_mean": jeffreys_posterior_mean(successes, trials),
                "pass_at_k": {
                    str(k): unbiased_pass_at_k(successes, trials, k)
                    for k in _PASS_K
                    if k <= trials
                },
            }
        )

    panels, families = {}, []
    for role in ("targeted", "sentinel"):
        role_items = [row for row in items if row["panel_role"] == role]
        role_generations = [
            row[0]
            for rows in by_task.values()
            for row in rows
            if row[0].panel_role == role
        ]
        role_rewards = [reward_by_sample[row.sample_id] for row in role_generations]
        length_stops = [
            row
            for row in role_generations
            if row.sampling_max_tokens is not None
            and row.sampled_tokens >= row.sampling_max_tokens
        ]
        looped_length_stops = sum(
            detected_loop(row.completion_token_ids or ()) for row in length_stops
        )
        if not role_items or not role_generations:
            raise RunnerGateError(f"pre-B profile has no {role} evidence")
        observations = tuple(
            BinomialObservation(row["successes"], row["trials"]) for row in role_items
        )
        valid_ks = tuple(
            k for k in _PASS_K if all(k <= row["trials"] for row in role_items)
        )
        failure_counts = Counter(
            row.exact_verification.failure_code.value
            for row in role_rewards
            if row.exact_verification.failure_code is not None
        )
        strategy_counts = Counter(
            row.exact_verification.strategy_family_id
            for row in role_rewards
            if row.reward == 1.0
        )
        logprobs = [
            value
            for row in role_generations
            for value in (row.completion_logprobs or (None,) * row.sampled_tokens)
        ]
        panels[role] = {
            "item_count": len(role_items),
            "completion_count": len(role_rewards),
            "successes": sum(row["successes"] for row in role_items),
            "exact_success_rate": fsum(row["successes"] for row in role_items)
            / fsum(row["trials"] for row in role_items),
            "equal_item_posterior_mean": equal_item_posterior_mean(observations),
            "cover_curve": [
                {
                    "tau": tau,
                    "posterior_soft_cover": posterior_soft_cover(observations, tau),
                }
                for tau in cover_thresholds
            ],
            "mean_pass_at_k": {
                str(k): fsum(row["pass_at_k"][str(k)] for row in role_items)
                / len(role_items)
                for k in valid_ks
            },
            "syntactically_invalid_rate": sum(
                not row.exact_verification.valid_syntax for row in role_rewards
            )
            / len(role_rewards),
            "failure_code_counts": dict(sorted(failure_counts.items())),
            "length_stop_count": len(length_stops),
            "length_stop_rate": len(length_stops) / len(role_generations),
            "looped_length_stop_count": looped_length_stops,
            "loop_fraction_among_length_stops": (
                looped_length_stops / len(length_stops) if length_stops else 0.0
            ),
            "completion_token_length": _length_summary(role_generations),
            "unique_completion_count": len(
                {row.completion_text for row in role_generations}
            ),
            "sampled_token_surprisal": (
                asdict(summarize_token_surprisal(logprobs)) if logprobs else None
            ),
            "verified_strategy_family_counts": dict(sorted(strategy_counts.items())),
            "unique_verified_strategy_family_count": len(strategy_counts),
        }
        for family_id in sorted({row["assigned_family_id"] for row in role_items}):
            subset = [
                row for row in role_items if row["assigned_family_id"] == family_id
            ]
            family_observations = tuple(
                BinomialObservation(row["successes"], row["trials"]) for row in subset
            )
            families.append(
                {
                    "panel_role": role,
                    "assigned_family_id": family_id,
                    "item_count": len(subset),
                    "successes": sum(row["successes"] for row in subset),
                    "trials": sum(row["trials"] for row in subset),
                    "equal_item_posterior_mean": equal_item_posterior_mean(
                        family_observations
                    ),
                }
            )
    return {
        "schema_version": "duraseed-pre-b-capability-profile-v1",
        "origin_kind": origin_kind,
        "seed": seed,
        "method": method,
        "analysis_role": "descriptive_only_not_checkpoint_matching_or_selection",
        "matching_metric": (
            "targeted_raw_exact_success_rate_one_draw_cadence"
            if origin_kind == "post_hoc_overlap_matched"
            else "targeted_equal_item_jeffreys_posterior_mean"
        ),
        "source_generation_sha256s": source_hashes,
        "panels": panels,
        "families": families,
        "items": items,
    }


def fixed_pre_b_profiles(
    root: Path, seed_sources: tuple[Any, ...], cover_thresholds: tuple[float, ...]
) -> tuple[dict, ...]:
    return tuple(
        pre_b_capability_profile(
            origin_kind="fixed_budget_stage_a",
            seed=source.seed,
            method=method,
            manifest=source.a_validation,
            evaluation_directories=(
                root / f"seed-{source.seed}" / method / "steps-25-50" / "a-validation",
            ),
            cover_thresholds=cover_thresholds,
        )
        for source in seed_sources
        for method in ("B-S", "B-G")
    )


def matched_pre_b_profiles(
    root: Path,
    selection: dict,
    inputs: Any,
) -> tuple[dict, ...]:
    manifests = {source.seed: source.a_validation for source in inputs.seed_sources}
    cover_thresholds = tuple(
        float(value) for value in inputs.config.evaluation["reliability_tau_report"]
    )
    profiles = []
    for cell in selection["cells"]:
        selected = cell["selected"]
        directory = (
            root
            / "candidates"
            / f"seed-{cell['seed']}"
            / cell["method"]
            / f"step-{selected['step']}"
        )
        draws = [directory / "initial" / "a-validation"]
        if selection["global_extension_to_32"]:
            draws.append(directory / "extension" / "a-validation")
        try:
            combined_bytes = (directory / "combined.json").read_bytes()
            combined = json.loads(combined_bytes)
        except (OSError, ValueError) as error:
            raise RunnerGateError("matched pre-B combination is unreadable") from error
        profile = pre_b_capability_profile(
            origin_kind="matched_target_stage_a",
            seed=cell["seed"],
            method=cell["method"],
            manifest=manifests[cell["seed"]],
            evaluation_directories=tuple(draws),
            cover_thresholds=cover_thresholds,
            expected_sampler_path=selected["sampler_path"],
        )
        counts = [
            {key: row[key] for key in ("task_id", "panel_role", "successes", "trials")}
            for row in profile["items"]
        ]
        if (
            sha256_bytes(combined_bytes) != selected["combined_sha256"]
            or combined.get("source_generation_sha256s")
            != list(profile["source_generation_sha256s"])
            or combined.get("item_counts") != counts
        ):
            raise RunnerGateError("matched pre-B profile differs from selection")
        profiles.append(profile)
    return tuple(profiles)


def matched_retention_grid(
    pilot_root: Path, stage_b_root: Path, seed: int, method: str, selected: dict
) -> tuple[dict, ...]:
    """Join the selected origin monitor to its matched Stage-B monitor series."""

    step = int(selected["step"])
    if step not in STAGE_A_GRID:
        raise RunnerGateError("matched retention origin is off the Stage-A grid")
    if step == 0:
        origin = pilot_root / f"seed-{seed}" / "boundary-origin" / "a-monitor"
    else:
        prior = STAGE_A_GRID[STAGE_A_GRID.index(step) - 1]
        origin = (
            pilot_root / f"seed-{seed}" / method / f"steps-{prior}-{step}" / "a-monitor"
        )
    first = read_evaluation(origin)
    if (
        first is None
        or first.get("generation_sha256") != selected["monitor_generation_sha256"]
        or first.get("sampler_path") != selected["sampler_path"]
    ):
        raise RunnerGateError("matched retention origin differs from selection")
    results = [first]
    for start, stop in zip(STAGE_B_GRID[:-1], STAGE_B_GRID[1:], strict=True):
        result = read_evaluation(stage_b_root / f"steps-{start}-{stop}" / "a-retention")
        if result is None:
            raise RunnerGateError("matched retention grid is incomplete")
        results.append(result)
    return tuple(results)


__all__ = [
    "fixed_pre_b_profiles",
    "matched_pre_b_profiles",
    "matched_retention_grid",
    "pre_b_capability_profile",
]
