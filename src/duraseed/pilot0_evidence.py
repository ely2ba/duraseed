"""Authentication and exact-join recovery for Pilot-0 evaluations."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Literal

from duraseed.data.io import atomic_write_bytes
from duraseed.data.manifests import DatasetManifest
from duraseed.provenance import (
    canonical_json_bytes,
    derive_namespaced_seed,
    sha256_bytes,
)
from duraseed.run_records import GenerationRecord, MethodCode, RewardRecord
from duraseed.runners import RunnerGateError
from duraseed.runners.remote_journal import RemoteJournal


EvaluationStage = Literal["m0", "stage_a", "stage_b"]


def read_evaluation(directory: Path) -> dict | None:
    path = directory / "result.json"
    if not path.exists():
        if (directory / "remote-calls.jsonl").exists():
            RemoteJournal(directory)
            raise RunnerGateError(
                "incomplete Pilot-0 evaluation requires reconciliation"
            )
        return None
    try:
        result = json.loads(path.read_bytes())
        generation_bytes = (directory / "generations.jsonl").read_bytes()
        reward_bytes = (directory / "rewards.jsonl").read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("Pilot-0 evaluation artifact is unreadable") from error
    if result.get("generation_sha256") != sha256_bytes(generation_bytes) or result.get(
        "reward_sha256"
    ) != sha256_bytes(reward_bytes):
        raise RunnerGateError("Pilot-0 evaluation hash mismatch")
    try:
        generations = tuple(
            GenerationRecord.model_validate_json(row)
            for row in generation_bytes.decode().splitlines()
        )
        rewards = tuple(
            RewardRecord.model_validate_json(row)
            for row in reward_bytes.decode().splitlines()
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise RunnerGateError("Pilot-0 evaluation rows are unreadable") from error
    by_sample = {row.sample_id: row for row in generations}
    reward_by_sample = {row.sample_id: row for row in rewards}
    if (
        len(by_sample) != len(generations)
        or len(reward_by_sample) != len(rewards)
        or set(by_sample) != set(reward_by_sample)
    ):
        raise RunnerGateError("Pilot-0 evaluation exact join failed")
    counts: dict[str, dict[str, int | str]] = defaultdict(
        lambda: {"panel_role": "", "successes": 0, "trials": 0}
    )
    for sample_id, generation in by_sample.items():
        reward = reward_by_sample[sample_id]
        if (
            reward.task_id != generation.task_id
            or reward.reward != generation.reward
            or reward.reward_id != f"reward:{sample_id}"
            or reward.reward not in (0.0, 1.0)
        ):
            raise RunnerGateError("Pilot-0 evaluation joined rows disagree")
        values = counts[generation.task_id]
        if values["panel_role"] not in ("", generation.panel_role):
            raise RunnerGateError("Pilot-0 evaluation changed one item's panel role")
        values["panel_role"] = generation.panel_role or ""
        values["successes"] = int(values["successes"]) + int(reward.reward)
        values["trials"] = int(values["trials"]) + 1
    expected_counts = [
        {"task_id": task_id, **values} for task_id, values in sorted(counts.items())
    ]
    samples = result.get("samples_per_item")
    if (
        result.get("row_count") != len(by_sample)
        or result.get("item_count") != len(counts)
        or result.get("item_counts") != expected_counts
        or result.get("join_sha256")
        != sha256_bytes(canonical_json_bytes(sorted(by_sample)))
        or type(samples) is not int
        or samples < 1
        or any(int(values["trials"]) != samples for values in counts.values())
    ):
        raise RunnerGateError("Pilot-0 evaluation summary disagrees with joined rows")
    return result


def load_evaluation_prefix(
    directory: Path,
    samples_per_item: int,
    *,
    sample_index_start: int = 0,
    run_id: str,
    label: str,
    manifest: DatasetManifest,
    sampler_path: str,
    origin_sampler_path: str,
    method: MethodCode | None,
    checkpoint_stage: EvaluationStage,
    training_step: int,
    seed: int,
    seed_namespace: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    task_contracts: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], set[str], set[str], set[str]]:
    generation_path = directory / "generations.jsonl"
    reward_path = directory / "rewards.jsonl"
    if generation_path.exists() != reward_path.exists():
        raise RunnerGateError("Pilot-0 evaluation has one side of its exact join")
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"panel_role": "", "successes": 0, "trials": 0}
    )
    if not generation_path.exists():
        return counts, set(), set(), set()
    try:
        generations = tuple(
            GenerationRecord.model_validate_json(row)
            for row in generation_path.read_text().splitlines()
        )
        rewards = tuple(
            RewardRecord.model_validate_json(row)
            for row in reward_path.read_text().splitlines()
        )
    except (OSError, ValueError) as error:
        raise RunnerGateError("Pilot-0 partial evaluation is unreadable") from error
    by_sample = {row.sample_id: row for row in generations}
    reward_by_sample = {row.sample_id: row for row in rewards}
    if (
        len(by_sample) != len(generations)
        or len(reward_by_sample) != len(rewards)
        or set(by_sample) != set(reward_by_sample)
    ):
        raise RunnerGateError("Pilot-0 partial evaluation exact join failed")
    for sample_id, generation in by_sample.items():
        reward = reward_by_sample[sample_id]
        contract = task_contracts.get(generation.task_id)
        if contract is None:
            raise RunnerGateError("Pilot-0 partial evaluation contains an unknown task")
        expected_seed = derive_namespaced_seed(
            seed,
            seed_namespace,
            generation.task_family,
            generation.task_id,
            generation.item_index,
            generation.sample_index,
        )
        if (
            generation.task_id != reward.task_id
            or generation.reward != reward.reward
            or reward.reward_id != f"reward:{sample_id}"
            or generation.task_family != contract["task_family"]
            or generation.item_index != contract["item_index"]
            or generation.prompt_text != contract["prompt_text"]
            or generation.assigned_family_id != contract["assigned_family_id"]
            or generation.panel_role != contract["panel_role"]
            or generation.run_id != run_id
            or generation.task_manifest_id != manifest.manifest_id
            or generation.source_split != manifest.split
            or generation.sampler_checkpoint_path != sampler_path
            or generation.origin_sampler_checkpoint_path != origin_sampler_path
            or generation.method != method
            or generation.purpose != "evaluation"
            or generation.checkpoint_stage != checkpoint_stage
            or generation.training_step != training_step
            or generation.seed != seed
            or generation.sampling_seed != expected_seed
            or generation.sampling_max_tokens != max_tokens
            or generation.sampling_temperature != temperature
            or generation.sampling_top_p != top_p
            or generation.sample_id
            != (
                f"{run_id}:{label}:{generation.task_family}:"
                f"{generation.task_id}:{generation.item_index}:"
                f"sample-{generation.sample_index}:cap-{generation.sampling_max_tokens}"
            )
        ):
            raise RunnerGateError("Pilot-0 partial evaluation row mismatch")
        values = counts[generation.task_id]
        values["panel_role"] = generation.panel_role
        values["successes"] += int(reward.reward)
        values["trials"] += 1
    by_task_indices: dict[str, set[int]] = defaultdict(set)
    for generation in generations:
        by_task_indices[generation.task_id].add(generation.sample_index)
    expected_indices = set(
        range(sample_index_start, sample_index_start + samples_per_item)
    )
    if any(
        values["trials"] != samples_per_item
        or by_task_indices[task_id] != expected_indices
        for task_id, values in counts.items()
    ):
        raise RunnerGateError("Pilot-0 partial evaluation ended inside an item group")
    return counts, set(by_sample), set(reward_by_sample), set(counts)


def finish_evaluation(
    directory: Path,
    *,
    manifest: DatasetManifest,
    label: str,
    sampler_path: str,
    samples_per_item: int,
    counts: dict[str, dict[str, int]],
    sample_ids: set[str],
    reward_ids: set[str],
    coordinates: dict[str, Any],
) -> dict:
    if sample_ids != reward_ids:
        raise RunnerGateError("Pilot-0 generation/reward exact join failed")
    generation_bytes = (directory / "generations.jsonl").read_bytes()
    reward_bytes = (directory / "rewards.jsonl").read_bytes()
    result = {
        "schema_version": "duraseed-pilot0-evaluation-v1",
        "label": label,
        "manifest_id": manifest.manifest_id,
        "source_split": manifest.split,
        "sampler_path": sampler_path,
        "coordinates": coordinates,
        "samples_per_item": samples_per_item,
        "item_count": manifest.record_count,
        "row_count": len(sample_ids),
        "generation_sha256": sha256_bytes(generation_bytes),
        "reward_sha256": sha256_bytes(reward_bytes),
        "join_sha256": sha256_bytes(canonical_json_bytes(sorted(sample_ids))),
        "item_counts": [
            {
                "task_id": task_id,
                "panel_role": values["panel_role"],
                "successes": values["successes"],
                "trials": values["trials"],
            }
            for task_id, values in sorted(counts.items())
        ],
    }
    atomic_write_bytes(directory / "result.json", canonical_json_bytes(result))
    return result


__all__ = [
    "EvaluationStage",
    "finish_evaluation",
    "load_evaluation_prefix",
    "read_evaluation",
]
