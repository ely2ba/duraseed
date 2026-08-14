"""Durable file and exact-join binding for completed Pilot-0 segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duraseed.pilot0_sources import seed_source_ids
from duraseed.provenance import (
    canonical_json_hash,
    sha256_bytes,
    validate_sha256_id,
)
from duraseed.run_records import GenerationRecord, RewardRecord
from duraseed.runners import RunnerGateError


def _jsonl(path: Path) -> tuple[str, ...]:
    try:
        rows = tuple(line for line in path.read_text().splitlines() if line.strip())
        for row in rows:
            value = json.loads(row)
            if not isinstance(value, dict):
                raise ValueError
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RunnerGateError(f"Pilot-0 evidence is unreadable: {path.name}") from error
    return rows


def segment_coordinates(
    inputs: Any,
    source: Any,
    preflight_sha256: str,
    **values: Any,
) -> dict[str, Any]:
    """Bind a segment to the authorized run and exact seed source manifests."""

    validate_sha256_id(preflight_sha256)
    source_ids = seed_source_ids(source).model_dump(mode="json")
    if source_ids.pop("seed") != source.seed:
        raise RunnerGateError("Pilot-0 seed source identity changed")
    return {
        "run_id": inputs.run_id,
        "project_id": inputs.project_id,
        "preflight_sha256": preflight_sha256,
        "source_bundle_sha256": inputs.source_authentication.bundle_sha256,
        "seed": source.seed,
        "source_manifest_ids": source_ids,
        **values,
    }


def stage_b_segment_coordinates(
    inputs: Any,
    source: Any,
    preflight_sha256: str,
    stage_a: dict[str, Any],
    parent: dict[str, Any],
    *,
    method: str,
    start: int,
    stop: int,
    learning_rate: float,
) -> dict[str, Any]:
    values = segment_coordinates(
        inputs,
        source,
        preflight_sha256,
        kind="stage-b",
        method=method,
        start=start,
        stop=stop,
        origin_sampler_path=stage_a["selected_sampler_path"],
        origin_state_path=stage_a["selected_state_path"],
        parent_sampler_path=parent["sampler_path"],
        parent_state_path=parent["state_path"],
        learning_rate=learning_rate,
    )
    for name in ("matched_target_sha256", "matched_selection_sha256"):
        if name in stage_a:
            values[name] = stage_a[name]
    return values


def _validate_join(directory: Path, expected: dict[str, Any]) -> None:
    generation_path = directory / "generations.jsonl"
    reward_path = directory / "rewards.jsonl"
    if generation_path.exists() != reward_path.exists():
        raise RunnerGateError("Pilot-0 evidence contains one side of an exact join")
    if not generation_path.exists():
        return
    try:
        generations = tuple(
            GenerationRecord.model_validate_json(row) for row in _jsonl(generation_path)
        )
        rewards = tuple(
            RewardRecord.model_validate_json(row) for row in _jsonl(reward_path)
        )
    except ValueError as error:
        raise RunnerGateError("Pilot-0 evidence rows violate their schema") from error
    by_sample = {row.sample_id: row for row in generations}
    reward_by_sample = {row.sample_id: row for row in rewards}
    if (
        len(by_sample) != len(generations)
        or len(reward_by_sample) != len(rewards)
        or set(by_sample) != set(reward_by_sample)
    ):
        raise RunnerGateError("Pilot-0 durable generation/reward join failed")
    if any(
        generation.task_id != reward_by_sample[sample_id].task_id
        or generation.reward != reward_by_sample[sample_id].reward
        or reward_by_sample[sample_id].reward_id != f"reward:{sample_id}"
        for sample_id, generation in by_sample.items()
    ):
        raise RunnerGateError("Pilot-0 durable joined rows disagree")
    manifest_ids = set(expected.get("source_manifest_ids", {}).values())
    for generation in generations:
        if (
            generation.run_id != expected.get("run_id")
            or generation.seed != expected.get("seed")
            or generation.method != expected.get("method")
            or generation.task_manifest_id not in manifest_ids
        ):
            raise RunnerGateError("Pilot-0 durable rows changed run coordinates")
        if generation.purpose == "training" and (
            generation.checkpoint_stage != "stage_a"
            or generation.training_step <= int(expected.get("start", -1))
            or generation.training_step > int(expected.get("stop", -1))
            or generation.origin_sampler_checkpoint_path
            != expected.get("origin_sampler_path")
        ):
            raise RunnerGateError("Pilot-0 training rows changed segment coordinates")
    result_path = directory / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise RunnerGateError("Pilot-0 result evidence is unreadable") from error
        if (
            result.get("generation_sha256")
            != sha256_bytes(generation_path.read_bytes())
            or result.get("reward_sha256") != sha256_bytes(reward_path.read_bytes())
            or result.get("row_count") != len(generations)
        ):
            raise RunnerGateError("Pilot-0 result hashes disagree with durable rows")
        coordinates = result.get("coordinates")
        first = generations[0] if generations else None
        required = {
            "run_id",
            "label",
            "origin_sampler_path",
            "method",
            "checkpoint_stage",
            "training_step",
            "seed",
            "seed_namespace",
            "max_tokens",
            "temperature",
            "top_p",
            "task_contract_sha256",
        }
        try:
            if not isinstance(coordinates, dict) or set(coordinates) != required:
                raise ValueError
            validate_sha256_id(coordinates["task_contract_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise RunnerGateError(
                "Pilot-0 result omitted its full evaluation coordinates"
            ) from error
        if first is None or any(
            (
                row.run_id,
                row.seed,
                row.method,
                row.checkpoint_stage,
                row.training_step,
                row.origin_sampler_checkpoint_path,
                row.sampling_max_tokens,
                row.sampling_temperature,
                row.sampling_top_p,
                row.sampler_checkpoint_path,
                row.task_manifest_id,
                row.source_split,
                row.sample_id,
            )
            != (
                coordinates["run_id"],
                coordinates["seed"],
                coordinates["method"],
                coordinates["checkpoint_stage"],
                coordinates["training_step"],
                coordinates["origin_sampler_path"],
                coordinates["max_tokens"],
                coordinates["temperature"],
                coordinates["top_p"],
                result.get("sampler_path"),
                result.get("manifest_id"),
                result.get("source_split"),
                (
                    f"{coordinates['run_id']}:{coordinates['label']}:"
                    f"{row.task_family}:{row.task_id}:{row.item_index}:"
                    f"sample-{row.sample_index}:cap-{coordinates['max_tokens']}"
                ),
            )
            for row in generations
        ):
            raise RunnerGateError("Pilot-0 evaluation rows changed full coordinates")
        if (
            coordinates["run_id"] != expected.get("run_id")
            or coordinates["seed"] != expected.get("seed")
            or coordinates["method"] != expected.get("method")
            or coordinates["origin_sampler_path"] != expected.get("origin_sampler_path")
            or result.get("manifest_id") not in manifest_ids
            or result.get("sampler_path") != first.sampler_checkpoint_path
            or result.get("sampler_path") != expected.get("sampler_path")
        ):
            raise RunnerGateError("Pilot-0 evaluation is bound to another segment")
        kind = expected.get("kind")
        evaluation_step = (
            0 if kind in {"m0-evidence", "boundary-origin"} else expected.get("stop")
        )
        evaluation_stage = (
            "m0"
            if kind == "m0-evidence"
            else "stage_b"
            if kind == "stage-b"
            else "stage_a"
        )
        if (
            coordinates["training_step"] != evaluation_step
            or coordinates["checkpoint_stage"] != evaluation_stage
        ):
            raise RunnerGateError("Pilot-0 evaluation changed its segment stage")


def evidence_index(directory: Path, expected: dict[str, Any]) -> dict[str, str]:
    for path in directory.rglob("generations.jsonl"):
        _validate_join(path.parent, expected)
    for path in directory.rglob("rewards.jsonl"):
        _validate_join(path.parent, expected)
    for path in directory.rglob("*.jsonl"):
        _jsonl(path)
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != "segment.json"
        and not path.name.startswith(".segment.json.tmp-")
    )
    if not paths:
        raise RunnerGateError("Pilot-0 segment has no durable evidence")
    return {
        path.relative_to(directory).as_posix(): sha256_bytes(path.read_bytes())
        for path in paths
    }


def bind_segment_evidence(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    index = evidence_index(directory, payload)
    return {
        **payload,
        "evidence_file_count": len(index),
        "evidence_sha256": canonical_json_hash(index),
    }


def verify_segment_evidence(directory: Path, payload: dict[str, Any]) -> None:
    index = evidence_index(directory, payload)
    if payload.get("evidence_file_count") != len(index) or payload.get(
        "evidence_sha256"
    ) != canonical_json_hash(index):
        raise RunnerGateError("Pilot-0 completed segment evidence changed")


__all__ = [
    "bind_segment_evidence",
    "evidence_index",
    "segment_coordinates",
    "stage_b_segment_coordinates",
    "verify_segment_evidence",
]
