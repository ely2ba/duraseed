"""Exact partial-run checks for a counted Stage-A update-health failure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duraseed.runners import RunnerGateError
from duraseed.training.stage_a_update_health import (
    UPDATE_HEALTH_FILE,
    StageAUpdateHealthFailureEvidence,
    parse_stage_a_update_health_failure,
)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("invalid Stage-A update-health artifact") from error


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_bytes().splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("invalid Stage-A update-health stream") from error


def validate_update_health_attempt(
    attempt: Path,
    rows: list[dict[str, Any]],
    failure: StageAUpdateHealthFailureEvidence,
) -> int:
    """Validate the exact partial metric schedule and step-k rollout groups."""

    persisted = parse_stage_a_update_health_failure(_json(attempt / UPDATE_HEALTH_FILE))
    if persisted != failure:
        raise RunnerGateError("Stage-A update-health artifact differs")
    expected = {
        "B-S": tuple(range(1, (10 if failure.training_step <= 10 else 50) + 1)),
        "B-G": tuple(range(1, failure.training_step)),
    }
    for method, steps in expected.items():
        observed = tuple(
            row.get("training_step")
            for row in rows
            if row.get("method") == method
            and row.get("learning_rate") == (1e-4 if method == "B-S" else 1e-5)
        )
        if observed != steps:
            raise RunnerGateError("Stage-A update-health metric schedule differs")
    if len(rows) != sum(len(steps) for steps in expected.values()):
        raise RunnerGateError("Stage-A update-health metric schedule differs")

    generations = tuple(
        row
        for row in _jsonl(attempt / "generations.jsonl")
        if row.get("purpose") == "training"
        and row.get("method") == "B-G"
        and row.get("training_step") == failure.training_step
    )
    rewards = {
        row.get("sample_id"): row
        for row in _jsonl(attempt / "rewards.jsonl")
        if isinstance(row.get("sample_id"), str)
    }
    groups: dict[tuple[object, object], list[float]] = {}
    try:
        for generation in generations:
            key = (generation["task_id"], generation["item_index"])
            groups.setdefault(key, []).append(
                float(rewards[generation["sample_id"]]["reward"])
            )
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError("Stage-A update-health raw evidence differs") from error
    if (
        len(generations) != failure.rollout_sample_count
        or len(groups) != failure.group_count
        or any(len(values) != 8 for values in groups.values())
        or any(
            value not in {0.0, 1.0} for values in groups.values() for value in values
        )
    ):
        raise RunnerGateError("Stage-A update-health rollout groups differ")
    all_zero = sum(all(value == 0 for value in values) for values in groups.values())
    all_one = sum(all(value == 1 for value in values) for values in groups.values())
    mixed = len(groups) - all_zero - all_one
    if (mixed, all_zero, all_one) != (
        failure.mixed_group_count,
        failure.all_zero_group_count,
        failure.all_one_group_count,
    ):
        raise RunnerGateError("Stage-A update-health group counts differ")
    return len(rows)


def update_health_checkpoint_paths(
    rows: list[dict[str, Any]],
    failure: StageAUpdateHealthFailureEvidence,
) -> tuple[str, ...]:
    """Return only the candidates that can exist before the counted stop."""

    candidates = tuple(
        row["sampler"]
        for row in rows
        if "method" in row and isinstance(row.get("sampler"), str)
    )
    expected = {("B-S", 1e-4, 10)}
    if failure.training_step > 10:
        expected |= {("B-G", 1e-5, 10), ("B-S", 1e-4, 50)}
    try:
        observed = {
            (row["method"], float(row["learning_rate"]), int(row["step"]))
            for row in rows
            if "method" in row and isinstance(row.get("sampler"), str)
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError(
            "Stage-A update-health checkpoint lineage differs"
        ) from error
    if (
        observed != expected
        or len(candidates) != len(expected)
        or len(set(candidates)) != len(candidates)
    ):
        raise RunnerGateError("Stage-A update-health checkpoint lineage differs")
    return candidates


__all__ = ["update_health_checkpoint_paths", "validate_update_health_attempt"]
