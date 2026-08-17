"""End-of-action integrity checks for accepted calibration attempts."""

from __future__ import annotations

import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import GenerationRecord, RewardRecord, TrainingMetricRecord
from duraseed.runners import RunnerGateError
from duraseed.training.stage_a_calibration import STAGE_A_LEARNING_RATE_GRIDS
from duraseed.training.teacher_exposure import REPAIR_CHECKPOINT_UPDATES


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid calibration evidence: {path.name}") from error


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RunnerGateError(f"missing calibration stream: {path.name}") from error
    if raw and not raw.endswith(b"\n"):
        raise RunnerGateError(f"calibration stream is not durably terminated: {path}")
    values = []
    try:
        for line in raw.splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError
            values.append(value)
    except (json.JSONDecodeError, ValueError) as error:
        raise RunnerGateError(f"invalid calibration JSONL: {path}") from error
    return values


def _accepted_attempt(arm: Path) -> Path:
    completed = _json(arm / "completed.json")
    if (
        not isinstance(completed, dict)
        or completed.get("arm_id") != arm.name
        or type(completed.get("attempt")) is not int
        or completed["attempt"] < 1
    ):
        raise RunnerGateError(f"invalid completed calibration arm: {arm.name}")
    attempts = sorted(path for path in arm.glob("attempt-*") if path.is_dir())
    expected = [f"attempt-{number:04d}" for number in range(1, len(attempts) + 1)]
    if [path.name for path in attempts] != expected or completed["attempt"] != len(
        attempts
    ):
        raise RunnerGateError(
            f"accepted calibration attempt is not the final contiguous attempt: {arm.name}"
        )
    attempt = attempts[-1]
    state = _json(attempt / "remote-call-state.json")
    if not isinstance(state, dict) or state.get("pending") is not None:
        raise RunnerGateError(f"accepted calibration arm is still pending: {arm.name}")
    return attempt


def _join_paths(
    generation_path: Path,
    reward_path: Path,
    *,
    training_step: int | None = None,
) -> dict[str, int]:
    seen = set()
    count = 0
    try:
        with (
            generation_path.open("rb") as generations,
            reward_path.open("rb") as rewards,
        ):
            for generation_raw, reward_raw in zip_longest(generations, rewards):
                if generation_raw is None or reward_raw is None:
                    raise ValueError
                generation = GenerationRecord.model_validate_json(generation_raw)
                reward = RewardRecord.model_validate_json(reward_raw)
                if (
                    generation.sample_id in seen
                    or generation.sample_id != reward.sample_id
                    or generation.task_id != reward.task_id
                    or generation.reward != reward.reward
                    or (
                        training_step is not None
                        and generation.training_step != training_step
                    )
                ):
                    raise ValueError
                seen.add(generation.sample_id)
                count += 1
    except (OSError, ValueError) as error:
        raise RunnerGateError(
            "accepted calibration generation/reward join differs"
        ) from error
    return {"generation_count": count, "reward_count": count}


def _join(attempt: Path, *, training_step: int | None = None) -> dict[str, int]:
    return _join_paths(
        attempt / "generations.jsonl",
        attempt / "rewards.jsonl",
        training_step=training_step,
    )


def _raw_streams(root: Path) -> list[dict[str, Any]]:
    values = []
    for path in sorted(root.rglob("*.jsonl")):
        digest = hashlib.sha256()
        row_count = 0
        try:
            with path.open("rb") as source:
                for line in source:
                    if not line.endswith(b"\n"):
                        raise RunnerGateError(
                            f"calibration stream is not durably terminated: {path}"
                        )
                    digest.update(line)
                    row_count += 1
        except OSError as error:
            raise RunnerGateError(f"missing calibration stream: {path.name}") from error
        values.append(
            {
                "path": path.relative_to(root).as_posix(),
                "row_count": row_count,
                "sha256": f"sha256:{digest.hexdigest()}",
            }
        )
    return values


def _control_files(root: Path, accepted: list[dict[str, Any]]) -> list[dict[str, str]]:
    values = []
    for row in accepted:
        arm = root / row["arm_id"]
        attempt = arm / row["attempt"]
        for path in (
            arm / "completed.json",
            arm / "coordinate.json",
            attempt / "remote-call-state.json",
        ):
            try:
                digest = sha256_bytes(path.read_bytes())
            except OSError as error:
                raise RunnerGateError(
                    f"accepted calibration control file is missing: {path.name}"
                ) from error
            values.append({"path": path.relative_to(root).as_posix(), "sha256": digest})
    return values


def _teacher_metrics(attempt: Path, expected_steps: int) -> int:
    rows = tuple(
        TrainingMetricRecord.model_validate_json(canonical_json_bytes(row))
        for row in _jsonl(attempt / "metrics.jsonl")
    )
    if tuple(row.training_step for row in rows) != tuple(range(1, expected_steps + 1)):
        raise RunnerGateError("teacher-dose attempt lacks its exact step metrics")
    return len(rows)


def _stage_metrics(attempt: Path, *, screen_only: bool) -> int:
    rows = _jsonl(attempt / "metrics.jsonl")
    boundary = tuple(row for row in rows if row.get("subphase") == "boundary-seed")
    if boundary:
        raise RunnerGateError("direct-M0 Stage-A contains boundary-seed metrics")
    branch_rows = tuple(row for row in rows if "method" in row)
    expected_rates = {
        method: tuple(float(value) for value in rates)
        for method, rates in STAGE_A_LEARNING_RATE_GRIDS.items()
    }
    selected = {"B-S": 0, "B-G": 0}
    for method, rates in expected_rates.items():
        for rate in rates:
            steps = tuple(
                row.get("training_step")
                for row in branch_rows
                if row.get("method") == method
                and float(row.get("learning_rate")) == rate
            )
            if steps == tuple(range(1, 51)):
                selected[method] += 1
            elif steps != tuple(range(1, 11)):
                raise RunnerGateError("Stage-A branch metric schedule differs")
    expected_selected = {"B-S": 0, "B-G": 0} if screen_only else {"B-S": 1, "B-G": 1}
    expected_count = 60 if screen_only else 140
    if (
        selected != expected_selected
        or len(branch_rows) != expected_count
        or len(rows) != expected_count
    ):
        raise RunnerGateError("Stage-A selected continuation schedule differs")
    return len(rows)


def seal_calibration_action(
    action: str,
    root: Path,
    *,
    teacher_updates: int,
    stage_a_screen_only: bool = False,
) -> dict[str, Any]:
    """Validate accepted joins/steps and bind every raw JSONL byte stream."""

    accepted = []
    metric_count = 0
    if action == "teacher-dose":
        arms = tuple(
            path
            for path in sorted(root.iterdir())
            if path.is_dir() and (path / "completed.json").is_file()
        )
        if not arms:
            raise RunnerGateError("teacher-dose action has no completed arms")
        progressive = all(arm.name.startswith("trajectory-seed-") for arm in arms)
        if progressive and tuple(arm.name for arm in arms) != (
            "trajectory-seed-17",
            "trajectory-seed-37",
        ):
            raise RunnerGateError("teacher exposure omits a completed orientation")
        for arm in arms:
            attempt = _accepted_attempt(arm)
            if progressive:
                expected = tuple(
                    value
                    for value in REPAIR_CHECKPOINT_UPDATES
                    if value <= teacher_updates
                )
                observed = tuple(
                    sorted(
                        int(path.name.removeprefix("checkpoint-"))
                        for path in attempt.glob("checkpoint-*")
                        if path.is_dir()
                    )
                )
                if observed != expected:
                    raise RunnerGateError("teacher exposure checkpoint streams differ")
                joined = {"generation_count": 0, "reward_count": 0}
                for updates in observed:
                    point = _join(
                        attempt / f"checkpoint-{updates}", training_step=updates
                    )
                    if point != {"generation_count": 864, "reward_count": 864}:
                        raise RunnerGateError(
                            "teacher exposure checkpoint gate is incomplete"
                        )
                    for key in joined:
                        joined[key] += point[key]
            else:
                joined = _join(attempt)
            if not arm.name.startswith("baseline-"):
                metric_count += _teacher_metrics(attempt, teacher_updates)
            accepted.append({"arm_id": arm.name, "attempt": attempt.name, **joined})
    elif action == "stage-a":
        arm = root / "complete-bounded-stage-a"
        attempt = _accepted_attempt(arm)
        accepted.append({"arm_id": arm.name, "attempt": attempt.name, **_join(attempt)})
        metric_count = _stage_metrics(attempt, screen_only=stage_a_screen_only)
    else:
        raise ValueError("unknown calibration integrity action")
    result = {
        "schema_version": "duraseed-calibration-integrity-v1",
        "action": action,
        "accepted_attempts": accepted,
        "accepted_metric_count": metric_count,
        "control_files": _control_files(root, accepted),
        "raw_streams": _raw_streams(root),
    }
    payload = canonical_json_bytes(result)
    path = root / "integrity.json"
    if path.exists() and path.read_bytes() != payload:
        raise RunnerGateError("calibration integrity seal changed on resume")
    atomic_write_bytes(path, payload)
    return {**result, "artifact_sha256": sha256_bytes(payload)}


def validate_committed_action(
    action: str,
    root: Path,
    artifact: dict[str, Any],
    *,
    teacher_updates: int,
) -> None:
    """Recompute accepted evidence and bind it to the committed action artifact."""

    integrity = seal_calibration_action(action, root, teacher_updates=teacher_updates)
    ttl_path = root / "checkpoint-ttl-audit.json"
    try:
        ttl_hash = sha256_bytes(ttl_path.read_bytes())
    except OSError as error:
        raise RunnerGateError(
            "committed calibration action omits its TTL audit"
        ) from error
    if (
        artifact.get("integrity") != integrity
        or artifact.get("checkpoint_ttl_audit_sha256") != ttl_hash
    ):
        raise RunnerGateError("committed calibration integrity binding differs")


__all__ = ["seal_calibration_action", "validate_committed_action"]
