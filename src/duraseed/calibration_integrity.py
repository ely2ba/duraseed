"""End-of-action integrity checks for accepted calibration attempts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import GenerationRecord, RewardRecord, TrainingMetricRecord
from duraseed.runners import RunnerGateError
from duraseed.training.stage_a_calibration import STAGE_A_LEARNING_RATE_GRIDS


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


def _join(attempt: Path) -> dict[str, int]:
    generations = tuple(
        GenerationRecord.model_validate(row)
        for row in _jsonl(attempt / "generations.jsonl")
    )
    rewards = tuple(
        RewardRecord.model_validate(row) for row in _jsonl(attempt / "rewards.jsonl")
    )
    by_generation = {row.sample_id: row for row in generations}
    by_reward = {row.sample_id: row for row in rewards}
    if (
        len(by_generation) != len(generations)
        or len(by_reward) != len(rewards)
        or set(by_generation) != set(by_reward)
        or any(
            by_generation[key].task_id != by_reward[key].task_id
            or by_generation[key].reward != by_reward[key].reward
            for key in by_generation
        )
    ):
        raise RunnerGateError("accepted calibration generation/reward join differs")
    return {"generation_count": len(generations), "reward_count": len(rewards)}


def _raw_streams(root: Path) -> list[dict[str, Any]]:
    values = []
    for path in sorted(root.rglob("*.jsonl")):
        rows = _jsonl(path)
        values.append(
            {
                "path": path.relative_to(root).as_posix(),
                "row_count": len(rows),
                "sha256": sha256_bytes(path.read_bytes()),
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
        TrainingMetricRecord.model_validate(row)
        for row in _jsonl(attempt / "metrics.jsonl")
    )
    if expected_steps != 16 or tuple(row.training_step for row in rows) != tuple(
        range(1, 17)
    ):
        raise RunnerGateError("teacher-dose attempt lacks exact 16-step metrics")
    return len(rows)


def _stage_metrics(attempt: Path, expected_boundary_steps: int) -> int:
    rows = _jsonl(attempt / "metrics.jsonl")
    boundary = tuple(row for row in rows if row.get("subphase") == "boundary-seed")
    if expected_boundary_steps != 16 or tuple(
        row.get("step") for row in boundary
    ) != tuple(range(1, 17)):
        raise RunnerGateError("Stage-A origin lacks exact 16-step metrics")
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
    if selected != {"B-S": 1, "B-G": 1} or len(branch_rows) != 156 or len(rows) != 172:
        raise RunnerGateError("Stage-A selected continuation schedule differs")
    return len(rows)


def seal_calibration_action(
    action: str, root: Path, *, teacher_updates: int
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
        for arm in arms:
            attempt = _accepted_attempt(arm)
            joined = _join(attempt)
            if not arm.name.startswith("baseline-"):
                metric_count += _teacher_metrics(attempt, teacher_updates)
            accepted.append({"arm_id": arm.name, "attempt": attempt.name, **joined})
    elif action == "stage-a":
        arm = root / "complete-bounded-stage-a"
        attempt = _accepted_attempt(arm)
        accepted.append({"arm_id": arm.name, "attempt": attempt.name, **_join(attempt)})
        metric_count = _stage_metrics(attempt, teacher_updates)
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
