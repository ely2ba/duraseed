"""Local replay and stream seal for one completed capability-dose attempt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duraseed.provenance import sha256_bytes
from duraseed.run_records import GenerationRecord, RewardRecord, TrainingMetricRecord
from duraseed.runners import RunnerGateError
from duraseed.runners.capability_dose_evidence import panel_evidence
from duraseed.runtime import SampleObservation
from duraseed.training.capability_dose import decide_dose
from duraseed.training.capability_dose_evidence import CapabilityDoseLiveEvidence


def _jsonl(path: Path, model: type) -> tuple[Any, ...]:
    try:
        return tuple(
            model.model_validate_json(row) for row in path.read_bytes().splitlines()
        )
    except (OSError, ValueError) as error:
        raise RunnerGateError(f"invalid capability-dose stream: {path.name}") from error


def _metrics(path: Path) -> tuple[TrainingMetricRecord, ...]:
    rows = []
    try:
        for line in path.read_text().splitlines():
            value = json.loads(line)
            if (
                value.pop("method", None) != "B-S"
                or value.pop("learning_rate", None) != 1e-4
            ):
                raise RunnerGateError("capability-dose metric coordinate differs")
            rows.append(TrainingMetricRecord.model_validate(value))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RunnerGateError("invalid capability-dose metrics") from error
    return tuple(rows)


def _observation(
    generation: GenerationRecord, reward: RewardRecord
) -> SampleObservation:
    tokens = generation.completion_token_ids or ()
    logprobs = tuple(
        value for value in (generation.completion_logprobs or ()) if value is not None
    )
    if len(tokens) != len(logprobs):
        raise RunnerGateError("capability-dose raw sample lost log-probabilities")
    return SampleObservation(generation, reward, None, tokens, logprobs)


def _panel(
    rows: tuple[SampleObservation, ...], label: str, role: str
) -> tuple[SampleObservation, ...]:
    marker = f":{label}:"
    result = tuple(
        row
        for row in rows
        if marker in row.generation.sample_id and row.generation.panel_role == role
    )
    if not result:
        raise RunnerGateError(f"capability-dose raw evidence omits {label}/{role}")
    return result


def validate_capability_dose_attempt(
    attempt: Path, evidence: CapabilityDoseLiveEvidence
) -> dict[str, Any]:
    """Recompute every panel summary and bind the exact schedule/checkpoint."""

    generations = _jsonl(attempt / "generations.jsonl", GenerationRecord)
    rewards = _jsonl(attempt / "rewards.jsonl", RewardRecord)
    by_sample = {row.sample_id: row for row in rewards}
    if (
        len(by_sample) != len(rewards)
        or len(generations) != len(rewards)
        or {row.sample_id for row in generations} != set(by_sample)
    ):
        raise RunnerGateError("capability-dose generation/reward join differs")
    observations = tuple(
        _observation(row, by_sample[row.sample_id]) for row in generations
    )
    origin_target = _panel(observations, "origin-target", "targeted")
    origin_sentinel = _panel(observations, "origin-sentinel", "sentinel")
    expected_samples = len(origin_target) + len(origin_sentinel)
    for evaluation in evidence.evaluations:
        prefix = f"dose-{evaluation.phase}-step-{evaluation.update}"
        target = _panel(observations, f"{prefix}-target", "targeted")
        sentinel = (
            _panel(observations, f"{prefix}-sentinel", "sentinel")
            if evaluation.phase == "cadence"
            else None
        )
        if panel_evidence(origin_target, target) != evaluation.target or (
            sentinel is not None
            and panel_evidence(origin_sentinel, sentinel) != evaluation.sentinel
        ):
            raise RunnerGateError("capability-dose reduced panel differs from raw data")
        expected_samples += len(target) + (len(sentinel) if sentinel is not None else 0)
    metrics = _metrics(attempt / "metrics.jsonl")
    steps = tuple(row.training_step for row in metrics)
    decision = decide_dose(evidence.evaluations)
    checkpoint_rows = tuple(
        json.loads(line)
        for line in (attempt / "checkpoints.jsonl").read_text().splitlines()
    )
    ttl = json.loads((attempt / "checkpoint-ttl-audit.json").read_bytes())
    journal = json.loads((attempt / "remote-call-state.json").read_bytes())
    ttl_rows = ttl.get("rows") if isinstance(ttl, dict) else None
    paths = {
        evidence.retained_sampler_checkpoint_path,
        evidence.retained_state_checkpoint_path,
    }
    if (
        expected_samples != len(generations)
        or steps != tuple(range(1, evidence.decision.update + 1))
        or metrics != evidence.metrics
        or decision != evidence.decision
        or len(checkpoint_rows) != 1
        or checkpoint_rows[0].get("step") != evidence.decision.update
        or checkpoint_rows[0].get("method") != "B-S"
        or checkpoint_rows[0].get("learning_rate") != 1e-4
        or checkpoint_rows[0].get("selected_for_stage_b") is not True
        or {checkpoint_rows[0].get("sampler"), checkpoint_rows[0].get("state")} != paths
        or not isinstance(ttl_rows, list)
        or len(ttl_rows) != 2
        or {row.get("path") for row in ttl_rows if isinstance(row, dict)} != paths
        or any(row.get("ttl_seconds") != 604800 for row in ttl_rows)
        or journal.get("pending") is not None
    ):
        raise RunnerGateError("capability-dose schedule or checkpoint lineage differs")
    return {
        "schema_version": "duraseed-capability-dose-integrity-v1",
        "generation_count": len(generations),
        "reward_count": len(rewards),
        "metric_count": len(metrics),
        "generations_sha256": sha256_bytes(
            (attempt / "generations.jsonl").read_bytes()
        ),
        "rewards_sha256": sha256_bytes((attempt / "rewards.jsonl").read_bytes()),
        "metrics_sha256": sha256_bytes((attempt / "metrics.jsonl").read_bytes()),
        "checkpoint_ttl_audit_sha256": sha256_bytes(
            (attempt / "checkpoint-ttl-audit.json").read_bytes()
        ),
    }


__all__ = ["validate_capability_dose_attempt"]
