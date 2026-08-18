"""Counted scientific failures for an observed B-G update-health collapse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes
from duraseed.runners import RunnerGateError


UPDATE_HEALTH_FILE = "update-health-failure.json"


@dataclass(frozen=True, slots=True)
class StageAUpdateHealthFailureEvidence:
    method: Literal["B-G"]
    learning_rate: float
    training_step: int
    phase: Literal["screen", "continuation"]
    reason: Literal["zero_mixed_group", "nonfinite_training_metric"]
    last_valid_metric_step: int
    optimizer_update_completed: bool
    group_count: int
    rollout_sample_count: int
    mixed_group_count: int
    all_zero_group_count: int
    all_one_group_count: int
    metric_name: str | None = None
    schema_version: Literal["duraseed-stage-a-update-health-failure-v1"] = (
        "duraseed-stage-a-update-health-failure-v1"
    )

    def __post_init__(self) -> None:
        counts = (
            self.group_count,
            self.rollout_sample_count,
            self.mixed_group_count,
            self.all_zero_group_count,
            self.all_one_group_count,
        )
        if (
            self.method != "B-G"
            or self.learning_rate != 1e-5
            or type(self.training_step) is not int
            or not 1 <= self.training_step <= 50
            or self.phase != ("screen" if self.training_step <= 10 else "continuation")
            or self.last_valid_metric_step != self.training_step - 1
            or type(self.optimizer_update_completed) is not bool
            or any(type(value) is not int or value < 0 for value in counts)
            or self.group_count != 16
            or self.rollout_sample_count != 128
            or self.mixed_group_count
            + self.all_zero_group_count
            + self.all_one_group_count
            != self.group_count
        ):
            raise ValueError("invalid Stage-A update-health failure evidence")
        zero_mixed = self.reason == "zero_mixed_group"
        nonfinite = self.reason == "nonfinite_training_metric"
        if (
            not (zero_mixed or nonfinite)
            or zero_mixed
            and (
                self.mixed_group_count != 0
                or self.optimizer_update_completed
                or self.metric_name is not None
            )
            or nonfinite
            and (
                self.mixed_group_count < 1
                or not self.optimizer_update_completed
                or not isinstance(self.metric_name, str)
                or not self.metric_name.strip()
            )
        ):
            raise ValueError("incoherent Stage-A update-health failure reason")


class StageAUpdateHealthFailure(Exception):
    """A measured protocol failure, distinct from a remote interruption."""

    def __init__(self, evidence: StageAUpdateHealthFailureEvidence) -> None:
        super().__init__(evidence.reason)
        self.evidence = evidence


def parse_stage_a_update_health_failure(
    value: Any,
) -> StageAUpdateHealthFailureEvidence:
    if isinstance(value, StageAUpdateHealthFailureEvidence):
        return value
    if not isinstance(value, dict):
        raise RunnerGateError("Stage-A update-health evidence is not an object")
    try:
        return StageAUpdateHealthFailureEvidence(**value)
    except (TypeError, ValueError) as error:
        raise RunnerGateError("Stage-A update-health evidence is invalid") from error


def write_stage_a_update_health_failure(
    directory: Path, evidence: StageAUpdateHealthFailureEvidence
) -> None:
    path = directory / UPDATE_HEALTH_FILE
    payload = canonical_json_bytes(evidence)
    if path.exists() and path.read_bytes() != payload:
        raise RunnerGateError("Stage-A update-health failure evidence changed")
    atomic_write_bytes(path, payload)


__all__ = [
    "StageAUpdateHealthFailure",
    "StageAUpdateHealthFailureEvidence",
    "UPDATE_HEALTH_FILE",
    "parse_stage_a_update_health_failure",
    "write_stage_a_update_health_failure",
]
