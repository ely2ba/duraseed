"""Small run metadata and JSONL writers for Tinker experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import StrEnum
import json
import os
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import validate_sha256_id
from duraseed.schemas import VerificationResult


RUN_RECORD_FILE = "run.json"
MethodCode: TypeAlias = Literal["G-U", "G-B", "R-G", "B-S", "B-O", "B-G"]
RunKind: TypeAlias = Literal[
    "pilot_method",
    "engineering_smoke",
    "m0_calibration",
    "stage_a_calibration",
    "stage_b_calibration",
]
TrainingPhase: TypeAlias = Literal["m0", "stage_a", "stage_b"]
CheckpointPhase: TypeAlias = Literal["m0", "stage_a", "stage_b"]
CheckpointStage: TypeAlias = Literal["base", "m0", "stage_a", "stage_b"]


class _JsonlRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty")
    return value


class RunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RunRecord(BaseModel):
    """Identity, usage, and status for one experiment run."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    protocol_version: str
    git_commit: str
    resolved_config_hash: str
    run_kind: RunKind = "pilot_method"
    method: MethodCode | None
    seed: int = Field(ge=0)
    model_id: str = Field(description="Tinker base-checkpoint identity")
    renderer: str | None
    lora_rank: int = Field(gt=0)
    task_manifest_ids: dict[str, str]
    parent_tinker_checkpoint_path: str | None = None
    tinker_session_id: str | None = None
    tinker_training_run_id: str | None = None
    tinker_training_run_ids: tuple[str, ...] = ()
    final_sampler_checkpoint_path: str | None = None
    final_state_checkpoint_path: str | None = None
    status: RunStatus
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    sampled_tokens: int = Field(default=0, ge=0)
    train_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    deviations: list[str] = Field(default_factory=list)
    tinker_sdk_version: str | None = None
    tinker_cookbook_version: str | None = None
    project_id: str | None = None
    authorized_cost_usd: float | None = Field(default=None, ge=0)
    reserved_cost_usd: float | None = Field(default=None, ge=0)
    price_snapshot_id: str | None = None
    service_randomness_contract: str | None = None

    @field_validator(
        "protocol_version",
        "git_commit",
        "resolved_config_hash",
        "model_id",
    )
    @classmethod
    def required_text_is_nonempty(cls, value: str) -> str:
        return _nonempty(value)

    @field_validator(
        "tinker_sdk_version",
        "tinker_cookbook_version",
        "project_id",
        "price_snapshot_id",
        "service_randomness_contract",
        "tinker_training_run_id",
    )
    @classmethod
    def optional_text_is_nonempty(cls, value: str | None) -> str | None:
        return _nonempty(value) if value is not None else None

    @field_validator("tinker_training_run_ids")
    @classmethod
    def training_run_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("tinker_training_run_ids must not contain empty IDs")
        if len(value) != len(set(value)):
            raise ValueError("tinker_training_run_ids must be unique")
        return value

    @field_validator("started_at", "updated_at", "finished_at")
    @classmethod
    def timestamps_are_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamps must include a timezone")
        return value

    @field_validator("task_manifest_ids")
    @classmethod
    def manifest_ids_are_canonical(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(not key.strip() for key in value):
            raise ValueError("task_manifest_ids must contain non-empty names and IDs")
        for manifest_id in value.values():
            validate_sha256_id(manifest_id)
        return value

    @model_validator(mode="after")
    def run_kind_matches_method_and_cost_reservation(self) -> "RunRecord":
        if self.run_kind == "pilot_method" and self.method is None:
            raise ValueError("pilot_method runs require a method")
        if self.run_kind != "pilot_method" and self.method is not None:
            raise ValueError(f"{self.run_kind} runs require method=None")
        if (
            self.reserved_cost_usd is not None
            and self.authorized_cost_usd is not None
            and self.reserved_cost_usd > self.authorized_cost_usd
        ):
            raise ValueError("reserved_cost_usd must not exceed authorized_cost_usd")
        return self


class CheckpointRecord(_JsonlRecord):
    """One retained M0, Stage-A, or Stage-B Tinker checkpoint pair."""

    phase: CheckpointPhase
    training_step: int = Field(ge=0)
    sampler_checkpoint_path: str
    state_checkpoint_path: str
    selected_for_stage_b: bool = False
    m0_state_path: str | None = None
    parent_state_path: str | None = None
    optimizer_inheritance: (
        Literal["fresh", "full_state", "weights_only_fresh"] | None
    ) = None
    ttl_seconds: int | None = Field(default=None, ge=0)
    selected_origin: bool = False
    learning_rate: float | None = Field(default=None, gt=0)
    tinker_training_run_id: str | None = None

    @field_validator(
        "sampler_checkpoint_path",
        "state_checkpoint_path",
        "m0_state_path",
        "parent_state_path",
        "tinker_training_run_id",
    )
    @classmethod
    def checkpoint_paths_are_nonempty(cls, value: str | None) -> str | None:
        return _nonempty(value) if value is not None else None

    @model_validator(mode="after")
    def only_stage_a_can_be_selected(self) -> "CheckpointRecord":
        if self.selected_for_stage_b and self.phase != "stage_a":
            raise ValueError("only a Stage-A checkpoint can be selected for Stage B")
        return self


class StageBEvaluationRecord(_JsonlRecord):
    """One point on the retained Stage-B absolute and gain curves."""

    stage_b_step: int = Field(ge=0)
    sampler_checkpoint_path: str
    score: float = Field(ge=0.0, le=1.0)
    gain_from_zero_shot: float = Field(ge=-1.0, le=1.0)
    is_final: bool = False

    @field_validator("sampler_checkpoint_path")
    @classmethod
    def sampler_path_is_nonempty(cls, value: str) -> str:
        return _nonempty(value)

    @model_validator(mode="after")
    def zero_shot_point_has_zero_gain(self) -> "StageBEvaluationRecord":
        if self.stage_b_step == 0 and self.gain_from_zero_shot != 0.0:
            raise ValueError("the zero-shot Stage-B point must have zero gain")
        return self


class TrainingMetricRecord(_JsonlRecord):
    """Small per-step training history row for ``metrics.jsonl``."""

    phase: TrainingPhase
    training_step: int = Field(ge=0)
    metrics: dict[str, float] = Field(min_length=1)

    @field_validator("metrics")
    @classmethod
    def metric_names_are_nonempty(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not name.strip() for name in value):
            raise ValueError("metric names must not be empty")
        return value


class GenerationRecord(_JsonlRecord):
    """One raw output with explicit sample coordinates.

    ``sampling_seed=None`` means the service did not expose or control a
    request-level seed; the run metadata must then state that randomness
    contract rather than inventing determinism.
    """

    sample_id: str
    sample_index: int = Field(ge=0)
    sampling_seed: int | None = Field(default=None, ge=0)
    purpose: Literal["training", "evaluation"]
    checkpoint_stage: CheckpointStage
    training_step: int = Field(ge=0)
    sampler_checkpoint_path: str
    task_manifest_id: str
    task_id: str
    task_family: Literal["format", "tces", "maps"]
    source_split: str
    prompt_text: str
    completion_text: str
    prompt_tokens: int = Field(ge=0)
    sampled_tokens: int = Field(ge=0)
    sampling_temperature: float | None = Field(default=None, ge=0)
    sampling_top_p: float | None = Field(default=None, gt=0, le=1)
    sampling_max_tokens: int | None = Field(default=None, gt=0)
    run_id: str | None = None
    method: MethodCode | None = None
    seed: int | None = Field(default=None, ge=0)
    origin_sampler_checkpoint_path: str | None = None
    item_index: int | None = Field(default=None, ge=0)
    assigned_family_id: str | None = None
    family_id: str | None = None
    panel_role: str | None = None
    completion_token_ids: tuple[int, ...] | None = None
    completion_logprobs: tuple[float | None, ...] | None = None
    stop_reason: str | None = None
    renderer_termination: str | None = None
    reward: float | None = None
    advantage: float | None = None

    @field_validator(
        "sample_id",
        "sampler_checkpoint_path",
        "task_id",
        "source_split",
        "prompt_text",
        "run_id",
        "origin_sampler_checkpoint_path",
        "assigned_family_id",
        "family_id",
        "panel_role",
        "stop_reason",
        "renderer_termination",
    )
    @classmethod
    def generation_identity_is_nonempty(cls, value: str | None) -> str | None:
        return _nonempty(value) if value is not None else None

    @field_validator("task_manifest_id")
    @classmethod
    def task_manifest_id_is_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)

    @field_validator("completion_token_ids")
    @classmethod
    def completion_tokens_are_nonnegative(
        cls, value: tuple[int, ...] | None
    ) -> tuple[int, ...] | None:
        if value is not None and any(token_id < 0 for token_id in value):
            raise ValueError("completion_token_ids must be nonnegative")
        return value

    @model_validator(mode="after")
    def completion_arrays_match_sampled_tokens(self) -> "GenerationRecord":
        if self.completion_token_ids is not None:
            if len(self.completion_token_ids) != self.sampled_tokens:
                raise ValueError(
                    "completion_token_ids length must match sampled_tokens"
                )
            if self.completion_logprobs is not None and len(
                self.completion_logprobs
            ) != len(self.completion_token_ids):
                raise ValueError(
                    "completion_logprobs length must match completion_token_ids"
                )
        elif self.completion_logprobs is not None:
            raise ValueError(
                "completion_logprobs require completion_token_ids to be supplied"
            )
        return self


class RewardRecord(_JsonlRecord):
    """One exact reward linked to a raw generated sample."""

    reward_id: str
    sample_id: str
    task_id: str
    reward: Literal[0.0, 1.0]
    exact_verification: VerificationResult

    @field_validator("reward_id", "sample_id", "task_id")
    @classmethod
    def reward_identity_is_nonempty(cls, value: str) -> str:
        return _nonempty(value)

    @model_validator(mode="after")
    def reward_matches_verification(self) -> "RewardRecord":
        if self.reward != self.exact_verification.reward:
            raise ValueError("reward must match exact_verification")
        return self


def create_run_directory(root: str | os.PathLike[str], run_id: str) -> Path:
    """Create a unique run directory, failing if the run ID already exists."""

    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    run_directory = root_path / run_id
    run_directory.mkdir()
    return run_directory


def write_run_record(run_directory: str | os.PathLike[str], record: RunRecord) -> Path:
    """Atomically create or replace ``run.json`` in an existing run directory."""

    if not isinstance(record, RunRecord):
        raise TypeError("record must be a RunRecord")
    directory = Path(run_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {directory}")
    payload = (
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return atomic_write_bytes(directory / RUN_RECORD_FILE, payload)


def read_run_record(run_directory: str | os.PathLike[str]) -> RunRecord:
    """Load and validate ``run.json`` from a run directory."""

    return RunRecord.model_validate_json(
        (Path(run_directory) / RUN_RECORD_FILE).read_bytes()
    )


def _jsonl_line(record: Mapping[str, Any] | BaseModel) -> bytes:
    value = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def write_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[Mapping[str, Any] | BaseModel],
) -> Path:
    """Atomically replace a JSONL stream with the supplied records."""

    return atomic_write_bytes(path, b"".join(_jsonl_line(record) for record in records))


def append_jsonl(
    path: str | os.PathLike[str], record: Mapping[str, Any] | BaseModel
) -> Path:
    """Append one compact JSON object to a run stream."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as handle:
        handle.write(_jsonl_line(record))
        handle.flush()
        os.fsync(handle.fileno())
    return destination


__all__ = [
    "CheckpointRecord",
    "GenerationRecord",
    "MethodCode",
    "RUN_RECORD_FILE",
    "RewardRecord",
    "RunRecord",
    "RunStatus",
    "StageBEvaluationRecord",
    "TrainingMetricRecord",
    "append_jsonl",
    "create_run_directory",
    "read_run_record",
    "write_jsonl",
    "write_run_record",
]
