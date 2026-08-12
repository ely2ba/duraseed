"""Verified text sources for future Tinker supervised updates.

This module stops at verified text.  A later Tinker integration may convert a
``VerifiedSourceRecord`` to a Tinker Datum, but training-runtime, tokenization,
and optimizer concerns do not belong in the scientific data layer.
"""

from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from typing import Iterable, Literal, TypeAlias

from pydantic import ConfigDict, Field, field_validator, model_validator

from duraseed.data.format import FormatManifest, FormatRecord
from duraseed.data.manifests import (
    DatasetManifest,
    MAPSTaskManifestRecord,
    TCESTaskManifestRecord,
)
from duraseed.provenance import validate_sha256_id
from duraseed.schemas import MAPSTask, StrictModel, TCESTask, VerificationResult
from duraseed.tasks.maps.generator import render_prompt as render_maps_prompt
from duraseed.tasks.tces.generator import render_prompt as render_tces_prompt
from duraseed.training.reward import verify_task_completion


class SourceRecordError(ValueError):
    """Raised when text cannot be authenticated and exactly verified."""


class SourceKind(StrEnum):
    """Allowed origins for supervised examples."""

    TASK_AGNOSTIC_FORMAT = "task_agnostic_format"
    SOLVER_TEACHER = "solver_teacher"
    CURRENT_POLICY_VERIFIED = "current_policy_verified"


TaskFamily: TypeAlias = Literal["format", "tces", "maps"]
SourceSplit: TypeAlias = Literal[
    "format_train",
    "a_seed_train",
    "a_rl_train",
    "b_train",
]


class VerifiedSourceRecord(StrictModel):
    """One minimal, manifest-backed source example for later Tinker conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    prompt_text: str = Field(min_length=1)
    verified_completion_text: str = Field(min_length=1)
    task_id: str
    task_family: TaskFamily
    source_split: SourceSplit
    source_kind: SourceKind
    strategy_family_id: str | None
    exact_verification: VerificationResult | None
    source_manifest_id: str

    @field_validator("task_id", "source_manifest_id")
    @classmethod
    def identities_are_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)

    @field_validator("strategy_family_id")
    @classmethod
    def strategy_family_is_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("strategy_family_id must be nonempty when present")
        return value

    @model_validator(mode="after")
    def source_and_verification_are_coherent(self) -> "VerifiedSourceRecord":
        if self.source_kind is SourceKind.TASK_AGNOSTIC_FORMAT:
            if self.task_family != "format" or self.source_split != "format_train":
                raise ValueError("format warm-up sources require format_train")
            if (
                self.exact_verification is not None
                or self.strategy_family_id is not None
            ):
                raise ValueError("task-agnostic format sources have no task verifier")
            return self

        if self.task_family == "format":
            raise ValueError("task sources must use the TCES or MAPS family")
        if self.exact_verification is None:
            raise ValueError("task sources require an exact verification result")
        if self.exact_verification.reward != 1.0:
            raise ValueError("supervised task sources must pass the exact verifier")
        if self.strategy_family_id != self.exact_verification.strategy_family_id:
            raise ValueError("strategy family must match exact verification")
        if self.source_kind is SourceKind.CURRENT_POLICY_VERIFIED and (
            self.task_family != "tces" or self.source_split != "a_rl_train"
        ):
            raise ValueError("current-policy sources require TCES a_rl_train data")
        if self.task_family == "maps" and self.source_split != "b_train":
            raise ValueError("MAPS supervised sources require b_train")
        if self.task_family == "tces" and self.source_split not in {
            "a_seed_train",
            "a_rl_train",
        }:
            raise ValueError("TCES supervised sources require a training split")
        return self


TaskManifestRecord: TypeAlias = TCESTaskManifestRecord | MAPSTaskManifestRecord


def _render_task_prompt(task: TCESTask | MAPSTask) -> str:
    return (
        render_tces_prompt(task)
        if isinstance(task, TCESTask)
        else render_maps_prompt(task)
    )


def _authenticated_task(
    source_manifest: DatasetManifest,
    source_record: TaskManifestRecord,
    *,
    task_family: Literal["tces", "maps"],
    source_split: Literal["a_seed_train", "a_rl_train", "b_train"],
) -> TaskManifestRecord:
    if not isinstance(source_manifest, DatasetManifest):
        raise TypeError("source_manifest must be a DatasetManifest")
    expected_type = (
        TCESTaskManifestRecord if task_family == "tces" else MAPSTaskManifestRecord
    )
    if not isinstance(source_record, expected_type):
        raise SourceRecordError(f"{task_family.upper()} source record required")
    if (
        source_manifest.task_family != task_family
        or source_manifest.split != source_split
    ):
        raise SourceRecordError(f"source manifest must be {task_family} {source_split}")
    stored = next(
        (
            record
            for record in source_manifest.records
            if record.task_id == source_record.task_id
        ),
        None,
    )
    if stored != source_record:
        raise SourceRecordError("source record is not present in its manifest")
    return source_record


def _verified_task_record(
    source_manifest: DatasetManifest,
    source_record: TaskManifestRecord,
    completion: str,
    *,
    source_kind: SourceKind,
    task_family: Literal["tces", "maps"],
    source_split: Literal["a_seed_train", "a_rl_train", "b_train"],
) -> VerifiedSourceRecord:
    if not isinstance(completion, str) or not completion:
        raise SourceRecordError("completion must be nonempty text")
    record = _authenticated_task(
        source_manifest,
        source_record,
        task_family=task_family,
        source_split=source_split,
    )
    task = record.to_task()
    verification = verify_task_completion(completion, task)
    if verification.reward != 1.0 or verification.strategy_family_id is None:
        raise SourceRecordError("completion must pass the authoritative exact verifier")
    return VerifiedSourceRecord(
        prompt_text=_render_task_prompt(task),
        verified_completion_text=completion,
        task_id=record.task_id,
        task_family=task_family,
        source_split=source_split,
        source_kind=source_kind,
        strategy_family_id=verification.strategy_family_id,
        exact_verification=verification,
        source_manifest_id=source_manifest.manifest_id,
    )


def build_format_warmup_record(
    *,
    source_manifest: FormatManifest,
    source_record: FormatRecord,
) -> VerifiedSourceRecord:
    """Authenticate one wrapper-only example from the separate format manifest."""

    if not isinstance(source_manifest, FormatManifest):
        raise TypeError("source_manifest must be a FormatManifest")
    if not isinstance(source_record, FormatRecord):
        raise TypeError("source_record must be a FormatRecord")
    if source_manifest.split != "format_train":
        raise SourceRecordError("format warm-up sources require format_train")
    if source_record not in source_manifest.records:
        raise SourceRecordError("format source record is not present in its manifest")
    return VerifiedSourceRecord(
        prompt_text=source_record.prompt_text,
        verified_completion_text=source_record.verified_completion_text,
        task_id=source_record.record_id,
        task_family="format",
        source_split=source_manifest.split,
        source_kind=SourceKind.TASK_AGNOSTIC_FORMAT,
        strategy_family_id=None,
        exact_verification=None,
        source_manifest_id=source_manifest.manifest_id,
    )


def build_solver_teacher_record(
    *,
    source_manifest: DatasetManifest,
    source_record: TCESTaskManifestRecord,
    completion: str,
) -> VerifiedSourceRecord:
    """Build one exact TCES solver-teacher example for seed or static SFT."""

    if not isinstance(source_manifest, DatasetManifest):
        raise TypeError("source_manifest must be a DatasetManifest")
    if source_manifest.split not in {"a_seed_train", "a_rl_train"}:
        raise SourceRecordError("TCES teacher source must use a training split")
    source_split: Literal["a_seed_train", "a_rl_train"] = source_manifest.split
    return _verified_task_record(
        source_manifest,
        source_record,
        completion,
        source_kind=SourceKind.SOLVER_TEACHER,
        task_family="tces",
        source_split=source_split,
    )


def build_teacher_dose_records(
    *,
    source_manifest: DatasetManifest,
    solver_completions: Iterable[tuple[TCESTaskManifestRecord, str]],
    selected_families: Iterable[str],
    demonstrations_per_family: int,
) -> tuple[VerifiedSourceRecord, ...]:
    """Select exactly one deterministic verified-teacher dose per family.

    Candidate order is irrelevant.  Families are returned in canonical order and
    examples within each family follow manifest item order.  Each task can count
    as at most one demonstration, regardless of equivalent solver renderings.
    """

    families = tuple(selected_families)
    if not families or any(
        not isinstance(family, str) or not family for family in families
    ):
        raise ValueError("selected_families must contain nonempty strings")
    if len(families) != len(set(families)):
        raise ValueError("selected_families must be unique")
    families = tuple(sorted(families))
    if (
        isinstance(demonstrations_per_family, bool)
        or not isinstance(demonstrations_per_family, int)
        or demonstrations_per_family < 1
    ):
        raise ValueError("demonstrations_per_family must be a positive integer")

    selected = set(families)
    candidates: list[tuple[str, int, str, str, str, VerifiedSourceRecord]] = []
    for source_record, completion in solver_completions:
        built = build_solver_teacher_record(
            source_manifest=source_manifest,
            source_record=source_record,
            completion=completion,
        )
        family = built.strategy_family_id
        assert family is not None
        if family != source_record.intended_family:
            raise SourceRecordError(
                "teacher completion family differs from manifest intended_family"
            )
        if family not in selected:
            continue
        assert built.exact_verification is not None
        canonical = built.exact_verification.canonical_expression
        assert canonical is not None
        candidates.append(
            (
                family,
                source_record.item_index,
                built.task_id,
                canonical,
                built.verified_completion_text,
                built,
            )
        )

    by_family: dict[str, list[VerifiedSourceRecord]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for family, _, task_id, canonical, _, built in sorted(candidates):
        identity = (family, task_id)
        if identity in seen:
            continue
        seen.add(identity)
        by_family[family].append(built)

    shortages = {
        family: demonstrations_per_family - len(by_family[family])
        for family in families
        if len(by_family[family]) < demonstrations_per_family
    }
    if shortages:
        details = ", ".join(
            f"{family} needs {missing} more" for family, missing in shortages.items()
        )
        raise SourceRecordError(
            f"insufficient verified solver examples for teacher dose: {details}"
        )

    return tuple(
        record
        for family in families
        for record in by_family[family][:demonstrations_per_family]
    )


def build_current_policy_verified_record(
    *,
    source_manifest: DatasetManifest,
    source_record: TCESTaskManifestRecord,
    completion: str,
) -> VerifiedSourceRecord:
    """Build one exact-success TCES example sampled from the current policy."""

    return _verified_task_record(
        source_manifest,
        source_record,
        completion,
        source_kind=SourceKind.CURRENT_POLICY_VERIFIED,
        task_family="tces",
        source_split="a_rl_train",
    )


def build_stage_b_maps_record(
    *,
    source_manifest: DatasetManifest,
    source_record: MAPSTaskManifestRecord,
    completion: str,
) -> VerifiedSourceRecord:
    """Build one solver-backed, globally shortest MAPS Stage-B example."""

    built = _verified_task_record(
        source_manifest,
        source_record,
        completion,
        source_kind=SourceKind.SOLVER_TEACHER,
        task_family="maps",
        source_split="b_train",
    )
    assert built.exact_verification is not None
    if (
        built.exact_verification.canonical_expression
        not in source_record.shortest_programs
    ):
        raise SourceRecordError(
            "Stage-B completion must be a globally shortest program"
        )
    return built


__all__ = [
    "SourceKind",
    "SourceRecordError",
    "SourceSplit",
    "TaskFamily",
    "VerifiedSourceRecord",
    "build_current_policy_verified_record",
    "build_format_warmup_record",
    "build_solver_teacher_record",
    "build_teacher_dose_records",
    "build_stage_b_maps_record",
]
