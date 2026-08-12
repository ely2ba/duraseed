"""Deterministic, task-agnostic examples for the shared answer protocol."""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from duraseed.provenance import canonical_json_hash, validate_sha256_id
from duraseed.schemas import StrictModel


FORMAT_RECORD_SCHEMA_VERSION = "duraseed-format-record-v1"
FORMAT_MANIFEST_SCHEMA_VERSION = "duraseed-format-manifest-v1"
FormatSplit = Literal["format_train", "format_eval"]
FORMAT_SPLIT_SIZES = MappingProxyType(
    {
        "format_train": 1_024,
        "format_eval": 256,
    }
)


def _format_text(split: FormatSplit, item_index: int) -> tuple[str, str]:
    coordinate = canonical_json_hash(
        {
            "item_index": item_index,
            "kind": "task_agnostic_format_token",
            "split": split,
            "version": 1,
        }
    )
    token = f"format-{coordinate.removeprefix('sha256:')[:16]}"
    return (
        "Return the literal token below inside exactly one "
        f"<answer>...</answer> tag.\nToken: {token}",
        f"<answer>{token}</answer>",
    )


def _record_identity_payload(record: "FormatRecord") -> dict[str, object]:
    return {
        "item_index": record.item_index,
        "prompt_text": record.prompt_text,
        "schema_version": FORMAT_RECORD_SCHEMA_VERSION,
        "split": record.split,
        "verified_completion_text": record.verified_completion_text,
    }


class FormatRecord(StrictModel):
    """One fixed wrapper-only example with no TCES or MAPS task fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    record_id: str
    split: FormatSplit
    item_index: int = Field(ge=0)
    prompt_text: str = Field(min_length=1)
    verified_completion_text: str = Field(min_length=1)

    @field_validator("record_id")
    @classmethod
    def record_id_is_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)

    @field_validator("item_index", mode="before")
    @classmethod
    def item_index_is_an_integer(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("format item_index must be an integer")
        return value

    @model_validator(mode="after")
    def content_is_fixed_and_addressed(self) -> "FormatRecord":
        if self.item_index >= FORMAT_SPLIT_SIZES[self.split]:
            raise ValueError("format item_index exceeds the declared split size")
        expected_prompt, expected_completion = _format_text(
            self.split,
            self.item_index,
        )
        if (
            self.prompt_text != expected_prompt
            or self.verified_completion_text != expected_completion
        ):
            raise ValueError("format record does not match the task-agnostic template")
        expected_id = canonical_json_hash(_record_identity_payload(self))
        if self.record_id != expected_id:
            raise ValueError("format record_id does not match its content")
        return self


def build_format_record(split: FormatSplit, item_index: int) -> FormatRecord:
    """Build one deterministic answer-wrapper example."""

    if isinstance(item_index, bool) or not isinstance(item_index, int):
        raise ValueError("format item_index must be an integer")
    if item_index < 0:
        raise ValueError("format item_index must be non-negative")
    prompt, completion = _format_text(split, item_index)
    values = {
        "split": split,
        "item_index": item_index,
        "prompt_text": prompt,
        "verified_completion_text": completion,
    }
    provisional = FormatRecord.model_construct(record_id="", **values)
    return FormatRecord(
        record_id=canonical_json_hash(_record_identity_payload(provisional)),
        **values,
    )


def _manifest_identity_payload(manifest: "FormatManifest") -> dict[str, object]:
    return {
        "records": manifest.records,
        "schema_version": manifest.schema_version,
        "split": manifest.split,
    }


class FormatManifest(StrictModel):
    """A small content-addressed manifest separate from reasoning-task manifests."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["duraseed-format-manifest-v1"] = (
        FORMAT_MANIFEST_SCHEMA_VERSION
    )
    manifest_id: str
    split: FormatSplit
    records: tuple[FormatRecord, ...]

    @field_validator("manifest_id")
    @classmethod
    def manifest_id_is_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)

    @model_validator(mode="after")
    def records_are_canonical_and_addressed(self) -> "FormatManifest":
        if len(self.records) > FORMAT_SPLIT_SIZES[self.split]:
            raise ValueError("format manifest exceeds the declared split size")
        if any(record.split != self.split for record in self.records):
            raise ValueError("format record split differs from its manifest")
        if tuple(record.item_index for record in self.records) != tuple(
            range(len(self.records))
        ):
            raise ValueError("format manifest must contain a canonical index prefix")
        if len({record.record_id for record in self.records}) != len(self.records):
            raise ValueError("format manifest contains duplicate records")
        expected_id = canonical_json_hash(_manifest_identity_payload(self))
        if self.manifest_id != expected_id:
            raise ValueError("format manifest_id does not match its content")
        return self


def build_format_manifest(split: FormatSplit, size: int) -> FormatManifest:
    """Build a deterministic prefix of one format-only split."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("format manifest size must be a non-negative integer")
    if size > FORMAT_SPLIT_SIZES[split]:
        raise ValueError("format manifest exceeds the declared split size")
    records = tuple(build_format_record(split, index) for index in range(size))
    provisional = FormatManifest.model_construct(
        manifest_id="",
        split=split,
        records=records,
    )
    return FormatManifest(
        manifest_id=canonical_json_hash(_manifest_identity_payload(provisional)),
        split=split,
        records=records,
    )


__all__ = [
    "FORMAT_MANIFEST_SCHEMA_VERSION",
    "FORMAT_RECORD_SCHEMA_VERSION",
    "FORMAT_SPLIT_SIZES",
    "FormatManifest",
    "FormatRecord",
    "FormatSplit",
    "build_format_manifest",
    "build_format_record",
]
