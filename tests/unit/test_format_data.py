from __future__ import annotations

from pydantic import ValidationError
import pytest

from duraseed.data.format import (
    FORMAT_SPLIT_SIZES,
    FormatManifest,
    FormatRecord,
    build_format_manifest,
)
from duraseed.data.manifests import DatasetManifest


def test_format_manifest_is_deterministic_and_content_addressed() -> None:
    first = build_format_manifest("format_train", 3)
    repeated = build_format_manifest("format_train", 3)

    assert first == repeated
    assert first.manifest_id == repeated.manifest_id
    assert first.manifest_id.startswith("sha256:")
    assert len({record.record_id for record in first.records}) == 3
    assert all(record.record_id.startswith("sha256:") for record in first.records)

    changed = build_format_manifest("format_train", 2)
    assert changed.manifest_id != first.manifest_id

    tampered = first.records[0].model_dump(mode="python")
    tampered["prompt_text"] += " Target: 42"
    with pytest.raises(ValidationError, match="task-agnostic template"):
        FormatRecord.model_validate(tampered)


def test_format_examples_contain_only_the_common_response_protocol() -> None:
    manifest = build_format_manifest("format_eval", 2)
    forbidden_task_syntax = (
        "operands",
        "allowed binary operations",
        "target:",
        "start value",
        "modulus",
        "allowed instructions",
        "maximum program length",
        "add ",
        "mul ",
        "neg",
    )

    for record in manifest.records:
        combined = f"{record.prompt_text}\n{record.verified_completion_text}".lower()
        assert not any(marker in combined for marker in forbidden_task_syntax)
        assert set(type(record).model_fields) == {
            "record_id",
            "split",
            "item_index",
            "prompt_text",
            "verified_completion_text",
        }


def test_format_manifest_is_not_a_reasoning_task_manifest() -> None:
    manifest = build_format_manifest("format_eval", 1)

    assert not isinstance(manifest, DatasetManifest)
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(manifest.model_dump(mode="python"))


def test_format_manifest_fails_closed_on_wrong_content_or_size() -> None:
    manifest = build_format_manifest("format_train", 1)
    tampered = manifest.model_dump(mode="python")
    tampered["manifest_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="does not match its content"):
        FormatManifest.model_validate(tampered)

    with pytest.raises(ValueError, match="declared split size"):
        build_format_manifest("format_eval", FORMAT_SPLIT_SIZES["format_eval"] + 1)
