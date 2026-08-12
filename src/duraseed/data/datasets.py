"""Deterministic compressed JSONL dataset artifacts.

Every row carries one top-level split and every reader requires an explicit
execution context before returning final-test data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gzip
import io
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from duraseed.data.io import atomic_write_bytes
from duraseed.data.sealing import (
    ExecutionContext,
    FinalTestAccessDenied,
    guard_record_splits,
)
from duraseed.provenance import (
    IntegrityError,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256_id,
)


DEFAULT_MAX_COMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_RECORDS = 10_000_000
DATASET_LOGICAL_FORMAT = "duraseed-canonical-jsonl-v1"

T = TypeVar("T")


class DatasetError(ValueError):
    """Base class for deterministic dataset format/integrity failures."""


class DatasetFormatError(DatasetError):
    """Raised for malformed, noncanonical, or unclassified records."""


class DatasetIntegrityError(DatasetError, IntegrityError):
    """Raised when physical or logical content hashes do not match."""


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """Hashes and counts returned after a deterministic atomic write."""

    path: Path
    storage_format: str
    record_count: int
    logical_hash: str
    file_hash: str
    byte_size: int


def _record_value(record: Any) -> Any:
    # canonical_json_bytes understands Pydantic models and dataclasses directly.
    return record


def canonical_jsonl_bytes(records: Sequence[Any]) -> bytes:
    """Return canonical logical bytes while preserving caller-supplied order."""

    return b"".join(
        canonical_json_bytes(_record_value(record)) + b"\n" for record in records
    )


def logical_content_hash(records: Sequence[Any]) -> str:
    """Hash canonical JSONL bytes independently of the storage container."""

    return sha256_bytes(canonical_jsonl_bytes(records))


def _gzip_bytes(payload: bytes, *, compresslevel: int = 9) -> bytes:
    if (
        isinstance(compresslevel, bool)
        or not isinstance(compresslevel, int)
        or not 0 <= compresslevel <= 9
    ):
        raise ValueError("compresslevel must be between zero and nine")
    output = io.BytesIO()
    # filename="" suppresses FNAME and mtime=0 removes wall-clock metadata.
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=compresslevel,
        fileobj=output,
        mtime=0,
    ) as stream:
        stream.write(payload)
    return output.getvalue()


def _validate_gzip_header(payload: bytes) -> None:
    if len(payload) < 18 or payload[:3] != b"\x1f\x8b\x08":
        raise DatasetFormatError("artifact is not a complete gzip stream")
    flags = payload[3]
    if flags != 0:
        raise DatasetFormatError(
            "gzip stream contains nondeterministic optional metadata"
        )
    if payload[4:8] != b"\x00\x00\x00\x00":
        raise DatasetFormatError("gzip stream has a nonzero timestamp")


def _validate_expected_hash(actual: str, expected: str | None, label: str) -> None:
    if expected is None:
        return
    validate_sha256_id(expected)
    if actual != expected:
        raise DatasetIntegrityError(
            f"{label} mismatch: expected {expected}, got {actual}"
        )


def _artifact(
    *,
    path: str | os.PathLike[str],
    storage_format: str,
    record_count: int,
    logical: bytes,
    physical: bytes,
) -> DatasetArtifact:
    return DatasetArtifact(
        path=Path(path),
        storage_format=storage_format,
        record_count=record_count,
        logical_hash=sha256_bytes(logical),
        file_hash=sha256_bytes(physical),
        byte_size=len(physical),
    )


def write_jsonl_gzip(
    path: str | os.PathLike[str],
    records: Sequence[Any],
    *,
    compresslevel: int = 9,
) -> DatasetArtifact:
    """Atomically write deterministic gzip JSONL."""

    logical = canonical_jsonl_bytes(records)
    physical = _gzip_bytes(logical, compresslevel=compresslevel)
    atomic_write_bytes(path, physical)
    return _artifact(
        path=path,
        storage_format="jsonl.gz",
        record_count=len(records),
        logical=logical,
        physical=physical,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetFormatError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _record_split(record: Mapping[str, Any]) -> str:
    split = record.get("split")
    if not isinstance(split, str) or not split.strip():
        raise DatasetFormatError("every dataset row must declare a top-level split")
    return split.strip()


def _validate_typed_record(
    raw: Any,
    canonical_line: bytes,
    record_model: type[T] | None,
) -> T | Any:
    if record_model is None:
        return raw
    if not isinstance(record_model, type) or not issubclass(record_model, BaseModel):
        raise TypeError("record_model must be a Pydantic BaseModel subclass")
    validated = record_model.model_validate_json(canonical_line)
    if canonical_json_bytes(validated) != canonical_line:
        raise DatasetFormatError(
            "typed record serialization differs from authenticated row content"
        )
    return validated


def _parse_canonical_jsonl(
    logical: bytes,
    *,
    context: ExecutionContext | str,
    record_model: type[T] | None,
    max_records: int,
) -> tuple[T | Any, ...]:
    if (
        isinstance(max_records, bool)
        or not isinstance(max_records, int)
        or max_records < 0
    ):
        raise ValueError("max_records must be non-negative")
    if logical and not logical.endswith(b"\n"):
        raise DatasetFormatError("canonical JSONL must end with a newline")
    # Validate the role before parsing any row or invoking a typed model.
    guard_record_splits((), context)
    typed_records: list[T | Any] = []
    for line_number, line in enumerate(logical.splitlines(), start=1):
        if line_number > max_records:
            raise DatasetFormatError(f"dataset exceeds {max_records} records")
        if not line:
            raise DatasetFormatError(f"blank JSONL row at line {line_number}")
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DatasetFormatError(f"JSONL row {line_number} is not UTF-8") from error
        try:
            raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise DatasetFormatError(f"invalid JSON at line {line_number}") from error
        if not isinstance(raw, Mapping):
            raise DatasetFormatError(f"dataset row {line_number} must be an object")
        if canonical_json_bytes(raw) != line:
            raise DatasetFormatError(f"JSONL row {line_number} is not canonical")
        guard_record_splits((_record_split(raw),), context)
        typed_records.append(_validate_typed_record(raw, line, record_model))
    return tuple(typed_records)


def read_jsonl_gzip(
    path: str | os.PathLike[str],
    *,
    context: ExecutionContext | str,
    record_model: type[T] | None = None,
    expected_logical_hash: str | None = None,
    expected_file_hash: str | None = None,
    max_compressed_bytes: int = DEFAULT_MAX_COMPRESSED_BYTES,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> tuple[T | Any, ...]:
    """Read deterministic gzip JSONL and enforce embedded split access policy."""

    source = Path(path)
    limits = (max_compressed_bytes, max_uncompressed_bytes, max_records)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in limits):
        raise ValueError("dataset byte/record limits must be integers")
    if max_compressed_bytes < 1 or max_uncompressed_bytes < 0 or max_records < 0:
        raise ValueError("dataset byte limits are invalid")
    if source.stat().st_size > max_compressed_bytes:
        raise DatasetFormatError("compressed dataset exceeds configured byte limit")
    physical = source.read_bytes()
    if len(physical) > max_compressed_bytes:
        raise DatasetFormatError("compressed dataset exceeds configured byte limit")
    file_hash = sha256_bytes(physical)
    _validate_expected_hash(file_hash, expected_file_hash, "file hash")
    _validate_gzip_header(physical)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(physical), mode="rb") as stream:
            logical = stream.read(max_uncompressed_bytes + 1)
    except (EOFError, OSError) as error:
        raise DatasetFormatError("invalid gzip stream") from error
    if len(logical) > max_uncompressed_bytes:
        raise DatasetFormatError("uncompressed dataset exceeds configured byte limit")
    logical_hash = sha256_bytes(logical)
    _validate_expected_hash(logical_hash, expected_logical_hash, "logical hash")
    return _parse_canonical_jsonl(
        logical,
        context=context,
        record_model=record_model,
        max_records=max_records,
    )


__all__ = [
    "DATASET_LOGICAL_FORMAT",
    "DatasetArtifact",
    "DatasetError",
    "DatasetFormatError",
    "DatasetIntegrityError",
    "ExecutionContext",
    "FinalTestAccessDenied",
    "canonical_jsonl_bytes",
    "logical_content_hash",
    "read_jsonl_gzip",
    "write_jsonl_gzip",
]
