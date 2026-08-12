from __future__ import annotations

from pathlib import Path

import pytest

from duraseed.data.datasets import (
    DatasetFormatError,
    DatasetIntegrityError,
    read_jsonl_gzip,
    write_jsonl_gzip,
)
from duraseed.data.manifests import TCESTaskManifestRecord, build_tces_record
from duraseed.data.sealing import ExecutionContext, FinalTestAccessDenied
from duraseed.tasks.tces.generator import TCESGenerator, TCESGeneratorConfig


def _record(seed: int, split: str = "a_validation") -> TCESTaskManifestRecord:
    config = TCESGeneratorConfig(
        n_operands=3,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_valid_expressions=1_000,
        split=split,
    )
    return build_tces_record(TCESGenerator(seed, config).generate())


def test_gzip_is_byte_deterministic_timestamp_free_and_typed_round_trips(
    tmp_path: Path,
) -> None:
    records = [_record(101), _record(103)]
    first_path = tmp_path / "first.jsonl.gz"
    second_path = tmp_path / "second.jsonl.gz"
    first = write_jsonl_gzip(first_path, records)
    second = write_jsonl_gzip(second_path, records)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.file_hash == second.file_hash
    assert first.logical_hash == second.logical_hash
    assert first_path.read_bytes()[3] == 0
    assert first_path.read_bytes()[4:8] == b"\x00\x00\x00\x00"
    loaded = read_jsonl_gzip(
        first_path,
        context=ExecutionContext.SELECTION,
        record_model=TCESTaskManifestRecord,
        expected_file_hash=first.file_hash,
        expected_logical_hash=first.logical_hash,
    )
    assert loaded == tuple(records)
    repeat = write_jsonl_gzip(first_path, records)
    assert repeat == first


def test_final_rows_are_denied_outside_final_evaluation(tmp_path: Path) -> None:
    path = tmp_path / "ordinary-training-file.jsonl.gz"
    record = _record(127, split="a_test_single")
    write_jsonl_gzip(path, [record])
    for context in (
        ExecutionContext.TRAINING,
        ExecutionContext.SELECTION,
        ExecutionContext.DEBUGGING,
    ):
        with pytest.raises(FinalTestAccessDenied):
            read_jsonl_gzip(
                path,
                context=context,
                record_model=TCESTaskManifestRecord,
            )
    assert read_jsonl_gzip(
        path,
        context=ExecutionContext.FINAL_EVALUATION,
        record_model=TCESTaskManifestRecord,
    ) == (record,)


def test_gzip_reader_rejects_hash_mismatch(tmp_path: Path) -> None:
    record = _record(139)
    valid_path = tmp_path / "valid.gz"
    artifact = write_jsonl_gzip(valid_path, [record])
    with pytest.raises(DatasetIntegrityError, match="file hash"):
        read_jsonl_gzip(
            valid_path,
            context=ExecutionContext.SELECTION,
            expected_file_hash="sha256:" + "0" * 64,
        )
    assert artifact.logical_hash.startswith("sha256:")


def test_readers_require_a_top_level_split(tmp_path: Path) -> None:
    path = tmp_path / "unclassified.gz"
    write_jsonl_gzip(path, [{"value": 1}])
    with pytest.raises(DatasetFormatError, match="top-level split"):
        read_jsonl_gzip(path, context=ExecutionContext.TRAINING)
