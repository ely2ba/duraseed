from __future__ import annotations

from pathlib import Path

import pytest

from duraseed.data.datasets import (
    read_jsonl_gzip,
    write_jsonl_gzip,
)
from duraseed.data.leakage import audit_leakage
from duraseed.data.manifests import (
    DatasetManifest,
    TCESTaskManifestRecord,
    build_manifest,
    build_tces_record,
    manifest_bytes,
    read_manifest,
    write_manifest,
)
from duraseed.data.sealing import (
    ExecutionContext,
    FinalTestAccessDenied,
    open_final_test,
    seal_file,
)
from duraseed.data.splits import TCESSplitBuilder
from duraseed.tasks.tces import TCESGeneratorConfig


def _fast_config() -> TCESGeneratorConfig:
    return TCESGeneratorConfig(
        n_operands=3,
        operand_min=2,
        operand_max=12,
        target_min=-100,
        target_max=100,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_answer_length=64,
        max_attempts=128,
        exclude_target_in_operands=False,
    )


def _build_manifests(root_seed: int) -> dict[str, DatasetManifest]:
    split_items = TCESSplitBuilder(root_seed, _fast_config()).build_splits(
        {
            "a_seed_train": 2,
            "a_validation": 2,
            "a_test_single": 1,
        }
    )
    manifests: dict[str, DatasetManifest] = {}
    for split, items in split_items.items():
        records = [build_tces_record(item) for item in items]
        manifests[split] = build_manifest(
            name=f"tces-{split}-v1",
            split=split,
            generator_version="1.0.0",
            root_seed=root_seed,
            records=records,
        )
    return manifests


def test_full_data_pipeline_is_reproducible_and_guarded(
    tmp_path: Path,
) -> None:
    root_seed = 17291
    first = _build_manifests(root_seed)
    second = _build_manifests(root_seed)

    assert first == second
    assert {split: manifest_bytes(manifest) for split, manifest in first.items()} == {
        split: manifest_bytes(manifest) for split, manifest in second.items()
    }
    assert audit_leakage(first).clean

    validation = first["a_validation"]
    manifest_path = tmp_path / "validation-manifest.json"
    assert write_manifest(manifest_path, validation) == manifest_path
    assert (
        read_manifest(manifest_path, context=ExecutionContext.SELECTION) == validation
    )

    gzip_path = tmp_path / "validation.jsonl.gz"
    gzip_artifact = write_jsonl_gzip(gzip_path, validation.records)
    assert (
        read_jsonl_gzip(
            gzip_path,
            context=ExecutionContext.SELECTION,
            record_model=TCESTaskManifestRecord,
            expected_logical_hash=gzip_artifact.logical_hash,
            expected_file_hash=gzip_artifact.file_hash,
        )
        == validation.records
    )
    final_manifest = first["a_test_single"]
    # Neither a harmless filename nor direct access to the canonical manifest
    # can hide an embedded final-test split.
    disguised_path = tmp_path / "ordinary-training-data.json"
    write_manifest(disguised_path, final_manifest)
    with pytest.raises(FinalTestAccessDenied):
        read_manifest(disguised_path, context=ExecutionContext.SELECTION)

    plaintext_path = tmp_path / "final-plaintext.json"
    plaintext_path.write_bytes(manifest_bytes(final_manifest))
    sealed_path = tmp_path / "final.sealed"
    key = bytes(range(32))
    seal_file(
        plaintext_path,
        sealed_path,
        key=key,
        declared_split="a_test_single",
    )
    with pytest.raises(FinalTestAccessDenied):
        open_final_test(
            sealed_path,
            key=key,
            command=ExecutionContext.TRAINING,
        )
    assert (
        open_final_test(
            sealed_path,
            key=key,
            command=ExecutionContext.FINAL_EVALUATION,
            expected_split="a_test_single",
        )
        == plaintext_path.read_bytes()
    )
