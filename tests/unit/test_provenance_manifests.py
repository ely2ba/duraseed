from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from duraseed.data.manifests import (
    DatasetManifest,
    ManifestError,
    ManifestIntegrityError,
    NonCanonicalManifestError,
    TCESTaskManifestRecord,
    build_manifest,
    build_maps_record,
    build_tces_record,
    manifest_bytes,
    read_manifest,
    write_manifest,
)
from duraseed.data.sealing import ExecutionContext, FinalTestAccessDenied
from duraseed.provenance import (
    CanonicalJSONError,
    MAX_ROOT_SEED,
    SeedLedger,
    SeedNamespace,
    canonical_json_bytes,
    canonical_json_hash,
    derive_namespaced_seed,
    derive_seed_ledger,
    sha256_bytes,
)
from duraseed.schemas import ExactRational
from duraseed.tasks.maps import MAPSGenerator
from duraseed.tasks.tces.generator import TCESGenerator, TCESGeneratorConfig


def _small_tces_instance(
    dataset_root_seed: int,
    index: int = 0,
    *,
    split: str = "a_validation",
):
    config = TCESGeneratorConfig(
        n_operands=3,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_valid_expressions=1_000,
        split=split,
    )
    generator_seed = derive_namespaced_seed(
        dataset_root_seed,
        "dataset.tces.split",
        split,
    )
    return TCESGenerator(generator_seed, config).generate(index)


def test_canonical_json_is_ordered_exact_and_rejects_ambiguous_values() -> None:
    first = {"target": ExactRational(numerator=6, denominator=-8), "a": 1}
    second = {"a": 1, "target": {"denominator": 4, "numerator": -3}}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first) == (
        b'{"a":1,"target":{"denominator":4,"numerator":-3}}'
    )
    assert canonical_json_hash({"b": 2, "a": 1}) == (
        "sha256:43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )
    assert canonical_json_bytes(SeedNamespace.DATASET) == b'"dataset"'

    for invalid in ({1, 2}, b"binary", math.inf, math.nan):
        with pytest.raises(CanonicalJSONError):
            canonical_json_bytes(invalid)


def test_namespaced_seed_derivation_is_stable_distinct_and_bounded() -> None:
    root_seed = 11
    first = derive_namespaced_seed(root_seed, SeedNamespace.DATASET)
    assert first == derive_namespaced_seed(root_seed, "dataset")
    assert first != derive_namespaced_seed(root_seed, SeedNamespace.GENERATION)
    assert derive_namespaced_seed(root_seed, "generation", 1, 23) != (
        derive_namespaced_seed(root_seed, "generation", 12, 3)
    )
    assert 0 <= first <= MAX_ROOT_SEED

    for invalid in (-1, MAX_ROOT_SEED + 1, True, 1.0):
        with pytest.raises(ValueError):
            derive_namespaced_seed(invalid, "dataset")
    for namespace in ("", "UPPER", "two words", "../escape"):
        with pytest.raises(ValueError):
            derive_namespaced_seed(root_seed, namespace)


def test_seed_ledger_is_frozen_and_self_authenticating() -> None:
    ledger = derive_seed_ledger(29)
    assert ledger.dataset_seed == derive_namespaced_seed(29, "dataset")
    with pytest.raises(ValidationError):
        ledger.dataset_seed = 3  # type: ignore[misc]
    tampered = ledger.model_dump()
    tampered["dataset_seed"] += 1
    with pytest.raises(ValidationError):
        SeedLedger(**tampered)


def test_tces_record_covers_spec_and_preserves_exact_task_types() -> None:
    instance = _small_tces_instance(7)
    features = {
        "ood_profile": "held_out_operand_count",
        "held_out_dimension": "operand_count",
        "id_operand_count": 3,
        "ood_operand_count": 4,
    }
    record = build_tces_record(instance, difficulty_features=features)
    assert record.task_id == record.content_hash
    assert record.task_id.startswith("sha256:")
    assert record.target == instance.task.target
    assert isinstance(record.target, ExactRational)
    assert record.valid_family_count == len(record.valid_family_ids)
    assert record.minimum_depth == instance.enumeration.shortest_depth
    assert record.to_task().target.as_fraction() == instance.task.target.as_fraction()
    assert record.difficulty_features["ood_operand_count"] == 4

    payload = record.model_dump(mode="json")
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        TCESTaskManifestRecord.model_validate(payload)


def test_tces_builder_rejects_mismatched_or_incomplete_provenance() -> None:
    instance = _small_tces_instance(13)
    with pytest.raises(ValueError, match="content_hash"):
        build_tces_record(replace(instance, content_hash="sha256:" + "0" * 64))
    incomplete = replace(instance.enumeration, pruned=True)
    with pytest.raises(ValueError, match="incomplete"):
        build_tces_record(replace(instance, enumeration=incomplete))


def test_maps_record_is_canonical_solver_audited_and_sha_addressed() -> None:
    instance = MAPSGenerator(19).generate(2)
    record = build_maps_record(instance, split="validation")
    assert record.task_id == record.content_hash
    assert record.task_id == "sha256:" + instance.task.task_id.removeprefix("maps-")
    assert record.shortest_program_count == len(record.shortest_programs)
    assert record.to_task().target == instance.task.target
    assert tuple(
        instruction.canonical() for instruction in record.allowed_instructions
    ) == tuple(
        sorted(instruction.canonical() for instruction in record.allowed_instructions)
    )


def test_manifest_order_hash_and_bytes_are_independent_of_input_order() -> None:
    records = [build_tces_record(_small_tces_instance(31, index)) for index in range(2)]
    first = build_manifest(
        name="tces-validation-v1",
        task_family="tces",
        split="a_validation",
        generator_version="1.0.0",
        root_seed=31,
        records=records,
        metadata={"generation_metadata": {"id_operand_count": 3}},
    )
    second = build_manifest(
        name="tces-validation-v1",
        task_family="tces",
        split="a_validation",
        generator_version="1.0.0",
        root_seed=31,
        records=list(reversed(records)),
        metadata={"generation_metadata": {"id_operand_count": 3}},
    )
    assert first == second
    assert first.records == tuple(sorted(first.records, key=lambda row: row.task_id))
    assert first.records_hash == canonical_json_hash(first.records)
    assert first.manifest_id == canonical_json_hash(
        {
            "dataset_seed": first.dataset_seed,
            "generator_version": first.generator_version,
            "metadata": first.metadata,
            "name": first.name,
            "parent_manifest_id": first.parent_manifest_id,
            "record_count": first.record_count,
            "records": first.records,
            "root_seed": first.root_seed,
            "schema_version": first.schema_version,
            "split": first.split,
            "task_family": first.task_family,
        }
    )
    assert manifest_bytes(first) == manifest_bytes(second)

    with pytest.raises(ValidationError, match="generator_seed"):
        build_manifest(
            name="wrong-root-v1",
            task_family="tces",
            split="a_validation",
            generator_version="1.0.0",
            root_seed=32,
            records=records,
        )


def test_manifest_write_read_is_canonical_atomic_and_context_guarded(
    tmp_path: Path,
) -> None:
    record = build_tces_record(_small_tces_instance(37))
    manifest = build_manifest(
        name="validation-v1",
        split="a_validation",
        generator_version="1.0.0",
        root_seed=37,
        records=[record],
    )
    path = tmp_path / "manifest.json"
    assert write_manifest(path, manifest) == path
    assert read_manifest(path, context=ExecutionContext.SELECTION) == manifest
    with pytest.raises(ValueError):
        read_manifest(
            path,
            context=ExecutionContext.SELECTION,
            max_bytes=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ManifestError, match="exceeds"):
        read_manifest(
            path,
            context=ExecutionContext.SELECTION,
            max_bytes=path.stat().st_size - 1,
        )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    with pytest.raises(NonCanonicalManifestError):
        read_manifest(noncanonical, context=ExecutionContext.SELECTION)

    other = build_manifest(
        name="other-v1",
        split="a_validation",
        generator_version="1.0.0",
        root_seed=37,
        records=[record],
    )
    write_manifest(path, other)
    assert read_manifest(path, context=ExecutionContext.SELECTION) == other


def test_renamed_final_manifest_is_denied_outside_final_evaluation(
    tmp_path: Path,
) -> None:
    instance = _small_tces_instance(41, split="a_test_single")
    record = build_tces_record(instance)
    manifest = build_manifest(
        name="innocent-looking-name",
        split="a_test_single",
        generator_version="1.0.0",
        root_seed=41,
        records=[record],
    )
    path = tmp_path / "training-data.json"
    write_manifest(path, manifest)
    for context in (
        ExecutionContext.TRAINING,
        ExecutionContext.SELECTION,
        ExecutionContext.DEBUGGING,
    ):
        with pytest.raises(FinalTestAccessDenied):
            read_manifest(path, context=context)
    assert read_manifest(path, context=ExecutionContext.FINAL_EVALUATION) == manifest


def test_manifest_tampering_fails_hash_validation() -> None:
    record = build_tces_record(_small_tces_instance(43))
    manifest = build_manifest(
        name="tamper-v1",
        split="a_validation",
        generator_version="1.0.0",
        root_seed=43,
        records=[record],
    )
    payload = manifest.model_dump(mode="json")
    payload["record_count"] = 2
    with pytest.raises((ValidationError, ManifestIntegrityError)):
        DatasetManifest.model_validate(payload)

    payload = manifest.model_dump(mode="python")
    payload["records_hash"] = "sha256:" + "0" * 64
    with pytest.raises((ValidationError, ManifestIntegrityError), match="records_hash"):
        DatasetManifest.model_validate(payload)


def test_sha256_bytes_requires_bytes() -> None:
    with pytest.raises(TypeError):
        sha256_bytes(bytearray(b"mutable"))  # type: ignore[arg-type]
