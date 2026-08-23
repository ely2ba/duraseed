from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from duraseed.data.boundary_confirmation import build_confirmation_manifest
from duraseed.data.boundary_protocol import build_broad_manifest
from duraseed.data.leakage import audit_leakage
from duraseed.data.panel_split_manifest import build_panel_split_manifest
from duraseed.data.panel_split_production import (
    PanelSplitEquivalenceError,
    PanelSplitManifests,
    build_production_panel_splits,
    production_forbidden_records,
    write_equivalent_panel_splits,
)
from duraseed.data.panels import (
    FamilyPanelArtifact,
    PanelLabel,
    SeedBlockPanelAssignment,
)
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.tasks.tces import TCESGeneratorConfig


def _fixture():
    generator = TCESGeneratorConfig(
        n_operands=3,
        operand_min=2,
        operand_max=20,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_attempts=512,
    )
    broad = build_broad_manifest(
        generator,
        family_count=3,
        items_per_family=2,
        template_scan_ceiling=32,
        variant_index_offset=64,
    )
    families = tuple(sorted({record.intended_family for record in broad.records})[:2])
    confirmation = build_confirmation_manifest(generator, broad, families)
    artifact = FamilyPanelArtifact(
        m0_checkpoint_path="tinker://m0/sampler",
        candidate_family_table_manifest_id="sha256:" + "a" * 64,
        panel_matching_report_id="sha256:" + "b" * 64,
        allocation_seed=3,
        panel_a_family_ids=(families[0],),
        panel_b_family_ids=(families[1],),
        seed_block_assignments=(
            SeedBlockPanelAssignment(
                training_seed=17,
                targeted_panel=PanelLabel.A,
                sentinel_panel=PanelLabel.B,
            ),
            SeedBlockPanelAssignment(
                training_seed=37,
                targeted_panel=PanelLabel.B,
                sentinel_panel=PanelLabel.A,
            ),
        ),
    )
    config = SimpleNamespace(
        seed=5,
        tasks=SimpleNamespace(
            tces=SimpleNamespace(generator_kwargs=lambda: asdict(generator))
        ),
    )
    return config, artifact, broad, confirmation, families


def test_panel_split_manifests_match_the_accepted_v0_fixture() -> None:
    config, artifact, broad, confirmation, families = _fixture()
    source_records = (*broad.records, *confirmation.records)

    train = build_panel_split_manifest(
        config,
        artifact=artifact,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
        split="a_seed_train",
        items_per_family=2,
        forbidden_records=source_records,
    )
    repeated = build_panel_split_manifest(
        config,
        artifact=artifact,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
        split="a_seed_train",
        items_per_family=2,
        forbidden_records=source_records,
    )
    gate = build_panel_split_manifest(
        config,
        artifact=artifact,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
        split="a_seed_gate",
        items_per_family=2,
        forbidden_records=(*source_records, *train.records),
    )

    assert train == repeated
    assert len(train.records) == len(gate.records) == 4
    assert all(
        set(record.valid_family_ids).intersection(families) == {record.intended_family}
        for record in (*train.records, *gate.records)
    )
    assert audit_leakage(
        {"a_seed_train": train.records, "a_seed_gate": gate.records}
    ).clean

    train_bytes = canonical_json_bytes(train.model_dump(mode="json"))
    gate_bytes = canonical_json_bytes(gate.model_dump(mode="json"))
    combined = canonical_json_bytes(
        {
            "a_seed_train": train.model_dump(mode="json"),
            "a_seed_gate": gate.model_dump(mode="json"),
        }
    )
    assert (len(train_bytes), sha256_bytes(train_bytes)) == (
        4_384,
        "sha256:942ff8bc3e0d909ea2049cf474f5b2745e11edc2034aa4168b8daa1077a171b8",
    )
    assert (len(gate_bytes), sha256_bytes(gate_bytes)) == (
        4_415,
        "sha256:5d42e5fbf9e1acec338084890211d2e629c2aaed9456b6a622f7a73ce7eab6bc",
    )
    assert (len(combined), sha256_bytes(combined)) == (
        8_831,
        "sha256:2f8d1a47d1eb01534cf471101a09db34bd8187b0a2d9f53d41165aa5ab6a32c0",
    )


def test_parallel_candidate_reduction_preserves_panel_manifest() -> None:
    config, artifact, broad, confirmation, _ = _fixture()
    source_records = (*broad.records, *confirmation.records)

    class InlineExecutor:
        @staticmethod
        def map(function, values):
            return map(function, values)

    serial = build_panel_split_manifest(
        config,
        artifact=artifact,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
        split="a_seed_train",
        items_per_family=2,
        forbidden_records=source_records,
    )
    parallel_path = build_panel_split_manifest(
        config,
        artifact=artifact,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
        split="a_seed_train",
        items_per_family=2,
        forbidden_records=source_records,
        executor=InlineExecutor(),  # type: ignore[arg-type]
    )

    assert parallel_path == serial


def test_production_pair_preserves_archived_build_order(monkeypatch) -> None:
    import duraseed.data.panel_split_production as production

    config, artifact, broad, confirmation, families = _fixture()
    original = production.build_panel_split_manifest
    calls = []

    def capture(*args, **kwargs):
        calls.append(
            (
                kwargs["split"],
                kwargs["items_per_family"],
                tuple(kwargs["forbidden_records"]),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(production, "build_panel_split_manifest", capture)
    pair = build_production_panel_splits(
        config,
        artifact=artifact,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
    )
    source = (*broad.records, *confirmation.records)

    assert [(split, count) for split, count, _ in calls] == [
        ("a_seed_train", 16),
        ("a_seed_gate", 8),
    ]
    assert calls[0][2] == source
    assert calls[1][2] == (*source, *pair.train.records)
    assert len(pair.train.records) == 16 * len(families)
    assert len(pair.gate.records) == 8 * len(families)
    assert production_forbidden_records(broad, confirmation) == source
    assert production_forbidden_records(broad, confirmation, (broad, confirmation)) == (
        *source,
        *broad.records,
        *confirmation.records,
    )


def test_production_equivalence_writes_only_canonical_downstream_files(
    tmp_path,
) -> None:
    config, artifact, broad, confirmation, _ = _fixture()
    pair = build_production_panel_splits(
        config,
        artifact=artifact,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
    )
    forbidden = production_forbidden_records(broad, confirmation)

    result = write_equivalent_panel_splits(
        tmp_path,
        old=pair,
        new=pair,
        artifact=artifact,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
        forbidden_records=forbidden,
    )

    assert {path.name for path in tmp_path.iterdir()} == {
        "a_seed_train_manifest.json",
        "a_seed_gate_manifest.json",
        "panel_split_equivalence.json",
    }
    assert result["old_new_identical"] is True
    assert result["comparisons"]["a_seed_train"]["complete_object_equal"] is True
    assert result["comparisons"]["a_seed_gate"]["canonical_bytes_equal"] is True
    assert result["a_seed_train_sha256"] == sha256_bytes(
        (tmp_path / "a_seed_train_manifest.json").read_bytes()
    )
    assert result["a_seed_gate_sha256"] == sha256_bytes(
        (tmp_path / "a_seed_gate_manifest.json").read_bytes()
    )


def test_production_equivalence_hard_stops_before_writing_on_divergence(
    tmp_path,
) -> None:
    config, artifact, broad, confirmation, _ = _fixture()
    pair = build_production_panel_splits(
        config,
        artifact=artifact,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
    )
    divergent = PanelSplitManifests(
        pair.train,
        pair.gate.model_copy(update={"name": "divergent-v0-gate"}),
    )

    with pytest.raises(PanelSplitEquivalenceError, match="diverged"):
        write_equivalent_panel_splits(
            tmp_path,
            old=divergent,
            new=pair,
            artifact=artifact,
            broad_manifest=broad,
            confirmation_manifest=confirmation,
            forbidden_records=production_forbidden_records(broad, confirmation),
        )

    assert list(tmp_path.iterdir()) == []
