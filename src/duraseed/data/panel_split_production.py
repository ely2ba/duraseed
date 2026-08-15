"""Production construction and exact v0/v1 comparison for panel splits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duraseed.config import PilotConfig
from duraseed.data.io import atomic_write_bytes
from duraseed.data.manifests import (
    DatasetManifest,
    TCESTaskManifestRecord,
    manifest_bytes,
    write_manifest,
)
from duraseed.data.panel_split_manifest import build_panel_split_manifest
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.provenance import canonical_json_bytes, canonical_json_hash, sha256_bytes


PANEL_TRAIN_ITEMS_PER_FAMILY = 16
PANEL_GATE_ITEMS_PER_FAMILY = 8


class PanelSplitEquivalenceError(ValueError):
    """The archived and extracted builders did not produce identical outputs."""


@dataclass(frozen=True, slots=True)
class PanelSplitManifests:
    train: DatasetManifest
    gate: DatasetManifest


def production_forbidden_records(
    broad: DatasetManifest,
    confirmation: DatasetManifest,
    prior_manifests: Sequence[DatasetManifest] = (),
) -> tuple[TCESTaskManifestRecord, ...]:
    """Return the archived broad, confirmation, RL, monitor exclusion order."""

    return tuple(
        record
        for manifest in (broad, confirmation, *prior_manifests)
        for record in manifest.records
        if isinstance(record, TCESTaskManifestRecord)
    )


def build_production_panel_splits(
    config: PilotConfig,
    *,
    artifact: FamilyPanelArtifact,
    broad_manifest: DatasetManifest,
    confirmation_manifest: DatasetManifest,
    prior_manifests: Sequence[DatasetManifest] = (),
) -> PanelSplitManifests:
    """Build 16 train then 8 unseen gate items per selected family."""

    forbidden = production_forbidden_records(
        broad_manifest, confirmation_manifest, prior_manifests
    )
    train = build_panel_split_manifest(
        config,
        artifact=artifact,
        broad_manifest=broad_manifest,
        confirmation_manifest=confirmation_manifest,
        split="a_seed_train",
        items_per_family=PANEL_TRAIN_ITEMS_PER_FAMILY,
        forbidden_records=forbidden,
    )
    gate = build_panel_split_manifest(
        config,
        artifact=artifact,
        broad_manifest=broad_manifest,
        confirmation_manifest=confirmation_manifest,
        split="a_seed_gate",
        items_per_family=PANEL_GATE_ITEMS_PER_FAMILY,
        forbidden_records=(*forbidden, *train.records),
    )
    return PanelSplitManifests(train, gate)


def _comparison(old: DatasetManifest, new: DatasetManifest) -> dict[str, Any]:
    old_bytes = canonical_json_bytes(old)
    new_bytes = canonical_json_bytes(new)
    return {
        "old_record_count": old.record_count,
        "new_record_count": new.record_count,
        "old_canonical_byte_length": len(old_bytes),
        "new_canonical_byte_length": len(new_bytes),
        "old_canonical_sha256": sha256_bytes(old_bytes),
        "new_canonical_sha256": sha256_bytes(new_bytes),
        "complete_object_equal": old == new,
        "canonical_bytes_equal": old_bytes == new_bytes,
    }


def verify_panel_split_equivalence(
    old: PanelSplitManifests,
    new: PanelSplitManifests,
    *,
    artifact: FamilyPanelArtifact,
    broad_manifest: DatasetManifest,
    confirmation_manifest: DatasetManifest,
    forbidden_records: Sequence[TCESTaskManifestRecord],
) -> dict[str, Any]:
    """Compare complete manifests and canonical bytes; fail on any divergence."""

    train = _comparison(old.train, new.train)
    gate = _comparison(old.gate, new.gate)
    old_combined = canonical_json_bytes(
        {"a_seed_train": old.train, "a_seed_gate": old.gate}
    )
    new_combined = canonical_json_bytes(
        {"a_seed_train": new.train, "a_seed_gate": new.gate}
    )
    if not (
        train["complete_object_equal"]
        and train["canonical_bytes_equal"]
        and gate["complete_object_equal"]
        and gate["canonical_bytes_equal"]
        and old_combined == new_combined
    ):
        raise PanelSplitEquivalenceError(
            "archived v0 and extracted v1 panel splits diverged"
        )
    return {
        "schema_version": "duraseed-calibration-panel-split-equivalence-v1",
        "status": "passed",
        "old_new_identical": True,
        "source_broad_manifest_id": broad_manifest.manifest_id,
        "source_confirmation_manifest_id": confirmation_manifest.manifest_id,
        "panel_artifact_id": canonical_json_hash(artifact),
        "forbidden_records_sha256": canonical_json_hash(tuple(forbidden_records)),
        "a_seed_train_manifest_id": new.train.manifest_id,
        "a_seed_gate_manifest_id": new.gate.manifest_id,
        "a_seed_train_sha256": sha256_bytes(manifest_bytes(new.train)),
        "a_seed_gate_sha256": sha256_bytes(manifest_bytes(new.gate)),
        "comparisons": {
            "a_seed_train": train,
            "a_seed_gate": gate,
            "combined": {
                "old_canonical_byte_length": len(old_combined),
                "new_canonical_byte_length": len(new_combined),
                "old_canonical_sha256": sha256_bytes(old_combined),
                "new_canonical_sha256": sha256_bytes(new_combined),
                "canonical_bytes_equal": True,
            },
        },
    }


def write_equivalent_panel_splits(
    directory: str | Path,
    *,
    old: PanelSplitManifests,
    new: PanelSplitManifests,
    artifact: FamilyPanelArtifact,
    broad_manifest: DatasetManifest,
    confirmation_manifest: DatasetManifest,
    forbidden_records: Sequence[TCESTaskManifestRecord],
) -> dict[str, Any]:
    """Write only the two canonical manifests and their passed comparison."""

    result = verify_panel_split_equivalence(
        old,
        new,
        artifact=artifact,
        broad_manifest=broad_manifest,
        confirmation_manifest=confirmation_manifest,
        forbidden_records=forbidden_records,
    )
    root = Path(directory)
    write_manifest(root / "a_seed_train_manifest.json", new.train)
    write_manifest(root / "a_seed_gate_manifest.json", new.gate)
    atomic_write_bytes(
        root / "panel_split_equivalence.json", canonical_json_bytes(result)
    )
    return result


__all__ = [
    "PANEL_GATE_ITEMS_PER_FAMILY",
    "PANEL_TRAIN_ITEMS_PER_FAMILY",
    "PanelSplitEquivalenceError",
    "PanelSplitManifests",
    "build_production_panel_splits",
    "production_forbidden_records",
    "verify_panel_split_equivalence",
    "write_equivalent_panel_splits",
]
