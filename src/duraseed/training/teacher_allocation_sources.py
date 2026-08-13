"""Object-level authentication for the local teacher-allocation freezer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from duraseed.config import PilotConfig
from duraseed.data.leakage import audit_leakage
from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.provenance import canonical_json_hash


RANDOM_FAMILY_ROWS = 16
_BOUNDARY_ROOT_SEED = 5


class TeacherAllocationSourceError(ValueError):
    """Supplied scientific objects do not match the frozen allocation inputs."""


@dataclass(frozen=True, slots=True)
class TeacherAllocationSources:
    config: PilotConfig
    panel: FamilyPanelArtifact
    broad_manifest: DatasetManifest
    confirmation_manifest: DatasetManifest
    a_rl_train_manifest: DatasetManifest
    a_monitor_manifest: DatasetManifest
    target_train_manifest: DatasetManifest
    gate_manifest: DatasetManifest
    selected_dose: int | None
    optimizer_updates: int

    @property
    def all_manifests(self) -> tuple[DatasetManifest, ...]:
        return (
            self.broad_manifest,
            self.confirmation_manifest,
            self.a_rl_train_manifest,
            self.a_monitor_manifest,
            self.target_train_manifest,
            self.gate_manifest,
        )


def _require_tces_manifest(
    manifest: DatasetManifest,
    *,
    split: str,
    root_seed: int,
) -> None:
    if (
        not isinstance(manifest, DatasetManifest)
        or manifest.task_family != "tces"
        or manifest.split != split
        or manifest.root_seed != root_seed
        or any(
            not isinstance(record, TCESTaskManifestRecord)
            for record in manifest.records
        )
    ):
        raise TeacherAllocationSourceError(
            f"source manifest must be authenticated TCES/{split} data"
        )


def validate_teacher_allocation_sources(
    source: TeacherAllocationSources,
) -> TeacherAllocationSources:
    validate_teacher_allocation_base_sources(source)
    if (
        type(source.selected_dose) is not int
        or source.selected_dose
        not in source.config.teacher_dose.demonstrations_per_family
        or source.optimizer_updates != source.config.teacher_dose.calibration_updates
    ):
        raise TeacherAllocationSourceError("teacher-dose recipe is off protocol")
    return source


def validate_teacher_allocation_base_sources(
    source: TeacherAllocationSources,
) -> TeacherAllocationSources:
    """Authenticate dose-independent inputs before empirical dose selection."""

    if not isinstance(source, TeacherAllocationSources):
        raise TypeError("sources must be TeacherAllocationSources")
    if not isinstance(source.config, PilotConfig):
        raise TypeError("sources.config must be PilotConfig")
    if not isinstance(source.panel, FamilyPanelArtifact):
        raise TypeError("sources.panel must be FamilyPanelArtifact")
    panel_ids = (*source.panel.panel_a_family_ids, *source.panel.panel_b_family_ids)
    if len(panel_ids) != 24 or len(set(panel_ids)) != 24:
        raise TeacherAllocationSourceError(
            "teacher allocation requires unique 12/12 panels"
        )

    for manifest in (source.broad_manifest, source.confirmation_manifest):
        _require_tces_manifest(
            manifest,
            split="a_candidate",
            root_seed=_BOUNDARY_ROOT_SEED,
        )
    for manifest, split in (
        (source.a_rl_train_manifest, "a_rl_train"),
        (source.a_monitor_manifest, "a_monitor"),
        (source.target_train_manifest, "a_seed_train"),
        (source.gate_manifest, "a_seed_gate"),
    ):
        _require_tces_manifest(manifest, split=split, root_seed=source.config.seed)

    if source.optimizer_updates != source.config.teacher_dose.calibration_updates:
        raise TeacherAllocationSourceError("teacher-dose update count is off protocol")
    panel_families = frozenset(panel_ids)
    train_counts = Counter(
        record.intended_family for record in source.target_train_manifest.records
    )
    gate_counts = Counter(
        record.intended_family for record in source.gate_manifest.records
    )
    if (
        set(train_counts) != panel_families
        or set(train_counts.values()) != {RANDOM_FAMILY_ROWS}
        or set(gate_counts) != panel_families
        or set(gate_counts.values())
        != {source.config.teacher_dose.gate_items_per_family}
    ):
        raise TeacherAllocationSourceError(
            "teacher-dose manifests omit the exact per-panel populations"
        )
    panel_id = canonical_json_hash(source.panel)
    for manifest in (source.target_train_manifest, source.gate_manifest):
        if any(
            panel_families.intersection(record.valid_family_ids)
            != {record.intended_family}
            for record in manifest.records
        ):
            raise TeacherAllocationSourceError(
                "teacher-dose row exposes another protected panel family"
            )
        if (
            manifest.metadata.get("panel_artifact_id") != panel_id
            or manifest.metadata.get("source_broad_manifest_id")
            != source.broad_manifest.manifest_id
            or manifest.metadata.get("source_confirmation_manifest_id")
            != source.confirmation_manifest.manifest_id
        ):
            raise TeacherAllocationSourceError(
                "teacher-dose manifest lineage is inconsistent"
            )
    audit_leakage(source.all_manifests).assert_clean()
    return source


__all__ = [
    "RANDOM_FAMILY_ROWS",
    "TeacherAllocationSourceError",
    "TeacherAllocationSources",
    "validate_teacher_allocation_base_sources",
    "validate_teacher_allocation_sources",
]
