"""Persist the single offline capacity pass before boundary confirmation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from duraseed.boundary_confirmation_resume import (
    BoundaryConfirmationResumeSnapshot,
    CONFIRMATION_RESUME_GROUPS,
    CONFIRMATION_RESUME_SAMPLES,
)
from duraseed.boundary_live_sources import (
    capacity_cleared_confirmation,
    reconstruct_family_templates,
)
from duraseed.config import PilotConfig
from duraseed.data.boundary import assess_refinement_finalist_gate
from duraseed.data.boundary_confirmation import build_confirmation_manifest
from duraseed.data.io import atomic_write_bytes
from duraseed.data.manifests import DatasetManifest, write_manifest
from duraseed.data.panel_capacity import FamilyCapacityAudit, FamilySplitCapacity
from duraseed.runners import RunnerGateError
from duraseed.tasks.tces import TCESGeneratorConfig


CAPACITY_AUDITS_FILE = "extension2_capacity_audits.json"
CONFIRMATION_PREPARATION_FILE = "extension2_confirmation_preparation.json"


@dataclass(frozen=True, slots=True)
class PreparedBoundaryConfirmation:
    snapshot: BoundaryConfirmationResumeSnapshot
    manifest: DatasetManifest
    capacity_audits: tuple[FamilyCapacityAudit, ...]


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    atomic_write_bytes(path, payload + b"\n")


def _load_audits(snapshot: BoundaryConfirmationResumeSnapshot):
    try:
        values = json.loads(
            (snapshot.directory / CAPACITY_AUDITS_FILE).read_text(encoding="utf-8")
        )
        return tuple(
            FamilyCapacityAudit(
                row["family_id"],
                tuple(
                    FamilySplitCapacity(**split) for split in row["split_capacities"]
                ),
                row["available_disjoint_instances"],
                row["passed"],
            )
            for row in values
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerGateError(
            "saved Extension-2 capacity audits are invalid"
        ) from error


def _finalists(snapshot: BoundaryConfirmationResumeSnapshot) -> tuple[str, ...]:
    return tuple(
        sorted(
            row.intended_family_id
            for row in snapshot.refinement_summaries
            if assess_refinement_finalist_gate(row).eligible
        )
    )


def _from_saved(
    snapshot: BoundaryConfirmationResumeSnapshot,
    generator: TCESGeneratorConfig,
    *,
    git_commit: str,
) -> PreparedBoundaryConfirmation | None:
    marker_path = snapshot.directory / CONFIRMATION_PREPARATION_FILE
    if not marker_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest = DatasetManifest.model_validate_json(
            (snapshot.directory / "extension2_confirmation_manifest.json").read_bytes()
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RunnerGateError(
            "saved Extension-2 confirmation preparation is invalid"
        ) from error
    audits = _load_audits(snapshot)
    finalists = _finalists(snapshot)
    cleared = tuple(row.family_id for row in audits if row.passed)
    templates = reconstruct_family_templates(snapshot.extension2, finalists)
    expected_manifest = build_confirmation_manifest(
        generator,
        snapshot.extension2,
        cleared,
        templates={family_id: templates[family_id] for family_id in cleared},
    )
    expected_marker = {
        "schema_version": 1,
        "run_id": snapshot.directory.name,
        "source_group_count": sum(CONFIRMATION_RESUME_GROUPS.values()),
        "source_sample_count": CONFIRMATION_RESUME_SAMPLES,
        "git_commit": git_commit,
        "capacity_family_ids": list(finalists),
        "capacity_passed_family_ids": list(cleared),
        "confirmation_manifest_id": manifest.manifest_id,
    }
    if (
        marker != expected_marker
        or tuple(row.family_id for row in audits) != finalists
        or manifest != expected_manifest
    ):
        raise RunnerGateError("saved Extension-2 confirmation preparation changed")
    return PreparedBoundaryConfirmation(snapshot, manifest, audits)


def prepare_boundary_confirmation(
    snapshot: BoundaryConfirmationResumeSnapshot,
    config: PilotConfig,
    *,
    git_commit: str,
) -> PreparedBoundaryConfirmation:
    """Run or reload the one ordered capacity pass before any remote client."""

    generator = TCESGeneratorConfig(**config.tasks.tces.generator_kwargs())
    saved = _from_saved(snapshot, generator, git_commit=git_commit)
    if saved is not None:
        return saved
    manifest, audits = capacity_cleared_confirmation(
        generator, snapshot.extension2, snapshot.refinement_summaries
    )
    finalists = _finalists(snapshot)
    if tuple(row.family_id for row in audits) != finalists:
        raise RunnerGateError("Extension-2 capacity audit order changed")
    write_manifest(
        snapshot.directory / "extension2_confirmation_manifest.json", manifest
    )
    _write_json(
        snapshot.directory / CAPACITY_AUDITS_FILE, [asdict(row) for row in audits]
    )
    _write_json(
        snapshot.directory / CONFIRMATION_PREPARATION_FILE,
        {
            "schema_version": 1,
            "run_id": snapshot.directory.name,
            "source_group_count": sum(CONFIRMATION_RESUME_GROUPS.values()),
            "source_sample_count": CONFIRMATION_RESUME_SAMPLES,
            "git_commit": git_commit,
            "capacity_family_ids": list(finalists),
            "capacity_passed_family_ids": [
                row.family_id for row in audits if row.passed
            ],
            "confirmation_manifest_id": manifest.manifest_id,
        },
    )
    return PreparedBoundaryConfirmation(snapshot, manifest, audits)


__all__ = [
    "CAPACITY_AUDITS_FILE",
    "CONFIRMATION_PREPARATION_FILE",
    "PreparedBoundaryConfirmation",
    "prepare_boundary_confirmation",
]
