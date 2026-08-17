"""Persistence for already authenticated calibration source objects."""

from __future__ import annotations

from pathlib import Path

from duraseed.data.io import atomic_write_bytes
from duraseed.data.stage_a_prompt_pools import (
    StageAPromptPoolBundle,
    write_stage_a_prompt_pool_bundle,
)
from duraseed.provenance import canonical_json_bytes, canonical_json_hash
from duraseed.training.teacher_allocation_sources import TeacherAllocationSources


def write_calibration_sources(
    directory: str | Path,
    sources: TeacherAllocationSources,
    prompt_pools: StageAPromptPoolBundle,
) -> None:
    """Persist the exact local manifests consumed by direct-M0 Stage A."""

    root = Path(directory)
    write_stage_a_prompt_pool_bundle(root, prompt_pools)
    atomic_write_bytes(
        root / "source-identities.json",
        canonical_json_bytes(
            {
                "panel_artifact_id": canonical_json_hash(sources.panel),
                "broad_manifest_id": sources.broad_manifest.manifest_id,
                "confirmation_manifest_id": sources.confirmation_manifest.manifest_id,
            }
        ),
    )


__all__ = ["write_calibration_sources"]
