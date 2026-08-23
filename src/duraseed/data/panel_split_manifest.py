"""Deterministic teacher-source manifests for the frozen family panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import replace

from duraseed.config import PilotConfig
from duraseed.data.boundary_confirmation import regenerate_family_templates
from duraseed.data.family_generation import (
    FamilyGenerationJob,
    FilteredFamilyGenerationJob,
    generate_filtered_family,
    select_family_candidates,
)
from duraseed.data.manifests import (
    DatasetManifest,
    TCESTaskManifestRecord,
    build_manifest,
    build_tces_record,
)
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.data.splits import derive_tces_split_seed, tces_numeric_key
from duraseed.provenance import canonical_json_hash
from duraseed.tasks.tces import (
    GeneratedTCESInstance,
    TCESFamilyGenerator,
    TCESGenerationError,
    TCESGeneratorConfig,
)


PANEL_SPLIT_SCAN_MULTIPLIER = 32


class PanelSplitManifestError(ValueError):
    """The selected panel cannot supply the requested isolated records."""


def build_panel_split_manifest(
    config: PilotConfig,
    *,
    artifact: FamilyPanelArtifact,
    broad_manifest: DatasetManifest,
    confirmation_manifest: DatasetManifest,
    split: str,
    items_per_family: int,
    forbidden_records: Sequence[TCESTaskManifestRecord],
    executor: Executor | None = None,
    templates: Mapping[str, GeneratedTCESInstance] | None = None,
) -> DatasetManifest:
    """Build one panel-only split without reusing numeric or semantic content."""

    family_ids = tuple(
        sorted((*artifact.panel_a_family_ids, *artifact.panel_b_family_ids))
    )
    generator_config = TCESGeneratorConfig(**config.tasks.tces.generator_kwargs())
    templates = (
        regenerate_family_templates(generator_config, broad_manifest, family_ids)
        if templates is None
        else {family_id: templates[family_id] for family_id in family_ids}
    )
    split_seed = derive_tces_split_seed(config.seed, split)
    split_config = replace(
        generator_config,
        split=split,
        min_valid_families=1,
        max_valid_families=None,
    )
    protected = frozenset(family_ids)
    used_numeric = {tces_numeric_key(record) for record in forbidden_records}
    used_content = {record.content_hash for record in forbidden_records}
    records: list[TCESTaskManifestRecord] = []
    scan_limit = items_per_family * PANEL_SPLIT_SCAN_MULTIPLIER
    if executor is None:
        candidate_sets = (None,) * len(family_ids)
    else:
        candidate_sets = tuple(
            executor.map(
                generate_filtered_family,
                (
                    FilteredFamilyGenerationJob(
                        FamilyGenerationJob(
                            templates[family_id],
                            generator_config,
                            config.seed,
                            split,
                            scan_limit,
                        ),
                        items_per_family,
                        protected.difference({family_id}),
                        frozenset(used_numeric),
                        frozenset(used_content),
                    )
                    for family_id in family_ids
                ),
            )
        )
    for family_id, candidates in zip(family_ids, candidate_sets, strict=True):
        if candidates is None:
            generator = TCESFamilyGenerator(
                split_seed, templates[family_id], split_config
            )
            accepted = 0
            for item_index in range(scan_limit):
                try:
                    instance = generator.generate(item_index)
                except TCESGenerationError:
                    continue
                if (
                    instance.intended_family != family_id
                    or frozenset(instance.valid_family_ids).intersection(protected)
                    != {family_id}
                    or tces_numeric_key(instance) in used_numeric
                    or instance.content_hash in used_content
                ):
                    continue
                record = build_tces_record(instance)
                records.append(record)
                used_numeric.add(tces_numeric_key(instance))
                used_content.add(instance.content_hash)
                accepted += 1
                if accepted == items_per_family:
                    break
        else:
            selected = select_family_candidates(
                candidates,
                count=items_per_family,
                forbidden_valid_families=protected.difference({family_id}),
                used_numeric=used_numeric,
                used_content=used_content,
            )
            accepted = 0 if selected is None else len(selected)
            if selected is None:
                generator = TCESFamilyGenerator(
                    split_seed, templates[family_id], split_config
                )
                selected_records: list[TCESTaskManifestRecord] = []
                for item_index in range(scan_limit):
                    try:
                        instance = generator.generate(item_index)
                    except TCESGenerationError:
                        continue
                    if (
                        instance.intended_family != family_id
                        or frozenset(instance.valid_family_ids).intersection(protected)
                        != {family_id}
                        or tces_numeric_key(instance) in used_numeric
                        or instance.content_hash in used_content
                    ):
                        continue
                    record = build_tces_record(instance)
                    selected_records.append(record)
                    used_numeric.add(tces_numeric_key(instance))
                    used_content.add(instance.content_hash)
                    if len(selected_records) == items_per_family:
                        break
                accepted = len(selected_records)
                records.extend(selected_records)
            else:
                records.extend(selected)
        if accepted != items_per_family:
            raise PanelSplitManifestError(
                f"{split} cannot supply {items_per_family} isolated items for "
                f"panel family {family_id}"
            )
    return build_manifest(
        name=f"teacher-dose-{split}",
        split=split,
        generator_version="1.0.0",
        root_seed=config.seed,
        records=records,
        metadata={
            "scope": "teacher_dose_calibration",
            "scientific_manifest": False,
            "panel_artifact_id": canonical_json_hash(artifact.model_dump(mode="json")),
            "families": list(family_ids),
            "items_per_family": items_per_family,
            "panel_family_isolation": (
                "valid_panel_family_intersection_is_intended_only"
            ),
            "source_broad_manifest_id": broad_manifest.manifest_id,
            "source_confirmation_manifest_id": confirmation_manifest.manifest_id,
        },
    )


__all__ = [
    "PANEL_SPLIT_SCAN_MULTIPLIER",
    "PanelSplitManifestError",
    "build_panel_split_manifest",
]
