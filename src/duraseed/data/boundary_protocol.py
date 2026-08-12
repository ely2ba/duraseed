"""Deterministic cohort construction for the TCES boundary protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from duraseed.data.leakage import LeakageAuditError, audit_leakage
from duraseed.data.manifests import (
    DatasetManifest,
    build_manifest,
    build_tces_record,
)
from duraseed.data.splits import TCESSplitBuilder, derive_tces_split_seed
from duraseed.tasks.tces import TCESFamilyGenerator, TCESGeneratorConfig


BOUNDARY_ENGINEERING_SEED = 5
BOUNDARY_BROAD_FAMILY_COUNT = 64
BOUNDARY_BROAD_ITEMS_PER_FAMILY = 4
BOUNDARY_BROAD_SAMPLES_PER_ITEM = 4
BOUNDARY_BROAD_TEMPLATE_SCAN_CEILING = 512
BOUNDARY_BROAD_VARIANT_INDEX_OFFSET = 512
BOUNDARY_BROAD_EXTENSION_1_VARIANT_INDEX_OFFSET = 2048
BOUNDARY_BROAD_EXTENSION_2_VARIANT_INDEX_OFFSET = 2304
BOUNDARY_BROAD_INITIAL_COHORT = "initial"
BOUNDARY_BROAD_EXTENSION_1_COHORT = "extension_1"
BOUNDARY_BROAD_EXTENSION_2_COHORT = "extension_2"
BoundaryBroadCohort = Literal["initial", "extension_1", "extension_2"]
BOUNDARY_BROAD_COHORTS: tuple[BoundaryBroadCohort, ...] = (
    BOUNDARY_BROAD_INITIAL_COHORT,
    BOUNDARY_BROAD_EXTENSION_1_COHORT,
    BOUNDARY_BROAD_EXTENSION_2_COHORT,
)


class BoundaryProtocolError(ValueError):
    """A cohort differs from the frozen deterministic protocol."""


def cohort_variant_index_offset(cohort: BoundaryBroadCohort) -> int:
    if cohort == BOUNDARY_BROAD_INITIAL_COHORT:
        return BOUNDARY_BROAD_VARIANT_INDEX_OFFSET
    if cohort == BOUNDARY_BROAD_EXTENSION_1_COHORT:
        return BOUNDARY_BROAD_EXTENSION_1_VARIANT_INDEX_OFFSET
    if cohort == BOUNDARY_BROAD_EXTENSION_2_COHORT:
        return BOUNDARY_BROAD_EXTENSION_2_VARIANT_INDEX_OFFSET
    raise ValueError("unknown broad boundary cohort")


def build_broad_manifest(
    generator_config: TCESGeneratorConfig,
    *,
    root_seed: int = BOUNDARY_ENGINEERING_SEED,
    family_count: int = BOUNDARY_BROAD_FAMILY_COUNT,
    items_per_family: int = BOUNDARY_BROAD_ITEMS_PER_FAMILY,
    template_scan_ceiling: int = BOUNDARY_BROAD_TEMPLATE_SCAN_CEILING,
    variant_index_offset: int | None = None,
    cohort: BoundaryBroadCohort = BOUNDARY_BROAD_INITIAL_COHORT,
) -> DatasetManifest:
    """Materialize exact-family numeric variants under one split seed."""

    if family_count < 1 or items_per_family < 2:
        raise ValueError("boundary manifest requires families and repeated items")
    if cohort not in BOUNDARY_BROAD_COHORTS:
        raise ValueError("unknown broad boundary cohort")
    cohort_ordinal = BOUNDARY_BROAD_COHORTS.index(cohort)
    if cohort != BOUNDARY_BROAD_INITIAL_COHORT and (
        family_count != BOUNDARY_BROAD_FAMILY_COUNT
        or items_per_family != BOUNDARY_BROAD_ITEMS_PER_FAMILY
    ):
        raise ValueError("extension cohorts use the fixed broad-screen dimensions")
    family_ordinal_stop = cohort_ordinal * BOUNDARY_BROAD_FAMILY_COUNT + family_count
    if template_scan_ceiling < family_ordinal_stop:
        raise ValueError("template scan ceiling is smaller than the family count")
    expected_offset = cohort_variant_index_offset(cohort)
    variant_index_offset = (
        expected_offset if variant_index_offset is None else variant_index_offset
    )
    if cohort != BOUNDARY_BROAD_INITIAL_COHORT and (
        variant_index_offset != expected_offset
    ):
        raise ValueError("extension cohort variant offset is fixed")
    if variant_index_offset < template_scan_ceiling:
        raise ValueError("variant indices must follow the template scan range")

    split = "a_candidate"
    candidates = TCESSplitBuilder(root_seed, generator_config).lazy_split(
        split, size=template_scan_ceiling
    )
    distinct_templates = []
    seen_families: set[str] = set()
    for accepted_index in range(template_scan_ceiling):
        template = candidates[accepted_index]
        if template.intended_family in seen_families:
            continue
        distinct_templates.append(template)
        seen_families.add(template.intended_family)
        if len(distinct_templates) == family_ordinal_stop:
            break
    if len(distinct_templates) != family_ordinal_stop:
        raise BoundaryProtocolError(
            "candidate prefix did not contain enough distinct exact TCES families"
        )

    split_seed = derive_tces_split_seed(root_seed, split)
    family_config = replace(generator_config, split=split)
    manifests: list[DatasetManifest] = []
    prior_family_ids: set[str] = set()
    prior_task_ids: set[str] = set()
    for block_ordinal in range(cohort_ordinal + 1):
        block_cohort = BOUNDARY_BROAD_COHORTS[block_ordinal]
        block_family_start = block_ordinal * BOUNDARY_BROAD_FAMILY_COUNT
        block_family_count = (
            family_count
            if block_cohort == BOUNDARY_BROAD_INITIAL_COHORT
            else BOUNDARY_BROAD_FAMILY_COUNT
        )
        block_items_per_family = (
            items_per_family
            if block_cohort == BOUNDARY_BROAD_INITIAL_COHORT
            else BOUNDARY_BROAD_ITEMS_PER_FAMILY
        )
        block_templates = distinct_templates[
            block_family_start : block_family_start + block_family_count
        ]
        block_offset = (
            variant_index_offset
            if block_cohort == cohort
            else cohort_variant_index_offset(block_cohort)
        )
        records = []
        template_metadata = []
        for family_ordinal, template in enumerate(block_templates):
            start_index = block_offset + family_ordinal * block_items_per_family
            variants = TCESFamilyGenerator(
                split_seed, template, family_config
            ).generate_many(block_items_per_family, start_index=start_index)
            records.extend(build_tces_record(variant) for variant in variants)
            template_metadata.append(
                {
                    "family_id": template.intended_family,
                    "template_content_hash": template.content_hash,
                    "template_item_index": template.item_index,
                    "variant_item_indices": [
                        variant.item_index for variant in variants
                    ],
                }
            )
        family_ids = {record.intended_family for record in records}
        task_ids = {record.task_id for record in records}
        if family_ids.intersection(prior_family_ids):
            raise BoundaryProtocolError("broad cohorts overlap in exact family")
        if task_ids.intersection(prior_task_ids):
            raise BoundaryProtocolError("broad cohorts overlap in numeric task")
        metadata: dict[str, Any] = {
            "scope": "m0_boundary_broad_screen",
            "scientific_manifest": False,
            "generation_mode": "exact_family_numeric_variants_v1",
            "family_count": block_family_count,
            "items_per_family": block_items_per_family,
            "templates": template_metadata,
        }
        parent_id = manifests[-1].manifest_id if manifests else None
        name = "m0-boundary-broad-a-candidate"
        if block_cohort != BOUNDARY_BROAD_INITIAL_COHORT:
            name = f"{name}-{block_cohort.replace('_', '-')}"
            metadata["cohort_provenance"] = {
                "cohort_id": block_cohort,
                "cohort_ordinal": block_ordinal,
                "distinct_family_ordinal_start": block_family_start,
                "distinct_family_ordinal_stop": block_family_start + block_family_count,
                "selection_order": "first_appearance_in_a_candidate",
                "prior_cohort_ids": list(BOUNDARY_BROAD_COHORTS[:block_ordinal]),
                "prior_family_count": len(prior_family_ids),
                "prior_task_count": len(prior_task_ids),
                "immediate_parent_manifest_id": parent_id,
                "variant_item_index_start": block_offset,
                "variant_item_index_stop": block_offset
                + block_family_count * block_items_per_family,
            }
        manifests.append(
            build_manifest(
                name=name,
                split=split,
                generator_version="1.0.0",
                root_seed=root_seed,
                records=records,
                parent_manifest_id=parent_id,
                metadata=metadata,
            )
        )
        prior_family_ids.update(family_ids)
        prior_task_ids.update(task_ids)
    return manifests[-1]


def audit_new_broad_cohort(
    manifest: DatasetManifest,
    prior_broad_manifests: Sequence[DatasetManifest] = (),
    prior_confirmation_manifests: Sequence[DatasetManifest] = (),
) -> dict[str, Any]:
    """Check exact-family and numeric disjointness before sampling a cohort."""

    broad = tuple(prior_broad_manifests)
    confirmation = tuple(prior_confirmation_manifests)
    if len(broad) != len(confirmation):
        raise ValueError("prior broad and confirmation cohorts must be paired")
    if broad and manifest.parent_manifest_id != broad[-1].manifest_id:
        raise BoundaryProtocolError(
            "new broad manifest does not extend the prior chain"
        )
    prior_family_ids = {
        record.intended_family for prior in broad for record in prior.records
    }
    if prior_family_ids.intersection(
        record.intended_family for record in manifest.records
    ):
        raise BoundaryProtocolError("new broad cohort overlaps a prior exact family")
    records = (
        tuple(
            record
            for pair in zip(broad, confirmation, strict=True)
            for prior in pair
            for record in prior.records
        )
        + manifest.records
    )
    try:
        audit_leakage(records).assert_clean()
    except LeakageAuditError as error:
        raise BoundaryProtocolError(
            "new broad cohort overlaps prior task, content, or numeric evidence"
        ) from error
    return {
        "status": "passed",
        "prior_cohort_count": len(broad),
        "prior_record_count": len(records) - len(manifest.records),
        "new_record_count": len(manifest.records),
        "checked_dimensions": [
            "intended_family",
            "task_id",
            "content_hash",
            "numeric_identity",
        ],
    }


def validate_broad_cohort_provenance(
    manifest: DatasetManifest,
    plan: Mapping[str, Any],
    *,
    expected_cohort: BoundaryBroadCohort,
    expected_parent_manifest_id: str | None,
) -> None:
    """Authenticate a prior cohort's manifest, lineage, and frozen plan fields."""

    if expected_cohort not in BOUNDARY_BROAD_COHORTS:
        raise ValueError("unknown broad boundary cohort")
    cohort_ordinal = BOUNDARY_BROAD_COHORTS.index(expected_cohort)
    ordinal_start = cohort_ordinal * BOUNDARY_BROAD_FAMILY_COUNT
    ordinal_stop = ordinal_start + BOUNDARY_BROAD_FAMILY_COUNT
    if (
        manifest.parent_manifest_id != expected_parent_manifest_id
        or manifest.root_seed != BOUNDARY_ENGINEERING_SEED
        or manifest.split != "a_candidate"
        or manifest.record_count
        != BOUNDARY_BROAD_FAMILY_COUNT * BOUNDARY_BROAD_ITEMS_PER_FAMILY
    ):
        raise BoundaryProtocolError(
            "prior broad manifest differs from its fixed cohort"
        )
    families = {record.intended_family for record in manifest.records}
    templates = manifest.metadata.get("templates")
    if (
        len(families) != BOUNDARY_BROAD_FAMILY_COUNT
        or not isinstance(templates, list)
        or len(templates) != BOUNDARY_BROAD_FAMILY_COUNT
        or any(not isinstance(row, Mapping) for row in templates)
        or [str(row.get("family_id", "")) for row in templates]
        != list(dict.fromkeys(str(row.get("family_id", "")) for row in templates))
        or {str(row.get("family_id", "")) for row in templates} != families
    ):
        raise BoundaryProtocolError("prior broad family provenance is incomplete")
    measurement = plan.get("measurement")
    if (
        not isinstance(measurement, Mapping)
        or measurement.get("manifest_id") != manifest.manifest_id
    ):
        raise BoundaryProtocolError("prior broad preflight differs from its manifest")
    if expected_cohort == BOUNDARY_BROAD_INITIAL_COHORT:
        if (
            manifest.name != "m0-boundary-broad-a-candidate"
            or "cohort_provenance" in manifest.metadata
        ):
            raise BoundaryProtocolError("prior initial broad manifest identity changed")
        return
    provenance = manifest.metadata.get("cohort_provenance")
    recorded = measurement.get("cohort_provenance")
    expected_range = cohort_variant_index_offset(expected_cohort)
    expected_values = {
        "cohort_id": expected_cohort,
        "cohort_ordinal": cohort_ordinal,
        "distinct_family_ordinal_start": ordinal_start,
        "distinct_family_ordinal_stop": ordinal_stop,
        "selection_order": "first_appearance_in_a_candidate",
        "variant_item_index_start": expected_range,
        "variant_item_index_stop": expected_range
        + BOUNDARY_BROAD_FAMILY_COUNT * BOUNDARY_BROAD_ITEMS_PER_FAMILY,
    }
    if not isinstance(provenance, Mapping) or not isinstance(recorded, Mapping):
        raise BoundaryProtocolError("prior extension omitted cohort provenance")
    if any(
        provenance.get(key) != value or recorded.get(key) != value
        for key, value in expected_values.items()
    ) or (
        provenance.get("immediate_parent_manifest_id") != expected_parent_manifest_id
        or recorded.get("parent_manifest_id") != expected_parent_manifest_id
    ):
        raise BoundaryProtocolError("prior extension cohort provenance changed")


__all__ = [
    "BOUNDARY_BROAD_COHORTS",
    "BOUNDARY_BROAD_EXTENSION_1_COHORT",
    "BOUNDARY_BROAD_EXTENSION_2_COHORT",
    "BOUNDARY_BROAD_FAMILY_COUNT",
    "BOUNDARY_BROAD_INITIAL_COHORT",
    "BOUNDARY_BROAD_ITEMS_PER_FAMILY",
    "BOUNDARY_BROAD_SAMPLES_PER_ITEM",
    "BOUNDARY_ENGINEERING_SEED",
    "BoundaryBroadCohort",
    "BoundaryProtocolError",
    "audit_new_broad_cohort",
    "build_broad_manifest",
    "cohort_variant_index_offset",
    "validate_broad_cohort_provenance",
]
