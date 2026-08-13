"""Deterministic refinement and confirmation inputs for boundary cohorts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from duraseed.data.boundary import (
    BoundaryFamilySummary,
    assess_confirmation_observation_gate,
    assess_refinement_finalist_gate,
)
from duraseed.data.boundary_protocol import (
    BOUNDARY_BROAD_EXTENSION_1_COHORT,
    BOUNDARY_BROAD_EXTENSION_2_COHORT,
    BOUNDARY_BROAD_INITIAL_COHORT,
    BOUNDARY_ENGINEERING_SEED,
)
from duraseed.data.manifests import (
    DatasetManifest,
    build_manifest,
    build_tces_record,
)
from duraseed.data.panel_capacity import FamilyCapacityAudit
from duraseed.data.splits import TCESSplitBuilder, derive_tces_split_seed
from duraseed.provenance import derive_namespaced_seed
from duraseed.tasks.tces import (
    GeneratedTCESInstance,
    TCESFamilyGenerator,
    TCESGeneratorConfig,
)


BOUNDARY_REFINEMENT_AUDIT_FAMILY_COUNT = 12
BOUNDARY_REFINEMENT_TOTAL_SAMPLES_PER_ITEM = 16
BOUNDARY_CONFIRMATION_ITEMS_PER_FAMILY = 4
BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM = 16
BOUNDARY_CONFIRMATION_VARIANT_INDEX_OFFSET = 1024
_AUDIT_SEED_NAMESPACE = "tinker.boundary_refinement.zero_family_audit"
_BROAD_COHORT_STARTS = {
    BOUNDARY_BROAD_INITIAL_COHORT: 0,
    BOUNDARY_BROAD_EXTENSION_1_COHORT: 64,
    BOUNDARY_BROAD_EXTENSION_2_COHORT: 128,
}


class BoundaryConfirmationError(ValueError):
    """Confirmation inputs or evidence differ from the frozen protocol."""


@dataclass(frozen=True, slots=True)
class ConfirmationEvidence:
    stage2_finalist_family_ids: tuple[str, ...]
    capacity_cleared_family_ids: tuple[str, ...]
    observation_eligible_family_ids: tuple[str, ...]
    final_eligible_family_ids: tuple[str, ...]


def choose_refinement_family_ids(
    family_successes: Mapping[str, int],
    *,
    audit_family_count: int = BOUNDARY_REFINEMENT_AUDIT_FAMILY_COUNT,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep every success-positive family plus a seeded zero-success audit."""

    if not family_successes or any(
        not isinstance(family_id, str)
        or not family_id.strip()
        or isinstance(successes, bool)
        or not isinstance(successes, int)
        or successes < 0
        for family_id, successes in family_successes.items()
    ):
        raise ValueError("family_successes must contain named nonnegative counts")
    if (
        isinstance(audit_family_count, bool)
        or not isinstance(audit_family_count, int)
        or audit_family_count < 1
    ):
        raise ValueError("audit_family_count must be a positive integer")
    candidates = tuple(
        sorted(
            family_id
            for family_id, successes in family_successes.items()
            if successes > 0
        )
    )
    rejected = tuple(
        sorted(
            family_id
            for family_id, successes in family_successes.items()
            if successes == 0
        )
    )
    audit = tuple(
        sorted(
            rejected,
            key=lambda family_id: (
                derive_namespaced_seed(
                    BOUNDARY_ENGINEERING_SEED,
                    _AUDIT_SEED_NAMESPACE,
                    family_id,
                ),
                family_id,
            ),
        )[:audit_family_count]
    )
    return candidates, audit


def broad_cohort(manifest: DatasetManifest) -> tuple[str, int]:
    """Return a manifest's frozen cohort ID and global family ordinal start."""

    provenance = manifest.metadata.get("cohort_provenance")
    if provenance is None:
        return BOUNDARY_BROAD_INITIAL_COHORT, 0
    if not isinstance(provenance, Mapping):
        raise BoundaryConfirmationError("broad cohort provenance is malformed")
    cohort = provenance.get("cohort_id")
    start = provenance.get("distinct_family_ordinal_start")
    stop = provenance.get("distinct_family_ordinal_stop")
    if (
        not isinstance(cohort, str)
        or cohort not in _BROAD_COHORT_STARTS
        or isinstance(start, bool)
        or not isinstance(start, int)
        or start != _BROAD_COHORT_STARTS[cohort]
        or stop != start + 64
    ):
        raise BoundaryConfirmationError("broad cohort ordinal range is off protocol")
    return cohort, start


def regenerate_family_templates(
    generator_config: TCESGeneratorConfig,
    broad_manifest: DatasetManifest,
    family_ids: Sequence[str],
) -> dict[str, Any]:
    requested = tuple(sorted(set(family_ids)))
    if not requested or len(requested) != len(tuple(family_ids)):
        raise ValueError("family IDs must be nonempty and unique")
    templates_metadata = broad_manifest.metadata.get("templates")
    if not isinstance(templates_metadata, list):
        raise BoundaryConfirmationError("broad manifest omitted template provenance")
    metadata_by_family: dict[str, Mapping[str, Any]] = {}
    for value in templates_metadata:
        if not isinstance(value, Mapping) or not isinstance(
            value.get("family_id"), str
        ):
            raise BoundaryConfirmationError("invalid broad template provenance")
        metadata_by_family[str(value["family_id"])] = value
    if not set(requested).issubset(metadata_by_family):
        raise BoundaryConfirmationError("a requested family is absent from provenance")
    largest_index = max(
        int(metadata_by_family[family_id]["template_item_index"])
        for family_id in requested
    )
    candidates = TCESSplitBuilder(
        BOUNDARY_ENGINEERING_SEED, generator_config
    ).lazy_split("a_candidate", size=largest_index + 1)
    templates = {}
    for family_id in requested:
        metadata = metadata_by_family[family_id]
        template = candidates[int(metadata["template_item_index"])]
        if (
            template.intended_family != family_id
            or template.content_hash != metadata.get("template_content_hash")
        ):
            raise BoundaryConfirmationError(
                "regenerated template differs from broad provenance"
            )
        templates[family_id] = template
    return templates


def build_confirmation_manifest(
    generator_config: TCESGeneratorConfig,
    broad_manifest: DatasetManifest,
    finalist_family_ids: Sequence[str],
    *,
    templates: Mapping[str, GeneratedTCESInstance] | None = None,
) -> DatasetManifest:
    """Generate four new deterministic numeric items for each finalist."""

    supplied = tuple(finalist_family_ids)
    finalists = tuple(sorted(set(supplied)))
    if len(finalists) != len(supplied):
        raise ValueError("finalist family IDs must be unique")
    if broad_manifest.task_family != "tces" or broad_manifest.split != "a_candidate":
        raise BoundaryConfirmationError("confirmation requires the broad manifest")
    templates_metadata = broad_manifest.metadata.get("templates")
    if not isinstance(templates_metadata, list):
        raise BoundaryConfirmationError("broad manifest omitted template provenance")
    metadata_by_family = {
        str(value["family_id"]): value
        for value in templates_metadata
        if isinstance(value, Mapping) and isinstance(value.get("family_id"), str)
    }
    cohort, global_start = broad_cohort(broad_manifest)
    if not finalists and cohort == BOUNDARY_BROAD_INITIAL_COHORT:
        raise ValueError("the initial confirmation requires finalist family IDs")
    if templates is not None and set(templates) != set(finalists):
        raise BoundaryConfirmationError("template overrides differ from finalists")
    templates = templates or (
        regenerate_family_templates(generator_config, broad_manifest, finalists)
        if finalists
        else {}
    )
    split = "a_candidate"
    split_seed = derive_tces_split_seed(BOUNDARY_ENGINEERING_SEED, split)
    family_config = replace(generator_config, split=split)
    broad_task_ids = {record.task_id for record in broad_manifest.records}
    records = []
    held_out_metadata = []
    for family_id in finalists:
        metadata = metadata_by_family[family_id]
        template = templates[family_id]
        local_ordinal = next(
            index
            for index, value in enumerate(templates_metadata)
            if value["family_id"] == family_id
        )
        start_index = (
            BOUNDARY_CONFIRMATION_VARIANT_INDEX_OFFSET
            + (global_start + local_ordinal) * BOUNDARY_CONFIRMATION_ITEMS_PER_FAMILY
        )
        variants = TCESFamilyGenerator(
            split_seed, template, family_config
        ).generate_many(BOUNDARY_CONFIRMATION_ITEMS_PER_FAMILY, start_index=start_index)
        records.extend(build_tces_record(variant) for variant in variants)
        held_out_metadata.append(
            {
                "family_id": family_id,
                "template_item_index": int(metadata["template_item_index"]),
                "held_out_item_indices": [variant.item_index for variant in variants],
            }
        )
    if broad_task_ids.intersection(record.task_id for record in records):
        raise BoundaryConfirmationError("held-out confirmation repeats a broad task")
    metadata: dict[str, Any] = {
        "scope": "m0_boundary_finalist_confirmation",
        "scientific_manifest": False,
        "generation_mode": "exact_family_held_out_numeric_variants_v1",
        "family_count": len(finalists),
        "items_per_family": BOUNDARY_CONFIRMATION_ITEMS_PER_FAMILY,
        "source_broad_manifest_id": broad_manifest.manifest_id,
        "families": held_out_metadata,
    }
    name = "m0-boundary-finalist-confirmation-a-candidate"
    if cohort != BOUNDARY_BROAD_INITIAL_COHORT:
        name = f"{name}-{cohort.replace('_', '-')}"
        metadata["cohort_provenance"] = {
            "cohort_id": cohort,
            "distinct_family_ordinal_start": global_start,
            "distinct_family_ordinal_stop": global_start + 64,
            "held_out_item_index_start": BOUNDARY_CONFIRMATION_VARIANT_INDEX_OFFSET
            + global_start * BOUNDARY_CONFIRMATION_ITEMS_PER_FAMILY,
            "held_out_item_index_stop": BOUNDARY_CONFIRMATION_VARIANT_INDEX_OFFSET
            + (global_start + 64) * BOUNDARY_CONFIRMATION_ITEMS_PER_FAMILY,
        }
    return build_manifest(
        name=name,
        split=split,
        generator_version="1.0.0",
        root_seed=BOUNDARY_ENGINEERING_SEED,
        records=records,
        task_family="tces",
        metadata=metadata,
    )


def reduce_confirmation_evidence(
    refinement_summaries: Sequence[BoundaryFamilySummary],
    confirmation_summaries: Sequence[BoundaryFamilySummary],
    refined_family_ids: Sequence[str],
    capacity_audits: Sequence[FamilyCapacityAudit],
) -> ConfirmationEvidence:
    """Reduce frozen gates and capacity rows without selecting panels."""

    refined = set(refined_family_ids)
    if not refined or len(refined) != len(tuple(refined_family_ids)):
        raise BoundaryConfirmationError(
            "refined family IDs must be nonempty and unique"
        )
    refinement_by_family = {row.intended_family_id: row for row in refinement_summaries}
    confirmation_by_family = {
        row.intended_family_id: row for row in confirmation_summaries
    }
    if (
        len(refinement_by_family) != len(tuple(refinement_summaries))
        or len(confirmation_by_family) != len(tuple(confirmation_summaries))
        or not refined.issubset(refinement_by_family)
    ):
        raise BoundaryConfirmationError("refinement summary grid is ambiguous")
    if any(
        refinement_by_family[family_id].item_count != 4
        or any(
            item.trials != BOUNDARY_REFINEMENT_TOTAL_SAMPLES_PER_ITEM
            for item in refinement_by_family[family_id].items
        )
        for family_id in refined
    ):
        raise BoundaryConfirmationError("refinement evidence grid is incomplete")
    stage2 = tuple(
        sorted(
            family_id
            for family_id in refined
            if assess_refinement_finalist_gate(refinement_by_family[family_id]).eligible
        )
    )
    capacity_by_family = {row.family_id: row for row in capacity_audits}
    if tuple(row.family_id for row in capacity_audits) != stage2:
        raise BoundaryConfirmationError(
            "capacity evidence differs from Stage-2 finalist order"
        )
    cleared = tuple(
        family_id
        for family_id in stage2
        if family_id in capacity_by_family and capacity_by_family[family_id].passed
    )
    if not set(cleared).issubset(confirmation_by_family):
        raise BoundaryConfirmationError("confirmation summary grid is incomplete")
    if any(
        confirmation_by_family[family_id].item_count != 8
        or any(
            item.trials != BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM
            for item in confirmation_by_family[family_id].items
        )
        for family_id in cleared
    ):
        raise BoundaryConfirmationError("confirmation item grid is incomplete")
    observation = tuple(
        sorted(
            family_id
            for family_id in cleared
            if assess_confirmation_observation_gate(
                confirmation_by_family[family_id]
            ).eligible
        )
    )
    return ConfirmationEvidence(
        stage2_finalist_family_ids=stage2,
        capacity_cleared_family_ids=cleared,
        observation_eligible_family_ids=observation,
        final_eligible_family_ids=tuple(
            family_id
            for family_id in observation
            if capacity_by_family[family_id].passed
        ),
    )


__all__ = [
    "BOUNDARY_CONFIRMATION_ITEMS_PER_FAMILY",
    "BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM",
    "BOUNDARY_REFINEMENT_AUDIT_FAMILY_COUNT",
    "BOUNDARY_REFINEMENT_TOTAL_SAMPLES_PER_ITEM",
    "BoundaryConfirmationError",
    "ConfirmationEvidence",
    "broad_cohort",
    "build_confirmation_manifest",
    "choose_refinement_family_ids",
    "reduce_confirmation_evidence",
    "regenerate_family_templates",
]
