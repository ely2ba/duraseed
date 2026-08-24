"""Parallel-safe TCES family candidate generation and deterministic reduction."""

from __future__ import annotations

from dataclasses import dataclass, replace

from duraseed.data.manifests import TCESTaskManifestRecord, build_tces_record
from duraseed.data.panel_capacity import (
    FamilyCapacityAudit,
    audit_family_split_capacity,
)
from duraseed.data.splits import derive_tces_split_seed, tces_numeric_key
from duraseed.tasks.tces import (
    GeneratedTCESInstance,
    TCESFamilyGenerator,
    TCESGenerator,
    TCESGenerationError,
    TCESGeneratorConfig,
)


@dataclass(frozen=True, slots=True)
class FamilyGenerationJob:
    """Everything needed to generate one fixed family prefix in a worker."""

    template: GeneratedTCESInstance
    generator_config: TCESGeneratorConfig
    root_seed: int
    split: str
    scan_limit: int


@dataclass(frozen=True, slots=True)
class IndexedSplitGenerationJob:
    """One independent indexed candidate from a deterministic split stream."""

    root_seed: int
    generator_config: TCESGeneratorConfig
    split: str
    item_index: int


@dataclass(frozen=True, slots=True)
class FilteredFamilyGenerationJob:
    """One family with filters that are independent of other family outputs."""

    generation: FamilyGenerationJob
    count: int
    forbidden_valid_families: frozenset[str]
    used_numeric: frozenset[object]
    used_content: frozenset[str]


@dataclass(frozen=True, slots=True)
class AuditedFilteredFamilyGenerationJob:
    """A capacity audit followed by an early-stopping filtered generation."""

    generation: FilteredFamilyGenerationJob
    requirements: tuple[tuple[str, int], ...]
    forbidden_records: tuple[TCESTaskManifestRecord, ...]


def generate_split_candidate(job: IndexedSplitGenerationJob) -> GeneratedTCESInstance:
    """Generate one split candidate without changing accepted-sequence ordering."""

    config = replace(job.generator_config, split=job.split)
    return TCESGenerator(
        derive_tces_split_seed(job.root_seed, job.split), config
    ).generate(job.item_index)


def generate_filtered_family(
    job: FilteredFamilyGenerationJob,
) -> tuple[TCESTaskManifestRecord, ...] | None:
    """Generate and reduce one independent family, stopping at the target count."""

    config = replace(
        job.generation.generator_config,
        split=job.generation.split,
        min_valid_families=1,
        max_valid_families=None,
    )
    generator = TCESFamilyGenerator(
        derive_tces_split_seed(job.generation.root_seed, job.generation.split),
        job.generation.template,
        config,
    )
    local_numeric: set[object] = set()
    local_content: set[str] = set()
    selected: list[TCESTaskManifestRecord] = []
    for item_index in range(job.generation.scan_limit):
        try:
            instance = generator.generate(item_index)
        except TCESGenerationError:
            continue
        record = build_tces_record(instance)
        if job.forbidden_valid_families.intersection(record.valid_family_ids):
            continue
        numeric = tces_numeric_key(record)
        if (
            numeric in job.used_numeric
            or numeric in local_numeric
            or record.content_hash in job.used_content
            or record.content_hash in local_content
        ):
            continue
        selected.append(record)
        local_numeric.add(numeric)
        local_content.add(record.content_hash)
        if len(selected) == job.count:
            return tuple(selected)
    return None


def audit_then_generate_filtered(
    job: AuditedFilteredFamilyGenerationJob,
) -> tuple[FamilyCapacityAudit, tuple[TCESTaskManifestRecord, ...] | None]:
    """Audit one family, then stop generation once its required rows exist."""

    generation = job.generation.generation
    audit = audit_family_split_capacity(
        generation.template,
        generation.generator_config,
        root_seed=generation.root_seed,
        requirements=job.requirements,
        forbidden_records=job.forbidden_records,
    )
    candidates = generate_filtered_family(job.generation) if audit.passed else None
    return audit, candidates


def select_family_candidates(
    candidates: tuple[TCESTaskManifestRecord, ...],
    *,
    count: int,
    forbidden_valid_families: frozenset[str],
    used_numeric: set[object],
    used_content: set[str],
) -> tuple[TCESTaskManifestRecord, ...] | None:
    """Apply the original ordered filters and commit identities only on success."""

    local_numeric: set[object] = set()
    local_content: set[str] = set()
    selected: list[TCESTaskManifestRecord] = []
    for record in candidates:
        if forbidden_valid_families.intersection(record.valid_family_ids):
            continue
        numeric = tces_numeric_key(record)
        if (
            numeric in used_numeric
            or numeric in local_numeric
            or record.content_hash in used_content
            or record.content_hash in local_content
        ):
            continue
        selected.append(record)
        local_numeric.add(numeric)
        local_content.add(record.content_hash)
        if len(selected) == count:
            used_numeric.update(local_numeric)
            used_content.update(local_content)
            return tuple(selected)
    return None


__all__ = [
    "AuditedFilteredFamilyGenerationJob",
    "FamilyGenerationJob",
    "FilteredFamilyGenerationJob",
    "IndexedSplitGenerationJob",
    "audit_then_generate_filtered",
    "generate_filtered_family",
    "generate_split_candidate",
    "select_family_candidates",
]
