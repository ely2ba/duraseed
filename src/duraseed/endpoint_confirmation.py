"""Fresh item-disjoint arithmetic confirmation using the existing family generator."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import os

from duraseed.data.family_generation import (
    FamilyGenerationJob,
    FilteredFamilyGenerationJob,
    generate_filtered_family,
)
from duraseed.data.manifests import build_manifest
from duraseed.data.splits import tces_numeric_key
from duraseed.pilot0_data import _family_expression
from duraseed.tasks.tces import GeneratedTCESInstance, generate_teacher_trace
from duraseed.tasks.tces.enumerate import enumerate_task

CONFIRMATION_SEED = 20260910
CONFIRMATION_SPLIT = "a_validation"


def source_records(source):
    """Exclude whole acquisition pools, not only prompts used by the new teacher."""
    return (
        *source.prompt_pools.a_rl_train_manifest.records,
        *source.prompt_pools.a_monitor_manifest.records,
        *source.a_validation.records,
    )


def _generate_family(job):
    record, count, config, forbidden, used_numeric, used_content = job
    expression = _family_expression(record)
    enumeration = enumerate_task(record.to_task())
    if (
        not enumeration.complete
        or enumeration.family_ids != record.valid_family_ids
        or len(enumeration.expressions) != record.valid_expression_count
    ):
        raise ValueError("confirmation template differs from the retained exact record")
    template = GeneratedTCESInstance(
        record.to_task(),
        record.content_hash,
        expression,
        record.intended_family,
        enumeration,
        generate_teacher_trace(expression),
        record.generator_seed,
        record.item_index,
        record.accepted_attempt,
    )
    rows = generate_filtered_family(
        FilteredFamilyGenerationJob(
            FamilyGenerationJob(
                template, config, CONFIRMATION_SEED, CONFIRMATION_SPLIT, count * 32
            ),
            count,
            forbidden,
            used_numeric,
            used_content,
        )
    )
    if rows is None:
        raise ValueError(
            f"confirmation family capacity unavailable: {record.intended_family}"
        )
    return rows


def validate_confirmation(source, manifest):
    existing = source_records(source)
    forbidden_numeric = {tces_numeric_key(row) for row in existing}
    forbidden_content = {row.content_hash for row in existing}
    target = set(source.prompt_pools.artifact.boundary_family_ids)
    sentinel = set(source.prompt_pools.artifact.sentinel_family_ids)
    protected = target | sentinel
    expected = Counter(row.intended_family for row in source.a_validation.records)
    if (
        manifest.record_count != 512
        or manifest.split != CONFIRMATION_SPLIT
        or manifest.root_seed != CONFIRMATION_SEED
        or Counter(row.intended_family for row in manifest.records) != expected
        or len({tces_numeric_key(row) for row in manifest.records}) != 512
        or sum(row.intended_family in target for row in manifest.records) != 256
        or sum(row.intended_family in sentinel for row in manifest.records) != 256
        or any(
            tces_numeric_key(row) in forbidden_numeric
            or row.content_hash in forbidden_content
            or set(row.valid_family_ids) & protected != {row.intended_family}
            for row in manifest.records
        )
    ):
        raise ValueError(
            "confirmation panel changed family balance or overlaps existing inputs"
        )


def build_confirmation(source, generator_config, *, workers=None):
    """Parallelize families; deterministic indexed generation and reduction are unchanged."""
    original = source.a_validation
    counts = Counter(row.intended_family for row in original.records)
    representatives = {row.intended_family: row for row in reversed(original.records)}
    existing = source_records(source)
    numeric = frozenset(tces_numeric_key(row) for row in existing)
    content = frozenset(row.content_hash for row in existing)
    protected = frozenset(counts)
    config = replace(generator_config, min_valid_families=1, max_valid_families=None)
    jobs = [
        (
            representatives[family],
            counts[family],
            config,
            protected - {family},
            numeric,
            content,
        )
        for family in sorted(counts)
    ]
    with ProcessPoolExecutor(
        max_workers=workers or min(8, os.cpu_count() or 1)
    ) as pool:
        groups = list(pool.map(_generate_family, jobs))
    manifest = build_manifest(
        name="endpoint-clone-confirmation",
        split=CONFIRMATION_SPLIT,
        generator_version=original.generator_version,
        root_seed=CONFIRMATION_SEED,
        records=[row for group in groups for row in group],
        metadata={
            "scope": "new item-disjoint endpoint-clone confirmation; not historical sealed test",
            "family_counts": dict(sorted(counts.items())),
            "excluded_manifests": [
                source.prompt_pools.a_rl_train_manifest.manifest_id,
                source.prompt_pools.a_monitor_manifest.manifest_id,
                source.a_validation.manifest_id,
            ],
        },
    )
    validate_confirmation(source, manifest)
    return manifest
