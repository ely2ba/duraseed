"""Ordered process-pool execution for independent boundary-capacity audits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from duraseed.data.panel_capacity import (
    PANEL_SPLIT_MINIMUMS,
    FamilyCapacityAudit,
    audit_family_split_capacity,
)
from duraseed.tasks.tces import GeneratedTCESInstance, TCESGeneratorConfig


_CAPACITY_AUDIT_WORKERS = 3


@dataclass(frozen=True, slots=True)
class _FamilyCapacityAuditJob:
    template: GeneratedTCESInstance
    generator_config: TCESGeneratorConfig
    root_seed: int
    requirements: tuple[tuple[str, int], ...]
    forbidden_records: tuple[object, ...]
    protected_family_ids: tuple[str, ...]


def _run_family_capacity_audit(job: _FamilyCapacityAuditJob) -> FamilyCapacityAudit:
    return audit_family_split_capacity(
        job.template,
        job.generator_config,
        root_seed=job.root_seed,
        requirements=job.requirements,
        forbidden_records=job.forbidden_records,
        protected_family_ids=job.protected_family_ids,
    )


def audit_family_split_capacities(
    templates: Mapping[str, GeneratedTCESInstance],
    family_ids: Sequence[str],
    generator_config: TCESGeneratorConfig,
    *,
    root_seed: int,
    requirements: Mapping[str, int] | tuple[tuple[str, int], ...] = (
        PANEL_SPLIT_MINIMUMS
    ),
    forbidden_records: Sequence[object] = (),
    protected_family_ids: Sequence[str] = (),
    max_workers: int | None = None,
) -> tuple[FamilyCapacityAudit, ...]:
    """Audit independent families in input order with a bounded process pool."""

    workers = _CAPACITY_AUDIT_WORKERS if max_workers is None else max_workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("max_workers must be a positive integer")
    ordered = tuple(family_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("family IDs must be unique")
    if not ordered:
        return ()
    normalized_requirements = tuple(dict(requirements).items())
    shared_forbidden = tuple(forbidden_records)
    shared_protected = tuple(protected_family_ids)
    jobs = tuple(
        _FamilyCapacityAuditJob(
            templates[family_id],
            generator_config,
            root_seed,
            normalized_requirements,
            shared_forbidden,
            shared_protected,
        )
        for family_id in ordered
    )
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        return tuple(executor.map(_run_family_capacity_audit, jobs))


__all__ = ["audit_family_split_capacities"]
