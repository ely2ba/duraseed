"""Deterministic three-cohort reduction, kept private until exact equivalence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from duraseed.boundary_capacity import audit_family_split_capacities
from duraseed.data.boundary import (
    BoundaryFamilySummary,
    assess_confirmation_observation_gate,
)
from duraseed.data.boundary_confirmation import regenerate_family_templates
from duraseed.data.boundary_protocol import (
    BOUNDARY_BROAD_EXTENSION_1_COHORT,
    BOUNDARY_BROAD_EXTENSION_2_COHORT,
    BOUNDARY_BROAD_INITIAL_COHORT,
)
from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
from duraseed.data.panel_capacity import (
    PANEL_SELECTED_TEST_SINGLE_MINIMUM,
    PANEL_SPLIT_MINIMUMS,
    FamilyCapacityAudit,
)
from duraseed.data.panel_matching import (
    FamilyPanelCandidate,
    build_family_panel_candidate,
    crossed_seed_assignments,
    match_family_panels,
)
from duraseed.data.boundary_freeze_contracts import (
    BoundaryFreezeCohort,
    BoundaryFreezeResult,
    BoundaryFreezeSettings,
)
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.provenance import canonical_json_hash
from duraseed.tasks.tces import TCESGeneratorConfig


BOUNDARY_PANEL_FREEZE_MINIMUM_CANDIDATES = 36
BOUNDARY_PANEL_FREEZE_EQUIVALENCE_STATUS = "pending_three_cohort_equivalence_check"


class BoundaryFreezeUnverifiedError(RuntimeError):
    """The public freeze output is unavailable before exact equivalence."""


class BoundaryFreezeReductionError(ValueError):
    """Authenticated inputs cannot support the frozen three-cohort reduction."""


def _combined_manifest(
    cohorts: tuple[BoundaryFreezeCohort, ...], *, confirmation: bool
) -> DatasetManifest:
    from duraseed.data.manifests import build_manifest

    sources = tuple(
        row.confirmation_manifest if confirmation else row.broad_manifest
        for row in cohorts
    )
    records = tuple(record for source in sources for record in source.records)
    metadata_key = "families" if confirmation else "templates"
    metadata_rows = [
        value for source in sources for value in source.metadata[metadata_key]
    ]
    first = sources[0]
    broad_id = (
        None
        if not confirmation
        else _combined_manifest(cohorts, confirmation=False).manifest_id
    )
    metadata: dict[str, Any] = {
        "scope": "m0_boundary_three_cohort_panel_freeze",
        "scientific_manifest": False,
        "generation_mode": (
            "exact_family_held_out_numeric_variants_v1"
            if confirmation
            else "exact_family_numeric_variants_v1"
        ),
        "family_count": len(metadata_rows),
        "items_per_family": 4,
        metadata_key: metadata_rows,
        "cohort_ids": [row.cohort_id for row in cohorts],
        (
            "source_confirmation_manifest_ids"
            if confirmation
            else "source_broad_manifest_ids"
        ): [source.manifest_id for source in sources],
    }
    if confirmation:
        metadata["source_broad_manifest_id"] = broad_id
    return build_manifest(
        name=(
            "m0-boundary-three-cohort-confirmation-a-candidate"
            if confirmation
            else "m0-boundary-three-cohort-a-candidate"
        ),
        split="a_candidate",
        generator_version=first.generator_version,
        root_seed=first.root_seed,
        records=records,
        parent_manifest_id=broad_id or cohorts[-1].broad_manifest.manifest_id,
        task_family="tces",
        metadata=metadata,
    )


def _ranked_candidates(
    candidates: tuple[FamilyPanelCandidate, ...], allocation_seed: int
) -> tuple[FamilyPanelCandidate, ...]:
    from duraseed.data.panel_matching import _selection_order

    return _selection_order(
        tuple(sorted(candidates, key=lambda row: row.family_id)), allocation_seed
    )


def _family_table(
    summaries: tuple[BoundaryFamilySummary, ...],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for summary in summaries:
        gate = assess_confirmation_observation_gate(summary)
        rows.append(
            {
                **asdict(summary),
                "observation_gate_eligible": gate.eligible,
                "observation_gate_exclusion_reasons": list(gate.exclusion_reasons),
                "selection_status": "resolved_by_fixed_three_cohort_freeze",
            }
        )
    return tuple(rows)


def _reduce_three_cohort_panels(
    cohorts: Sequence[BoundaryFreezeCohort],
    *,
    settings: BoundaryFreezeSettings,
    teacher_trace_token_counts: Mapping[str, Mapping[str, int]] | None = None,
    teacher_trace_token_counter: Callable[
        [BoundaryFamilySummary, tuple[TCESTaskManifestRecord, ...]], Mapping[str, int]
    ]
    | None = None,
) -> BoundaryFreezeResult:
    """Private extraction of the archived reducer for completed-evidence comparison."""

    values = tuple(cohorts)
    if tuple(row.cohort_id for row in values) != (
        BOUNDARY_BROAD_INITIAL_COHORT,
        BOUNDARY_BROAD_EXTENSION_1_COHORT,
        BOUNDARY_BROAD_EXTENSION_2_COHORT,
    ):
        raise BoundaryFreezeReductionError("the freeze requires three distinct cohorts")
    if (
        values[0].broad_manifest.parent_manifest_id is not None
        or values[1].broad_manifest.parent_manifest_id
        != values[0].broad_manifest.manifest_id
        or values[2].broad_manifest.parent_manifest_id
        != values[1].broad_manifest.manifest_id
        or len({row.sampler_checkpoint_path for row in values}) != 1
    ):
        raise BoundaryFreezeReductionError(
            "cohort source identity or parent chain differs"
        )
    broad = _combined_manifest(values, confirmation=False)
    confirmation = _combined_manifest(values, confirmation=True)
    source_records = (*broad.records, *confirmation.records)
    summaries = tuple(
        sorted(
            (summary for cohort in values for summary in cohort.finalist_summaries),
            key=lambda row: row.intended_family_id,
        )
    )
    finalist_ids = tuple(row.intended_family_id for row in summaries)
    summary_by_family = {row.intended_family_id: row for row in summaries}
    if len(finalist_ids) != len(set(finalist_ids)):
        raise BoundaryFreezeReductionError("cohort finalists are ambiguous")
    config = TCESGeneratorConfig(**dict(settings.generator_kwargs))
    templates = regenerate_family_templates(config, broad, finalist_ids)
    audits = audit_family_split_capacities(
        templates,
        finalist_ids,
        config,
        root_seed=settings.capacity_root_seed,
        forbidden_records=source_records,
        protected_family_ids=finalist_ids,
    )
    if tuple(row.family_id for row in audits) != finalist_ids:
        raise BoundaryFreezeReductionError("combined capacity audit order differs")
    observation = tuple(
        row.intended_family_id
        for row in summaries
        if assess_confirmation_observation_gate(row).eligible
    )
    locked = values[0].locked_eligible_family_ids
    audit_by_family = {row.family_id: row for row in audits}
    if (
        len(locked) != 15
        or len(set(locked)) != 15
        or not set(locked).issubset(observation)
        or any(not audit_by_family[family_id].passed for family_id in locked)
    ):
        raise BoundaryFreezeReductionError(
            "combined capacity reclassifies a locked passer"
        )
    eligible = tuple(
        family_id for family_id in observation if audit_by_family[family_id].passed
    )
    records_by_family: dict[str, list[TCESTaskManifestRecord]] = {
        family_id: [] for family_id in eligible
    }
    for record in source_records:
        if (
            isinstance(record, TCESTaskManifestRecord)
            and record.intended_family in records_by_family
        ):
            records_by_family[record.intended_family].append(record)
    if (teacher_trace_token_counts is None) == (teacher_trace_token_counter is None):
        raise BoundaryFreezeReductionError(
            "supply exactly one teacher token-count source"
        )
    counts = (
        {
            family_id: dict(
                teacher_trace_token_counter(
                    summary_by_family[family_id],
                    tuple(records_by_family[family_id]),
                )
            )
            for family_id in eligible
        }
        if teacher_trace_token_counter is not None
        else {
            family_id: dict(values)
            for family_id, values in (teacher_trace_token_counts or {}).items()
        }
    )
    if set(counts) != set(eligible):
        raise BoundaryFreezeReductionError(
            "teacher token counts differ from eligible families"
        )
    candidates = tuple(
        build_family_panel_candidate(
            summary_by_family[family_id],
            records_by_family[family_id],
            teacher_trace_token_counts=counts[family_id],
            available_disjoint_instances=audit_by_family[
                family_id
            ].available_disjoint_instances,
        )
        for family_id in eligible
    )
    ranked = _ranked_candidates(candidates, settings.allocation_seed)
    match = None
    candidate_payload = None
    candidate_id = None
    report = None
    report_id = None
    selected_audits: tuple[FamilyCapacityAudit, ...] = ()
    artifact = None
    if len(candidates) >= BOUNDARY_PANEL_FREEZE_MINIMUM_CANDIDATES:
        match = match_family_panels(
            candidates,
            panel_size=settings.panel_size,
            allocation_seed=settings.allocation_seed,
        )
        candidate_payload = {
            "schema": "duraseed-panel-candidates-v1",
            "matching_covariates": list(settings.matching_covariates),
            "candidates": [
                asdict(row) for row in sorted(candidates, key=lambda row: row.family_id)
            ],
        }
        report = {
            "schema": "duraseed-panel-match-v1",
            "selection_rule": "highest confirmed I8, then split capacity; exact ties use the frozen allocation seed",
            "distance": "equal_weight_gower",
            "match": asdict(match),
        }
        selected = match.selected_family_ids
        selected_templates = regenerate_family_templates(config, broad, selected)
        selected_audits = audit_family_split_capacities(
            selected_templates,
            selected,
            config,
            root_seed=settings.capacity_root_seed,
            requirements={
                **dict(PANEL_SPLIT_MINIMUMS),
                "a_test_single": PANEL_SELECTED_TEST_SINGLE_MINIMUM,
            },
            forbidden_records=source_records,
            protected_family_ids=selected,
        )
        if tuple(row.family_id for row in selected_audits) != selected:
            raise BoundaryFreezeReductionError("selected capacity audit order differs")
        if selected_audits and all(row.passed for row in selected_audits):
            candidate_id = canonical_json_hash(candidate_payload)
            report_id = canonical_json_hash(report)
            artifact = FamilyPanelArtifact(
                m0_checkpoint_path=values[0].sampler_checkpoint_path,
                candidate_family_table_manifest_id=candidate_id,
                panel_matching_report_id=report_id,
                allocation_seed=settings.allocation_seed,
                panel_a_family_ids=match.panel_a_family_ids,
                panel_b_family_ids=match.panel_b_family_ids,
                seed_block_assignments=crossed_seed_assignments(
                    settings.training_seeds, allocation_seed=settings.allocation_seed
                ),
            )
        else:
            match = candidate_payload = report = None
    summary = {
        "status": (
            "confirmed_panels_frozen"
            if artifact is not None
            else "confirmation_complete_panel_selection_unresolved"
        ),
        "source_kind": "three_cohort_boundary_freeze_v1",
        "selection_performed": artifact is not None,
        "panel_assignment_performed": artifact is not None,
        "pilot_started": False,
        "cohort_ids": [row.cohort_id for row in values],
        "finalist_family_count": len(summaries),
        "observation_eligible_family_count": len(observation),
        "observation_eligible_family_ids": list(observation),
        "split_capacity_requirements": dict(PANEL_SPLIT_MINIMUMS),
        "split_capacity_eligible_family_count": sum(row.passed for row in audits),
        "split_capacity_eligible_family_ids": [
            row.family_id for row in audits if row.passed
        ],
        "panel_candidate_family_count": len(candidates),
        "panel_candidate_family_ids": [row.family_id for row in candidates],
        "minimum_required_for_panel_construction": BOUNDARY_PANEL_FREEZE_MINIMUM_CANDIDATES,
        "selected_panel_split_capacity_passed": bool(selected_audits)
        and all(row.passed for row in selected_audits),
        "panel_a_family_ids": list(artifact.panel_a_family_ids) if artifact else [],
        "panel_b_family_ids": list(artifact.panel_b_family_ids) if artifact else [],
        "candidate_table_id": candidate_id,
        "panel_matching_report_id": report_id,
        "next_step": (
            "continue remaining pre-pilot calibration"
            if artifact
            else "stop Phase 3 unresolved; do not extend or weaken the design"
        ),
    }
    return BoundaryFreezeResult(
        broad,
        confirmation,
        summaries,
        _family_table(summaries),
        audits,
        observation,
        eligible,
        counts,
        candidates,
        tuple(row.family_id for row in ranked),
        candidate_payload,
        candidate_id,
        match,
        report,
        report_id,
        (
            tuple(sorted(row.family_id for row in ranked[24:36]))
            if artifact is not None
            else ()
        ),
        selected_audits,
        artifact,
        summary,
    )


def freeze_three_cohort_panels(
    cohorts: Sequence[object],
    candidates: Sequence[FamilyPanelCandidate],
    *,
    panel_size: int,
    allocation_seed: int,
    training_seeds: Sequence[int],
    m0_checkpoint_path: str,
) -> None:
    """Fail closed until the completed-evidence exact comparison is accepted."""

    del cohorts, candidates, panel_size, allocation_seed, training_seeds
    del m0_checkpoint_path
    raise BoundaryFreezeUnverifiedError(
        "three-cohort freeze is pending the completed-evidence exact check"
    )


__all__ = [
    "BOUNDARY_PANEL_FREEZE_EQUIVALENCE_STATUS",
    "BOUNDARY_PANEL_FREEZE_MINIMUM_CANDIDATES",
    "BoundaryFreezeUnverifiedError",
    "freeze_three_cohort_panels",
]
