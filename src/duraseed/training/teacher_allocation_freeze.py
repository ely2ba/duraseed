"""Exact local search for the frozen random-teacher candidate universe."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from duraseed.data.leakage import LeakageAuditReport, audit_leakage
from duraseed.data.manifests import (
    GENERATOR_VERSION,
    DatasetManifest,
    TCESTaskManifestRecord,
    build_manifest,
)
from duraseed.data.matching import FamilyBlockMatchReport, FamilyBlockMatchStatus
from duraseed.data.panels import PanelLabel
from duraseed.data.splits import TCES_SPLIT_SIZES, TCESSplitBuilder, tces_numeric_key
from duraseed.tasks.tces import TCESGeneratorConfig
from duraseed.training.teacher_allocation import (
    PanelTeacherAllocation,
    TeacherTokenMeasurer,
    TeacherTraceCandidate,
    build_crossed_teacher_allocations,
)
from duraseed.training.teacher_allocation_sources import (
    RANDOM_FAMILY_ROWS,
    TeacherAllocationSources,
    validate_teacher_allocation_sources,
)
from duraseed.training.teacher_allocation_stream import (
    TeacherAllocationFreezeError,
    candidate_rows,
    family_structure_key,
    generate_random_family,
    match_orientation,
    random_manifest,
    target_blocks,
)


RANDOM_FAMILY_SCAN_MULTIPLIER = 32
_BOUNDARY_ENGINEERING_SEED = 5


@dataclass(frozen=True, slots=True)
class TeacherAllocationFreezeResult:
    status: FamilyBlockMatchStatus
    random_candidate_manifest: DatasetManifest
    allocations: tuple[PanelTeacherAllocation, PanelTeacherAllocation] | None
    panel_a_report: FamilyBlockMatchReport | None
    panel_b_report: FamilyBlockMatchReport | None
    a_candidate_prefix_length: int
    exclusion_counts: tuple[tuple[str, int], ...]
    leakage_report: LeakageAuditReport
    failure: str | None

    @property
    def selected(self) -> bool:
        return self.status is FamilyBlockMatchStatus.SELECTED


def _complete_result(
    *,
    source: TeacherAllocationSources,
    random_records: Sequence[TCESTaskManifestRecord],
    random_candidates: Sequence[TeacherTraceCandidate],
    target_candidates: Sequence[TeacherTraceCandidate],
    report_a: FamilyBlockMatchReport | None,
    report_b: FamilyBlockMatchReport | None,
    status: FamilyBlockMatchStatus,
    prefix_length: int,
    exclusion_counts: Mapping[str, int],
    token_measurer: TeacherTokenMeasurer,
    failure: str | None,
) -> TeacherAllocationFreezeResult:
    manifest = random_manifest(source, random_records, prefix_length)
    leakage = audit_leakage((*source.all_manifests, manifest)).assert_clean()
    allocations: tuple[PanelTeacherAllocation, PanelTeacherAllocation] | None = None
    if status is FamilyBlockMatchStatus.SELECTED:
        if report_a is None or report_b is None:
            raise TeacherAllocationFreezeError("selected prefix omitted match reports")
        final_random = tuple(
            replace(candidate, source_manifest=manifest)
            for candidate in random_candidates
        )
        allocations = build_crossed_teacher_allocations(
            panel_artifact=source.panel,
            demonstrations_per_family=source.selected_dose,
            candidates=(*target_candidates, *final_random),
            optimizer_updates=source.optimizer_updates,
            token_measurer=token_measurer,
        )
        final_by_panel = {
            row.targeted_panel: row.matching_report for row in allocations
        }
        for panel, preliminary in (
            (PanelLabel.A, report_a),
            (PanelLabel.B, report_b),
        ):
            final = final_by_panel[panel]
            if (
                final.status is not FamilyBlockMatchStatus.SELECTED
                or final.target_record_ids != preliminary.target_record_ids
                or final.random_record_ids != preliminary.random_record_ids
            ):
                raise TeacherAllocationFreezeError(
                    "authoritative allocation replay changed the first passing match"
                )
    return TeacherAllocationFreezeResult(
        status=status,
        random_candidate_manifest=manifest,
        allocations=allocations,
        panel_a_report=report_a,
        panel_b_report=report_b,
        a_candidate_prefix_length=prefix_length,
        exclusion_counts=tuple(sorted(exclusion_counts.items())),
        leakage_report=leakage,
        failure=failure,
    )


def build_teacher_allocation_freeze(
    *,
    sources: TeacherAllocationSources,
    token_measurer: TeacherTokenMeasurer,
    template_ceiling: int | None = None,
    family_scan_multiplier: int = RANDOM_FAMILY_SCAN_MULTIPLIER,
) -> TeacherAllocationFreezeResult:
    """Return the earliest jointly matched allocation or exact bounded failure."""

    source = validate_teacher_allocation_sources(sources)
    declared_ceiling = TCES_SPLIT_SIZES["a_candidate"]
    ceiling = declared_ceiling if template_ceiling is None else template_ceiling
    if type(ceiling) is not int or not 1 <= ceiling <= declared_ceiling:
        raise ValueError(f"template_ceiling must be in [1, {declared_ceiling}]")
    if (
        type(family_scan_multiplier) is not int
        or not 1 <= family_scan_multiplier <= RANDOM_FAMILY_SCAN_MULTIPLIER
    ):
        raise ValueError(
            "family_scan_multiplier must not exceed the frozen production bound"
        )
    generator_config = TCESGeneratorConfig(
        **source.config.tasks.tces.generator_kwargs()
    )
    panel_ids = (*source.panel.panel_a_family_ids, *source.panel.panel_b_family_ids)
    target_candidates = candidate_rows(
        source.target_train_manifest, panel_ids, token_measurer
    )
    target_a = target_blocks(source, target_candidates, PanelLabel.A)
    target_b = target_blocks(source, target_candidates, PanelLabel.B)
    relevant_structures = {row.structure_key for row in (*target_a, *target_b)}
    used_numeric = {
        tces_numeric_key(record)
        for manifest in source.all_manifests
        for record in manifest.records
    }
    used_content = {
        record.content_hash
        for manifest in source.all_manifests
        for record in manifest.records
    }
    stream = TCESSplitBuilder(_BOUNDARY_ENGINEERING_SEED, generator_config).lazy_split(
        "a_candidate", size=ceiling
    )
    seen_families: set[str] = set()
    random_records: list[TCESTaskManifestRecord] = []
    random_candidates: list[TeacherTraceCandidate] = []
    random_blocks = []
    exclusions: Counter[str] = Counter()
    panel_set = frozenset(panel_ids)
    report_a: FamilyBlockMatchReport | None = None
    report_b: FamilyBlockMatchReport | None = None
    max_random_families = TCES_SPLIT_SIZES["a_seed_train"] // RANDOM_FAMILY_ROWS

    def finish(
        status: FamilyBlockMatchStatus,
        prefix: int,
        failure: str | None,
        *,
        clear_panel_b: bool = False,
    ) -> TeacherAllocationFreezeResult:
        return _complete_result(
            source=source,
            random_records=random_records,
            random_candidates=random_candidates,
            target_candidates=target_candidates,
            report_a=report_a,
            report_b=None if clear_panel_b else report_b,
            status=status,
            prefix_length=prefix,
            exclusion_counts=exclusions,
            token_measurer=token_measurer,
            failure=failure,
        )

    for accepted_index in range(ceiling):
        template = stream[accepted_index]
        prefix = accepted_index + 1
        family_id = template.intended_family
        if family_id in seen_families:
            continue
        seen_families.add(family_id)
        if family_id in panel_set:
            exclusions["protected_panel_family"] += 1
            continue
        if (
            family_structure_key(family_id, generator_config.n_operands)
            not in relevant_structures
        ):
            exclusions["not_structurally_eligible"] += 1
            continue
        if len(random_records) // RANDOM_FAMILY_ROWS >= max_random_families:
            return finish(
                FamilyBlockMatchStatus.INFEASIBLE,
                prefix,
                "a_seed_train_ceiling_reached_without_joint_match",
            )
        family_rows = generate_random_family(
            source,
            template,
            generator_config,
            used_numeric,
            used_content,
            family_scan_multiplier,
        )
        if family_rows is None:
            exclusions["insufficient_clean_a_seed_train_rows"] += 1
            continue
        family_manifest = build_manifest(
            name="random-teacher-family-candidate",
            split="a_seed_train",
            generator_version=GENERATOR_VERSION,
            root_seed=source.config.seed,
            records=list(family_rows),
            parent_manifest_id=source.broad_manifest.manifest_id,
            metadata={
                "scope": "phase4_random_teacher_family_candidate",
                "family_id": family_id,
                "items_per_family": RANDOM_FAMILY_ROWS,
                "a_candidate_prefix_length": prefix,
            },
        )
        family_candidates = candidate_rows(family_manifest, panel_ids, token_measurer)
        blocks = tuple(row.family_block_record for row in family_candidates)
        if any(block is None for block in blocks):
            raise TeacherAllocationFreezeError(
                "random teacher candidate omitted its authenticated family block"
            )
        random_records.extend(family_rows)
        random_candidates.extend(family_candidates)
        random_blocks.extend(block for block in blocks if block is not None)
        report_a = match_orientation(target_a, random_blocks, source)
        if report_a.status is FamilyBlockMatchStatus.SEARCH_EXHAUSTED:
            return finish(
                report_a.status,
                prefix,
                "panel_a_match_search_exhausted",
                clear_panel_b=True,
            )
        report_b = match_orientation(target_b, random_blocks, source)
        if report_b.status is FamilyBlockMatchStatus.SEARCH_EXHAUSTED:
            return finish(report_b.status, prefix, "panel_b_match_search_exhausted")
        if report_a.passed and report_b.passed:
            return finish(FamilyBlockMatchStatus.SELECTED, prefix, None)

    return finish(
        FamilyBlockMatchStatus.INFEASIBLE,
        ceiling,
        "a_candidate_ceiling_reached_without_joint_match",
    )


__all__ = [
    "TeacherAllocationFreezeError",
    "TeacherAllocationFreezeResult",
    "build_teacher_allocation_freeze",
]
