"""Pure deterministic teacher allocation from authenticated TCES candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
from duraseed.data.matching import (
    FamilyBlockMatchPolicy,
    FamilyBlockMatchReport,
    FamilyBlockRecord,
    StructuralCovariates,
    TeacherExampleRecord,
    match_teacher_family_blocks,
)
from duraseed.data.panel_matching import parse_tces_family_structure
from duraseed.data.panels import FamilyPanelArtifact, PanelLabel
from duraseed.provenance import SeedNamespace, derive_namespaced_seed
from duraseed.tasks.tces import (
    extract_answer_span,
    generate_teacher_trace,
    parse_expression,
)
from duraseed.training.sft import (
    SourceRecordError,
    VerifiedSourceRecord,
    build_teacher_dose_records,
)


PILOT_ROOT_SEED = 11
RANDOM_TEACHER_ALLOCATION_SEED = derive_namespaced_seed(
    PILOT_ROOT_SEED, SeedNamespace.RANDOM_TEACHER_ALLOCATION
)
_PANEL_SIZE = 12
_TRACE_FORMAT = "concise_derivation_v1"
_POLICY_ID = "core_family_v1"
_OPERATOR_SYMBOLS = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}


class TeacherAllocationError(ValueError):
    """Candidate evidence cannot support the frozen crossed allocation."""


@dataclass(frozen=True, slots=True)
class TeacherTokenCounts:
    prompt: int
    target: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1 for value in (self.prompt, self.target)
        ):
            raise ValueError("teacher token counts must be positive integers")


TeacherTokenMeasurer = Callable[[VerifiedSourceRecord], TeacherTokenCounts]


@dataclass(frozen=True, slots=True)
class TeacherTraceCandidate:
    source_manifest: DatasetManifest
    source_record: TCESTaskManifestRecord
    completion: str
    matching_record: TeacherExampleRecord
    family_block_record: FamilyBlockRecord | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_manifest, DatasetManifest):
            raise TypeError("source_manifest must be a DatasetManifest")
        if not isinstance(self.source_record, TCESTaskManifestRecord):
            raise TypeError("source_record must be a TCESTaskManifestRecord")
        if not isinstance(self.completion, str) or not self.completion:
            raise TeacherAllocationError("completion must be nonempty text")
        if not isinstance(self.matching_record, TeacherExampleRecord):
            raise TypeError("matching_record must be a TeacherExampleRecord")
        if self.family_block_record is not None:
            if not isinstance(self.family_block_record, FamilyBlockRecord):
                raise TypeError("family_block_record must be a FamilyBlockRecord")
            if self.family_block_record.record != self.matching_record:
                raise TeacherAllocationError(
                    "family block and legacy matching records must be identical"
                )


@dataclass(frozen=True, slots=True)
class PanelTeacherAllocation:
    targeted_panel: PanelLabel
    sentinel_panel: PanelLabel
    targeted_family_ids: tuple[str, ...]
    targeted_records: tuple[VerifiedSourceRecord, ...]
    random_records: tuple[VerifiedSourceRecord, ...]
    teacher_trace_format: str
    matching_policy_id: Literal["core_family_v1"]
    target_optimizer_updates: int
    random_optimizer_updates: int
    matching_report: FamilyBlockMatchReport


@dataclass(frozen=True, slots=True)
class _Candidate:
    candidate: TeacherTraceCandidate
    verified: VerifiedSourceRecord


def _candidate_key(candidate: TeacherTraceCandidate) -> tuple[str, str]:
    return candidate.source_manifest.manifest_id, candidate.source_record.task_id


def _authenticate_source(
    *,
    source_manifest: DatasetManifest,
    source: TCESTaskManifestRecord,
    completion: str,
    panel_families: frozenset[str],
    token_measurer: TeacherTokenMeasurer,
) -> _Candidate:
    try:
        verified = build_teacher_dose_records(
            source_manifest=source_manifest,
            solver_completions=((source, completion),),
            selected_families=(source.intended_family,),
            demonstrations_per_family=1,
        )[0]
        expression = parse_expression(extract_answer_span(completion).text)
    except (SourceRecordError, TypeError, ValueError) as error:
        raise TeacherAllocationError(
            "teacher candidate is not an authenticated manifested TCES trace"
        ) from error
    if completion != generate_teacher_trace(expression):
        raise TeacherAllocationError(
            "teacher candidate must use the canonical concise derivation"
        )
    overlap = panel_families.intersection(source.valid_family_ids)
    targeted = source.intended_family in panel_families
    if overlap != ({source.intended_family} if targeted else set()):
        raise TeacherAllocationError(
            "teacher candidate exposes a protected panel family"
        )
    counts = token_measurer(verified)
    if not isinstance(counts, TeacherTokenCounts):
        raise TypeError("token measurer must return TeacherTokenCounts")
    structure = parse_tces_family_structure(source.intended_family)
    matching = TeacherExampleRecord(
        record_id=source.task_id,
        family_id=source.intended_family,
        allocation_group="boundary" if targeted else "random",
        covariates=StructuralCovariates(
            tree_depth=structure.tree_depth,
            operand_count=len(source.operands),
            operator_multiset=tuple(
                _OPERATOR_SYMBOLS[value] for value in structure.operator_multiset
            ),
            noncommutative_count=structure.noncommutative_operation_count,
            fractional_intermediate="F" in structure.fractional_intermediate_profile,
            target_magnitude_bin="diagnostic_only",
            valid_family_count_bin="diagnostic_only",
            teacher_trace_token_bin="diagnostic_only",
        ),
        teacher_prompt_tokens=counts.prompt,
        teacher_target_tokens=counts.target,
        teacher_trace_format=_TRACE_FORMAT,
    )
    block = FamilyBlockRecord(
        record=matching,
        fractional_profile=structure.fractional_intermediate_profile,
        absolute_target=abs(source.target.numerator / source.target.denominator),
        valid_family_count=source.valid_family_count,
    )
    return _Candidate(
        candidate=TeacherTraceCandidate(
            source_manifest=source_manifest,
            source_record=source,
            completion=completion,
            matching_record=matching,
            family_block_record=block,
        ),
        verified=verified,
    )


def build_teacher_trace_candidate(
    *,
    source_manifest: DatasetManifest,
    source_record: TCESTaskManifestRecord,
    completion: str,
    panel_family_ids: Sequence[str],
    token_measurer: TeacherTokenMeasurer,
) -> TeacherTraceCandidate:
    """Authenticate and measure one manifested teacher trace."""

    supplied = tuple(panel_family_ids)
    if not supplied or len(supplied) != len(set(supplied)):
        raise ValueError("panel_family_ids must be nonempty and unique")
    return _authenticate_source(
        source_manifest=source_manifest,
        source=source_record,
        completion=completion,
        panel_families=frozenset(supplied),
        token_measurer=token_measurer,
    ).candidate


def _orientation(
    *,
    targeted_panel: PanelLabel,
    sentinel_panel: PanelLabel,
    family_ids: tuple[str, ...],
    candidates: tuple[_Candidate, ...],
    dose: int,
    optimizer_updates: int,
) -> PanelTeacherAllocation:
    by_family: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_family.setdefault(candidate.candidate.matching_record.family_id, []).append(
            candidate
        )
    targeted: list[_Candidate] = []
    for family_id in family_ids:
        by_manifest: dict[str, list[_Candidate]] = {}
        for value in by_family.get(family_id, ()):
            by_manifest.setdefault(value.verified.source_manifest_id, []).append(value)
        options = []
        for manifest_id, values in by_manifest.items():
            try:
                dose_records = build_teacher_dose_records(
                    source_manifest=values[0].candidate.source_manifest,
                    solver_completions=(
                        (row.candidate.source_record, row.candidate.completion)
                        for row in values
                    ),
                    selected_families=(family_id,),
                    demonstrations_per_family=dose,
                )
            except SourceRecordError:
                continue
            by_id = {row.verified.task_id: row for row in values}
            selected = tuple(by_id[row.task_id] for row in dose_records)
            if len(selected) == dose:
                options.append(
                    (
                        tuple(row.verified.task_id for row in selected),
                        manifest_id,
                        selected,
                    )
                )
        if not options:
            raise TeacherAllocationError(
                f"insufficient authenticated teacher dose for family {family_id}"
            )
        targeted.extend(min(options, key=lambda option: (option[0], option[1]))[2])
    random_blocks = tuple(
        candidate.candidate.family_block_record
        for candidate in candidates
        if candidate.candidate.matching_record.allocation_group == "random"
    )
    if any(block is None for block in random_blocks):
        raise TeacherAllocationError("teacher candidate omitted its family block")
    report = match_teacher_family_blocks(
        tuple(
            candidate.candidate.family_block_record
            for candidate in targeted
            if candidate.candidate.family_block_record is not None
        ),
        tuple(block for block in random_blocks if block is not None),
        policy=FamilyBlockMatchPolicy(
            dose=dose,
            allocation_seed=RANDOM_TEACHER_ALLOCATION_SEED,
        ),
        target_optimizer_updates=optimizer_updates,
        random_optimizer_updates=optimizer_updates,
    )
    if not report.passed:
        raise TeacherAllocationError(
            "teacher allocation did not satisfy core family-block matching: "
            + report.status.value
        )
    by_id = {candidate.verified.task_id: candidate.verified for candidate in candidates}
    return PanelTeacherAllocation(
        targeted_panel=targeted_panel,
        sentinel_panel=sentinel_panel,
        targeted_family_ids=family_ids,
        targeted_records=tuple(by_id[value] for value in report.target_record_ids),
        random_records=tuple(by_id[value] for value in report.random_record_ids),
        teacher_trace_format=_TRACE_FORMAT,
        matching_policy_id=_POLICY_ID,
        target_optimizer_updates=optimizer_updates,
        random_optimizer_updates=optimizer_updates,
        matching_report=report,
    )


def build_crossed_teacher_allocations(
    *,
    panel_artifact: FamilyPanelArtifact,
    demonstrations_per_family: int,
    candidates: Sequence[TeacherTraceCandidate],
    optimizer_updates: int,
    token_measurer: TeacherTokenMeasurer,
) -> tuple[PanelTeacherAllocation, PanelTeacherAllocation]:
    """Build both matched orientations from authenticated, remeasured traces."""

    if not isinstance(panel_artifact, FamilyPanelArtifact):
        raise TypeError("panel_artifact must be a FamilyPanelArtifact")
    panel_sizes = tuple(
        map(
            len,
            (panel_artifact.panel_a_family_ids, panel_artifact.panel_b_family_ids),
        )
    )
    if panel_sizes != (_PANEL_SIZE, _PANEL_SIZE):
        raise TeacherAllocationError("teacher allocation requires frozen 12/12 panels")
    if type(demonstrations_per_family) is not int or demonstrations_per_family < 1:
        raise ValueError("demonstrations_per_family must be a positive integer")
    if type(optimizer_updates) is not int or optimizer_updates < 1:
        raise ValueError("optimizer_updates must be a positive integer")
    supplied = tuple(candidates)
    identities = tuple(_candidate_key(candidate) for candidate in supplied)
    task_ids = tuple(candidate.source_record.task_id for candidate in supplied)
    if (
        not supplied
        or len(identities) != len(set(identities))
        or len(task_ids) != len(set(task_ids))
    ):
        raise TeacherAllocationError("teacher candidates must be nonempty and unique")
    panel_families = frozenset(
        (*panel_artifact.panel_a_family_ids, *panel_artifact.panel_b_family_ids)
    )
    canonical = tuple(
        _authenticate_source(
            source_manifest=candidate.source_manifest,
            source=candidate.source_record,
            completion=candidate.completion,
            panel_families=panel_families,
            token_measurer=token_measurer,
        )
        for candidate in sorted(supplied, key=_candidate_key)
    )
    return (
        _orientation(
            targeted_panel=PanelLabel.A,
            sentinel_panel=PanelLabel.B,
            family_ids=panel_artifact.panel_a_family_ids,
            candidates=canonical,
            dose=demonstrations_per_family,
            optimizer_updates=optimizer_updates,
        ),
        _orientation(
            targeted_panel=PanelLabel.B,
            sentinel_panel=PanelLabel.A,
            family_ids=panel_artifact.panel_b_family_ids,
            candidates=canonical,
            dose=demonstrations_per_family,
            optimizer_updates=optimizer_updates,
        ),
    )


__all__ = [
    "PanelTeacherAllocation",
    "RANDOM_TEACHER_ALLOCATION_SEED",
    "TeacherAllocationError",
    "TeacherTokenCounts",
    "TeacherTokenMeasurer",
    "TeacherTraceCandidate",
    "build_crossed_teacher_allocations",
    "build_teacher_trace_candidate",
]
