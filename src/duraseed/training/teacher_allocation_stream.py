"""Deterministic record construction used by the teacher-allocation freezer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from duraseed.data.manifests import (
    GENERATOR_VERSION,
    DatasetManifest,
    TCESTaskManifestRecord,
    build_manifest,
    build_tces_record,
)
from duraseed.data.matching import (
    FamilyBlockMatchPolicy,
    FamilyBlockMatchReport,
    FamilyBlockRecord,
    match_teacher_family_blocks,
)
from duraseed.data.panel_matching import parse_tces_family_structure
from duraseed.data.panels import PanelLabel
from duraseed.data.splits import derive_tces_split_seed, tces_numeric_key
from duraseed.provenance import canonical_json_hash
from duraseed.tasks.tces import (
    GeneratedTCESInstance,
    TCESFamilyGenerator,
    TCESGenerationError,
    TCESGeneratorConfig,
    enumerate_task,
    generate_teacher_trace,
)
from duraseed.training.reward import verify_task_completion
from duraseed.training.sft import build_teacher_dose_records
from duraseed.training.teacher_allocation import (
    RANDOM_TEACHER_ALLOCATION_SEED,
    TeacherTokenMeasurer,
    TeacherTraceCandidate,
    build_teacher_trace_candidate,
)
from duraseed.training.teacher_allocation_sources import (
    RANDOM_FAMILY_ROWS,
    TeacherAllocationSources,
)


_OPERATOR_SYMBOLS = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}
_OPERATOR_ORDER = {"+": 0, "-": 1, "*": 2, "/": 3}


class TeacherAllocationFreezeError(RuntimeError):
    """Authenticated evidence cannot support the frozen allocation search."""


def teacher_completion(record: TCESTaskManifestRecord) -> str:
    enumeration = enumerate_task(record.to_task())
    if (
        not enumeration.complete
        or enumeration.family_ids != record.valid_family_ids
        or len(enumeration.expressions) != record.valid_expression_count
        or enumeration.shortest_depth != record.minimum_depth
    ):
        raise TeacherAllocationFreezeError(
            f"teacher task {record.task_id} differs from its exact enumeration"
        )
    representative = enumeration.family_representatives.get(record.intended_family)
    if representative is None:
        raise TeacherAllocationFreezeError(
            f"teacher task {record.task_id} lacks its intended-family solution"
        )
    completion = generate_teacher_trace(representative)
    result = verify_task_completion(completion, record.to_task())
    if result.reward != 1.0 or result.strategy_family_id != record.intended_family:
        raise TeacherAllocationFreezeError(
            f"teacher task {record.task_id} failed exact verification"
        )
    return completion


def family_structure_key(family_id: str, operand_count: int) -> tuple[object, ...]:
    structure = parse_tces_family_structure(family_id)
    operators = tuple(
        sorted(
            (_OPERATOR_SYMBOLS[value] for value in structure.operator_multiset),
            key=_OPERATOR_ORDER.__getitem__,
        )
    )
    return (
        structure.tree_depth,
        operand_count,
        operators,
        structure.noncommutative_operation_count,
        structure.fractional_intermediate_profile,
        "concise_derivation_v1",
    )


def candidate_rows(
    manifest: DatasetManifest,
    panel_family_ids: tuple[str, ...],
    token_measurer: TeacherTokenMeasurer,
) -> tuple[TeacherTraceCandidate, ...]:
    return tuple(
        build_teacher_trace_candidate(
            source_manifest=manifest,
            source_record=record,
            completion=teacher_completion(record),
            panel_family_ids=panel_family_ids,
            token_measurer=token_measurer,
        )
        for record in manifest.records
        if isinstance(record, TCESTaskManifestRecord)
    )


def target_blocks(
    source: TeacherAllocationSources,
    candidates: Sequence[TeacherTraceCandidate],
    panel: PanelLabel,
) -> tuple[FamilyBlockRecord, ...]:
    family_ids = (
        source.panel.panel_a_family_ids
        if panel is PanelLabel.A
        else source.panel.panel_b_family_ids
    )
    selected = build_teacher_dose_records(
        source_manifest=source.target_train_manifest,
        solver_completions=(
            (candidate.source_record, candidate.completion) for candidate in candidates
        ),
        selected_families=family_ids,
        demonstrations_per_family=source.selected_dose,
    )
    by_id = {candidate.source_record.task_id: candidate for candidate in candidates}
    blocks = tuple(by_id[row.task_id].family_block_record for row in selected)
    if any(block is None for block in blocks):
        raise TeacherAllocationFreezeError("target candidate omitted its family block")
    return tuple(block for block in blocks if block is not None)


def match_orientation(
    target: Sequence[FamilyBlockRecord],
    random: Sequence[FamilyBlockRecord],
    source: TeacherAllocationSources,
) -> FamilyBlockMatchReport:
    return match_teacher_family_blocks(
        target,
        random,
        policy=FamilyBlockMatchPolicy(
            dose=source.selected_dose,
            allocation_seed=RANDOM_TEACHER_ALLOCATION_SEED,
        ),
        target_optimizer_updates=source.optimizer_updates,
        random_optimizer_updates=source.optimizer_updates,
    )


def generate_random_family(
    source: TeacherAllocationSources,
    template: GeneratedTCESInstance,
    generator_config: TCESGeneratorConfig,
    used_numeric: set[object],
    used_content: set[str],
    scan_multiplier: int,
) -> tuple[TCESTaskManifestRecord, ...] | None:
    panel = frozenset(
        (*source.panel.panel_a_family_ids, *source.panel.panel_b_family_ids)
    )
    config = replace(
        generator_config,
        split="a_seed_train",
        min_valid_families=1,
        max_valid_families=None,
    )
    generator = TCESFamilyGenerator(
        derive_tces_split_seed(source.config.seed, "a_seed_train"), template, config
    )
    local_numeric: set[object] = set()
    local_content: set[str] = set()
    selected: list[TCESTaskManifestRecord] = []
    for item_index in range(RANDOM_FAMILY_ROWS * scan_multiplier):
        try:
            instance = generator.generate(item_index)
        except TCESGenerationError:
            continue
        numeric = tces_numeric_key(instance)
        if (
            instance.intended_family != template.intended_family
            or panel.intersection(instance.valid_family_ids)
            or numeric in used_numeric
            or numeric in local_numeric
            or instance.content_hash in used_content
            or instance.content_hash in local_content
        ):
            continue
        selected.append(build_tces_record(instance))
        local_numeric.add(numeric)
        local_content.add(instance.content_hash)
        if len(selected) == RANDOM_FAMILY_ROWS:
            used_numeric.update(local_numeric)
            used_content.update(local_content)
            return tuple(selected)
    return None


def random_manifest(
    source: TeacherAllocationSources,
    records: Sequence[TCESTaskManifestRecord],
    prefix_length: int,
) -> DatasetManifest:
    return build_manifest(
        name="random-teacher-a-seed-train",
        split="a_seed_train",
        generator_version=GENERATOR_VERSION,
        root_seed=source.config.seed,
        records=list(records),
        task_family="tces",
        parent_manifest_id=source.broad_manifest.manifest_id,
        metadata={
            "scope": "phase4_random_teacher_candidate_universe",
            "scientific_manifest": True,
            "matching_policy_id": "core_family_v1",
            "source_broad_manifest_id": source.broad_manifest.manifest_id,
            "source_confirmation_manifest_id": source.confirmation_manifest.manifest_id,
            "source_target_train_manifest_id": source.target_train_manifest.manifest_id,
            "source_gate_manifest_id": source.gate_manifest.manifest_id,
            "panel_artifact_id": canonical_json_hash(source.panel),
            "selected_dose": source.selected_dose,
            "optimizer_updates": source.optimizer_updates,
            "items_per_family": RANDOM_FAMILY_ROWS,
            "a_candidate_prefix_length": prefix_length,
        },
    )


__all__ = [
    "TeacherAllocationFreezeError",
    "candidate_rows",
    "family_structure_key",
    "generate_random_family",
    "match_orientation",
    "random_manifest",
    "target_blocks",
]
