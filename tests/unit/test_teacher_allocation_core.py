from __future__ import annotations

from dataclasses import replace

from duraseed.data.matching import FamilyBlockMatchStatus, TeacherExampleRecord
from duraseed.data.panel_matching import parse_tces_family_structure
from duraseed.data.panels import PanelLabel
from duraseed.provenance import canonical_json_value
from duraseed.tasks.tces import (
    extract_answer_span,
    generate_teacher_trace,
    parse_expression,
)
from duraseed.training.teacher_allocation import (
    RANDOM_TEACHER_ALLOCATION_SEED,
    TeacherTraceCandidate,
    build_crossed_teacher_allocations,
    build_teacher_trace_candidate,
)
from tests.unit.teacher_allocation_fixtures import (
    _teacher_fixture,
    fixed_token_measurer,
)


def _with_matching(
    candidate: TeacherTraceCandidate,
    matching: TeacherExampleRecord,
) -> TeacherTraceCandidate:
    family_block = candidate.family_block_record
    assert family_block is not None
    return replace(
        candidate,
        matching_record=matching,
        family_block_record=replace(family_block, record=matching),
    )


def test_candidate_factory_matches_v0_canonical_candidate() -> None:
    artifact, candidates = _teacher_fixture()
    source = next(
        candidate
        for candidate in candidates
        if candidate.source_record.intended_family in artifact.panel_a_family_ids
    )
    completion = generate_teacher_trace(
        parse_expression(extract_answer_span(source.completion).text)
    )
    built = build_teacher_trace_candidate(
        source_manifest=source.source_manifest,
        source_record=source.source_record,
        completion=completion,
        panel_family_ids=(*artifact.panel_a_family_ids, *artifact.panel_b_family_ids),
        token_measurer=fixed_token_measurer,
    )

    structure = parse_tces_family_structure(source.source_record.intended_family)
    matching = built.matching_record
    assert matching.record_id == source.source_record.task_id
    assert matching.family_id == source.source_record.intended_family
    assert matching.allocation_group == "boundary"
    assert matching.teacher_trace_format == "concise_derivation_v1"
    assert matching.covariates.tree_depth == structure.tree_depth
    assert matching.covariates.operator_multiset == ("+", "+", "+", "+")
    assert matching.covariates.target_magnitude_bin == "diagnostic_only"
    assert matching.teacher_prompt_tokens == 3
    assert matching.teacher_target_tokens == 2
    block = built.family_block_record
    assert block is not None and block.record is matching
    assert (
        canonical_json_value(built)["matching_record"]["record_id"]
        == matching.record_id
    )


def test_crossed_allocations_match_v0_and_preserve_manifest_item_order() -> None:
    artifact, candidates = _teacher_fixture()
    first = build_crossed_teacher_allocations(
        panel_artifact=artifact,
        demonstrations_per_family=2,
        candidates=candidates,
        optimizer_updates=16,
        token_measurer=fixed_token_measurer,
    )
    repeated = build_crossed_teacher_allocations(
        panel_artifact=artifact,
        demonstrations_per_family=2,
        candidates=tuple(reversed(candidates)),
        optimizer_updates=16,
        token_measurer=fixed_token_measurer,
    )

    assert RANDOM_TEACHER_ALLOCATION_SEED == 8974530496190395558
    assert first == repeated
    assert tuple(row.targeted_panel for row in first) == (PanelLabel.A, PanelLabel.B)
    for row in first:
        assert len(row.targeted_records) == len(row.random_records) == 24
        assert row.matching_report.status is FamilyBlockMatchStatus.SELECTED
        assert row.matching_report.exact_family_count
        assert row.matching_report.exact_rows_per_family
        assert row.matching_report.exact_family_structure
        assert row.matching_report.exact_prompt_token_budget
        assert row.matching_report.exact_optimizer_updates
        assert row.matching_report.target_token_relative_difference == 0

    dose_one = build_crossed_teacher_allocations(
        panel_artifact=artifact,
        demonstrations_per_family=1,
        candidates=candidates,
        optimizer_updates=16,
        token_measurer=fixed_token_measurer,
    )
    source_by_id = {
        candidate.source_record.task_id: candidate for candidate in candidates
    }
    differing_orders = 0
    for row in dose_one:
        for record in row.targeted_records:
            family_rows = [
                candidate.source_record
                for candidate in candidates
                if candidate.source_record.intended_family == record.strategy_family_id
            ]
            by_item = min(
                family_rows, key=lambda value: (value.item_index, value.task_id)
            )
            by_task = min(family_rows, key=lambda value: value.task_id)
            differing_orders += by_item.task_id != by_task.task_id
            assert source_by_id[record.task_id].source_record == by_item
    assert differing_orders > 0


def test_allocation_recomputes_untrusted_derived_candidate_metadata() -> None:
    artifact, candidates = _teacher_fixture()
    baseline = build_crossed_teacher_allocations(
        panel_artifact=artifact,
        demonstrations_per_family=1,
        candidates=candidates,
        optimizer_updates=16,
        token_measurer=fixed_token_measurer,
    )
    falsified = tuple(
        _with_matching(
            candidate,
            replace(
                candidate.matching_record,
                teacher_prompt_tokens=999,
                teacher_target_tokens=777,
            ),
        )
        for candidate in candidates
    )
    rebuilt = build_crossed_teacher_allocations(
        panel_artifact=artifact,
        demonstrations_per_family=1,
        candidates=falsified,
        optimizer_updates=16,
        token_measurer=fixed_token_measurer,
    )

    assert rebuilt == baseline
    assert all(
        row.matching_report.target_ledger.prompt_tokens == 36
        and row.matching_report.target_ledger.target_tokens == 24
        and row.matching_report.random_ledger.prompt_tokens == 36
        and row.matching_report.random_ledger.target_tokens == 24
        for row in rebuilt
    )
