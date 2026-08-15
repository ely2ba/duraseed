from __future__ import annotations

from itertools import product
from pathlib import Path

from duraseed.data.boundary_panel_amendment import (
    algebraic_family_classes,
    reduce_panel_amendment,
)
from duraseed.data.panel_capacity import (
    PANEL_CAPACITY_PROBE_MULTIPLIER,
    PANEL_FILTERED_SPLIT_SCAN_MULTIPLIER,
    PANEL_SELECTED_TEST_SINGLE_MINIMUM,
    PANEL_SPLIT_MINIMUMS,
    FamilyCapacityAudit,
    FamilySplitCapacity,
)
from duraseed.data.panel_matching import (
    FamilyPanelCandidate,
    parse_tces_family_structure,
)


ALLOCATION_SEED = 6448342238137851489
OPERATORS = ("ADD", "SUB", "MUL", "DIV")


def _candidate(
    family_id: str,
    *,
    informative: float,
    capacity: int,
    operators: tuple[str, ...] = ("ADD", "ADD", "ADD", "ADD"),
    profile: tuple[str, ...] = ("I", "I", "I", "I"),
) -> FamilyPanelCandidate:
    ordered = tuple(name for name in OPERATORS for _ in range(operators.count(name)))
    return FamilyPanelCandidate(
        family_id=family_id,
        m0_posterior_mean_success=0.1,
        informative_group_probability_i8=informative,
        tree_depth=5,
        operator_multiset=ordered,
        noncommutative_operation_count=sum(name in {"SUB", "DIV"} for name in ordered),
        fractional_intermediate_profile=profile,
        target_magnitude=10,
        valid_family_multiplicity=1,
        teacher_trace_length=20,
        available_disjoint_instances=capacity,
    )


def _unique_candidates(count: int) -> tuple[FamilyPanelCandidate, ...]:
    selected: list[FamilyPanelCandidate] = []
    for index, operators in enumerate(product(OPERATORS, repeat=4)):
        expression = (
            f"{operators[0]}(r1,{operators[1]}(r2,"
            f"{operators[2]}(r3,{operators[3]}(r4,r5))))"
        )
        candidate = _candidate(
            f"{expression}|intermediates=I,I,I,I",
            informative=1.0 - index / 1_000,
            capacity=300 - index,
            operators=operators,
        )
        if len(algebraic_family_classes((*selected, candidate), ALLOCATION_SEED)) > len(
            selected
        ):
            selected.append(candidate)
        if len(selected) == count:
            return tuple(selected)
    raise AssertionError(
        "fixture could not produce enough algebraically unique families"
    )


def _audit(family_id: str, *, test_passed: bool) -> FamilyCapacityAudit:
    requirements = (
        *PANEL_SPLIT_MINIMUMS,
        ("a_test_single", PANEL_SELECTED_TEST_SINGLE_MINIMUM),
    )
    rows = []
    for split, required in requirements:
        passed = test_passed or split != "a_test_single"
        available = required if passed else 0
        multiplier = (
            PANEL_FILTERED_SPLIT_SCAN_MULTIPLIER
            if split == "a_test_single"
            else PANEL_CAPACITY_PROBE_MULTIPLIER
        )
        rows.append(
            FamilySplitCapacity(
                split=split,
                required_instances=required,
                probe_indices=required * multiplier,
                available_disjoint_instances=available,
                first_generation_failure_index=None,
                passed=passed,
            )
        )
    return FamilyCapacityAudit(
        family_id=family_id,
        split_capacities=tuple(rows),
        available_disjoint_instances=sum(
            row.available_disjoint_instances for row in rows
        ),
        passed=all(row.passed for row in rows),
    )


def test_exact_algebraic_classes_use_frozen_rank_representative() -> None:
    first = _candidate(
        "ADD(ADD(r1,r2),ADD(r3,ADD(r4,r5)))|intermediates=I,I,I,I",
        informative=0.5,
        capacity=250,
    )
    equivalent = _candidate(
        "ADD(r5,ADD(r4,ADD(r3,ADD(r2,r1))))|intermediates=I,I,I,I",
        informative=0.9,
        capacity=240,
    )
    different_operators = _candidate(
        "SUB(r1,ADD(r2,ADD(r3,ADD(r4,r5))))|intermediates=I,I,I,I",
        informative=0.8,
        capacity=260,
        operators=("SUB", "ADD", "ADD", "ADD"),
    )
    same_value_different_operators = _candidate(
        "SUB(SUB(SUB(SUB(r1,r2),r3),r4),r5)|intermediates=I,I,I,I",
        informative=0.7,
        capacity=260,
        operators=("SUB", "SUB", "SUB", "SUB"),
    )
    classes = algebraic_family_classes(
        (
            first,
            equivalent,
            different_operators,
            same_value_different_operators,
        ),
        ALLOCATION_SEED,
    )

    assert sorted(len(row.member_family_ids) for row in classes) == [2, 2]
    additive = next(row for row in classes if first.family_id in row.member_family_ids)
    assert additive.representative_family_id == equivalent.family_id
    assert additive.member_family_ids == tuple(
        sorted((first.family_id, equivalent.family_id))
    )
    subtractive = next(
        row for row in classes if different_operators.family_id in row.member_family_ids
    )
    assert same_value_different_operators.family_id in subtractive.member_family_ids


def test_completed_source_collapses_49_families_to_37_classes() -> None:
    source = Path(__file__).parents[1] / "fixtures/boundary_panel_49_family_ids.txt"
    family_ids = tuple(source.read_text(encoding="utf-8").splitlines())
    candidates = []
    for index, family_id in enumerate(family_ids):
        structure = parse_tces_family_structure(family_id)
        candidates.append(
            _candidate(
                family_id,
                informative=0.9 - index / 100,
                capacity=300 - index,
                operators=structure.operator_multiset,
                profile=structure.fractional_intermediate_profile,
            )
        )

    classes = algebraic_family_classes(candidates, ALLOCATION_SEED)

    assert len(family_ids) == 49
    assert len(classes) == 37
    assert sorted(
        (
            len(row.member_family_ids)
            for row in classes
            if len(row.member_family_ids) > 1
        ),
        reverse=True,
    ) == [6, 3, 3, 2, 2, 2]


def test_intermediate_pool_uses_full_ordinary_representative_universe() -> None:
    candidates = _unique_candidates(37)
    classes = algebraic_family_classes(candidates, ALLOCATION_SEED)
    representative_ids = tuple(row.representative_family_id for row in classes)
    panel_failed_but_ordinary_eligible = representative_ids[0]
    audits = tuple(
        _audit(
            family_id,
            test_passed=family_id != panel_failed_but_ordinary_eligible,
        )
        for family_id in representative_ids
    )

    result = reduce_panel_amendment(
        candidates,
        audits,
        panel_size=12,
        intermediate_size=12,
        allocation_seed=ALLOCATION_SEED,
        training_seeds=(17, 37),
        m0_checkpoint_path="tinker://m0/sampler",
    )

    assert result.panel_artifact is not None
    assert result.match is not None
    assert panel_failed_but_ordinary_eligible not in result.match.selected_family_ids
    assert panel_failed_but_ordinary_eligible in result.intermediate_family_ids
    assert len(result.selected_capacity_audits) == 24
    assert all(row.passed for row in result.selected_capacity_audits)
    assert result.amendment_payload["intermediate_eligibility"] == {
        "requirements": dict(PANEL_SPLIT_MINIMUMS),
        "a_test_single_required": False,
        "excluded_panel_family_ids": sorted(result.match.selected_family_ids),
        "selection": "first 12 full-representative-rank families outside panels",
        "selected_family_ids": list(result.intermediate_family_ids),
    }
    assert len(result.candidate_payload["candidates"]) == 37
