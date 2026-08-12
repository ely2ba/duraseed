"""Focused tests for deterministic teacher-allocation matching reports."""

import pytest

from duraseed.provenance import MAX_ROOT_SEED
from duraseed.data.matching import (
    FamilyBlockMatchPolicy,
    FamilyBlockMatchStatus,
    FamilyBlockRecord,
    MatchingFailure,
    StructuralCovariates,
    TeacherExampleRecord,
    compare_teacher_allocations,
    match_teacher_family_blocks,
    match_teacher_allocations,
)


def _covariates(
    *,
    depth: int = 3,
    operators: tuple[str, ...] = ("+", "-"),
    fractional: bool = False,
    target_bin: str = "medium",
    family_bin: str = "2-4",
    trace_bin: str = "96-112",
) -> StructuralCovariates:
    return StructuralCovariates(
        tree_depth=depth,
        operand_count=len(operators) + 1,
        operator_multiset=operators,
        noncommutative_count=sum(operator in {"-", "/"} for operator in operators),
        fractional_intermediate=fractional,
        target_magnitude_bin=target_bin,
        valid_family_count_bin=family_bin,
        teacher_trace_token_bin=trace_bin,
    )


def _record(
    record_id: str,
    *,
    group: str,
    prompt_tokens: int,
    target_tokens: int,
    covariates: StructuralCovariates | None = None,
    trace_format: str = "postorder_v1",
) -> TeacherExampleRecord:
    return TeacherExampleRecord(
        record_id=record_id,
        family_id=f"family:{record_id}",
        allocation_group=group,
        covariates=covariates or _covariates(),
        teacher_prompt_tokens=prompt_tokens,
        teacher_target_tokens=target_tokens,
        teacher_trace_format=trace_format,
    )


def _family_block(
    family_id: str,
    *,
    group: str,
    dose: int = 2,
    prompt_tokens: tuple[int, ...] = (10, 11),
    target_tokens: tuple[int, ...] = (50, 50),
    fractional_profile: tuple[str, ...] = ("I", "I"),
    diagnostic_bin: str = "unused",
    valid_family_count: int = 2,
) -> tuple[FamilyBlockRecord, ...]:
    assert len(prompt_tokens) == len(target_tokens) == dose
    fractional = "F" in fractional_profile
    return tuple(
        FamilyBlockRecord(
            record=TeacherExampleRecord(
                record_id=f"{family_id}:row-{index:02d}",
                family_id=family_id,
                allocation_group=group,
                covariates=_covariates(
                    fractional=fractional,
                    target_bin=diagnostic_bin,
                    family_bin=diagnostic_bin,
                    trace_bin=diagnostic_bin,
                ),
                teacher_prompt_tokens=prompt_tokens[index],
                teacher_target_tokens=target_tokens[index],
                teacher_trace_format="postorder_v1",
            ),
            fractional_profile=fractional_profile,
            absolute_target=float(index + valid_family_count),
            valid_family_count=valid_family_count,
        )
        for index in range(dose)
    )


def _production_blocks(
    prefix: str,
    *,
    group: str,
    family_count: int,
    **kwargs: object,
) -> tuple[FamilyBlockRecord, ...]:
    return tuple(
        row
        for family_index in range(family_count)
        for row in _family_block(
            f"{prefix}-{family_index:02d}",
            group=group,
            **kwargs,  # type: ignore[arg-type]
        )
    )


def test_exact_subset_search_finds_viable_aggregate_budget_match() -> None:
    targets = (
        _record("target-1", group="boundary", prompt_tokens=20, target_tokens=100),
        _record("target-2", group="boundary", prompt_tokens=20, target_tokens=100),
    )
    # Per-item nearest matching is attracted to 20 and 21 prompt tokens.  The
    # exact aggregate solution is instead 10 + 30.
    candidates = (
        _record("candidate-10", group="random", prompt_tokens=10, target_tokens=99),
        _record("candidate-30", group="random", prompt_tokens=30, target_tokens=101),
        _record("candidate-20", group="random", prompt_tokens=20, target_tokens=100),
        _record("candidate-21", group="random", prompt_tokens=21, target_tokens=100),
    )

    allocation = match_teacher_allocations(
        targets,
        candidates,
        allocation_seed=29,
        target_optimizer_updates=40,
        matched_optimizer_updates=40,
    )

    assert {record.record_id for record in allocation.records} == {
        "candidate-10",
        "candidate-30",
    }
    report = allocation.report
    assert report.passed
    assert report.exact_example_count
    assert report.exact_prompt_token_budget
    assert report.exact_optimizer_updates
    assert report.target_ledger.prompt_tokens == report.matched_ledger.prompt_tokens
    assert report.target_token_relative_difference == 0


def test_matching_is_input_order_independent_and_seed_deterministic() -> None:
    targets = (
        _record("target-a", group="boundary", prompt_tokens=12, target_tokens=100),
        _record("target-b", group="boundary", prompt_tokens=12, target_tokens=100),
    )
    candidates = tuple(
        _record(
            f"candidate-{index}",
            group="random",
            prompt_tokens=12,
            target_tokens=100,
        )
        for index in range(6)
    )

    first = match_teacher_allocations(
        targets,
        candidates,
        allocation_seed=47,
        target_optimizer_updates=20,
        matched_optimizer_updates=20,
    )
    second = match_teacher_allocations(
        tuple(reversed(targets)),
        tuple(reversed(candidates)),
        allocation_seed=47,
        target_optimizer_updates=20,
        matched_optimizer_updates=20,
    )

    assert first.records == second.records
    assert first.report.to_dict() == second.report.to_dict()


def test_family_block_match_selects_auditable_12_by_d_control() -> None:
    targets = _production_blocks(
        "target",
        group="boundary",
        family_count=12,
        diagnostic_bin="target-only-bin",
        valid_family_count=2,
    )
    candidates = _production_blocks(
        "random",
        group="random",
        family_count=14,
        dose=16,
        prompt_tokens=(9,) * 8 + (12,) * 8,
        target_tokens=(49,) * 8 + (51,) * 8,
        diagnostic_bin="different-random-bin",
        valid_family_count=9,
    )
    policy = FamilyBlockMatchPolicy(dose=2, allocation_seed=47)

    first = match_teacher_family_blocks(
        targets,
        candidates,
        policy=policy,
        target_optimizer_updates=40,
        random_optimizer_updates=40,
    )
    second = match_teacher_family_blocks(
        tuple(reversed(targets)),
        tuple(reversed(candidates)),
        policy=policy,
        target_optimizer_updates=40,
        random_optimizer_updates=40,
    )

    assert first == second
    assert first.status is FamilyBlockMatchStatus.SELECTED
    assert first.passed
    assert len(first.target_family_ids) == len(set(first.target_family_ids)) == 12
    assert len(first.random_family_ids) == len(set(first.random_family_ids)) == 12
    assert len(first.records) == 24
    assert all(count == 2 for _, count in first.target_family_row_counts)
    assert all(count == 2 for _, count in first.random_family_row_counts)
    assert first.exact_family_count
    assert first.exact_rows_per_family
    assert first.exact_family_structure
    assert first.exact_prompt_token_budget
    assert first.exact_optimizer_updates
    assert first.target_token_relative_difference == 0
    serialized_policy = first.to_dict()["policy"]
    assert isinstance(serialized_policy, dict)
    assert serialized_policy["policy_id"] == "core_family_v1"
    assert {diagnostic.covariate for diagnostic in first.diagnostics} == {
        "absolute_target",
        "valid_family_count",
        "supervised_target_tokens",
    }


def test_family_block_match_jointly_selects_rows_within_candidate_family() -> None:
    targets = _production_blocks(
        "target",
        group="boundary",
        family_count=12,
        dose=8,
        prompt_tokens=(10,) * 8,
        target_tokens=(50,) * 8,
    )
    candidates = (
        *_family_block(
            "random-subset",
            group="random",
            dose=16,
            prompt_tokens=(9,) * 8 + (11,) * 8,
            target_tokens=(50,) * 16,
        ),
        *_production_blocks(
            "random",
            group="random",
            family_count=11,
            dose=16,
            prompt_tokens=(10,) * 16,
            target_tokens=(50,) * 16,
        ),
    )
    policy = FamilyBlockMatchPolicy(dose=8, allocation_seed=47)

    first = match_teacher_family_blocks(
        targets,
        candidates,
        policy=policy,
        target_optimizer_updates=40,
        random_optimizer_updates=40,
    )
    second = match_teacher_family_blocks(
        tuple(reversed(targets)),
        tuple(reversed(candidates)),
        policy=policy,
        target_optimizer_updates=40,
        random_optimizer_updates=40,
    )

    selected_subset_ids = {
        row.record.record_id
        for row in first.records
        if row.record.family_id == "random-subset"
    }
    assert first == second
    assert first.status is FamilyBlockMatchStatus.SELECTED
    assert selected_subset_ids == {
        "random-subset:row-00",
        "random-subset:row-01",
        "random-subset:row-02",
        "random-subset:row-03",
        "random-subset:row-08",
        "random-subset:row-09",
        "random-subset:row-10",
        "random-subset:row-11",
    }
    assert first.exact_prompt_token_budget


def test_family_block_match_uses_full_ordered_fractional_profile() -> None:
    targets = (
        *_family_block(
            "target-profile",
            group="boundary",
            fractional_profile=("F", "I"),
        ),
        *_production_blocks(
            "target",
            group="boundary",
            family_count=11,
        ),
    )
    candidates = (
        *_family_block(
            "random-profile",
            group="random",
            dose=16,
            prompt_tokens=(10,) * 16,
            target_tokens=(50,) * 16,
            fractional_profile=("I", "F"),
        ),
        *_production_blocks(
            "random",
            group="random",
            family_count=11,
            dose=16,
            prompt_tokens=(10,) * 16,
            target_tokens=(50,) * 16,
        ),
    )

    result = match_teacher_family_blocks(
        targets,
        candidates,
        policy=FamilyBlockMatchPolicy(dose=2, allocation_seed=3),
        target_optimizer_updates=20,
        random_optimizer_updates=20,
    )

    assert result.status is FamilyBlockMatchStatus.INFEASIBLE
    assert not result.records
    assert any("family_structure_shortage" in item for item in result.failures)


def test_family_block_match_distinguishes_infeasible_from_search_exhaustion() -> None:
    targets = _production_blocks(
        "target",
        group="boundary",
        family_count=12,
    )
    wrong_prompt_candidates = _production_blocks(
        "random",
        group="random",
        family_count=12,
        dose=16,
        prompt_tokens=(12,) * 16,
        target_tokens=(50,) * 16,
    )
    infeasible = match_teacher_family_blocks(
        targets,
        wrong_prompt_candidates,
        policy=FamilyBlockMatchPolicy(dose=2, allocation_seed=9),
        target_optimizer_updates=20,
        random_optimizer_updates=20,
    )
    exhausted = match_teacher_family_blocks(
        targets,
        _production_blocks(
            "random",
            group="random",
            family_count=13,
            dose=16,
            prompt_tokens=(10,) * 16,
            target_tokens=(50,) * 16,
        ),
        policy=FamilyBlockMatchPolicy(
            dose=2,
            allocation_seed=9,
            max_search_states=1,
        ),
        target_optimizer_updates=20,
        random_optimizer_updates=20,
    )

    assert infeasible.status is FamilyBlockMatchStatus.INFEASIBLE
    assert not infeasible.random_family_ids
    assert exhausted.status is FamilyBlockMatchStatus.SEARCH_EXHAUSTED
    assert not exhausted.random_family_ids
    assert exhausted.search_states > exhausted.policy.max_search_states
    assert any("option_search_exhausted" in item for item in exhausted.failures)


def test_report_contains_every_declared_covariate_and_token_ledger() -> None:
    target = (_record("target", group="boundary", prompt_tokens=25, target_tokens=100),)
    candidate = (
        _record("candidate", group="random", prompt_tokens=25, target_tokens=101),
    )
    report = compare_teacher_allocations(
        target,
        candidate,
        allocation_seed=11,
        target_optimizer_updates=8,
        matched_optimizer_updates=8,
    )

    names = {statistic.covariate for statistic in report.standardized_differences}
    assert names == {
        "tree_depth",
        "operand_count",
        "operator_multiset",
        "noncommutative_count",
        "fractional_intermediate",
        "target_magnitude_bin",
        "valid_family_count_bin",
        "teacher_trace_token_bin",
        "teacher_trace_format",
        "teacher_prompt_tokens",
        "teacher_target_tokens",
    }
    assert report.target_ledger.example_count == 1
    assert report.matched_ledger.prompt_tokens == 25
    assert report.target_token_relative_difference == pytest.approx(0.01)
    assert report.to_dict()["maximum_absolute_standardized_difference"] is None


def test_exact_two_percent_target_token_difference_is_accepted() -> None:
    targets = (
        _record("target", group="boundary", prompt_tokens=20, target_tokens=200),
    )
    matched = (_record("matched", group="random", prompt_tokens=20, target_tokens=204),)
    report = compare_teacher_allocations(
        targets,
        matched,
        allocation_seed=0,
        target_optimizer_updates=12,
        matched_optimizer_updates=12,
    )
    assert report.passed
    assert report.target_token_relative_difference == pytest.approx(0.02)


def test_prompt_target_and_optimizer_mismatches_fail_with_a_report() -> None:
    targets = (
        _record("target", group="boundary", prompt_tokens=20, target_tokens=100),
    )
    matched = (_record("matched", group="random", prompt_tokens=21, target_tokens=103),)

    with pytest.raises(MatchingFailure) as caught:
        compare_teacher_allocations(
            targets,
            matched,
            allocation_seed=1,
            target_optimizer_updates=10,
            matched_optimizer_updates=9,
        )

    report = caught.value.report
    assert not report.passed
    assert not report.exact_prompt_token_budget
    assert not report.exact_optimizer_updates
    assert report.target_token_relative_difference == pytest.approx(0.03)
    assert any("teacher_prompt_token_mismatch" in item for item in report.failures)
    assert any("teacher_target_token_mismatch" in item for item in report.failures)
    assert any("optimizer_update_mismatch" in item for item in report.failures)


def test_insufficient_stratum_and_trace_format_mismatch_fail_closed() -> None:
    alternate = _covariates(target_bin="large")
    targets = (
        _record("target-a", group="boundary", prompt_tokens=10, target_tokens=50),
        _record(
            "target-b",
            group="boundary",
            prompt_tokens=10,
            target_tokens=50,
            covariates=alternate,
        ),
    )
    candidates = (
        _record("candidate-a", group="random", prompt_tokens=10, target_tokens=50),
        _record(
            "candidate-wrong-format",
            group="random",
            prompt_tokens=10,
            target_tokens=50,
            covariates=alternate,
            trace_format="freeform_v0",
        ),
    )

    with pytest.raises(MatchingFailure) as caught:
        match_teacher_allocations(
            targets,
            candidates,
            allocation_seed=3,
            target_optimizer_updates=10,
            matched_optimizer_updates=10,
        )

    report = caught.value.report
    assert report.stratum_mismatches
    assert not report.exact_example_count
    assert any("stratum_mismatch" in failure for failure in report.failures)


def test_non_pre_treatment_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="pre-treatment"):
        TeacherExampleRecord(
            record_id="bad",
            family_id="family",
            allocation_group="random",
            covariates=_covariates(),
            teacher_prompt_tokens=10,
            teacher_target_tokens=10,
            teacher_trace_format="postorder_v1",
            measurement_stage="post_A",
        )


def test_matching_inputs_reject_coercible_or_mutable_field_types() -> None:
    with pytest.raises(ValueError, match="tree_depth"):
        _covariates(depth=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="operator_multiset"):
        _covariates(operators=["+", "-"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="teacher_prompt_tokens"):
        _record(
            "bad-token-count",
            group="random",
            prompt_tokens="10",  # type: ignore[arg-type]
            target_tokens=10,
        )


@pytest.mark.parametrize(
    ("seed", "rate"),
    [
        (-1, 0.02),
        (True, 0.02),
        (MAX_ROOT_SEED + 1, 0.02),
        (0, True),
        (0, -0.01),
        (0, 1.01),
        (0, float("nan")),
    ],
)
def test_seed_and_tolerance_validation_is_strict(seed: object, rate: object) -> None:
    target = (_record("target", group="boundary", prompt_tokens=10, target_tokens=10),)
    candidate = (
        _record("candidate", group="random", prompt_tokens=10, target_tokens=10),
    )
    with pytest.raises(ValueError):
        match_teacher_allocations(
            target,
            candidate,
            allocation_seed=seed,  # type: ignore[arg-type]
            target_optimizer_updates=1,
            matched_optimizer_updates=1,
            max_target_token_relative_difference=rate,  # type: ignore[arg-type]
        )
