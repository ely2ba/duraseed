"""Unit checks for TCES canonical families, enumeration, and teachers."""

from fractions import Fraction
import hashlib

import pytest

from duraseed.schemas import ExactRational, TCESConstraints, TCESTask
from duraseed.tasks.tces.ast import (
    BinaryExpression,
    BinaryOperator,
    evaluate_expression,
    leaf_values,
)
from duraseed.tasks.tces.canonicalize import (
    canonicalize_expression,
    render_canonical_expression,
)
from duraseed.tasks.tces.enumerate import (
    EnumerationConstraints,
    enumerate_solutions,
)
from duraseed.tasks.tces.parser import parse_expression
from duraseed.tasks.tces.strategies import (
    strategy_family_id,
    structural_signature,
)
from duraseed.tasks.tces.teacher import (
    build_teacher_trace,
    generate_teacher_trace,
    replay_teacher_steps,
    verify_teacher_trace,
)
from duraseed.tasks.tces.verifier import verify_completion


def test_specification_example_has_exact_canonical_family_and_teacher() -> None:
    expression = parse_expression("((14-8)*(11+(3-7)))")
    operands = (3, 7, 8, 11, 14)

    assert render_canonical_expression(expression) == "((14-8)*(11+(3-7)))"
    assert structural_signature(expression, operands) == (
        "MUL(SUB(r5,r3),ADD(r4,SUB(r1,r2)))"
    )
    assert strategy_family_id(expression, operands) == (
        "MUL(SUB(r5,r3),ADD(r4,SUB(r1,r2)))|intermediates=I,I,I,I"
    )

    expected_trace = "\n".join(
        (
            "14 - 8 = 6.",
            "3 - 7 = -4.",
            "11 + (-4) = 7.",
            "6 * 7 = 42.",
            "<answer>((14-8)*(11+(3-7)))</answer>",
        )
    )
    trace = build_teacher_trace(expression)
    assert trace.render() == expected_trace
    assert replay_teacher_steps(trace.steps) == Fraction(42)
    assert verify_teacher_trace(expression, expected_trace)


def test_canonicalization_is_idempotent_and_normalizes_only_commutative_order() -> None:
    left = parse_expression("(11-(7+3))")
    right = parse_expression("(2*5)")
    forward = BinaryExpression(BinaryOperator.MUL, left, right)
    swapped = BinaryExpression(BinaryOperator.MUL, right, left)

    canonical = canonicalize_expression(forward)
    assert canonicalize_expression(canonical) == canonical
    assert canonicalize_expression(swapped) == canonical
    assert render_canonical_expression(swapped) == render_canonical_expression(forward)

    subtraction = parse_expression("(7-3)")
    reversed_subtraction = parse_expression("(3-7)")
    assert render_canonical_expression(subtraction) != render_canonical_expression(
        reversed_subtraction
    )
    assert strategy_family_id(subtraction, (3, 7)) != strategy_family_id(
        reversed_subtraction, (3, 7)
    )


def test_fractional_intermediate_profile_is_part_of_primary_family_id() -> None:
    integral_path = parse_expression("((4/2)+7)")
    fractional_path = parse_expression("((5/3)+7)")

    assert structural_signature(integral_path, (2, 4, 7)) == (
        structural_signature(fractional_path, (3, 5, 7))
    )
    assert strategy_family_id(integral_path, (2, 4, 7)).endswith("|intermediates=I,I")
    assert strategy_family_id(fractional_path, (3, 5, 7)).endswith("|intermediates=F,F")


def test_repeated_operands_receive_deterministic_positional_rank_labels() -> None:
    expression = parse_expression("((2+2)*3)")

    assert structural_signature(expression, (2, 2, 3)) == ("MUL(r3,ADD(r1,r2))")
    assert strategy_family_id(expression, (2, 2, 3)) == strategy_family_id(
        expression, (2, 2, 3)
    )


def test_enumerator_uses_exact_fraction_arithmetic_and_immediate_constraints() -> None:
    fractional = enumerate_solutions((2, 3), Fraction(2, 3))
    integer_only = enumerate_solutions(
        (2, 3),
        Fraction(2, 3),
        constraints=EnumerationConstraints(allow_fractional_intermediates=False),
    )
    denominator_limited = enumerate_solutions(
        (2, 3),
        Fraction(2, 3),
        constraints=EnumerationConstraints(max_denominator=2),
    )

    assert [item.canonical_expression for item in fractional.expressions] == ["(2/3)"]
    assert fractional.expressions[0].value == Fraction(2, 3)
    assert not integer_only.solvable
    assert not denominator_limited.solvable


@pytest.mark.parametrize("operands", [(True, 3), (-2, 3), (10**1024,)])
def test_enumerator_rejects_operands_outside_the_parser_grammar(
    operands: tuple[int, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        enumerate_solutions(operands, 1)


def test_opt_in_expression_cap_is_deterministic_and_discloses_incompleteness() -> None:
    constraints = EnumerationConstraints(max_expressions_per_value=1)

    first = enumerate_solutions((2, 3, 4, 5), 14, constraints=constraints)
    second = enumerate_solutions((2, 3, 4, 5), 14, constraints=constraints)

    assert first == second
    assert first.pruned
    assert not first.complete
    assert [item.canonical_expression for item in first.expressions] == [
        "((2+3)+(4+5))"
    ]
    with pytest.raises(ValueError, match="pruned enumeration"):
        _ = first.complete_family_set


@pytest.fixture(scope="module")
def complete_n5_enumeration():  # type: ignore[no-untyped-def]
    return enumerate_solutions((3, 7, 8, 11, 14), 42)


def test_n5_family_set_is_complete_and_matches_independent_frozen_oracle(
    complete_n5_enumeration,  # type: ignore[no-untyped-def]
) -> None:
    result = complete_n5_enumeration
    expression_digest = hashlib.sha256(
        "\n".join(
            sorted(item.canonical_expression for item in result.expressions)
        ).encode("utf-8")
    ).hexdigest()

    # This oracle was produced by a separate permutation/tree-shape brute-force
    # implementation, not by the subset-DP recurrence.
    assert result.complete
    assert result.exact_expression_tree_count == 105
    assert len(result.complete_family_set) == 105
    assert result.shortest_depth == 4
    assert expression_digest == (
        "6af28fd80cf1d45f060f225881b3928b3cff491a547a3e6affa7f27fe7322a1f"
    )


def test_every_n5_family_representative_has_a_teacher_that_exact_verifier_accepts(
    complete_n5_enumeration,  # type: ignore[no-untyped-def]
) -> None:
    result = complete_n5_enumeration
    task = TCESTask(
        operands=result.operands,
        target=ExactRational(numerator=42),
        constraints=TCESConstraints(
            max_abs_intermediate=10_000,
            max_denominator=1_000,
            max_tree_depth=5,
            max_ast_nodes=31,
        ),
    )

    for solution in result.expressions:
        trace = generate_teacher_trace(solution.expression)
        assert verify_teacher_trace(solution.expression, trace)
        verified = verify_completion(trace, task)
        assert verified.reward == 1.0
        assert verified.canonical_expression == solution.canonical_expression
        assert verified.strategy_family_id == solution.family_id
        assert evaluate_expression(solution.expression) == Fraction(42)
        assert sorted(leaf_values(solution.expression)) == sorted(result.operands)
