"""Property-style cross-checks for the exact TCES solver components."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from hypothesis import given, settings, strategies as st

from duraseed.schemas import ExactRational, TCESConstraints, TCESTask
from duraseed.tasks.tces.ast import (
    BinaryExpression,
    BinaryOperator,
    Expression,
    IntegerLiteral,
    evaluate_expression,
    leaf_values,
    tree_depth,
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
from duraseed.tasks.tces.strategies import strategy_family_id
from duraseed.tasks.tces.teacher import (
    build_teacher_trace,
    generate_teacher_trace,
    verify_teacher_trace,
)
from duraseed.tasks.tces.verifier import verify_completion


@dataclass(frozen=True, slots=True)
class _BruteItem:
    expression: Expression
    value: Fraction


def _brute_apply(
    operator: BinaryOperator, left: Fraction, right: Fraction
) -> Fraction | None:
    if operator is BinaryOperator.ADD:
        return left + right
    if operator is BinaryOperator.SUB:
        return left - right
    if operator is BinaryOperator.MUL:
        return left * right
    if right == 0:
        return None
    return left / right


def _brute_identity(operator: BinaryOperator, left: Fraction, right: Fraction) -> bool:
    return (
        (operator is BinaryOperator.ADD and (left == 0 or right == 0))
        or (operator is BinaryOperator.SUB and right == 0)
        or (operator is BinaryOperator.MUL and (left == 1 or right == 1))
        or (operator is BinaryOperator.DIV and right == 1)
    )


def _brute_force_solutions(
    operands: tuple[int, ...],
    target: Fraction,
    operators: tuple[BinaryOperator, ...],
    constraints: EnumerationConstraints,
) -> frozenset[str]:
    """Independent pair-reduction brute force; intentionally not subset DP."""

    found: set[str] = set()

    def search(items: tuple[_BruteItem, ...]) -> None:
        if len(items) == 1:
            if items[0].value == target:
                found.add(render_canonical_expression(items[0].expression))
            return

        for left_index in range(len(items)):
            for right_index in range(left_index + 1, len(items)):
                first = items[left_index]
                second = items[right_index]
                remainder = tuple(
                    item
                    for index, item in enumerate(items)
                    if index not in (left_index, right_index)
                )
                for operator in operators:
                    orientations: Iterable[tuple[_BruteItem, _BruteItem]]
                    if operator in (BinaryOperator.ADD, BinaryOperator.MUL):
                        orientations = ((first, second),)
                    else:
                        orientations = ((first, second), (second, first))
                    for left, right in orientations:
                        if (
                            constraints.exclude_trivial_identity_steps
                            and _brute_identity(operator, left.value, right.value)
                        ):
                            continue
                        value = _brute_apply(operator, left.value, right.value)
                        if value is None:
                            continue
                        if abs(value) > constraints.max_abs_intermediate:
                            continue
                        if (
                            constraints.max_denominator is not None
                            and value.denominator > constraints.max_denominator
                        ):
                            continue
                        if (
                            not constraints.allow_fractional_intermediates
                            and value.denominator != 1
                        ):
                            continue
                        expression = canonicalize_expression(
                            BinaryExpression(
                                operator, left.expression, right.expression
                            )
                        )
                        if tree_depth(expression) > constraints.max_tree_depth:
                            continue
                        search(remainder + (_BruteItem(expression, value),))

    search(
        tuple(_BruteItem(IntegerLiteral(value), Fraction(value)) for value in operands)
    )
    return frozenset(found)


@st.composite
def _enumeration_cases(draw):  # type: ignore[no-untyped-def]
    operands = tuple(
        draw(
            st.lists(
                st.integers(min_value=2, max_value=9),
                min_size=2,
                max_size=4,
                unique=True,
            )
        )
    )
    operator_symbols = draw(st.sets(st.sampled_from(("+", "-", "*", "/")), min_size=1))
    operators = tuple(
        operator for operator in BinaryOperator if operator.value in operator_symbols
    )
    numerator = draw(st.integers(min_value=-30, max_value=50))
    denominator = draw(st.integers(min_value=1, max_value=6))
    return operands, Fraction(numerator, denominator), operators


@given(_enumeration_cases())
@settings(max_examples=30, deadline=None)
def test_subset_dp_matches_independent_brute_force_for_n_at_most_four(
    case: tuple[tuple[int, ...], Fraction, tuple[BinaryOperator, ...]],
) -> None:
    operands, target, operators = case
    constraints = EnumerationConstraints(max_tree_depth=5)

    dynamic = enumerate_solutions(
        operands,
        target,
        operators,
        constraints=constraints,
    )
    brute = _brute_force_solutions(operands, target, operators, constraints)

    assert frozenset(item.canonical_expression for item in dynamic.expressions) == brute
    assert all(item.value == target for item in dynamic.expressions)
    assert all(
        sorted(leaf_values(item.expression)) == sorted(operands)
        for item in dynamic.expressions
    )


_EXPRESSION_TREES = st.recursive(
    st.integers(min_value=2, max_value=20).map(IntegerLiteral),
    lambda children: st.builds(
        BinaryExpression,
        operator=st.sampled_from(tuple(BinaryOperator)),
        left=children,
        right=children,
    ),
    max_leaves=6,
)


@given(
    left=_EXPRESSION_TREES,
    right=_EXPRESSION_TREES,
    operator=st.sampled_from((BinaryOperator.ADD, BinaryOperator.MUL)),
)
@settings(max_examples=100, deadline=None)
def test_commutative_swap_and_canonicalization_idempotence(
    left: Expression,
    right: Expression,
    operator: BinaryOperator,
) -> None:
    forward = BinaryExpression(operator, left, right)
    swapped = BinaryExpression(operator, right, left)

    canonical = canonicalize_expression(forward)
    assert canonicalize_expression(canonical) == canonical
    assert canonicalize_expression(swapped) == canonical


@given(
    st.lists(
        st.integers(min_value=2, max_value=12),
        min_size=2,
        max_size=5,
        unique=True,
    )
)
@settings(max_examples=40, deadline=None)
def test_generated_teacher_replays_parses_and_exactly_verifies(
    operands_list: list[int],
) -> None:
    operands = tuple(operands_list)
    expression: Expression = BinaryExpression(
        BinaryOperator.DIV,
        IntegerLiteral(operands[0]),
        IntegerLiteral(operands[1]),
    )
    for index, operand in enumerate(operands[2:]):
        operator = BinaryOperator.ADD if index % 2 == 0 else BinaryOperator.SUB
        expression = BinaryExpression(operator, expression, IntegerLiteral(operand))
    expression = canonicalize_expression(expression)
    value = evaluate_expression(expression)
    trace = generate_teacher_trace(expression)

    assert verify_teacher_trace(expression, trace)
    structured = build_teacher_trace(expression)
    parsed = parse_expression(structured.answer_expression)
    assert evaluate_expression(parsed) == value
    assert strategy_family_id(parsed, operands) == strategy_family_id(
        expression, operands
    )

    task = TCESTask(
        operands=operands,
        target=ExactRational(
            numerator=value.numerator,
            denominator=value.denominator,
        ),
        constraints=TCESConstraints(
            max_abs_intermediate=100_000,
            max_denominator=10_000,
            max_tree_depth=8,
            max_ast_nodes=31,
        ),
    )
    verified = verify_completion(trace, task)
    assert verified.reward == 1.0
    assert verified.canonical_expression == structured.answer_expression
    assert verified.strategy_family_id == strategy_family_id(expression, operands)
