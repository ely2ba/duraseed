"""Value-abstracted strategy-family signatures for TCES expressions."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .ast import BinaryExpression, BinaryOperator, Expression, IntegerLiteral
from .canonicalize import canonicalize_expression


_OPERATOR_NAMES = {
    BinaryOperator.ADD: "ADD",
    BinaryOperator.SUB: "SUB",
    BinaryOperator.MUL: "MUL",
    BinaryOperator.DIV: "DIV",
}


@dataclass(frozen=True, slots=True)
class StrategyFamilySignature:
    """All components of the primary TCES strategy-family signature.

    ``intermediate_profile`` is in canonical post-order and contains ``I`` for
    an integral binary result or ``F`` for a non-integral rational result.  It
    includes the root result so that every binary node has exactly one marker.
    The optional sign profile uses ``N``, ``Z``, and ``P`` in the same order.
    """

    structure: str
    intermediate_profile: tuple[str, ...]
    sign_profile: tuple[str, ...] | None = None

    def family_id(self) -> str:
        """Return the stable serialized identifier used in artifacts."""

        profile = ",".join(self.intermediate_profile)
        identifier = f"{self.structure}|intermediates={profile}"
        if self.sign_profile is not None:
            identifier += f"|signs={','.join(self.sign_profile)}"
        return identifier

    def __str__(self) -> str:
        return self.family_id()


def _canonical_leaf_labels(
    expression: Expression, prompt_operands: Sequence[int]
) -> dict[int, deque[str]]:
    leaves: list[int] = []

    def collect(node: Expression) -> None:
        if isinstance(node, IntegerLiteral):
            leaves.append(node.value)
            return
        if isinstance(node, BinaryExpression):
            collect(node.left)
            collect(node.right)
            return
        raise TypeError(f"unsupported TCES expression node: {type(node)!r}")

    collect(expression)
    if Counter(leaves) != Counter(prompt_operands):
        raise ValueError(
            "expression leaf multiset must equal the prompt operand multiset"
        )

    labels: dict[int, deque[str]] = defaultdict(deque)
    # The original position breaks ties deterministically for the non-primary
    # repeated-operand setting.  Equal leaves are then consumed in canonical
    # left-to-right order because an expression does not carry source position.
    ranked = sorted(enumerate(prompt_operands), key=lambda item: (item[1], item[0]))
    for rank, (_, value) in enumerate(ranked, start=1):
        labels[value].append(f"r{rank}")
    return labels


def structural_signature(expression: Expression, prompt_operands: Sequence[int]) -> str:
    """Return the canonical tree/operator/operand-rank signature.

    This is the human-readable component shown in the project specification,
    for example ``MUL(SUB(r5,r3),ADD(r4,SUB(r1,r2)))``.  Use
    :func:`strategy_family_id` for the complete primary identifier, which also
    carries the integer-versus-fractional intermediate profile.
    """

    canonical = canonicalize_expression(expression)
    labels = _canonical_leaf_labels(canonical, prompt_operands)

    def build(node: Expression) -> str:
        if isinstance(node, IntegerLiteral):
            return labels[node.value].popleft()
        if isinstance(node, BinaryExpression):
            return (
                f"{_OPERATOR_NAMES[node.operator]}("
                f"{build(node.left)},{build(node.right)})"
            )
        raise TypeError(f"unsupported TCES expression node: {type(node)!r}")

    return build(canonical)


def _apply(operator: BinaryOperator, left: Fraction, right: Fraction) -> Fraction:
    if operator is BinaryOperator.ADD:
        return left + right
    if operator is BinaryOperator.SUB:
        return left - right
    if operator is BinaryOperator.MUL:
        return left * right
    if operator is BinaryOperator.DIV:
        if right == 0:
            raise ZeroDivisionError("division by zero in TCES strategy tree")
        return left / right
    raise ValueError(f"unsupported TCES operator: {operator!r}")


def _profiles(
    expression: Expression,
) -> tuple[Fraction, tuple[str, ...], tuple[str, ...]]:
    if isinstance(expression, IntegerLiteral):
        return Fraction(expression.value), (), ()
    if not isinstance(expression, BinaryExpression):
        raise TypeError(f"unsupported TCES expression node: {type(expression)!r}")

    left_value, left_types, left_signs = _profiles(expression.left)
    right_value, right_types, right_signs = _profiles(expression.right)
    value = _apply(expression.operator, left_value, right_value)
    result_type = "I" if value.denominator == 1 else "F"
    result_sign = "N" if value < 0 else "P" if value > 0 else "Z"
    return (
        value,
        left_types + right_types + (result_type,),
        left_signs + right_signs + (result_sign,),
    )


def intermediate_profile(expression: Expression) -> tuple[str, ...]:
    """Return canonical post-order integer/fraction markers."""

    _, profile, _ = _profiles(canonicalize_expression(expression))
    return profile


def intermediate_sign_profile(expression: Expression) -> tuple[str, ...]:
    """Return canonical post-order negative/zero/positive markers."""

    _, _, profile = _profiles(canonicalize_expression(expression))
    return profile


def strategy_family_signature(
    expression: Expression,
    prompt_operands: Sequence[int],
    *,
    include_sign_pattern: bool = False,
) -> StrategyFamilySignature:
    """Build the complete primary strategy-family signature."""

    canonical = canonicalize_expression(expression)
    _, value_types, signs = _profiles(canonical)
    return StrategyFamilySignature(
        structure=structural_signature(canonical, prompt_operands),
        intermediate_profile=value_types,
        sign_profile=signs if include_sign_pattern else None,
    )


def strategy_family_id(
    expression: Expression,
    prompt_operands: Sequence[int],
    *,
    include_sign_pattern: bool = False,
) -> str:
    """Return the stable family identifier for a verifier-valid expression."""

    return strategy_family_signature(
        expression,
        prompt_operands,
        include_sign_pattern=include_sign_pattern,
    ).family_id()


# Alias matching the shorter terminology used in analysis code.
family_signature = strategy_family_id


__all__ = [
    "StrategyFamilySignature",
    "family_signature",
    "intermediate_profile",
    "intermediate_sign_profile",
    "strategy_family_id",
    "strategy_family_signature",
    "structural_signature",
]
