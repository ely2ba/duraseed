"""Small, closed AST for Template-Controlled Expression Synthesis.

Only decimal integer leaves and the four declared binary operators are
representable.  Keeping the tree closed in this way is an important verifier
invariant: model text is never converted into Python code or a general-purpose
symbolic expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Iterator, TypeAlias


class BinaryOperator(StrEnum):
    """The complete TCES operator vocabulary."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


@dataclass(frozen=True, slots=True)
class IntegerLiteral:
    """A decimal integer leaf from the prompt's operand multiset."""

    value: int


@dataclass(frozen=True, slots=True)
class BinaryExpression:
    """A binary arithmetic operation.

    Unary operations are intentionally absent.  In particular, negative
    intermediate values must be produced by binary subtraction rather than a
    signed numeric literal.
    """

    operator: BinaryOperator
    left: Expression
    right: Expression


Expression: TypeAlias = IntegerLiteral | BinaryExpression


def iter_nodes(expression: Expression) -> Iterator[Expression]:
    """Yield nodes in deterministic pre-order without recursive traversal."""

    stack: list[Expression] = [expression]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, BinaryExpression):
            stack.append(node.right)
            stack.append(node.left)


def iter_postorder(expression: Expression) -> Iterator[Expression]:
    """Yield children before parents in deterministic left-to-right order."""

    stack: list[tuple[Expression, bool]] = [(expression, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            yield node
            continue
        stack.append((node, True))
        if isinstance(node, BinaryExpression):
            stack.append((node.right, False))
            stack.append((node.left, False))


def leaf_values(expression: Expression) -> tuple[int, ...]:
    """Return integer leaves in source-tree order."""

    return tuple(
        node.value
        for node in iter_nodes(expression)
        if isinstance(node, IntegerLiteral)
    )


def operators(expression: Expression) -> tuple[BinaryOperator, ...]:
    """Return operators in source-tree pre-order."""

    return tuple(
        node.operator
        for node in iter_nodes(expression)
        if isinstance(node, BinaryExpression)
    )


def node_count(expression: Expression) -> int:
    """Count integer and binary-operation AST nodes."""

    return sum(1 for _ in iter_nodes(expression))


def tree_depth(expression: Expression) -> int:
    """Return AST depth using one for a leaf.

    Parentheses do not create AST nodes, so redundant wrapping does not change
    this value.  Parser nesting is guarded separately.
    """

    maximum = 0
    stack: list[tuple[Expression, int]] = [(expression, 1)]
    while stack:
        node, depth = stack.pop()
        maximum = max(maximum, depth)
        if isinstance(node, BinaryExpression):
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))
    return maximum


def evaluate_expression(expression: Expression) -> Fraction:
    """Evaluate an AST with exact rational arithmetic.

    This helper has no task-specific magnitude guards.  The verifier uses a
    guarded evaluator so it can classify resource-limit failures precisely.
    """

    values: dict[int, Fraction] = {}
    for node in iter_postorder(expression):
        if isinstance(node, IntegerLiteral):
            values[id(node)] = Fraction(node.value)
            continue

        left = values[id(node.left)]
        right = values[id(node.right)]
        if node.operator is BinaryOperator.ADD:
            value = left + right
        elif node.operator is BinaryOperator.SUB:
            value = left - right
        elif node.operator is BinaryOperator.MUL:
            value = left * right
        else:
            value = left / right
        values[id(node)] = value
    return values[id(expression)]


def render_expression(expression: Expression) -> str:
    """Render a fully parenthesized expression while preserving child order."""

    rendered: dict[int, str] = {}
    for node in iter_postorder(expression):
        if isinstance(node, IntegerLiteral):
            rendered[id(node)] = str(node.value)
        else:
            rendered[id(node)] = (
                f"({rendered[id(node.left)]}{node.operator.value}"
                f"{rendered[id(node.right)]})"
            )
    return rendered[id(expression)]


def canonical_expression(expression: Expression) -> str:
    """Return a minimal local canonical rendering.

    Commutative children are ordered lexicographically; subtraction and
    division retain direction.  This is deliberately only an integration hook.
    The full project canonicalizer can replace it without changing parser or
    verifier semantics.
    """

    rendered: dict[int, str] = {}
    for node in iter_postorder(expression):
        if isinstance(node, IntegerLiteral):
            rendered[id(node)] = str(node.value)
            continue

        left = rendered[id(node.left)]
        right = rendered[id(node.right)]
        if node.operator in (BinaryOperator.ADD, BinaryOperator.MUL) and right < left:
            left, right = right, left
        rendered[id(node)] = f"({left}{node.operator.value}{right})"
    return rendered[id(expression)]


def ast_to_dict(expression: Expression) -> dict[str, object]:
    """Convert the closed AST into a JSON-serializable nested mapping."""

    encoded: dict[int, dict[str, object]] = {}
    for node in iter_postorder(expression):
        if isinstance(node, IntegerLiteral):
            encoded[id(node)] = {"type": "integer", "value": node.value}
        else:
            encoded[id(node)] = {
                "type": "binary",
                "operator": node.operator.value,
                "left": encoded[id(node.left)],
                "right": encoded[id(node.right)],
            }
    return encoded[id(expression)]
