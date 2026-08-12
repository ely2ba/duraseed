"""Canonical forms for TCES expression trees.

Canonicalization is deliberately syntactic.  It recursively sorts the two
children of addition and multiplication, but it does not apply algebraic
rewrites such as reassociation, distribution, cancellation, or constant
folding.  Consequently, the full binary-tree shape used by a strategy family
is preserved.
"""

from __future__ import annotations

from typing import TypeAlias

from .ast import BinaryExpression, BinaryOperator, Expression, IntegerLiteral


CanonicalKey: TypeAlias = tuple[object, ...]

_COMMUTATIVE_OPERATORS = frozenset({BinaryOperator.ADD, BinaryOperator.MUL})


def _node_count(expression: Expression) -> int:
    if isinstance(expression, IntegerLiteral):
        return 1
    if isinstance(expression, BinaryExpression):
        return 1 + _node_count(expression.left) + _node_count(expression.right)
    raise TypeError(f"unsupported TCES expression node: {type(expression)!r}")


def _structural_key(expression: Expression) -> CanonicalKey:
    """Return a totally ordered key for an already-canonical expression."""

    if isinstance(expression, IntegerLiteral):
        return ("literal", expression.value)
    if isinstance(expression, BinaryExpression):
        return (
            "binary",
            expression.operator.value,
            _structural_key(expression.left),
            _structural_key(expression.right),
        )
    raise TypeError(f"unsupported TCES expression node: {type(expression)!r}")


def _child_sort_key(expression: Expression) -> tuple[int, CanonicalKey]:
    # Sorting smaller subtrees first makes the normal form easy to inspect and
    # is independent of object identity or hash randomization.
    return (_node_count(expression), _structural_key(expression))


def canonicalize_expression(expression: Expression) -> Expression:
    """Return the deterministic, commutative-normalized form of ``expression``.

    New immutable nodes are returned.  Applying this function twice returns an
    equal tree, and swapping children below ``+`` or ``*`` cannot change the
    result.  Child order below ``-`` and ``/`` is always retained.
    """

    if isinstance(expression, IntegerLiteral):
        return expression
    if not isinstance(expression, BinaryExpression):
        raise TypeError(f"unsupported TCES expression node: {type(expression)!r}")

    left = canonicalize_expression(expression.left)
    right = canonicalize_expression(expression.right)
    if expression.operator in _COMMUTATIVE_OPERATORS and _child_sort_key(
        right
    ) < _child_sort_key(left):
        left, right = right, left
    return BinaryExpression(expression.operator, left, right)


def canonical_key(expression: Expression) -> tuple[int, CanonicalKey]:
    """Return a stable comparison/deduplication key for an expression."""

    canonical = canonicalize_expression(expression)
    return _child_sort_key(canonical)


def render_canonical_expression(expression: Expression) -> str:
    """Render a canonical expression with explicit binary parentheses."""

    canonical = canonicalize_expression(expression)

    def render(node: Expression) -> str:
        if isinstance(node, IntegerLiteral):
            return str(node.value)
        if isinstance(node, BinaryExpression):
            return f"({render(node.left)}{node.operator.value}{render(node.right)})"
        raise TypeError(f"unsupported TCES expression node: {type(node)!r}")

    return render(canonical)


# Short, discoverable aliases used by verifier/generator call sites.
canonicalize = canonicalize_expression
canonicalize_ast = canonicalize_expression
canonical_expression = render_canonical_expression


__all__ = [
    "CanonicalKey",
    "canonical_expression",
    "canonical_key",
    "canonicalize",
    "canonicalize_ast",
    "canonicalize_expression",
    "render_canonical_expression",
]
