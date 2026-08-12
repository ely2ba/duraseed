"""Deterministic exact-solver teacher traces for TCES."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

from .ast import BinaryExpression, BinaryOperator, Expression, IntegerLiteral
from .canonicalize import canonicalize_expression, render_canonical_expression


_LINE_PATTERN = re.compile(
    r"^(?P<left>\(?-?\d+(?:/\d+)?\)?) "
    r"(?P<operator>[+*/-]) "
    r"(?P<right>\(?-?\d+(?:/\d+)?\)?) = "
    r"(?P<result>-?\d+(?:/\d+)?)\.$"
)


@dataclass(frozen=True, slots=True)
class TeacherStep:
    """One post-order arithmetic step derived from a binary AST node."""

    operator: BinaryOperator
    left: Fraction
    right: Fraction
    result: Fraction
    subexpression: str

    def render(self) -> str:
        return (
            f"{_format_operand(self.left)} {self.operator.value} "
            f"{_format_operand(self.right)} = {_format_value(self.result)}."
        )


@dataclass(frozen=True, slots=True)
class TeacherTrace:
    """A structured trace whose rendered form is suitable for SFT data."""

    expression: Expression
    steps: tuple[TeacherStep, ...]
    answer_expression: str

    @property
    def value(self) -> Fraction:
        if self.steps:
            return self.steps[-1].result
        if isinstance(self.expression, IntegerLiteral):
            return Fraction(self.expression.value)
        raise AssertionError("binary teacher trace unexpectedly has no steps")

    def render(self) -> str:
        lines = [step.render() for step in self.steps]
        lines.append(f"<answer>{self.answer_expression}</answer>")
        return "\n".join(lines)


def _format_value(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _format_operand(value: Fraction) -> str:
    rendered = _format_value(value)
    return f"({rendered})" if value < 0 else rendered


def _parse_value(source: str) -> Fraction:
    normalized = source
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1]
    if "/" in normalized:
        numerator, denominator = normalized.split("/", maxsplit=1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(int(normalized))


def _apply(operator: BinaryOperator, left: Fraction, right: Fraction) -> Fraction:
    if operator is BinaryOperator.ADD:
        return left + right
    if operator is BinaryOperator.SUB:
        return left - right
    if operator is BinaryOperator.MUL:
        return left * right
    if operator is BinaryOperator.DIV:
        if right == 0:
            raise ZeroDivisionError("division by zero in teacher tree")
        return left / right
    raise ValueError(f"unsupported TCES operator: {operator!r}")


def build_teacher_trace(expression: Expression) -> TeacherTrace:
    """Build a concise canonical post-order derivation from one exact AST."""

    canonical = canonicalize_expression(expression)
    steps: list[TeacherStep] = []

    def visit(node: Expression) -> Fraction:
        if isinstance(node, IntegerLiteral):
            return Fraction(node.value)
        if not isinstance(node, BinaryExpression):
            raise TypeError(f"unsupported TCES expression node: {type(node)!r}")
        left = visit(node.left)
        right = visit(node.right)
        result = _apply(node.operator, left, right)
        steps.append(
            TeacherStep(
                operator=node.operator,
                left=left,
                right=right,
                result=result,
                subexpression=render_canonical_expression(node),
            )
        )
        return result

    visit(canonical)
    return TeacherTrace(
        expression=canonical,
        steps=tuple(steps),
        answer_expression=render_canonical_expression(canonical),
    )


def generate_teacher_trace(expression: Expression) -> str:
    """Return the deterministic SFT completion for ``expression``."""

    return build_teacher_trace(expression).render()


def replay_teacher_steps(steps: tuple[TeacherStep, ...]) -> Fraction | None:
    """Recompute every structured step, returning the final result."""

    for step in steps:
        if _apply(step.operator, step.left, step.right) != step.result:
            raise ValueError(f"inconsistent teacher step: {step.render()}")
    return steps[-1].result if steps else None


def verify_teacher_trace(expression: Expression, trace: str | None = None) -> bool:
    """Replay a rendered trace and prove it was derived from ``expression``.

    Verification is intentionally strict: arithmetic is reparsed as exact
    ``Fraction`` values, the post-order sequence must match the supplied AST,
    and the final answer must be that same canonical AST.
    """

    expected = build_teacher_trace(expression)
    rendered = expected.render() if trace is None else trace
    lines = rendered.splitlines()
    if len(lines) != len(expected.steps) + 1:
        return False
    if lines[-1] != f"<answer>{expected.answer_expression}</answer>":
        return False

    for source, step in zip(lines[:-1], expected.steps, strict=True):
        match = _LINE_PATTERN.fullmatch(source)
        if match is None:
            return False
        operator = next(
            (
                candidate
                for candidate in BinaryOperator
                if candidate.value == match.group("operator")
            ),
            None,
        )
        if operator is None:
            return False
        left = _parse_value(match.group("left"))
        right = _parse_value(match.group("right"))
        result = _parse_value(match.group("result"))
        if (
            operator is not step.operator
            or left != step.left
            or right != step.right
            or result != step.result
            or _apply(operator, left, right) != result
        ):
            return False

    replayed = replay_teacher_steps(expected.steps)
    return (
        replayed == expected.value
        if expected.steps
        else expected.value == Fraction(expected.expression.value)
    )


# Concise aliases used by data-generation code.
teacher_trace = generate_teacher_trace
trace_is_consistent = verify_teacher_trace


__all__ = [
    "TeacherStep",
    "TeacherTrace",
    "build_teacher_trace",
    "generate_teacher_trace",
    "replay_teacher_steps",
    "teacher_trace",
    "trace_is_consistent",
    "verify_teacher_trace",
]
