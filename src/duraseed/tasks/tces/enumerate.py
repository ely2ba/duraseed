"""Complete subset-DP enumeration for Template-Controlled Expression Synthesis.

The primary ``n=5`` path has no implicit expression cap.  If callers opt into
``max_expressions_per_value``, pruning is deterministic and the result is
explicitly marked incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from duraseed.schemas import TCES_MAX_INTEGER_DIGITS

from .ast import BinaryExpression, BinaryOperator, Expression, IntegerLiteral
from .canonicalize import canonicalize_expression, render_canonical_expression
from .strategies import strategy_family_id


_OPERATOR_ORDER = (
    BinaryOperator.ADD,
    BinaryOperator.SUB,
    BinaryOperator.MUL,
    BinaryOperator.DIV,
)

_OPERATOR_ALIASES = {
    "+": BinaryOperator.ADD,
    "add": BinaryOperator.ADD,
    "ADD": BinaryOperator.ADD,
    "-": BinaryOperator.SUB,
    "sub": BinaryOperator.SUB,
    "SUB": BinaryOperator.SUB,
    "*": BinaryOperator.MUL,
    "mul": BinaryOperator.MUL,
    "MUL": BinaryOperator.MUL,
    "/": BinaryOperator.DIV,
    "div": BinaryOperator.DIV,
    "DIV": BinaryOperator.DIV,
}


@dataclass(frozen=True, slots=True)
class EnumerationConstraints:
    """Arithmetic and resource constraints applied during DP insertion."""

    allow_fractional_intermediates: bool = True
    max_abs_intermediate: int = 10_000
    max_denominator: int | None = 1_000
    max_tree_depth: int = 5
    max_ast_nodes: int = 31
    # No-op identities are a latent-generator difficulty filter, not part of
    # the verifier language.  Complete solution enumeration therefore retains
    # them unless a caller explicitly asks for the narrower generator view.
    exclude_trivial_identity_steps: bool = False
    require_positive_intermediates: bool = False
    max_expressions_per_value: int | None = None

    def __post_init__(self) -> None:
        if self.max_abs_intermediate < 1:
            raise ValueError("max_abs_intermediate must be positive")
        if self.max_denominator is not None and self.max_denominator < 1:
            raise ValueError("max_denominator must be positive when provided")
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be positive")
        if self.max_ast_nodes < 1:
            raise ValueError("max_ast_nodes must be positive")
        if (
            self.max_expressions_per_value is not None
            and self.max_expressions_per_value < 1
        ):
            raise ValueError("max_expressions_per_value must be positive")


@dataclass(frozen=True, slots=True)
class EnumeratedExpression:
    """One distinct canonical exact solution."""

    expression: Expression
    value: Fraction
    canonical_expression: str
    family_id: str
    depth: int
    derivation_count: int


@dataclass(frozen=True, slots=True)
class FamilyEnumeration:
    """Counts and a deterministic teacher candidate for one family."""

    family_id: str
    expression_count: int
    derivation_count: int
    representative: Expression
    representative_expression: str
    shortest_depth: int


@dataclass(frozen=True, slots=True)
class EnumerationResult:
    """Complete target-specific output of the subset-DP enumerator."""

    operands: tuple[int, ...]
    target: Fraction
    allowed_ops: tuple[BinaryOperator, ...]
    constraints: EnumerationConstraints
    expressions: tuple[EnumeratedExpression, ...]
    families: tuple[FamilyEnumeration, ...]
    shortest_depth: int | None
    exact_expression_tree_count: int
    derivation_count: int
    pruned: bool

    @property
    def solvable(self) -> bool:
        return bool(self.expressions)

    @property
    def complete(self) -> bool:
        return not self.pruned

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(family.family_id for family in self.families)

    @property
    def complete_family_set(self) -> frozenset[str]:
        if self.pruned:
            raise ValueError("a pruned enumeration has no complete family set")
        return frozenset(self.family_ids)

    @property
    def family_counts(self) -> dict[str, int]:
        return {family.family_id: family.expression_count for family in self.families}

    @property
    def family_representatives(self) -> dict[str, Expression]:
        return {family.family_id: family.representative for family in self.families}


@dataclass(frozen=True, slots=True)
class _DPRecord:
    expression: Expression
    value: Fraction
    canonical_expression: str
    depth: int
    node_count: int
    derivation_count: int = 1


def _as_fraction(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    as_fraction = getattr(value, "as_fraction", None)
    if callable(as_fraction):
        result = as_fraction()
        if isinstance(result, Fraction):
            return result
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if isinstance(numerator, int) and isinstance(denominator, int):
        return Fraction(numerator, denominator)
    raise TypeError(f"cannot convert {type(value)!r} to Fraction")


def _normalize_operators(
    operators: Iterable[BinaryOperator | str],
) -> tuple[BinaryOperator, ...]:
    normalized: set[BinaryOperator] = set()
    for operator in operators:
        if isinstance(operator, BinaryOperator):
            normalized.add(operator)
            continue
        try:
            normalized.add(_OPERATOR_ALIASES[operator])
        except KeyError as error:
            raise ValueError(f"unsupported TCES operator: {operator!r}") from error
    if not normalized:
        raise ValueError("at least one TCES operator is required")
    return tuple(operator for operator in _OPERATOR_ORDER if operator in normalized)


def _is_trivial_identity(
    operator: BinaryOperator, left: Fraction, right: Fraction
) -> bool:
    if operator is BinaryOperator.ADD:
        return left == 0 or right == 0
    if operator is BinaryOperator.SUB:
        return right == 0
    if operator is BinaryOperator.MUL:
        return left == 1 or right == 1
    if operator is BinaryOperator.DIV:
        return right == 1
    raise ValueError(f"unsupported TCES operator: {operator!r}")


def _apply(operator: BinaryOperator, left: Fraction, right: Fraction) -> Fraction:
    if operator is BinaryOperator.ADD:
        return left + right
    if operator is BinaryOperator.SUB:
        return left - right
    if operator is BinaryOperator.MUL:
        return left * right
    if operator is BinaryOperator.DIV:
        if right == 0:
            raise ZeroDivisionError("division by zero")
        return left / right
    raise ValueError(f"unsupported TCES operator: {operator!r}")


def _value_is_allowed(value: Fraction, constraints: EnumerationConstraints) -> bool:
    if abs(value) > constraints.max_abs_intermediate:
        return False
    if (
        constraints.max_denominator is not None
        and value.denominator > constraints.max_denominator
    ):
        return False
    if not constraints.allow_fractional_intermediates and value.denominator != 1:
        return False
    if constraints.require_positive_intermediates and value <= 0:
        return False
    return True


def _record_order(record: _DPRecord) -> tuple[int, str]:
    return (record.depth, record.canonical_expression)


def _insert_record(
    bucket: dict[str, _DPRecord],
    candidate: _DPRecord,
    cap: int | None,
) -> bool:
    """Insert/merge a record and return whether cap pruning occurred."""

    previous = bucket.get(candidate.canonical_expression)
    if previous is not None:
        bucket[candidate.canonical_expression] = _DPRecord(
            expression=previous.expression,
            value=previous.value,
            canonical_expression=previous.canonical_expression,
            depth=previous.depth,
            node_count=previous.node_count,
            derivation_count=(previous.derivation_count + candidate.derivation_count),
        )
        return False

    bucket[candidate.canonical_expression] = candidate
    if cap is None or len(bucket) <= cap:
        return False

    worst_key = max(bucket, key=lambda key: _record_order(bucket[key]))
    del bucket[worst_key]
    return True


def _bipartitions(mask: int) -> tuple[tuple[int, int], ...]:
    partitions: list[tuple[int, int]] = []
    left = (mask - 1) & mask
    while left:
        right = mask ^ left
        if right and left < right:
            partitions.append((left, right))
        left = (left - 1) & mask
    return tuple(sorted(partitions))


def _oriented_operations(
    left: _DPRecord,
    right: _DPRecord,
    allowed_ops: tuple[BinaryOperator, ...],
) -> Iterable[tuple[BinaryOperator, _DPRecord, _DPRecord]]:
    for operator in allowed_ops:
        if operator in (BinaryOperator.ADD, BinaryOperator.MUL):
            yield operator, left, right
        else:
            yield operator, left, right
            yield operator, right, left


def _combine(
    operator: BinaryOperator,
    left: _DPRecord,
    right: _DPRecord,
    constraints: EnumerationConstraints,
) -> _DPRecord | None:
    if operator is BinaryOperator.DIV and right.value == 0:
        return None
    if constraints.exclude_trivial_identity_steps and _is_trivial_identity(
        operator, left.value, right.value
    ):
        return None

    value = _apply(operator, left.value, right.value)
    if not _value_is_allowed(value, constraints):
        return None
    depth = max(left.depth, right.depth) + 1
    if depth > constraints.max_tree_depth:
        return None
    node_count = left.node_count + right.node_count + 1
    if node_count > constraints.max_ast_nodes:
        return None

    expression = canonicalize_expression(
        BinaryExpression(
            operator=operator,
            left=left.expression,
            right=right.expression,
        )
    )
    canonical = render_canonical_expression(expression)
    return _DPRecord(
        expression=expression,
        value=value,
        canonical_expression=canonical,
        depth=depth,
        node_count=node_count,
        derivation_count=left.derivation_count * right.derivation_count,
    )


def _enumerate_full_table(
    operands: tuple[int, ...],
    allowed_ops: tuple[BinaryOperator, ...],
    constraints: EnumerationConstraints,
) -> tuple[Mapping[Fraction, dict[str, _DPRecord]], bool]:
    # mask -> exact value -> canonical expression -> record
    table: dict[int, dict[Fraction, dict[str, _DPRecord]]] = {}
    for position, operand in enumerate(operands):
        mask = 1 << position
        value = Fraction(operand)
        expression = IntegerLiteral(operand)
        record = _DPRecord(
            expression=expression,
            value=value,
            canonical_expression=str(operand),
            depth=1,
            node_count=1,
        )
        table[mask] = {value: {record.canonical_expression: record}}

    pruned = False
    full_mask = (1 << len(operands)) - 1
    for subset_size in range(2, len(operands) + 1):
        for mask in range(1, full_mask + 1):
            if mask.bit_count() != subset_size:
                continue
            value_buckets: dict[Fraction, dict[str, _DPRecord]] = {}
            for left_mask, right_mask in _bipartitions(mask):
                left_values = table[left_mask]
                right_values = table[right_mask]
                for left_value in sorted(left_values):
                    left_records = left_values[left_value]
                    for right_value in sorted(right_values):
                        right_records = right_values[right_value]
                        for left_key in sorted(left_records):
                            left_record = left_records[left_key]
                            for right_key in sorted(right_records):
                                right_record = right_records[right_key]
                                for (
                                    operator,
                                    oriented_left,
                                    oriented_right,
                                ) in _oriented_operations(
                                    left_record, right_record, allowed_ops
                                ):
                                    candidate = _combine(
                                        operator,
                                        oriented_left,
                                        oriented_right,
                                        constraints,
                                    )
                                    if candidate is None:
                                        continue
                                    bucket = value_buckets.setdefault(
                                        candidate.value, {}
                                    )
                                    pruned = (
                                        _insert_record(
                                            bucket,
                                            candidate,
                                            constraints.max_expressions_per_value,
                                        )
                                        or pruned
                                    )
            table[mask] = value_buckets
    return table[full_mask], pruned


def enumerate_solutions(
    operands: Sequence[int],
    target: object,
    allowed_ops: Iterable[BinaryOperator | str] = _OPERATOR_ORDER,
    *,
    constraints: EnumerationConstraints | None = None,
) -> EnumerationResult:
    """Exhaustively enumerate verifier-valid expressions for one target.

    Operand positions, rather than numeric values, define DP subsets.  This is
    necessary for correct multiset handling when repeated operands are enabled.
    Canonical output strings collapse positional derivations that are
    indistinguishable in the task language while retaining their count.
    """

    normalized_operands = tuple(operands)
    if not normalized_operands:
        raise ValueError("at least one operand is required")
    if any(
        isinstance(operand, bool) or not isinstance(operand, int)
        for operand in normalized_operands
    ):
        raise TypeError("TCES operands must be integers")
    if any(operand < 0 for operand in normalized_operands):
        raise ValueError("TCES operands must be unsigned decimal integers")
    if any(
        len(str(operand)) > TCES_MAX_INTEGER_DIGITS for operand in normalized_operands
    ):
        raise ValueError(
            f"TCES operands may have at most {TCES_MAX_INTEGER_DIGITS} digits"
        )

    normalized_target = _as_fraction(target)
    normalized_ops = _normalize_operators(allowed_ops)
    active_constraints = constraints or EnumerationConstraints()

    full_table, pruned = _enumerate_full_table(
        normalized_operands, normalized_ops, active_constraints
    )
    target_records = tuple(
        sorted(
            full_table.get(normalized_target, {}).values(),
            key=_record_order,
        )
    )

    enumerated: list[EnumeratedExpression] = []
    family_members: dict[str, list[_DPRecord]] = {}
    for record in target_records:
        identifier = strategy_family_id(record.expression, normalized_operands)
        enumerated.append(
            EnumeratedExpression(
                expression=record.expression,
                value=record.value,
                canonical_expression=record.canonical_expression,
                family_id=identifier,
                depth=record.depth,
                derivation_count=record.derivation_count,
            )
        )
        family_members.setdefault(identifier, []).append(record)

    families: list[FamilyEnumeration] = []
    for identifier in sorted(family_members):
        members = sorted(family_members[identifier], key=_record_order)
        representative = members[0]
        families.append(
            FamilyEnumeration(
                family_id=identifier,
                expression_count=len(members),
                derivation_count=sum(member.derivation_count for member in members),
                representative=representative.expression,
                representative_expression=representative.canonical_expression,
                shortest_depth=representative.depth,
            )
        )

    shortest_depth = min((record.depth for record in target_records), default=None)
    return EnumerationResult(
        operands=normalized_operands,
        target=normalized_target,
        allowed_ops=normalized_ops,
        constraints=active_constraints,
        expressions=tuple(enumerated),
        families=tuple(families),
        shortest_depth=shortest_depth,
        exact_expression_tree_count=len(enumerated),
        derivation_count=sum(record.derivation_count for record in target_records),
        pruned=pruned,
    )


def enumerate_task(
    task: object,
    *,
    allow_fractional_intermediates: bool = True,
    exclude_trivial_identity_steps: bool = False,
    require_positive_intermediates: bool = False,
    max_expressions_per_value: int | None = None,
) -> EnumerationResult:
    """Enumerate a :class:`~duraseed.schemas.TCESTask` without importing schemas."""

    task_constraints = getattr(task, "constraints")
    constraints = EnumerationConstraints(
        allow_fractional_intermediates=allow_fractional_intermediates,
        max_abs_intermediate=task_constraints.max_abs_intermediate,
        max_denominator=task_constraints.max_denominator,
        max_tree_depth=task_constraints.max_tree_depth,
        max_ast_nodes=task_constraints.max_ast_nodes,
        exclude_trivial_identity_steps=exclude_trivial_identity_steps,
        require_positive_intermediates=require_positive_intermediates,
        max_expressions_per_value=max_expressions_per_value,
    )
    return enumerate_solutions(
        getattr(task, "operands"),
        getattr(task, "target"),
        getattr(task, "allowed_ops"),
        constraints=constraints,
    )


# Explicit aliases used in CLI and older experiment notes.
enumerate_expressions = enumerate_solutions
subset_dp_enumerate = enumerate_solutions


__all__ = [
    "EnumeratedExpression",
    "EnumerationConstraints",
    "EnumerationResult",
    "FamilyEnumeration",
    "enumerate_expressions",
    "enumerate_solutions",
    "enumerate_task",
    "subset_dp_enumerate",
]
