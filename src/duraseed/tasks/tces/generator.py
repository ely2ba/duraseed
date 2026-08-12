"""Deterministic, exhaustive-solver-backed TCES task generation.

Each candidate attempt is an independent pseudorandom stream derived from the
root seed, item index, and attempt index.  Rejection in one item therefore
cannot perturb any other item.  A latent expression is used only to construct a
solvable prompt: the complete valid expression and family sets are obtained
from the subset-DP enumerator before any family metadata is attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import random
from typing import Iterable

from duraseed.schemas import (
    ExactRational,
    TCESConstraints,
    TCES_MAX_INTEGER_DIGITS,
    TCESTask,
)

from .ast import (
    BinaryExpression,
    BinaryOperator,
    Expression,
    IntegerLiteral,
    iter_postorder,
    leaf_values,
    node_count,
    tree_depth,
)
from .enumerate import EnumerationResult, enumerate_task
from .strategies import strategy_family_id
from .teacher import generate_teacher_trace, verify_teacher_trace
from .verifier import verify_completion


GENERATOR_VERSION = "1.0.0"
PROMPT_TEMPLATE_ID = "tces_v1"
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
        except (KeyError, TypeError) as error:
            raise ValueError(f"unsupported TCES operator: {operator!r}") from error
    if not normalized:
        raise ValueError("at least one TCES operator is required")
    return tuple(operator for operator in _OPERATOR_ORDER if operator in normalized)


@dataclass(frozen=True, slots=True)
class TCESGeneratorConfig:
    """Frozen generation and rejection-filter controls.

    The defaults are the primary five-operand specification.  Family filtering
    defaults to any non-empty complete family set so ordinary generation does
    not spend attempts searching for a calibration bucket.  Single- and
    multi-family datasets can set the minimum and maximum explicitly.
    """

    n_operands: int = 5
    operand_min: int = 2
    operand_max: int = 25
    require_distinct_operands: bool = True
    target_min: int = -250
    target_max: int = 250
    require_integer_target: bool = True
    allowed_ops: tuple[BinaryOperator | str, ...] = _OPERATOR_ORDER
    allow_fractional_intermediates: bool = True
    max_abs_intermediate: int = 10_000
    max_denominator: int = 1_000
    max_tree_depth: int = 5
    max_ast_nodes: int = 31
    max_answer_length: int = 1_024
    exclude_target_in_operands: bool = True
    exclude_trivial_identity_steps: bool = True
    require_positive_intermediates: bool = False
    min_valid_families: int = 1
    max_valid_families: int | None = None
    min_valid_expressions: int = 1
    max_valid_expressions: int | None = None
    split: str | None = None
    max_attempts: int = 2_048

    def __post_init__(self) -> None:
        integer_fields = (
            "n_operands",
            "operand_min",
            "operand_max",
            "target_min",
            "target_max",
            "max_abs_intermediate",
            "max_denominator",
            "max_tree_depth",
            "max_ast_nodes",
            "max_answer_length",
            "min_valid_families",
            "min_valid_expressions",
            "max_attempts",
        )
        if any(
            isinstance(getattr(self, field), bool)
            or not isinstance(getattr(self, field), int)
            for field in integer_fields
        ):
            raise ValueError("generator numeric controls must be integers")
        for field in ("max_valid_families", "max_valid_expressions"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError("optional family/expression caps must be integers")
        boolean_fields = (
            "require_distinct_operands",
            "require_integer_target",
            "allow_fractional_intermediates",
            "exclude_target_in_operands",
            "exclude_trivial_identity_steps",
            "require_positive_intermediates",
        )
        if any(not isinstance(getattr(self, field), bool) for field in boolean_fields):
            raise ValueError("generator feature switches must be booleans")
        if not 2 <= self.n_operands <= 16:
            raise ValueError("n_operands must be an integer between 2 and 16")
        if self.operand_min < 0:
            raise ValueError("operand_min must be non-negative")
        if len(str(self.operand_max)) > TCES_MAX_INTEGER_DIGITS:
            raise ValueError(
                f"operands may have at most {TCES_MAX_INTEGER_DIGITS} digits"
            )
        if self.operand_min > self.operand_max:
            raise ValueError("operand_min cannot exceed operand_max")
        operand_population = self.operand_max - self.operand_min + 1
        if self.require_distinct_operands and operand_population < self.n_operands:
            raise ValueError("operand range is too small for distinct operands")
        if self.target_min > self.target_max:
            raise ValueError("target_min cannot exceed target_max")
        if self.max_abs_intermediate < 1 or self.max_denominator < 1:
            raise ValueError("exact-rational guards must be positive")
        if not 1 <= self.max_tree_depth <= 64:
            raise ValueError("max_tree_depth must be between 1 and 64")
        if self.n_operands > 2 ** (self.max_tree_depth - 1):
            raise ValueError("max_tree_depth cannot contain the requested leaf count")
        minimum_nodes = 2 * self.n_operands - 1
        if self.max_ast_nodes < minimum_nodes:
            raise ValueError(
                f"max_ast_nodes must be at least {minimum_nodes} for a full tree"
            )
        if self.max_answer_length < 1:
            raise ValueError("max_answer_length must be positive")
        canonical_answer_bound = self.n_operands * len(str(self.operand_max)) + 3 * (
            self.n_operands - 1
        )
        if self.max_answer_length < canonical_answer_bound:
            raise ValueError(
                "max_answer_length cannot hold the largest canonical task answer"
            )
        if self.min_valid_families < 1:
            raise ValueError("min_valid_families must be positive")
        if (
            self.max_valid_families is not None
            and self.max_valid_families < self.min_valid_families
        ):
            raise ValueError("invalid valid-family count interval")
        if self.min_valid_expressions < 1:
            raise ValueError("min_valid_expressions must be positive")
        if (
            self.max_valid_expressions is not None
            and self.max_valid_expressions < self.min_valid_expressions
        ):
            raise ValueError("invalid valid-expression count interval")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if self.split is not None and (
            not isinstance(self.split, str) or not self.split.strip()
        ):
            raise ValueError("split must be non-empty when supplied")

        object.__setattr__(self, "allowed_ops", _normalize_operators(self.allowed_ops))


@dataclass(frozen=True, slots=True)
class GeneratedTCESInstance:
    """One task plus auditable latent, exhaustive, and teacher provenance."""

    task: TCESTask
    content_hash: str
    latent_expression: Expression
    intended_family: str
    enumeration: EnumerationResult
    teacher_trace: str
    root_seed: int
    item_index: int
    accepted_attempt: int

    @property
    def intended_family_id(self) -> str:
        """Explicit alias matching strategy-module terminology."""

        return self.intended_family

    @property
    def valid_family_count(self) -> int:
        return len(self.enumeration.families)

    @property
    def valid_expression_count(self) -> int:
        return len(self.enumeration.expressions)

    @property
    def valid_family_ids(self) -> tuple[str, ...]:
        return self.enumeration.family_ids


class TCESGenerationError(RuntimeError):
    """Raised when deterministic rejection sampling cannot produce an item."""


class _RejectCandidate(Exception):
    """Internal control flow for a scientifically invalid latent candidate."""


def _attempt_rng(root_seed: int, item_index: int, attempt: int) -> random.Random:
    material = f"duraseed:tces:v1:{root_seed}:{item_index}:{attempt}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(seed)


def _sample_operands(
    rng: random.Random, config: TCESGeneratorConfig
) -> tuple[int, ...]:
    population = range(config.operand_min, config.operand_max + 1)
    if config.require_distinct_operands:
        sampled = rng.sample(population, config.n_operands)
    else:
        sampled = [
            rng.randint(config.operand_min, config.operand_max)
            for _ in range(config.n_operands)
        ]
    # Prompt order is canonical.  The independently shuffled leaf order below
    # still samples rank permutations for the intended strategy family.
    return tuple(sorted(sampled))


def _sample_full_binary_tree(
    rng: random.Random,
    operands: tuple[int, ...],
    allowed_ops: tuple[BinaryOperator, ...],
) -> Expression:
    leaf_order = list(operands)
    rng.shuffle(leaf_order)

    def build(values: tuple[int, ...]) -> Expression:
        if len(values) == 1:
            return IntegerLiteral(values[0])
        split = rng.randint(1, len(values) - 1)
        left = build(values[:split])
        right = build(values[split:])
        return BinaryExpression(
            operator=rng.choice(allowed_ops),
            left=left,
            right=right,
        )

    return build(tuple(leaf_order))


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
            raise _RejectCandidate
        return left / right
    raise ValueError(f"unsupported TCES operator: {operator!r}")


def _validate_and_evaluate_latent(
    expression: Expression, config: TCESGeneratorConfig
) -> Fraction:
    if node_count(expression) > config.max_ast_nodes:
        raise _RejectCandidate
    if tree_depth(expression) > config.max_tree_depth:
        raise _RejectCandidate

    values: dict[int, Fraction] = {}
    for node in iter_postorder(expression):
        if isinstance(node, IntegerLiteral):
            values[id(node)] = Fraction(node.value)
            continue

        left = values[id(node.left)]
        right = values[id(node.right)]
        if config.exclude_trivial_identity_steps and _is_trivial_identity(
            node.operator, left, right
        ):
            raise _RejectCandidate
        value = _apply(node.operator, left, right)
        if abs(value) > config.max_abs_intermediate:
            raise _RejectCandidate
        if value.denominator > config.max_denominator:
            raise _RejectCandidate
        if not config.allow_fractional_intermediates and value.denominator != 1:
            raise _RejectCandidate
        if config.require_positive_intermediates and value <= 0:
            raise _RejectCandidate
        values[id(node)] = value
    return values[id(expression)]


def _task_semantic_payload(task: TCESTask) -> dict[str, object]:
    constraints = task.constraints
    allowed = frozenset(task.allowed_ops)
    return {
        "allowed_ops": [
            operator.value for operator in _OPERATOR_ORDER if operator.value in allowed
        ],
        "constraints": {
            "max_abs_intermediate": constraints.max_abs_intermediate,
            "max_answer_length": constraints.max_answer_length,
            "max_ast_nodes": constraints.max_ast_nodes,
            "max_denominator": constraints.max_denominator,
            "max_tree_depth": constraints.max_tree_depth,
            "use_each_once": constraints.use_each_once,
        },
        "operands": sorted(task.operands),
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "target": {
            "denominator": task.target.denominator,
            "numerator": task.target.numerator,
        },
        "task_family": "tces",
        "version": 1,
    }


def _content_digest(task: TCESTask) -> str:
    encoded = json.dumps(
        _task_semantic_payload(task),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def task_content_hash(task: TCESTask) -> str:
    """Return the split-independent semantic content hash."""

    return "sha256:" + _content_digest(task)


def _task_id(task: TCESTask) -> str:
    return "tces-" + _content_digest(task)


def _count_is_allowed(value: int, minimum: int, maximum: int | None) -> bool:
    return value >= minimum and (maximum is None or value <= maximum)


def _task_constraints(config: TCESGeneratorConfig) -> TCESConstraints:
    return TCESConstraints(
        use_each_once=True,
        max_abs_intermediate=config.max_abs_intermediate,
        max_denominator=config.max_denominator,
        max_tree_depth=config.max_tree_depth,
        max_ast_nodes=config.max_ast_nodes,
        max_answer_length=config.max_answer_length,
    )


def _materialize_instance(
    *,
    root_seed: int,
    item_index: int,
    attempt: int,
    operands: tuple[int, ...],
    latent: Expression,
    config: TCESGeneratorConfig,
    required_family: str | None = None,
) -> GeneratedTCESInstance:
    """Validate one latent candidate and attach exhaustive provenance."""

    target = _validate_and_evaluate_latent(latent, config)
    if config.require_integer_target and target.denominator != 1:
        raise _RejectCandidate
    if not Fraction(config.target_min) <= target <= Fraction(config.target_max):
        raise _RejectCandidate
    if config.exclude_target_in_operands and target in map(Fraction, operands):
        raise _RejectCandidate

    provisional_task = TCESTask(
        operands=operands,
        target=ExactRational.from_fraction(target),
        allowed_ops=tuple(operator.value for operator in config.allowed_ops),
        constraints=_task_constraints(config),
        split=config.split,
    )

    # Family-conditioned generation can reject a changed intermediate profile
    # before paying for exhaustive enumeration. The family is recomputed below
    # before it is attached to an accepted record.
    if (
        required_family is not None
        and strategy_family_id(latent, operands) != required_family
    ):
        raise _RejectCandidate

    trace = generate_teacher_trace(latent)
    if not verify_teacher_trace(latent, trace):
        raise TCESGenerationError("generated teacher trace failed replay")

    enumeration = enumerate_task(
        provisional_task,
        # These three switches constrain latent construction only. They are
        # absent from TCESTask and the public prompt/verifier, so using them
        # here would make the advertised valid set incomplete.
        allow_fractional_intermediates=True,
        # Identity steps are rejected only in the sampled latent tree. The
        # public prompt/verifier do not prohibit them, so exhaustive family
        # labeling must retain every verifier-valid solution.
        exclude_trivial_identity_steps=False,
        require_positive_intermediates=False,
        max_expressions_per_value=None,
    )
    if not enumeration.complete:
        raise TCESGenerationError("primary TCES enumeration was pruned")

    if not _count_is_allowed(
        len(enumeration.families),
        config.min_valid_families,
        config.max_valid_families,
    ):
        raise _RejectCandidate
    if not _count_is_allowed(
        len(enumeration.expressions),
        config.min_valid_expressions,
        config.max_valid_expressions,
    ):
        raise _RejectCandidate

    # Family labeling occurs only after exhaustive enumeration.
    intended_family = strategy_family_id(latent, operands)
    if required_family is not None and intended_family != required_family:
        raise _RejectCandidate
    if intended_family not in enumeration.complete_family_set:
        raise TCESGenerationError(
            "latent family is absent from the exhaustive family set"
        )

    verification = verify_completion(trace, provisional_task)
    if verification.reward != 1.0:
        raise TCESGenerationError(
            "latent teacher failed the exact task verifier: "
            f"{verification.failure_code}"
        )
    if verification.strategy_family_id != intended_family:
        raise TCESGenerationError(
            "verifier and generator disagree on the intended family"
        )

    digest = _content_digest(provisional_task)
    task = TCESTask(
        operands=provisional_task.operands,
        target=provisional_task.target,
        allowed_ops=provisional_task.allowed_ops,
        constraints=provisional_task.constraints,
        task_id="tces-" + digest,
        split=provisional_task.split,
    )
    return GeneratedTCESInstance(
        task=task,
        content_hash="sha256:" + digest,
        latent_expression=latent,
        intended_family=intended_family,
        enumeration=enumeration,
        teacher_trace=trace,
        root_seed=root_seed,
        item_index=item_index,
        accepted_attempt=attempt,
    )


class TCESGenerator:
    """Generate independent indexed TCES tasks with exhaustive provenance."""

    def __init__(
        self,
        root_seed: int,
        config: TCESGeneratorConfig | None = None,
    ) -> None:
        if isinstance(root_seed, bool) or not isinstance(root_seed, int):
            raise TypeError("root_seed must be an integer")
        self.root_seed = root_seed
        self.config = config or TCESGeneratorConfig()

    def generate(self, item_index: int = 0) -> GeneratedTCESInstance:
        """Generate one accepted, completely enumerated instance."""

        if (
            isinstance(item_index, bool)
            or not isinstance(item_index, int)
            or item_index < 0
        ):
            raise ValueError("item_index must be a non-negative integer")
        config = self.config
        allowed_ops = tuple(config.allowed_ops)

        for attempt in range(config.max_attempts):
            rng = _attempt_rng(self.root_seed, item_index, attempt)
            operands = _sample_operands(rng, config)
            latent = _sample_full_binary_tree(rng, operands, allowed_ops)
            try:
                return _materialize_instance(
                    root_seed=self.root_seed,
                    item_index=item_index,
                    attempt=attempt,
                    operands=operands,
                    latent=latent,
                    config=config,
                )
            except _RejectCandidate:
                continue

        raise TCESGenerationError(
            "could not generate an accepted TCES instance within "
            f"{config.max_attempts} deterministic attempts for item {item_index}"
        )

    def generate_task(self, item_index: int = 0) -> TCESTask:
        """Generate one task while discarding exhaustive provenance."""

        return self.generate(item_index).task

    def generate_many(
        self, count: int, *, start_index: int = 0
    ) -> tuple[GeneratedTCESInstance, ...]:
        """Generate an indexed batch and fail closed on a content duplicate."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or start_index < 0
        ):
            raise ValueError("start_index must be a non-negative integer")
        instances = tuple(
            self.generate(index) for index in range(start_index, start_index + count)
        )
        hashes = [instance.content_hash for instance in instances]
        if len(hashes) != len(set(hashes)):
            raise TCESGenerationError("generated batch contains a duplicate task")
        return instances


def _substitute_operand_ranks(
    expression: Expression,
    template_ranks: dict[int, int],
    operands: tuple[int, ...],
) -> Expression:
    if isinstance(expression, IntegerLiteral):
        return IntegerLiteral(operands[template_ranks[expression.value]])
    if isinstance(expression, BinaryExpression):
        return BinaryExpression(
            expression.operator,
            _substitute_operand_ranks(expression.left, template_ranks, operands),
            _substitute_operand_ranks(expression.right, template_ranks, operands),
        )
    raise TypeError(f"unsupported TCES expression node: {type(expression)!r}")


class TCESFamilyGenerator:
    """Generate distinct numeric tasks within one exact canonical family.

    The template fixes the latent tree, operators, operand-rank placement, and
    integral/fractional intermediate profile. New operands are sampled
    deterministically and rejected unless the complete family identifier is
    unchanged. Every accepted task is then exhaustively enumerated and exact
    verified through the same path as ordinary TCES generation.
    """

    def __init__(
        self,
        root_seed: int,
        template: GeneratedTCESInstance,
        config: TCESGeneratorConfig,
    ) -> None:
        if isinstance(root_seed, bool) or not isinstance(root_seed, int):
            raise TypeError("root_seed must be an integer")
        if not isinstance(template, GeneratedTCESInstance):
            raise TypeError("template must be a GeneratedTCESInstance")
        if not config.require_distinct_operands:
            raise ValueError("family-conditioned generation requires distinct operands")

        template_operands = tuple(sorted(template.task.operands))
        if len(template_operands) != config.n_operands:
            raise ValueError("template and generator operand counts differ")
        if len(set(template_operands)) != len(template_operands):
            raise ValueError("family template operands must be distinct")
        if tuple(sorted(leaf_values(template.latent_expression))) != template_operands:
            raise ValueError("family template latent leaves do not match its task")
        if frozenset(template.task.allowed_ops) != frozenset(
            operator.value for operator in config.allowed_ops
        ):
            raise ValueError("template and generator allowed operations differ")
        if template.task.constraints != _task_constraints(config):
            raise ValueError("template and generator task constraints differ")
        if (
            strategy_family_id(template.latent_expression, template_operands)
            != template.intended_family
        ):
            raise ValueError("template intended family is inconsistent")

        self.root_seed = root_seed
        self.template = template
        self.config = config
        self._template_operands = template_operands
        self._template_ranks = {
            operand: rank for rank, operand in enumerate(template_operands)
        }

    def generate(self, item_index: int = 0) -> GeneratedTCESInstance:
        """Generate one independently indexed numeric instance of the family."""

        if (
            isinstance(item_index, bool)
            or not isinstance(item_index, int)
            or item_index < 0
        ):
            raise ValueError("item_index must be a non-negative integer")

        for attempt in range(self.config.max_attempts):
            rng = _attempt_rng(self.root_seed, item_index, attempt)
            operands = _sample_operands(rng, self.config)
            if operands == self._template_operands:
                continue
            latent = _substitute_operand_ranks(
                self.template.latent_expression,
                self._template_ranks,
                operands,
            )
            try:
                return _materialize_instance(
                    root_seed=self.root_seed,
                    item_index=item_index,
                    attempt=attempt,
                    operands=operands,
                    latent=latent,
                    config=self.config,
                    required_family=self.template.intended_family,
                )
            except _RejectCandidate:
                continue

        raise TCESGenerationError(
            "could not generate an accepted instance of the requested TCES family "
            f"within {self.config.max_attempts} deterministic attempts for item "
            f"{item_index}"
        )

    def generate_many(
        self, count: int, *, start_index: int = 0
    ) -> tuple[GeneratedTCESInstance, ...]:
        """Generate a batch and fail closed on any numeric task duplicate."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or start_index < 0
        ):
            raise ValueError("start_index must be a non-negative integer")
        instances = tuple(
            self.generate(index) for index in range(start_index, start_index + count)
        )
        hashes = [instance.content_hash for instance in instances]
        if len(hashes) != len(set(hashes)):
            raise TCESGenerationError(
                "generated family batch contains a duplicate task"
            )
        return instances


def generate_instance(
    root_seed: int,
    item_index: int = 0,
    config: TCESGeneratorConfig | None = None,
) -> GeneratedTCESInstance:
    """Functional entry point for deterministic instance generation."""

    return TCESGenerator(root_seed, config).generate(item_index)


def generate_task(
    root_seed: int,
    item_index: int = 0,
    config: TCESGeneratorConfig | None = None,
) -> TCESTask:
    """Functional entry point returning only the shared task schema."""

    return TCESGenerator(root_seed, config).generate_task(item_index)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def render_prompt(task: TCESTask) -> str:
    """Render the canonical Stage-A prompt contract."""

    allowed = ", ".join(task.allowed_ops)
    target = _format_fraction(task.target.as_fraction())
    return (
        "Solve this exact expression-synthesis problem.\n\n"
        f"Numbers: {list(task.operands)}\n"
        f"Target: {target}\n"
        f"Allowed binary operations: {allowed}\n\n"
        "Rules:\n"
        "1. Use every listed number exactly once.\n"
        "2. Do not use any other numeric constants.\n"
        "3. Parentheses are allowed.\n"
        "4. Division uses exact rational arithmetic; division by zero is forbidden.\n"
        "5. The final expression must equal the target exactly.\n"
        "6. Return a concise derivation and exactly one final tag:\n"
        "   <answer>EXPRESSION</answer>"
    )


def render_teacher_completion(instance: GeneratedTCESInstance) -> str:
    """Return the deterministic, exact-verifier-approved teacher trace."""

    return instance.teacher_trace


__all__ = [
    "GENERATOR_VERSION",
    "PROMPT_TEMPLATE_ID",
    "GeneratedTCESInstance",
    "TCESGenerationError",
    "TCESFamilyGenerator",
    "TCESGenerator",
    "TCESGeneratorConfig",
    "generate_instance",
    "generate_task",
    "render_prompt",
    "render_teacher_completion",
    "task_content_hash",
]
