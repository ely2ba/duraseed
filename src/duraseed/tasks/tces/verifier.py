"""Fail-closed exact verifier for Template-Controlled Expression Synthesis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Callable, Sequence

from duraseed.schemas import (
    TCES_MAX_INTEGER_DIGITS,
    TCESTask,
    VerificationFailure,
    VerificationResult,
)
from duraseed.tasks.tces.ast import (
    BinaryOperator,
    Expression,
    IntegerLiteral,
    ast_to_dict,
    iter_postorder,
    leaf_values,
    operators,
)
from duraseed.tasks.tces.canonicalize import render_canonical_expression
from duraseed.tasks.tces.lexer import (
    LexErrorCode,
    LexerConfig,
    LexerError,
    tokenize,
)
from duraseed.tasks.tces.parser import ParseError, ParseErrorCode, Parser, ParserLimits
from duraseed.tasks.tces.strategies import strategy_family_id


OPEN_ANSWER_TAG = "<answer>"
CLOSE_ANSWER_TAG = "</answer>"

Canonicalizer = Callable[[Expression], str]
StrategyFamilyMapper = Callable[[Expression, Sequence[int]], str]


@dataclass(frozen=True, slots=True)
class AnswerSpan:
    """The exact model substring designated as the authoritative answer."""

    text: str
    start: int
    end: int


class AnswerTagError(ValueError):
    """Failure to extract exactly one well-ordered answer-tag pair."""

    def __init__(self, failure_code: VerificationFailure, message: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


class EvaluationErrorCode(StrEnum):
    DIVISION_BY_ZERO = "division_by_zero"
    MAGNITUDE_LIMIT_EXCEEDED = "magnitude_limit_exceeded"
    DENOMINATOR_LIMIT_EXCEEDED = "denominator_limit_exceeded"


class EvaluationError(ArithmeticError):
    """A classified exact-arithmetic failure."""

    def __init__(self, code: EvaluationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def extract_answer_span(completion: str) -> AnswerSpan:
    """Extract exactly one literal ``<answer>...</answer>`` pair.

    Free-form derivation outside the pair is permitted.  Tags are deliberately
    case-sensitive and attribute-free, and extra opening or closing tags are
    rejected even if one pair would otherwise be usable.
    """

    opening_count = completion.count(OPEN_ANSWER_TAG)
    closing_count = completion.count(CLOSE_ANSWER_TAG)
    if opening_count > 1 or closing_count > 1:
        raise AnswerTagError(
            VerificationFailure.MULTIPLE_ANSWER_TAGS,
            "completion contains multiple answer tags",
        )
    if opening_count != 1 or closing_count != 1:
        raise AnswerTagError(
            VerificationFailure.MISSING_ANSWER_TAG,
            "completion must contain exactly one opening and closing answer tag",
        )

    start = completion.find(OPEN_ANSWER_TAG) + len(OPEN_ANSWER_TAG)
    end = completion.find(CLOSE_ANSWER_TAG)
    if end < start:
        raise AnswerTagError(
            VerificationFailure.INVALID_SYNTAX,
            "closing answer tag precedes opening answer tag",
        )
    return AnswerSpan(text=completion[start:end], start=start, end=end)


def _evaluate_with_guards(
    expression: Expression,
    *,
    max_abs_intermediate: int,
    max_denominator: int,
) -> Fraction:
    """Evaluate exactly and enforce bounds after every binary operation."""

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
            if right == 0:
                raise EvaluationError(
                    EvaluationErrorCode.DIVISION_BY_ZERO,
                    "division by zero",
                )
            value = left / right

        if abs(value) > max_abs_intermediate:
            raise EvaluationError(
                EvaluationErrorCode.MAGNITUDE_LIMIT_EXCEEDED,
                f"intermediate magnitude exceeds {max_abs_intermediate}",
            )
        if value.denominator > max_denominator:
            raise EvaluationError(
                EvaluationErrorCode.DENOMINATOR_LIMIT_EXCEEDED,
                f"intermediate denominator exceeds {max_denominator}",
            )
        values[id(node)] = value

    return values[id(expression)]


def _lexer_failure(error: LexerError) -> VerificationFailure:
    if error.code in (LexErrorCode.NON_ASCII, LexErrorCode.INVALID_CHARACTER):
        return VerificationFailure.INVALID_CHARACTER
    return VerificationFailure.INVALID_TOKEN


def _parser_failure(error: ParseError) -> VerificationFailure:
    if error.code in (
        ParseErrorCode.AST_NODE_LIMIT_EXCEEDED,
        ParseErrorCode.AST_DEPTH_LIMIT_EXCEEDED,
        ParseErrorCode.PARENTHESIS_DEPTH_LIMIT_EXCEEDED,
    ):
        return VerificationFailure.AST_LIMIT_EXCEEDED
    return VerificationFailure.INVALID_SYNTAX


class TCESVerifier:
    """Exact binary-reward verifier with injectable analysis hooks.

    ``canonicalizer`` and ``strategy_family_mapper`` never affect arithmetic
    acceptance.  Defaults use the project's canonicalization and complete
    strategy-family signature, while injectable hooks keep focused tests and
    future versioned family definitions explicit.
    """

    def __init__(
        self,
        *,
        canonicalizer: Canonicalizer = render_canonical_expression,
        strategy_family_mapper: StrategyFamilyMapper = strategy_family_id,
        max_parenthesis_depth: int = 64,
    ) -> None:
        if max_parenthesis_depth < 1:
            raise ValueError("max_parenthesis_depth must be positive")
        self._canonicalizer = canonicalizer
        self._strategy_family_mapper = strategy_family_mapper
        self._max_parenthesis_depth = max_parenthesis_depth

    def verify(self, completion: str, task: TCESTask) -> VerificationResult:
        """Verify one model completion against one immutable TCES task."""

        if not task.constraints.use_each_once:
            raise ValueError("TCESVerifier requires use_each_once=true")
        if not isinstance(completion, str):
            return VerificationResult(failure_code=VerificationFailure.INVALID_TOKEN)

        try:
            span = extract_answer_span(completion)
        except AnswerTagError as error:
            return VerificationResult(failure_code=error.failure_code)

        if len(span.text) > task.constraints.max_answer_length:
            return VerificationResult(
                valid_answer_tag=True,
                failure_code=VerificationFailure.ANSWER_TOO_LONG,
            )
        if not span.text.strip():
            return VerificationResult(
                valid_answer_tag=True,
                failure_code=VerificationFailure.EMPTY_ANSWER,
            )

        lexer_config = LexerConfig(
            max_integer_digits=min(
                task.constraints.max_answer_length,
                TCES_MAX_INTEGER_DIGITS,
            ),
            max_tokens=task.constraints.max_answer_length + 1,
        )
        try:
            tokens = tokenize(span.text, lexer_config)
        except LexerError as error:
            return VerificationResult(
                valid_answer_tag=True,
                failure_code=_lexer_failure(error),
            )

        parser_limits = ParserLimits(
            max_ast_nodes=task.constraints.max_ast_nodes,
            max_ast_depth=task.constraints.max_tree_depth,
            max_parenthesis_depth=self._max_parenthesis_depth,
        )
        try:
            expression = Parser(tokens, parser_limits).parse()
        except ParseError as error:
            return VerificationResult(
                valid_answer_tag=True,
                valid_lexing=True,
                failure_code=_parser_failure(error),
            )

        canonical = self._canonicalizer(expression)
        encoded_ast = ast_to_dict(expression)
        observed_operands = Counter(leaf_values(expression))
        expected_operands = Counter(task.operands)
        operand_multiset_valid = observed_operands == expected_operands
        allowed = frozenset(task.allowed_ops)
        operations_valid = all(
            operator.value in allowed for operator in operators(expression)
        )

        common = {
            "valid_answer_tag": True,
            "valid_lexing": True,
            "valid_syntax": True,
            "valid_operand_multiset": operand_multiset_valid,
            "valid_operations": operations_valid,
            "canonical_expression": canonical,
            "ast_json": encoded_ast,
        }
        if not operand_multiset_valid:
            return VerificationResult(
                **common,
                failure_code=VerificationFailure.OPERAND_MULTISET_MISMATCH,
            )
        if not operations_valid:
            return VerificationResult(
                **common,
                failure_code=VerificationFailure.DISALLOWED_OPERATOR,
            )

        try:
            value = _evaluate_with_guards(
                expression,
                max_abs_intermediate=task.constraints.max_abs_intermediate,
                max_denominator=task.constraints.max_denominator,
            )
        except EvaluationError as error:
            failure = (
                VerificationFailure.DIVISION_BY_ZERO
                if error.code is EvaluationErrorCode.DIVISION_BY_ZERO
                else VerificationFailure.INTERMEDIATE_LIMIT_EXCEEDED
            )
            return VerificationResult(**common, failure_code=failure)

        value_fields = {
            "value_num": value.numerator,
            "value_den": value.denominator,
        }
        if value != task.target.as_fraction():
            return VerificationResult(
                **common,
                **value_fields,
                valid_intermediates=True,
                failure_code=VerificationFailure.WRONG_TARGET,
            )

        family_id = self._strategy_family_mapper(expression, task.operands)

        return VerificationResult(
            **common,
            **value_fields,
            valid_intermediates=True,
            correct_target=True,
            reward=1.0,
            strategy_family_id=family_id,
        )


def verify_completion(
    completion: str,
    task: TCESTask,
    *,
    canonicalizer: Canonicalizer = render_canonical_expression,
    strategy_family_mapper: StrategyFamilyMapper = strategy_family_id,
) -> VerificationResult:
    """Convenience wrapper for one-off verification."""

    return TCESVerifier(
        canonicalizer=canonicalizer,
        strategy_family_mapper=strategy_family_mapper,
    ).verify(completion, task)
