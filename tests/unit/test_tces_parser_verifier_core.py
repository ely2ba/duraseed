"""Focused security and semantics tests for the Vertical Slice 1 TCES core."""

from fractions import Fraction
from typing import Any

import pytest

from duraseed.schemas import (
    ExactRational,
    TCESConstraints,
    TCESTask,
    VerificationFailure,
)
from duraseed.tasks.tces.ast import (
    BinaryExpression,
    BinaryOperator,
    evaluate_expression,
    leaf_values,
    node_count,
    tree_depth,
)
from duraseed.tasks.tces.lexer import LexErrorCode, LexerError, tokenize
from duraseed.tasks.tces.parser import ParseError, ParseErrorCode, parse_expression
from duraseed.tasks.tces.verifier import extract_answer_span, verify_completion


def _task(
    operands: tuple[int, ...],
    target: int | Fraction,
    *,
    allowed_ops: tuple[str, ...] = ("+", "-", "*", "/"),
    max_abs_intermediate: int = 10_000,
    max_denominator: int = 1_000,
    max_tree_depth: int = 16,
    max_ast_nodes: int = 127,
    max_answer_length: int = 1_024,
) -> TCESTask:
    exact_target = Fraction(target)
    return TCESTask(
        operands=operands,
        target=ExactRational(
            numerator=exact_target.numerator,
            denominator=exact_target.denominator,
        ),
        allowed_ops=allowed_ops,
        constraints=TCESConstraints(
            max_abs_intermediate=max_abs_intermediate,
            max_denominator=max_denominator,
            max_tree_depth=max_tree_depth,
            max_ast_nodes=max_ast_nodes,
            max_answer_length=max_answer_length,
        ),
    )


def test_parser_obeys_precedence_and_ast_metrics() -> None:
    expression = parse_expression("2 + 3 * 4")

    assert isinstance(expression, BinaryExpression)
    assert expression.operator is BinaryOperator.ADD
    assert evaluate_expression(expression) == Fraction(14)
    assert leaf_values(expression) == (2, 3, 4)
    assert node_count(expression) == 5
    assert tree_depth(expression) == 3


@pytest.mark.parametrize("source", ["-3", "+3", "2*-3", "2--3", "(-3)"])
def test_unary_operators_are_always_rejected(source: str) -> None:
    with pytest.raises(ParseError) as caught:
        parse_expression(source)

    assert caught.value.code is ParseErrorCode.UNARY_OPERATOR_NOT_ALLOWED


def test_negative_intermediate_is_available_through_binary_subtraction() -> None:
    result = verify_completion(
        "3 - 7 = -4.\n<answer>(3-7)</answer>",
        _task((3, 7), -4),
    )

    assert result.reward == 1.0
    assert result.value_num == -4
    assert result.value_den == 1


@pytest.mark.parametrize("source", ["1.0+2", "1e2", "x+2", "2%3"])
def test_lexer_rejects_every_character_outside_the_tiny_grammar(source: str) -> None:
    with pytest.raises(LexerError) as caught:
        tokenize(source)

    assert caught.value.code is LexErrorCode.INVALID_CHARACTER


@pytest.mark.parametrize("source", ["2−1", "2×3", "４+2", "2\u00a0+3"])
def test_lexer_rejects_unicode_lookalikes_and_whitespace(source: str) -> None:
    with pytest.raises(LexerError) as caught:
        tokenize(source)

    assert caught.value.code is LexErrorCode.NON_ASCII


def test_exactly_one_literal_answer_tag_pair_is_required() -> None:
    task = _task((2, 3), 5)

    missing = verify_completion("2+3", task)
    repeated = verify_completion("<answer>2+3</answer><answer>2+3</answer>", task)
    reversed_tags = verify_completion("</answer>2+3<answer>", task)

    assert missing.failure_code is VerificationFailure.MISSING_ANSWER_TAG
    assert repeated.failure_code is VerificationFailure.MULTIPLE_ANSWER_TAGS
    assert reversed_tags.failure_code is VerificationFailure.INVALID_SYNTAX
    assert not missing.valid_answer_tag
    assert not repeated.valid_answer_tag
    assert not reversed_tags.valid_answer_tag


@pytest.mark.parametrize("completion", [None, b"<answer>2+3</answer>", 5])
def test_non_text_completions_fail_closed(completion: Any) -> None:
    result = verify_completion(completion, _task((2, 3), 5))

    assert result.reward == 0.0
    assert result.failure_code is VerificationFailure.INVALID_TOKEN


def test_answer_span_preserves_expression_but_ignores_derivation() -> None:
    span = extract_answer_span("work may be free form\n<answer> (2 + 3) </answer>done")

    assert span.text == " (2 + 3) "


def test_fraction_evaluation_and_repeated_operand_multiset_are_exact() -> None:
    result = verify_completion(
        "<answer>((3/2)+(5/2))</answer>",
        _task((3, 2, 5, 2), 4),
    )

    assert result.reward == 1.0
    assert result.value_num == 4
    assert result.value_den == 1
    assert result.valid_operand_multiset
    assert result.ast_json is not None
    assert result.strategy_family_id is not None
    assert "intermediates=" in result.strategy_family_id


@pytest.mark.parametrize(
    ("answer", "operands"),
    [
        ("2+4", (2, 3)),
        ("2+2", (2, 3)),
        ("2", (2, 3)),
        ("2+3+3", (2, 3)),
    ],
)
def test_operand_multiset_rejects_added_omitted_and_duplicated_constants(
    answer: str, operands: tuple[int, ...]
) -> None:
    result = verify_completion(f"<answer>{answer}</answer>", _task(operands, 5))

    assert result.failure_code is VerificationFailure.OPERAND_MULTISET_MISMATCH
    assert not result.valid_operand_multiset
    assert result.reward == 0.0


def test_task_operator_allowlist_is_enforced_after_safe_parsing() -> None:
    result = verify_completion(
        "<answer>(8/2)</answer>",
        _task((8, 2), 4, allowed_ops=("+", "-", "*")),
    )

    assert result.failure_code is VerificationFailure.DISALLOWED_OPERATOR
    assert result.valid_operand_multiset
    assert not result.valid_operations


def test_division_by_zero_is_a_structured_failure() -> None:
    result = verify_completion(
        "<answer>(4/(3-3))</answer>",
        _task((4, 3, 3), 0),
    )

    assert result.failure_code is VerificationFailure.DIVISION_BY_ZERO
    assert not result.valid_intermediates
    assert result.reward == 0.0


@pytest.mark.parametrize(
    ("answer", "task"),
    [
        ("20*20", _task((20, 20), 400, max_abs_intermediate=100)),
        ("2/3", _task((2, 3), Fraction(2, 3), max_denominator=2)),
    ],
)
def test_intermediate_magnitude_and_denominator_guards(
    answer: str, task: TCESTask
) -> None:
    result = verify_completion(f"<answer>{answer}</answer>", task)

    assert result.failure_code is VerificationFailure.INTERMEDIATE_LIMIT_EXCEEDED
    assert not result.valid_intermediates


@pytest.mark.parametrize(
    ("answer", "task"),
    [
        (
            "((1+2)+(3+4))",
            _task((1, 2, 3, 4), 10, max_tree_depth=2),
        ),
        (
            "1+2+3",
            _task((1, 2, 3), 6, max_ast_nodes=4),
        ),
    ],
)
def test_ast_depth_and_node_guards(answer: str, task: TCESTask) -> None:
    result = verify_completion(f"<answer>{answer}</answer>", task)

    assert result.failure_code is VerificationFailure.AST_LIMIT_EXCEEDED
    assert result.valid_lexing
    assert not result.valid_syntax


def test_answer_length_guard_runs_before_lexing() -> None:
    result = verify_completion(
        "<answer> 1 + 2 </answer>",
        _task((1, 2), 3, max_answer_length=5),
    )

    assert result.failure_code is VerificationFailure.ANSWER_TOO_LONG
    assert result.valid_answer_tag
    assert not result.valid_lexing


def test_wrong_target_preserves_exact_value_and_diagnostics() -> None:
    result = verify_completion("<answer>(8/2)</answer>", _task((8, 2), 5))

    assert result.failure_code is VerificationFailure.WRONG_TARGET
    assert result.value_num == 4
    assert result.value_den == 1
    assert result.valid_intermediates
    assert not result.correct_target
    assert result.canonical_expression is not None


@pytest.mark.parametrize(
    "answer",
    ["", "   ", "2 3", "(2+3", "2+", "()", "2//3", "2**3"],
)
def test_empty_and_malformed_answers_never_receive_reward(answer: str) -> None:
    result = verify_completion(f"<answer>{answer}</answer>", _task((2, 3), 5))

    assert result.reward == 0.0
    assert result.failure_code in {
        VerificationFailure.EMPTY_ANSWER,
        VerificationFailure.INVALID_SYNTAX,
        VerificationFailure.INVALID_CHARACTER,
    }


def test_strategy_family_hook_is_analysis_only_and_receives_prompt_operands() -> None:
    seen: list[tuple[tuple[int, ...], Fraction]] = []

    def family(expression: object, operands: tuple[int, ...]) -> str:
        seen.append((operands, evaluate_expression(expression)))  # type: ignore[arg-type]
        return "family:test"

    from duraseed.tasks.tces.verifier import TCESVerifier

    result = TCESVerifier(strategy_family_mapper=family).verify(
        "<answer>2+3</answer>", _task((2, 3), 5)
    )

    assert result.reward == 1.0
    assert result.strategy_family_id == "family:test"
    assert seen == [((2, 3), Fraction(5))]
