from fractions import Fraction

import pytest
from pydantic import ValidationError

from duraseed.schemas import (
    ExactRational,
    MAPSInstruction,
    MAPSInstructionKind,
    MAPSTask,
    TCESConstraints,
    TCESTask,
    VerificationResult,
)


def test_exact_rational_is_normalized() -> None:
    value = ExactRational(numerator=6, denominator=-8)
    assert (value.numerator, value.denominator) == (-3, 4)
    assert value.as_fraction() == Fraction(-3, 4)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExactRational(numerator=True),
        lambda: TCESTask(operands=(2, False), target=ExactRational(numerator=2)),
        lambda: TCESTask(operands=(-2, 3), target=ExactRational(numerator=1)),
        lambda: TCESTask(
            operands=(10**1024,),
            target=ExactRational(numerator=0),
            constraints=TCESConstraints(max_answer_length=2_000),
        ),
        lambda: TCESTask(
            operands=(2, 3),
            target=ExactRational(numerator=5),
            constraints=TCESConstraints(max_answer_length=4),
        ),
        lambda: TCESConstraints(use_each_once=False),
        lambda: MAPSInstruction(op=MAPSInstructionKind.ADD, argument=True),
        lambda: MAPSInstruction(
            op=MAPSInstructionKind.ADD,
            argument=10**32,
        ),
        lambda: MAPSTask(
            start=True,
            modulus=7,
            target=1,
            allowed_instructions=(
                MAPSInstruction(op=MAPSInstructionKind.ADD, argument=1),
            ),
        ),
        lambda: MAPSTask(
            start=0,
            modulus=1009,
            target=1,
            allowed_instructions=(
                MAPSInstruction(op=MAPSInstructionKind.ADD, argument=1),
            ),
            max_program_length=101,
        ),
    ],
)
def test_boolean_values_are_not_silently_coerced_to_integers(factory: object) -> None:
    with pytest.raises(ValidationError):
        factory()  # type: ignore[operator]


def test_unknown_fields_fail_closed() -> None:
    with pytest.raises(ValidationError):
        TCESTask(
            operands=(2, 3),
            target=ExactRational(numerator=5),
            typo_that_must_not_be_ignored=True,
        )


def test_scientific_schemas_are_immutable_after_validation() -> None:
    task = TCESTask(
        operands=(2, 3),
        target=ExactRational(numerator=5),
    )

    with pytest.raises(ValidationError):
        task.operands = (5,)  # type: ignore[misc]


def test_success_reward_requires_all_checks() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(reward=1.0)

    with pytest.raises(ValidationError):
        VerificationResult(
            valid_answer_tag=True,
            valid_lexing=True,
            valid_syntax=True,
            valid_operand_multiset=True,
            valid_operations=True,
            valid_intermediates=True,
            correct_target=True,
            reward=0.0,
        )

    with pytest.raises(ValidationError):
        VerificationResult(failure_code=None)


def _successful_result(**updates: object) -> VerificationResult:
    values: dict[str, object] = {
        "valid_answer_tag": True,
        "valid_lexing": True,
        "valid_syntax": True,
        "valid_operand_multiset": True,
        "valid_operations": True,
        "valid_intermediates": True,
        "correct_target": True,
        "reward": 1.0,
        "value_num": 1,
        "value_den": 1,
        "canonical_expression": "1",
        "strategy_family_id": "r1|intermediates=",
        "ast_json": {"type": "integer", "value": 1},
    }
    values.update(updates)
    return VerificationResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "updates",
    [
        {"valid_answer_tag": "true"},
        {"value_num": True},
        {"value_num": "1"},
        {"value_num": 1, "value_den": None},
        {"reward": 1},
    ],
)
def test_verification_results_reject_wrong_typed_artifacts(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _successful_result(**updates)


def test_verification_exact_value_metadata_is_normalized() -> None:
    result = _successful_result(value_num=2, value_den=-4)

    assert (result.value_num, result.value_den) == (-1, 2)


def test_maps_instruction_arity() -> None:
    with pytest.raises(ValidationError):
        MAPSInstruction(op=MAPSInstructionKind.ADD)
    with pytest.raises(ValidationError):
        MAPSInstruction(op=MAPSInstructionKind.NEG, argument=2)


def test_maps_task_normalizes_states_and_rejects_duplicates() -> None:
    add = MAPSInstruction(op=MAPSInstructionKind.ADD, argument=2)
    task = MAPSTask(
        start=38,
        modulus=31,
        target=-12,
        allowed_instructions=(add,),
    )
    assert task.start == 7
    assert task.target == 19

    with pytest.raises(ValidationError):
        MAPSTask(
            start=0,
            modulus=7,
            target=1,
            allowed_instructions=(add, add),
        )

    with pytest.raises(ValidationError):
        MAPSTask(
            start=7,
            modulus=31,
            target=38,
            allowed_instructions=(add,),
        )


def test_task_schemas_round_trip_through_canonical_json() -> None:
    tces = TCESTask(
        operands=(2, 3),
        target=ExactRational(numerator=5),
    )
    maps = MAPSTask(
        start=7,
        modulus=31,
        target=19,
        allowed_instructions=(
            MAPSInstruction(op=MAPSInstructionKind.ADD, argument=5),
            MAPSInstruction(op=MAPSInstructionKind.NEG),
        ),
    )

    assert TCESTask.model_validate_json(tces.model_dump_json()) == tces
    assert MAPSTask.model_validate_json(maps.model_dump_json()) == maps
