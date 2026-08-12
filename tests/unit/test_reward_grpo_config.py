from __future__ import annotations

from typing import Any

import pytest

from duraseed.schemas import (
    ExactRational,
    MAPSInstruction,
    MAPSInstructionKind,
    MAPSTask,
    TCESTask,
)
from duraseed.training.grpo import grouped_reward_diagnostics
from duraseed.training.reward import (
    verify_exact_completions,
    verify_task_completion,
)


def _tces_task() -> TCESTask:
    return TCESTask(
        operands=(2, 3),
        target=ExactRational(numerator=5),
    )


def _maps_task() -> MAPSTask:
    return MAPSTask(
        start=7,
        modulus=31,
        target=5,
        allowed_instructions=(
            MAPSInstruction(op=MAPSInstructionKind.ADD, argument=5),
            MAPSInstruction(op=MAPSInstructionKind.MUL, argument=3),
            MAPSInstruction(op=MAPSInstructionKind.NEG),
        ),
        max_program_length=3,
    )


@pytest.mark.parametrize(
    ("task", "completion", "expected_reward", "expected_failure"),
    (
        (_tces_task(), "<answer>3+2</answer>", 1.0, None),
        (_tces_task(), "<answer>3-2</answer>", 0.0, "wrong_target"),
        (_tces_task(), "<answer>eval('2+3')</answer>", 0.0, "invalid_character"),
        (
            _maps_task(),
            "<answer>MUL 3; ADD 5; NEG</answer>",
            1.0,
            None,
        ),
        (_maps_task(), "<answer>ADD 6</answer>", 0.0, "illegal_instruction"),
    ),
)
def test_exact_reward_dispatch_preserves_task_verifier_result(
    task: TCESTask | MAPSTask,
    completion: str,
    expected_reward: float,
    expected_failure: str | None,
) -> None:
    result = verify_task_completion(completion, task)
    assert result.reward == expected_reward
    assert (
        result.failure_code.value if result.failure_code is not None else None
    ) == expected_failure


def test_batch_verification_requires_exact_alignment() -> None:
    results = verify_exact_completions(
        ["<answer>2+3</answer>", "<answer>MUL 3; ADD 5; NEG</answer>"],
        [_tces_task(), _maps_task()],
    )
    assert tuple(result.reward for result in results) == (1.0, 1.0)

    with pytest.raises(ValueError, match="align one-to-one"):
        verify_exact_completions(["<answer>2+3</answer>"], [])
    with pytest.raises(TypeError, match="explicit sequence"):
        verify_exact_completions("<answer>2+3</answer>", [_tces_task()])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="completion must be text"):
        verify_exact_completions([object()], [_tces_task()])  # type: ignore[list-item]


def test_grouped_reward_diagnostic_centers_each_group_and_counts_modes() -> None:
    diagnostics = grouped_reward_diagnostics(
        (
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ),
        group_size=4,
    )

    assert diagnostics.group_means == (0.0, 1.0, 0.5)
    assert diagnostics.centered_advantages == (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (-0.5, 0.5, -0.5, 0.5),
    )
    assert diagnostics.all_zero_group_count == 1
    assert diagnostics.all_one_group_count == 1
    assert diagnostics.mixed_group_count == 1


@pytest.mark.parametrize(
    ("rewards", "group_size", "error", "message"),
    (
        ((), 4, ValueError, "positive multiple"),
        ((0.0, 1.0, 0.0), 2, ValueError, "positive multiple"),
        ((0.0, 0.5), 2, ValueError, "exact binary"),
        ((0, 1), 2, ValueError, "exact binary"),
        ((0.0, 1.0), 1, ValueError, "at least two"),
        ((0.0, 1.0), True, TypeError, "integer"),
    ),
)
def test_grouped_reward_diagnostic_rejects_invalid_inputs(
    rewards: Any,
    group_size: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        grouped_reward_diagnostics(rewards, group_size=group_size)
