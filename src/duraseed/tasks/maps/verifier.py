"""Structured, fail-closed verifier for MAPS model completions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from duraseed.schemas import MAPSTask, VerificationFailure, VerificationResult

from .interpreter import (
    DEFAULT_MAX_CONSTANT_DIGITS,
    Program,
    ProgramParseError,
    ProgramParseFailure,
    canonical_program,
    execute_program,
    parse_program,
    program_family_id,
)


DEFAULT_MAX_COMPLETION_CHARS = 4_096
_ANSWER_RE = re.compile(
    r"[ \t\r\n]*<answer>(?P<program>.*?)</answer>[ \t\r\n]*",
    flags=re.ASCII | re.DOTALL,
)

# Public task-specific name while retaining the repository-wide stable enum.
MAPSFailureCode = VerificationFailure


@dataclass(frozen=True, slots=True)
class MAPSVerificationResult:
    """Every verifier decision needed for reward and offline parity checks."""

    valid: bool
    reward: float
    failure_code: MAPSFailureCode | None
    message: str | None
    valid_answer_tag: bool
    valid_program: bool
    legal_instructions: bool
    within_length: bool
    correct_target: bool
    final_value: int | None = None
    canonical_program: str | None = None
    strategy_family_id: str | None = None
    program: Program | None = None

    def __post_init__(self) -> None:
        if self.valid:
            if self.reward != 1.0 or self.failure_code is not None:
                raise ValueError("valid MAPS result must have reward 1 and no failure")
            if not all(
                (
                    self.valid_answer_tag,
                    self.valid_program,
                    self.legal_instructions,
                    self.within_length,
                    self.correct_target,
                )
            ):
                raise ValueError("valid MAPS result requires every check to pass")
        elif self.reward != 0.0 or self.failure_code is None:
            raise ValueError("invalid MAPS result must have reward 0 and a failure")

    @property
    def accepted(self) -> bool:
        """Alias used by filtering/evaluation call sites."""

        return self.valid

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for raw generation records."""

        return {
            "valid": self.valid,
            "reward": self.reward,
            "failure_code": (
                self.failure_code.value if self.failure_code is not None else None
            ),
            "message": self.message,
            "valid_answer_tag": self.valid_answer_tag,
            "valid_program": self.valid_program,
            "legal_instructions": self.legal_instructions,
            "within_length": self.within_length,
            "correct_target": self.correct_target,
            "final_value": self.final_value,
            "canonical_program": self.canonical_program,
            "strategy_family_id": self.strategy_family_id,
        }

    def to_shared_result(self) -> VerificationResult:
        """Adapt the MAPS result to the current repository-wide result schema."""

        ast_json: dict[str, Any] | None = None
        if self.program is not None:
            ast_json = {
                "instructions": [
                    {"op": instruction.op.value, "argument": instruction.argument}
                    for instruction in self.program
                ]
            }
        return VerificationResult(
            valid_answer_tag=self.valid_answer_tag,
            valid_lexing=self.valid_program,
            valid_syntax=self.valid_program,
            # These two expression-oriented fields are vacuously true once a
            # MAPS program is syntactically valid and executable.
            valid_operand_multiset=self.valid_program,
            valid_operations=self.legal_instructions,
            valid_intermediates=self.valid_program,
            correct_target=self.correct_target,
            reward=self.reward,
            failure_code=self.failure_code,
            value_num=self.final_value,
            value_den=1 if self.final_value is not None else None,
            canonical_expression=self.canonical_program,
            strategy_family_id=self.strategy_family_id,
            ast_json=ast_json,
        )


def _failure(
    code: MAPSFailureCode,
    message: str,
    *,
    valid_answer_tag: bool = False,
    valid_program: bool = False,
    legal_instructions: bool = False,
    within_length: bool = False,
    correct_target: bool = False,
    final_value: int | None = None,
    canonical: str | None = None,
    family_id: str | None = None,
    program: Program | None = None,
) -> MAPSVerificationResult:
    return MAPSVerificationResult(
        valid=False,
        reward=0.0,
        failure_code=code,
        message=message,
        valid_answer_tag=valid_answer_tag,
        valid_program=valid_program,
        legal_instructions=legal_instructions,
        within_length=within_length,
        correct_target=correct_target,
        final_value=final_value,
        canonical_program=canonical,
        strategy_family_id=family_id,
        program=program,
    )


def verify_maps(
    completion: str,
    task: MAPSTask,
    *,
    max_completion_chars: int = DEFAULT_MAX_COMPLETION_CHARS,
) -> MAPSVerificationResult:
    """Verify one untrusted completion using only the closed MAPS DSL."""

    if not isinstance(completion, str):
        return _failure(MAPSFailureCode.INVALID_PROGRAM, "completion must be text")
    if max_completion_chars < 1:
        raise ValueError("max_completion_chars must be positive")
    if len(completion) > max_completion_chars:
        return _failure(
            MAPSFailureCode.ANSWER_TOO_LONG,
            f"completion exceeds {max_completion_chars} characters",
        )

    opening_tags = completion.count("<answer>")
    closing_tags = completion.count("</answer>")
    if opening_tags > 1 or closing_tags > 1:
        return _failure(
            MAPSFailureCode.MULTIPLE_ANSWER_TAGS,
            "completion must contain exactly one answer tag pair",
        )
    if opening_tags != 1 or closing_tags != 1:
        return _failure(
            MAPSFailureCode.MISSING_ANSWER_TAG,
            "completion must contain one <answer>...</answer> pair",
        )

    answer_match = _ANSWER_RE.fullmatch(completion)
    if answer_match is None:
        return _failure(
            MAPSFailureCode.INVALID_PROGRAM,
            "text outside the answer tag is not allowed",
        )
    program_text = answer_match.group("program")
    if not program_text.strip(" \t\r\n"):
        return _failure(
            MAPSFailureCode.EMPTY_ANSWER,
            "answer program is empty",
            valid_answer_tag=True,
        )

    try:
        program = parse_program(
            program_text,
            max_chars=max_completion_chars,
            # Parse one extra instruction so the verifier can return the
            # specific length failure rather than accepting a truncated parse.
            max_instructions=max(task.max_program_length + 1, 1),
            max_constant_digits=DEFAULT_MAX_CONSTANT_DIGITS,
        )
    except ProgramParseError as error:
        if error.code is ProgramParseFailure.EMPTY_PROGRAM:
            code = MAPSFailureCode.EMPTY_ANSWER
        elif error.code is ProgramParseFailure.TOO_MANY_INSTRUCTIONS:
            code = MAPSFailureCode.PROGRAM_TOO_LONG
        else:
            code = MAPSFailureCode.INVALID_PROGRAM
        return _failure(
            code,
            str(error),
            valid_answer_tag=True,
        )

    canonical = canonical_program(program)
    family_id = program_family_id(program)
    if len(program) > task.max_program_length:
        return _failure(
            MAPSFailureCode.PROGRAM_TOO_LONG,
            f"program exceeds the maximum length {task.max_program_length}",
            valid_answer_tag=True,
            valid_program=True,
            canonical=canonical,
            family_id=family_id,
            program=program,
        )

    allowed = {instruction.canonical() for instruction in task.allowed_instructions}
    for instruction in program:
        if instruction.canonical() not in allowed:
            return _failure(
                MAPSFailureCode.ILLEGAL_INSTRUCTION,
                f"instruction is not allowed: {instruction.canonical()}",
                valid_answer_tag=True,
                valid_program=True,
                within_length=True,
                canonical=canonical,
                family_id=family_id,
                program=program,
            )

    final_value = execute_program(task.start, task.modulus, program)
    if final_value != task.target:
        return _failure(
            MAPSFailureCode.WRONG_TARGET,
            f"program reaches {final_value}, expected {task.target}",
            valid_answer_tag=True,
            valid_program=True,
            legal_instructions=True,
            within_length=True,
            final_value=final_value,
            canonical=canonical,
            family_id=family_id,
            program=program,
        )

    return MAPSVerificationResult(
        valid=True,
        reward=1.0,
        failure_code=None,
        message=None,
        valid_answer_tag=True,
        valid_program=True,
        legal_instructions=True,
        within_length=True,
        correct_target=True,
        final_value=final_value,
        canonical_program=canonical,
        strategy_family_id=family_id,
        program=program,
    )


class MAPSVerifier:
    """Bind an immutable task for repeated online/offline verification."""

    def __init__(
        self,
        task: MAPSTask,
        *,
        max_completion_chars: int = DEFAULT_MAX_COMPLETION_CHARS,
    ) -> None:
        self.task = task
        self.max_completion_chars = max_completion_chars

    def verify(self, completion: str) -> MAPSVerificationResult:
        return verify_maps(
            completion,
            self.task,
            max_completion_chars=self.max_completion_chars,
        )


# Concise public alias used by reward/filter/evaluation code.
verify = verify_maps
