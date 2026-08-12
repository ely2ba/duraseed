"""Strict interpreter for the Modular Affine Program Synthesis DSL.

MAPS deliberately has a tiny grammar.  This module parses that grammar by
matching ASCII tokens and applies each instruction directly; model output is
never handed to ``eval`` or to any general-purpose language runtime.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
import re

from duraseed.schemas import (
    MAPSInstruction,
    MAPSInstructionKind,
    MAPS_MAX_CONSTANT_DIGITS,
)


Program = tuple[MAPSInstruction, ...]

DEFAULT_MAX_PROGRAM_CHARS = 4_096
DEFAULT_MAX_INSTRUCTIONS = 1_000
DEFAULT_MAX_CONSTANT_DIGITS = MAPS_MAX_CONSTANT_DIGITS
_ASCII_WHITESPACE = " \t\r\n"
_INTEGER_PATTERN = r"(?:0|-?[1-9][0-9]*)"
_INSTRUCTION_RE = re.compile(
    rf"(?:(?P<op>ADD|MUL) (?P<argument>{_INTEGER_PATTERN})|(?P<neg>NEG))",
    flags=re.ASCII,
)


class ProgramParseFailure(StrEnum):
    """Stable reasons why a program cannot be parsed safely."""

    EMPTY_PROGRAM = "empty_program"
    PROGRAM_TOO_LARGE = "program_too_large"
    TOO_MANY_INSTRUCTIONS = "too_many_instructions"
    INVALID_INSTRUCTION = "invalid_instruction"
    CONSTANT_TOO_LARGE = "constant_too_large"


class ProgramParseError(ValueError):
    """A fail-closed MAPS parse error with a machine-readable code."""

    def __init__(
        self,
        code: ProgramParseFailure,
        message: str,
        *,
        instruction_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.instruction_index = instruction_index


def parse_instruction(
    text: str,
    *,
    max_constant_digits: int = DEFAULT_MAX_CONSTANT_DIGITS,
) -> MAPSInstruction:
    """Parse one canonical instruction from the closed MAPS vocabulary.

    The only accepted forms are ``ADD <integer>``, ``MUL <integer>``, and
    ``NEG``.  Integers use canonical ASCII decimal notation: no leading plus,
    no leading zeroes, and no ``-0`` alias.
    """

    if not isinstance(text, str):
        raise ProgramParseError(
            ProgramParseFailure.INVALID_INSTRUCTION,
            "instruction must be text",
        )
    if max_constant_digits < 1:
        raise ValueError("max_constant_digits must be positive")

    match = _INSTRUCTION_RE.fullmatch(text)
    if match is None:
        raise ProgramParseError(
            ProgramParseFailure.INVALID_INSTRUCTION,
            f"invalid MAPS instruction: {text!r}",
        )

    if match.group("neg") is not None:
        return MAPSInstruction(op=MAPSInstructionKind.NEG)

    argument_text = match.group("argument")
    assert argument_text is not None  # Guaranteed by the regular expression.
    digit_count = len(argument_text.removeprefix("-"))
    if digit_count > max_constant_digits:
        raise ProgramParseError(
            ProgramParseFailure.CONSTANT_TOO_LARGE,
            f"instruction constant exceeds {max_constant_digits} digits",
        )
    op = MAPSInstructionKind(match.group("op"))
    return MAPSInstruction(op=op, argument=int(argument_text))


def parse_program(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_PROGRAM_CHARS,
    max_instructions: int = DEFAULT_MAX_INSTRUCTIONS,
    max_constant_digits: int = DEFAULT_MAX_CONSTANT_DIGITS,
) -> Program:
    """Parse a semicolon-separated MAPS program.

    ASCII spaces, tabs, and newlines may surround the whole program or a
    semicolon.  They are not accepted inside an instruction except for the one
    literal space separating an opcode and its integer argument.
    """

    if not isinstance(text, str):
        raise ProgramParseError(
            ProgramParseFailure.INVALID_INSTRUCTION,
            "program must be text",
        )
    if max_chars < 1 or max_instructions < 1:
        raise ValueError("parser resource limits must be positive")
    if len(text) > max_chars:
        raise ProgramParseError(
            ProgramParseFailure.PROGRAM_TOO_LARGE,
            f"program exceeds {max_chars} characters",
        )

    stripped = text.strip(_ASCII_WHITESPACE)
    if not stripped:
        raise ProgramParseError(
            ProgramParseFailure.EMPTY_PROGRAM,
            "program is empty",
        )

    raw_instructions = stripped.split(";")
    if len(raw_instructions) > max_instructions:
        raise ProgramParseError(
            ProgramParseFailure.TOO_MANY_INSTRUCTIONS,
            f"program exceeds {max_instructions} instructions",
        )

    parsed: list[MAPSInstruction] = []
    for index, raw_instruction in enumerate(raw_instructions):
        instruction_text = raw_instruction.strip(_ASCII_WHITESPACE)
        if not instruction_text:
            raise ProgramParseError(
                ProgramParseFailure.INVALID_INSTRUCTION,
                "empty instruction or trailing semicolon",
                instruction_index=index,
            )
        try:
            parsed.append(
                parse_instruction(
                    instruction_text,
                    max_constant_digits=max_constant_digits,
                )
            )
        except ProgramParseError as error:
            raise ProgramParseError(
                error.code,
                str(error),
                instruction_index=index,
            ) from error
    return tuple(parsed)


def canonical_instruction(instruction: MAPSInstruction) -> str:
    """Return the unique textual representation of an instruction."""

    return instruction.canonical()


def canonical_program(program: Sequence[MAPSInstruction]) -> str:
    """Return the unique semicolon-separated representation of a program."""

    return "; ".join(canonical_instruction(instruction) for instruction in program)


def program_family_id(program: Sequence[MAPSInstruction]) -> str:
    """Abstract constants while retaining the instruction-order template.

    Instruction-order templates are the MAPS analogue of strategy families
    and are one of the split/difficulty controls named in the specification.
    """

    if not program:
        return "maps:EMPTY"
    return "maps:" + ">".join(instruction.op.value for instruction in program)


def apply_instruction(value: int, instruction: MAPSInstruction, modulus: int) -> int:
    """Apply exactly one instruction and reduce the result modulo ``modulus``."""

    if isinstance(modulus, bool) or not isinstance(modulus, int) or modulus < 2:
        raise ValueError("modulus must be an integer greater than one")
    state = value % modulus
    # Comparing the serialized opcode also permits use with the repository's
    # shared Pydantic instruction schema once optional dependencies are loaded.
    op_value = getattr(instruction.op, "value", instruction.op)
    if op_value == MAPSInstructionKind.NEG.value:
        result = -state
    elif op_value == MAPSInstructionKind.ADD.value:
        assert instruction.argument is not None
        result = state + instruction.argument
    elif op_value == MAPSInstructionKind.MUL.value:
        assert instruction.argument is not None
        result = state * instruction.argument
    else:  # pragma: no cover - Pydantic prevents an unknown instruction kind.
        raise ValueError(f"unsupported instruction kind: {instruction.op!r}")
    return result % modulus


def execute_program(
    start: int,
    modulus: int,
    program: Iterable[MAPSInstruction],
) -> int:
    """Execute instructions from top to bottom, reducing after every step."""

    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError("start must be an integer")
    state = start % modulus
    for instruction in program:
        state = apply_instruction(state, instruction, modulus)
    return state


def parse_and_execute(
    text: str,
    *,
    start: int,
    modulus: int,
    max_chars: int = DEFAULT_MAX_PROGRAM_CHARS,
    max_instructions: int = DEFAULT_MAX_INSTRUCTIONS,
) -> int:
    """Convenience wrapper for strict parsing followed by exact execution."""

    program = parse_program(
        text,
        max_chars=max_chars,
        max_instructions=max_instructions,
    )
    return execute_program(start, modulus, program)
