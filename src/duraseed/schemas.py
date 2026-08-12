"""Fail-closed schemas shared by exact task environments.

The schemas intentionally reject unknown fields. Scientific artifacts should
never absorb misspelled configuration or result keys silently.
"""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from math import gcd
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAPS_MAX_CONSTANT_DIGITS = 32
MAPS_MAX_PROGRAM_LENGTH = 100
TCES_MAX_INTEGER_DIGITS = 1_024


class StrictModel(BaseModel):
    """Base schema that rejects undeclared fields and is immutable after parsing."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExactRational(StrictModel):
    """JSON-safe normalized exact rational value."""

    numerator: int
    denominator: int = 1

    @field_validator("numerator", "denominator", mode="before")
    @classmethod
    def components_must_be_integers(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("rational components must be integers")
        return value

    @field_validator("denominator")
    @classmethod
    def denominator_must_be_nonzero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("denominator must be nonzero")
        return value

    @model_validator(mode="after")
    def normalize(self) -> "ExactRational":
        numerator = self.numerator
        denominator = self.denominator
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        divisor = gcd(abs(numerator), denominator)
        object.__setattr__(self, "numerator", numerator // divisor)
        object.__setattr__(self, "denominator", denominator // divisor)
        return self

    @classmethod
    def from_fraction(cls, value: Fraction) -> "ExactRational":
        return cls(numerator=value.numerator, denominator=value.denominator)

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


class TCESConstraints(StrictModel):
    """Resource and arithmetic limits enforced by the TCES verifier."""

    use_each_once: Literal[True] = True
    max_abs_intermediate: int = Field(default=10_000, ge=1)
    max_denominator: int = Field(default=1_000, ge=1)
    max_tree_depth: int = Field(default=5, ge=1, le=64)
    max_ast_nodes: int = Field(default=31, ge=1, le=10_000)
    max_answer_length: int = Field(default=1_024, ge=1, le=1_000_000)

    @field_validator("use_each_once", mode="before")
    @classmethod
    def use_each_once_is_mandatory(cls, value: object) -> object:
        if value is not True:
            raise ValueError("TCES v1 requires use_each_once=true")
        return value


TCESOperator = Literal["+", "-", "*", "/"]


class TCESTask(StrictModel):
    """One immutable verifier input for expression synthesis."""

    operands: tuple[int, ...] = Field(min_length=1, max_length=16)
    target: ExactRational
    allowed_ops: tuple[TCESOperator, ...] = ("+", "-", "*", "/")
    constraints: TCESConstraints = Field(default_factory=TCESConstraints)
    task_id: str | None = None
    split: str | None = None

    @field_validator("operands", mode="before")
    @classmethod
    def operands_must_be_integers(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("operands must be a list or tuple of integers")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("operands must contain only integers")
        if any(item < 0 for item in value):
            raise ValueError("TCES operands must be unsigned decimal integers")
        if any(len(str(item)) > TCES_MAX_INTEGER_DIGITS for item in value):
            raise ValueError(
                f"TCES operands may have at most {TCES_MAX_INTEGER_DIGITS} digits"
            )
        return value

    @field_validator("allowed_ops")
    @classmethod
    def operators_must_be_unique(
        cls, value: tuple[TCESOperator, ...]
    ) -> tuple[TCESOperator, ...]:
        if not value:
            raise ValueError("at least one operator is required")
        if len(set(value)) != len(value):
            raise ValueError("allowed_ops must not contain duplicates")
        return value

    @model_validator(mode="after")
    def canonical_answer_must_fit_resource_limit(self) -> "TCESTask":
        # A fully parenthesized full-binary expression has one operator and
        # one parenthesis pair per internal node.  Requiring that canonical
        # representation to fit ensures every enumerated teacher can be
        # submitted to the verifier without a serialization-only mismatch.
        required = sum(len(str(operand)) for operand in self.operands)
        required += 3 * (len(self.operands) - 1)
        if self.constraints.max_answer_length < required:
            raise ValueError(
                "max_answer_length is too small for a canonical use-all answer"
            )
        return self


class VerificationFailure(StrEnum):
    """Stable failure taxonomy used by online reward and offline analysis."""

    ANSWER_TOO_LONG = "answer_too_long"
    MISSING_ANSWER_TAG = "missing_answer_tag"
    MULTIPLE_ANSWER_TAGS = "multiple_answer_tags"
    EMPTY_ANSWER = "empty_answer"
    INVALID_CHARACTER = "invalid_character"
    INVALID_TOKEN = "invalid_token"
    INVALID_SYNTAX = "invalid_syntax"
    DISALLOWED_OPERATOR = "disallowed_operator"
    OPERAND_MULTISET_MISMATCH = "operand_multiset_mismatch"
    DIVISION_BY_ZERO = "division_by_zero"
    AST_LIMIT_EXCEEDED = "ast_limit_exceeded"
    INTERMEDIATE_LIMIT_EXCEEDED = "intermediate_limit_exceeded"
    WRONG_TARGET = "wrong_target"
    INVALID_PROGRAM = "invalid_program"
    PROGRAM_TOO_LONG = "program_too_long"
    ILLEGAL_INSTRUCTION = "illegal_instruction"


class VerificationResult(StrictModel):
    """Exact, serializable result shared by reward and evaluation code."""

    valid_answer_tag: bool = False
    valid_lexing: bool = False
    valid_syntax: bool = False
    valid_operand_multiset: bool = False
    valid_operations: bool = False
    valid_intermediates: bool = False
    correct_target: bool = False
    reward: Literal[0.0, 1.0] = 0.0
    failure_code: VerificationFailure | None = None
    value_num: int | None = None
    value_den: int | None = None
    canonical_expression: str | None = None
    strategy_family_id: str | None = None
    ast_json: dict[str, Any] | None = None

    @field_validator(
        "valid_answer_tag",
        "valid_lexing",
        "valid_syntax",
        "valid_operand_multiset",
        "valid_operations",
        "valid_intermediates",
        "correct_target",
        mode="before",
    )
    @classmethod
    def check_flags_must_be_booleans(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("verification flags must be booleans")
        return value

    @field_validator("reward", mode="before")
    @classmethod
    def reward_must_be_binary_float(cls, value: object) -> object:
        if type(value) is not float or value not in (0.0, 1.0):
            raise ValueError("reward must be the float 0.0 or 1.0")
        return value

    @field_validator("value_num", "value_den", mode="before")
    @classmethod
    def exact_value_parts_must_be_integers_or_none(cls, value: object) -> object:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("exact verification values must be integers")
        return value

    @model_validator(mode="after")
    def enforce_exact_result_contract(self) -> "VerificationResult":
        if (self.value_num is None) != (self.value_den is None):
            raise ValueError("exact verification value numerator/denominator must pair")
        if self.value_num is not None and self.value_den is not None:
            if self.value_den == 0:
                raise ValueError("verification value denominator must be nonzero")
            numerator = self.value_num
            denominator = self.value_den
            if denominator < 0:
                numerator = -numerator
                denominator = -denominator
            divisor = gcd(abs(numerator), denominator)
            object.__setattr__(self, "value_num", numerator // divisor)
            object.__setattr__(self, "value_den", denominator // divisor)

        checks = (
            self.valid_answer_tag,
            self.valid_lexing,
            self.valid_syntax,
            self.valid_operand_multiset,
            self.valid_operations,
            self.valid_intermediates,
            self.correct_target,
        )
        succeeded = all(checks)
        expected_reward = 1.0 if succeeded else 0.0
        if self.reward != expected_reward:
            raise ValueError("reward must equal the conjunction of verifier checks")
        if succeeded:
            if self.failure_code is not None:
                raise ValueError("successful verification cannot have a failure code")
            required_metadata = (
                self.value_num,
                self.value_den,
                self.canonical_expression,
                self.strategy_family_id,
                self.ast_json,
            )
            if any(value is None for value in required_metadata):
                raise ValueError("successful verification requires exact metadata")
        elif self.failure_code is None:
            raise ValueError("failed verification requires a failure code")
        return self


class MAPSInstructionKind(StrEnum):
    ADD = "ADD"
    MUL = "MUL"
    NEG = "NEG"


class MAPSInstruction(StrictModel):
    """One instruction in the deliberately tiny MAPS DSL."""

    op: MAPSInstructionKind
    argument: int | None = None

    @field_validator("argument", mode="before")
    @classmethod
    def argument_must_be_an_integer_or_none(cls, value: object) -> object:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("instruction arguments must be integers")
        if value is not None and len(str(abs(value))) > MAPS_MAX_CONSTANT_DIGITS:
            raise ValueError(
                f"instruction arguments may have at most "
                f"{MAPS_MAX_CONSTANT_DIGITS} digits"
            )
        return value

    @model_validator(mode="after")
    def validate_arity(self) -> "MAPSInstruction":
        if self.op is MAPSInstructionKind.NEG and self.argument is not None:
            raise ValueError("NEG takes no argument")
        if self.op is not MAPSInstructionKind.NEG and self.argument is None:
            raise ValueError(f"{self.op} requires an integer argument")
        return self

    def canonical(self) -> str:
        return (
            self.op.value
            if self.argument is None
            else f"{self.op.value} {self.argument}"
        )


class MAPSTask(StrictModel):
    """One exact modular program-synthesis task."""

    start: int
    modulus: int = Field(ge=2)
    target: int
    allowed_instructions: tuple[MAPSInstruction, ...] = Field(min_length=1)
    max_program_length: int = Field(
        default=5,
        ge=1,
        le=MAPS_MAX_PROGRAM_LENGTH,
    )
    task_id: str | None = None
    split: str | None = None

    @field_validator("start", "modulus", "target", "max_program_length", mode="before")
    @classmethod
    def numeric_fields_must_be_integers(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("MAPS numeric fields must be integers")
        return value

    @model_validator(mode="after")
    def normalize_states(self) -> "MAPSTask":
        normalized_start = self.start % self.modulus
        normalized_target = self.target % self.modulus
        if normalized_start == normalized_target:
            raise ValueError("MAPS tasks must require at least one instruction")
        object.__setattr__(self, "start", normalized_start)
        object.__setattr__(self, "target", normalized_target)
        canonical = [
            instruction.canonical() for instruction in self.allowed_instructions
        ]
        if len(set(canonical)) != len(canonical):
            raise ValueError("allowed_instructions must not contain duplicates")
        return self
