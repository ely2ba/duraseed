"""Modular Affine Program Synthesis (MAPS).

The public surface keeps parsing, execution, solving, generation, and
verification available without exposing an executable general-purpose
language.
"""

from duraseed.schemas import MAPSInstruction, MAPSInstructionKind, MAPSTask

from .generator import (
    FROZEN_PRIMES_V1,
    GeneratedMAPSInstance,
    MAPSGenerationError,
    MAPSGenerator,
    MAPSGeneratorConfig,
    generate_instance,
    generate_task,
    render_prompt,
    render_teacher_answer,
)
from .interpreter import (
    Program,
    ProgramParseError,
    ProgramParseFailure,
    apply_instruction,
    canonical_instruction,
    canonical_program,
    execute_program,
    parse_and_execute,
    parse_instruction,
    parse_program,
    program_family_id,
)
from .solver import MAPSSolveResult, shortest_program, solve, solve_bfs
from .verifier import (
    MAPSFailureCode,
    MAPSVerificationResult,
    MAPSVerifier,
    verify,
    verify_maps,
)

__all__ = [
    "FROZEN_PRIMES_V1",
    "GeneratedMAPSInstance",
    "MAPSFailureCode",
    "MAPSGenerationError",
    "MAPSGenerator",
    "MAPSGeneratorConfig",
    "MAPSInstruction",
    "MAPSInstructionKind",
    "MAPSSolveResult",
    "MAPSTask",
    "MAPSVerificationResult",
    "MAPSVerifier",
    "Program",
    "ProgramParseError",
    "ProgramParseFailure",
    "apply_instruction",
    "canonical_instruction",
    "canonical_program",
    "execute_program",
    "generate_instance",
    "generate_task",
    "parse_and_execute",
    "parse_instruction",
    "parse_program",
    "program_family_id",
    "render_prompt",
    "render_teacher_answer",
    "shortest_program",
    "solve",
    "solve_bfs",
    "verify",
    "verify_maps",
]
