"""Template-Controlled Expression Synthesis (TCES)."""

from duraseed.schemas import (
    ExactRational,
    TCESConstraints,
    TCESTask,
    VerificationFailure,
    VerificationResult,
)

from .ast import (
    BinaryExpression,
    BinaryOperator,
    Expression,
    IntegerLiteral,
    evaluate_expression,
    render_expression,
)
from .canonicalize import (
    canonicalize_expression,
    render_canonical_expression,
)
from .enumerate import (
    EnumeratedExpression,
    EnumerationConstraints,
    EnumerationResult,
    FamilyEnumeration,
    enumerate_solutions,
    enumerate_task,
)
from .generator import (
    GeneratedTCESInstance,
    TCESGenerationError,
    TCESFamilyGenerator,
    TCESGenerator,
    TCESGeneratorConfig,
    generate_instance,
    generate_task,
    render_prompt,
    render_teacher_completion,
    task_content_hash,
)
from .lexer import LexerConfig, LexerError, Token, TokenKind, tokenize
from .parser import ParseError, ParserLimits, parse_expression
from .strategies import (
    StrategyFamilySignature,
    strategy_family_id,
    strategy_family_signature,
    structural_signature,
)
from .teacher import (
    TeacherStep,
    TeacherTrace,
    build_teacher_trace,
    generate_teacher_trace,
    verify_teacher_trace,
)
from .verifier import TCESVerifier, extract_answer_span, verify_completion

__all__ = [
    "BinaryExpression",
    "BinaryOperator",
    "EnumeratedExpression",
    "EnumerationConstraints",
    "EnumerationResult",
    "ExactRational",
    "Expression",
    "FamilyEnumeration",
    "GeneratedTCESInstance",
    "IntegerLiteral",
    "LexerConfig",
    "LexerError",
    "ParseError",
    "ParserLimits",
    "StrategyFamilySignature",
    "TCESConstraints",
    "TCESGenerationError",
    "TCESFamilyGenerator",
    "TCESGenerator",
    "TCESGeneratorConfig",
    "TCESTask",
    "TCESVerifier",
    "TeacherStep",
    "TeacherTrace",
    "Token",
    "TokenKind",
    "VerificationFailure",
    "VerificationResult",
    "build_teacher_trace",
    "canonicalize_expression",
    "enumerate_solutions",
    "enumerate_task",
    "evaluate_expression",
    "extract_answer_span",
    "generate_instance",
    "generate_task",
    "generate_teacher_trace",
    "parse_expression",
    "render_canonical_expression",
    "render_expression",
    "render_prompt",
    "render_teacher_completion",
    "strategy_family_id",
    "strategy_family_signature",
    "structural_signature",
    "task_content_hash",
    "tokenize",
    "verify_completion",
    "verify_teacher_trace",
]
