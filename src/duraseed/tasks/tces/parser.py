"""Recursive-descent parser for binary TCES arithmetic expressions.

Grammar::

    expression := product (("+" | "-") product)*
    product    := primary (("*" | "/") primary)*
    primary    := INTEGER | "(" expression ")"

Unary plus and unary minus are deliberately invalid.  TCES exposes only
binary operations, and allowing signed literals would weaken the exact prompt
operand-multiset invariant.  Negative intermediate values remain expressible
through binary subtraction, for example ``(3-7)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from duraseed.tasks.tces.ast import (
    BinaryExpression,
    BinaryOperator,
    Expression,
    IntegerLiteral,
    tree_depth,
)
from duraseed.tasks.tces.lexer import LexerConfig, Token, TokenKind, tokenize


class ParseErrorCode(StrEnum):
    EMPTY_EXPRESSION = "empty_expression"
    EXPECTED_OPERAND = "expected_operand"
    UNARY_OPERATOR_NOT_ALLOWED = "unary_operator_not_allowed"
    EXPECTED_CLOSING_PARENTHESIS = "expected_closing_parenthesis"
    UNEXPECTED_TOKEN = "unexpected_token"
    AST_NODE_LIMIT_EXCEEDED = "ast_node_limit_exceeded"
    AST_DEPTH_LIMIT_EXCEEDED = "ast_depth_limit_exceeded"
    PARENTHESIS_DEPTH_LIMIT_EXCEEDED = "parenthesis_depth_limit_exceeded"


class ParseError(ValueError):
    """A deterministic parser rejection tied to one token."""

    def __init__(self, code: ParseErrorCode, token: Token, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.token = token


@dataclass(frozen=True, slots=True)
class ParserLimits:
    max_ast_nodes: int = 127
    max_ast_depth: int = 32
    max_parenthesis_depth: int = 64

    def __post_init__(self) -> None:
        if self.max_ast_nodes < 1:
            raise ValueError("max_ast_nodes must be positive")
        if self.max_ast_depth < 1:
            raise ValueError("max_ast_depth must be positive")
        if self.max_parenthesis_depth < 1:
            raise ValueError("max_parenthesis_depth must be positive")


_ADDITIVE_OPERATORS: dict[TokenKind, BinaryOperator] = {
    TokenKind.PLUS: BinaryOperator.ADD,
    TokenKind.MINUS: BinaryOperator.SUB,
}
_MULTIPLICATIVE_OPERATORS: dict[TokenKind, BinaryOperator] = {
    TokenKind.STAR: BinaryOperator.MUL,
    TokenKind.SLASH: BinaryOperator.DIV,
}


class Parser:
    """Single-use recursive-descent parser over a frozen token sequence."""

    def __init__(
        self, tokens: tuple[Token, ...], limits: ParserLimits | None = None
    ) -> None:
        if not tokens or tokens[-1].kind is not TokenKind.EOF:
            raise ValueError("tokens must end with EOF")
        self._tokens = tokens
        self._limits = limits or ParserLimits()
        self._position = 0
        self._node_count = 0

    def parse(self) -> Expression:
        if self._current.kind is TokenKind.EOF:
            raise ParseError(
                ParseErrorCode.EMPTY_EXPRESSION,
                self._current,
                "answer expression is empty",
            )

        expression = self._parse_expression(parenthesis_depth=0)
        if self._current.kind is not TokenKind.EOF:
            raise ParseError(
                ParseErrorCode.UNEXPECTED_TOKEN,
                self._current,
                f"unexpected token {self._current.text!r}",
            )
        if tree_depth(expression) > self._limits.max_ast_depth:
            raise ParseError(
                ParseErrorCode.AST_DEPTH_LIMIT_EXCEEDED,
                self._current,
                f"AST exceeds maximum depth {self._limits.max_ast_depth}",
            )
        return expression

    @property
    def _current(self) -> Token:
        return self._tokens[self._position]

    def _advance(self) -> Token:
        token = self._current
        if token.kind is not TokenKind.EOF:
            self._position += 1
        return token

    def _record_node(self, token: Token) -> None:
        self._node_count += 1
        if self._node_count > self._limits.max_ast_nodes:
            raise ParseError(
                ParseErrorCode.AST_NODE_LIMIT_EXCEEDED,
                token,
                f"AST exceeds maximum node count {self._limits.max_ast_nodes}",
            )

    def _parse_expression(self, parenthesis_depth: int) -> Expression:
        expression = self._parse_product(parenthesis_depth)
        while self._current.kind in _ADDITIVE_OPERATORS:
            token = self._advance()
            right = self._parse_product(parenthesis_depth)
            self._record_node(token)
            expression = BinaryExpression(
                operator=_ADDITIVE_OPERATORS[token.kind],
                left=expression,
                right=right,
            )
        return expression

    def _parse_product(self, parenthesis_depth: int) -> Expression:
        expression = self._parse_primary(parenthesis_depth)
        while self._current.kind in _MULTIPLICATIVE_OPERATORS:
            token = self._advance()
            right = self._parse_primary(parenthesis_depth)
            self._record_node(token)
            expression = BinaryExpression(
                operator=_MULTIPLICATIVE_OPERATORS[token.kind],
                left=expression,
                right=right,
            )
        return expression

    def _parse_primary(self, parenthesis_depth: int) -> Expression:
        token = self._current
        if token.kind is TokenKind.INTEGER:
            self._advance()
            self._record_node(token)
            return IntegerLiteral(value=int(token.text, 10))

        if token.kind is TokenKind.LEFT_PAREN:
            if parenthesis_depth >= self._limits.max_parenthesis_depth:
                raise ParseError(
                    ParseErrorCode.PARENTHESIS_DEPTH_LIMIT_EXCEEDED,
                    token,
                    "parenthesis nesting limit exceeded",
                )
            self._advance()
            expression = self._parse_expression(parenthesis_depth + 1)
            if self._current.kind is not TokenKind.RIGHT_PAREN:
                raise ParseError(
                    ParseErrorCode.EXPECTED_CLOSING_PARENTHESIS,
                    self._current,
                    "expected closing parenthesis",
                )
            self._advance()
            return expression

        if token.kind in (TokenKind.PLUS, TokenKind.MINUS):
            raise ParseError(
                ParseErrorCode.UNARY_OPERATOR_NOT_ALLOWED,
                token,
                "unary plus and unary minus are not part of the TCES grammar",
            )

        raise ParseError(
            ParseErrorCode.EXPECTED_OPERAND,
            token,
            f"expected integer or '(', found {token.text!r}",
        )


def parse_expression(
    source: str,
    *,
    lexer_config: LexerConfig | None = None,
    parser_limits: ParserLimits | None = None,
) -> Expression:
    """Lex and parse an expression without invoking any general parser."""

    return Parser(tokenize(source, lexer_config), parser_limits).parse()
