"""ASCII-only lexer for the deliberately tiny TCES expression language."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TokenKind(StrEnum):
    INTEGER = "integer"
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    EOF = "eof"


@dataclass(frozen=True, slots=True)
class Token:
    """One lexeme with a half-open source span."""

    kind: TokenKind
    text: str
    start: int
    end: int


class LexErrorCode(StrEnum):
    NON_ASCII = "non_ascii"
    INVALID_CHARACTER = "invalid_character"
    INTEGER_LITERAL_TOO_LONG = "integer_literal_too_long"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"


class LexerError(ValueError):
    """A deterministic, source-located lexer rejection."""

    def __init__(self, code: LexErrorCode, position: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.position = position


@dataclass(frozen=True, slots=True)
class LexerConfig:
    """Resource limits independent of task arithmetic constraints."""

    max_integer_digits: int = 1_024
    max_tokens: int = 4_096

    def __post_init__(self) -> None:
        if self.max_integer_digits < 1:
            raise ValueError("max_integer_digits must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


_SINGLE_CHARACTER_TOKENS: dict[str, TokenKind] = {
    "+": TokenKind.PLUS,
    "-": TokenKind.MINUS,
    "*": TokenKind.STAR,
    "/": TokenKind.SLASH,
    "(": TokenKind.LEFT_PAREN,
    ")": TokenKind.RIGHT_PAREN,
}
_ASCII_WHITESPACE = " \t\n\r\f\v"


def tokenize(source: str, config: LexerConfig | None = None) -> tuple[Token, ...]:
    """Tokenize one answer expression using an explicit ASCII vocabulary.

    Python character classes such as ``str.isdigit`` are intentionally not
    used because they accept many Unicode code points that visually resemble
    the permitted grammar.
    """

    limits = config or LexerConfig()
    for position, character in enumerate(source):
        if ord(character) > 0x7F:
            raise LexerError(
                LexErrorCode.NON_ASCII,
                position,
                f"non-ASCII character at offset {position}",
            )

    tokens: list[Token] = []
    position = 0
    while position < len(source):
        character = source[position]
        if character in _ASCII_WHITESPACE:
            position += 1
            continue

        if "0" <= character <= "9":
            start = position
            while position < len(source) and "0" <= source[position] <= "9":
                position += 1
            text = source[start:position]
            if len(text) > limits.max_integer_digits:
                raise LexerError(
                    LexErrorCode.INTEGER_LITERAL_TOO_LONG,
                    start,
                    f"integer literal exceeds {limits.max_integer_digits} digits",
                )
            tokens.append(Token(TokenKind.INTEGER, text, start, position))
        else:
            kind = _SINGLE_CHARACTER_TOKENS.get(character)
            if kind is None:
                raise LexerError(
                    LexErrorCode.INVALID_CHARACTER,
                    position,
                    f"invalid character {character!r} at offset {position}",
                )
            tokens.append(Token(kind, character, position, position + 1))
            position += 1

        if len(tokens) > limits.max_tokens:
            raise LexerError(
                LexErrorCode.TOKEN_LIMIT_EXCEEDED,
                tokens[-1].start,
                f"expression exceeds {limits.max_tokens} tokens",
            )

    tokens.append(Token(TokenKind.EOF, "", len(source), len(source)))
    return tuple(tokens)
