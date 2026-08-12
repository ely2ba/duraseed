"""Small AES-GCM seal for self-blinded final-test data."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Iterable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from duraseed.data.io import atomic_write_bytes
from duraseed.data.splits import FINAL_TEST_SPLITS


SEAL_FORMAT = "duraseed-final-test-v1"
_KEY_BYTES = 32
_NONCE_BYTES = 12
_ENVELOPE_FIELDS = {
    "ciphertext",
    "declared_split",
    "nonce",
    "plaintext_sha256",
    "version",
}


class ExecutionContext(StrEnum):
    """Why a task artifact is being opened."""

    TRAINING = "training"
    SELECTION = "selection"
    DEBUGGING = "debugging"
    FINAL_EVALUATION = "final_evaluation"


class FinalTestAccessDenied(PermissionError):
    """Final-test data was requested outside the final-evaluation command."""


class SealError(ValueError):
    """A seal is malformed, unauthenticated, or inconsistent."""


@dataclass(frozen=True, slots=True)
class SealInfo:
    """Public metadata embedded in a sealed file."""

    declared_split: str
    plaintext_sha256: str
    version: str = SEAL_FORMAT


def is_final_test_split(split: str) -> bool:
    """Return whether ``split`` names a frozen Stage-A or Stage-B final split."""

    if not isinstance(split, str):
        raise TypeError("split must be a string")
    return split.strip().casefold() in FINAL_TEST_SPLITS


def _context(value: ExecutionContext | str) -> ExecutionContext:
    try:
        return value if isinstance(value, ExecutionContext) else ExecutionContext(value)
    except ValueError as error:
        raise ValueError(f"unknown execution context: {value!r}") from error


def guard_record_splits(splits: Iterable[str], context: ExecutionContext | str) -> None:
    """Deny final-test rows to training, selection, and debugging code."""

    access = _context(context)
    if access is not ExecutionContext.FINAL_EVALUATION and any(
        is_final_test_split(split) for split in splits
    ):
        raise FinalTestAccessDenied(
            "final-test data can only be opened by the final_evaluation command"
        )


def _key_bytes(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) != _KEY_BYTES:
        raise ValueError("AES-256 key must be exactly 32 bytes")
    return key


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metadata_bytes(
    *, declared_split: str, plaintext_sha256: str, version: str = SEAL_FORMAT
) -> bytes:
    return json.dumps(
        {
            "declared_split": declared_split,
            "plaintext_sha256": plaintext_sha256,
            "version": version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_envelope(path: str | os.PathLike[str]) -> dict[str, object]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("sealed file is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != _ENVELOPE_FIELDS:
        raise SealError("sealed file has missing or unknown fields")
    if value["version"] != SEAL_FORMAT:
        raise SealError("unsupported seal version")
    declared_split = value["declared_split"]
    digest = value["plaintext_sha256"]
    if not isinstance(declared_split, str) or not is_final_test_split(declared_split):
        raise SealError("sealed file does not declare a final-test split")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SealError("plaintext_sha256 must be 64 lowercase hexadecimal digits")
    try:
        nonce = base64.b64decode(value["nonce"], validate=True)
        ciphertext = base64.b64decode(value["ciphertext"], validate=True)
    except (TypeError, ValueError) as error:
        raise SealError("sealed file contains invalid base64") from error
    if len(nonce) != _NONCE_BYTES or len(ciphertext) < 16:
        raise SealError("sealed file contains invalid AES-GCM data")
    return value


def seal_file(
    source_path: str | os.PathLike[str],
    sealed_path: str | os.PathLike[str],
    *,
    key: bytes,
    declared_split: str,
) -> SealInfo:
    """Encrypt one final-test file with an externally supplied AES-256 key."""

    if not isinstance(declared_split, str) or not is_final_test_split(declared_split):
        raise ValueError("declared_split must name a frozen final-test split")
    plaintext = Path(source_path).read_bytes()
    digest = _sha256(plaintext)
    nonce = secrets.token_bytes(_NONCE_BYTES)
    ciphertext = AESGCM(_key_bytes(key)).encrypt(
        nonce,
        plaintext,
        _metadata_bytes(declared_split=declared_split, plaintext_sha256=digest),
    )
    envelope = {
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "declared_split": declared_split,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "plaintext_sha256": digest,
        "version": SEAL_FORMAT,
    }
    payload = (
        json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    atomic_write_bytes(sealed_path, payload)
    return SealInfo(declared_split=declared_split, plaintext_sha256=digest)


def inspect_seal(path: str | os.PathLike[str]) -> SealInfo:
    """Read public seal metadata without decrypting the final-test data."""

    envelope = _load_envelope(path)
    return SealInfo(
        declared_split=envelope["declared_split"],
        plaintext_sha256=envelope["plaintext_sha256"],
    )


def open_final_test(
    sealed_path: str | os.PathLike[str],
    *,
    key: bytes,
    command: ExecutionContext | str,
    expected_split: str | None = None,
) -> bytes:
    """Authenticate and decrypt only for the explicit final-evaluation command."""

    if _context(command) is not ExecutionContext.FINAL_EVALUATION:
        raise FinalTestAccessDenied(
            "final-test data can only be opened by the final_evaluation command"
        )
    envelope = _load_envelope(sealed_path)
    declared_split = envelope["declared_split"]
    digest = envelope["plaintext_sha256"]
    assert isinstance(declared_split, str)
    assert isinstance(digest, str)
    try:
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
        plaintext = AESGCM(_key_bytes(key)).decrypt(
            nonce,
            ciphertext,
            _metadata_bytes(
                declared_split=declared_split,
                plaintext_sha256=digest,
            ),
        )
    except InvalidTag as error:
        raise SealError("AES-GCM authentication failed") from error
    if _sha256(plaintext) != digest:
        raise SealError("plaintext SHA-256 does not match the authenticated seal")
    if expected_split is not None and expected_split != declared_split:
        raise SealError(
            f"sealed split is {declared_split!r}, expected {expected_split!r}"
        )
    return plaintext


__all__ = [
    "ExecutionContext",
    "FinalTestAccessDenied",
    "SEAL_FORMAT",
    "SealError",
    "SealInfo",
    "guard_record_splits",
    "inspect_seal",
    "is_final_test_split",
    "open_final_test",
    "seal_file",
]
