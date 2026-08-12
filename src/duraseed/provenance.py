"""Deterministic hashing and seed namespaces for scientific records.

This module is intentionally dependency-light apart from the project's schema
base.  Every derived seed is domain-separated before SHA-256 hashing, and all
content IDs use the same canonical JSON encoding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum, StrEnum
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from duraseed.schemas import StrictModel


CANONICAL_JSON_VERSION = "duraseed-canonical-json-v1"
SEED_DERIVATION_VERSION = "duraseed-seed-v1"
SHA256_PREFIX = "sha256:"
DERIVED_SEED_BITS = 63
MAX_ROOT_SEED = (1 << 63) - 1
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z", flags=re.ASCII)
_NAMESPACE_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z", flags=re.ASCII)


class ProvenanceError(ValueError):
    """Base class for deterministic provenance failures."""


class CanonicalJSONError(ProvenanceError):
    """Raised when a value cannot be represented by canonical JSON v1."""


class IntegrityError(ProvenanceError):
    """Raised when supplied content does not match its declared hash."""


class SeedNamespace(StrEnum):
    """Seed namespaces required by the reproducibility contract."""

    DATASET = "dataset"
    FAMILY_SELECTION = "family_selection"
    RANDOM_TEACHER_ALLOCATION = "random_teacher_allocation"
    TRAINING = "training"
    DATA_ORDER = "data_order"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    BOOTSTRAP = "bootstrap"


def _canonical_value(value: Any, *, location: str = "$") -> Any:
    """Convert supported values into an unambiguous JSON-compatible tree."""

    if isinstance(value, Enum):
        return _canonical_value(value.value, location=location)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError(f"non-finite float at {location}")
        # Normalizing negative zero avoids two encodings for the same numeric
        # value while retaining Python's deterministic shortest float spelling.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Fraction):
        return {
            "denominator": value.denominator,
            "numerator": value.numerator,
        }
    if isinstance(value, Path):
        raise CanonicalJSONError(
            f"filesystem paths require an explicit portable string at {location}"
        )
    if isinstance(value, BaseModel):
        model_fields = type(value).model_fields
        return {
            field_name: _canonical_value(
                getattr(value, field_name),
                location=f"{location}.{field_name}",
            )
            for field_name in model_fields
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(
                getattr(value, field.name),
                location=f"{location}.{field.name}",
            )
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError(
                    f"JSON object key at {location} must be a string"
                )
            canonical[key] = _canonical_value(item, location=f"{location}.{key}")
        return canonical
    if isinstance(value, (list, tuple)):
        return [
            _canonical_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        raise CanonicalJSONError(
            f"unordered collections are forbidden at {location}; sort explicitly"
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise CanonicalJSONError(
            f"binary data requires an explicit text encoding at {location}"
        )
    raise CanonicalJSONError(
        f"unsupported canonical JSON type at {location}: {type(value).__name__}"
    )


def canonical_json_value(value: Any) -> Any:
    """Return the canonical JSON-compatible representation of ``value``."""

    return _canonical_value(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value with stable field ordering and no insignificant bytes."""

    canonical = canonical_json_value(value)
    try:
        text = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:  # Defensive: normalization is strict.
        raise CanonicalJSONError(str(error)) from error
    return text.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase, algorithm-prefixed SHA-256 content ID."""

    if not isinstance(data, bytes):
        raise TypeError("sha256_bytes requires immutable bytes")
    return SHA256_PREFIX + hashlib.sha256(data).hexdigest()


def canonical_json_hash(value: Any) -> str:
    """Hash the canonical JSON representation of ``value``."""

    return sha256_bytes(canonical_json_bytes(value))


def validate_sha256_id(value: str) -> str:
    """Validate and return a canonical ``sha256:<hex>`` ID."""

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("expected canonical sha256:<64 lowercase hex> ID")
    return value


def verify_sha256(data: bytes, expected: str) -> None:
    """Raise ``IntegrityError`` unless ``data`` matches ``expected``."""

    validate_sha256_id(expected)
    actual = sha256_bytes(data)
    if actual != expected:
        raise IntegrityError(f"SHA-256 mismatch: expected {expected}, got {actual}")


def _validate_root_seed(root_seed: int) -> None:
    if (
        isinstance(root_seed, bool)
        or not isinstance(root_seed, int)
        or not 0 <= root_seed <= MAX_ROOT_SEED
    ):
        raise ValueError(f"root_seed must be an integer in [0, {MAX_ROOT_SEED}]")


def _validate_namespace(namespace: str) -> None:
    if not isinstance(namespace, str) or _NAMESPACE_RE.fullmatch(namespace) is None:
        raise ValueError("seed namespace must match [a-z][a-z0-9_.-]{0,127}")


def derive_namespaced_seed(
    root_seed: int,
    namespace: SeedNamespace | str,
    *coordinates: Any,
) -> int:
    """Derive one stable non-negative 63-bit seed using namespaced SHA-256.

    Coordinates make independent indexed streams possible without consuming a
    mutable PRNG sequence.  Their order is significant and encoded as canonical
    JSON, so ``(1, 23)`` cannot collide with ``(12, 3)``.
    """

    _validate_root_seed(root_seed)
    namespace_text = (
        namespace.value if isinstance(namespace, SeedNamespace) else namespace
    )
    _validate_namespace(namespace_text)
    material = canonical_json_bytes(
        {
            "coordinates": list(coordinates),
            "namespace": namespace_text,
            "root_seed": root_seed,
            "version": SEED_DERIVATION_VERSION,
        }
    )
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << DERIVED_SEED_BITS) - 1)


# Concise alias for call sites that already operate in provenance context.
derive_seed = derive_namespaced_seed


class SeedLedger(StrictModel):
    """Immutable, self-validating record of every required seed namespace."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    root_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    dataset_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    family_selection_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    random_teacher_allocation_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    training_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    data_order_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    generation_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    evaluation_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    bootstrap_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    derivation_version: str = SEED_DERIVATION_VERSION

    @field_validator(
        "root_seed",
        "dataset_seed",
        "family_selection_seed",
        "random_teacher_allocation_seed",
        "training_seed",
        "data_order_seed",
        "generation_seed",
        "evaluation_seed",
        "bootstrap_seed",
        mode="before",
    )
    @classmethod
    def seeds_must_be_exact_integers(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("seed values must be integers")
        return value

    @model_validator(mode="after")
    def derived_values_must_match_root(self) -> "SeedLedger":
        if self.derivation_version != SEED_DERIVATION_VERSION:
            raise ValueError("unsupported seed derivation version")
        for namespace in SeedNamespace:
            field_name = f"{namespace.value}_seed"
            expected = derive_namespaced_seed(self.root_seed, namespace)
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} does not match root_seed")
        values = [
            getattr(self, f"{namespace.value}_seed") for namespace in SeedNamespace
        ]
        if len(values) != len(set(values)):
            raise ValueError("derived seed namespaces unexpectedly collided")
        return self


def derive_seed_ledger(root_seed: int) -> SeedLedger:
    """Derive and validate all seed namespaces required by the spec."""

    _validate_root_seed(root_seed)
    values = {
        f"{namespace.value}_seed": derive_namespaced_seed(root_seed, namespace)
        for namespace in SeedNamespace
    }
    return SeedLedger(root_seed=root_seed, **values)


__all__ = [
    "CANONICAL_JSON_VERSION",
    "CanonicalJSONError",
    "DERIVED_SEED_BITS",
    "IntegrityError",
    "MAX_ROOT_SEED",
    "SEED_DERIVATION_VERSION",
    "SHA256_PREFIX",
    "SeedLedger",
    "SeedNamespace",
    "canonical_json_bytes",
    "canonical_json_hash",
    "canonical_json_value",
    "derive_namespaced_seed",
    "derive_seed",
    "derive_seed_ledger",
    "sha256_bytes",
    "validate_sha256_id",
    "verify_sha256",
]
