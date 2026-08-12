"""Canonical, content-addressed task and dataset manifests.

Task IDs in manifests follow the authoritative ``sha256:<hex>`` schema.  They
equal the split-independent semantic ``content_hash`` in v1; legacy in-memory
generator IDs such as ``tces-<hex>`` are accepted as inputs to builders but are
never persisted as manifest IDs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import os
import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import ConfigDict, Field, field_validator, model_validator

from duraseed.data.io import atomic_write_bytes
from duraseed.data.sealing import ExecutionContext, guard_record_splits
from duraseed.provenance import (
    SeedNamespace,
    canonical_json_bytes,
    canonical_json_hash,
    derive_namespaced_seed,
    validate_sha256_id,
)
from duraseed.schemas import (
    ExactRational,
    MAPSInstruction,
    MAPSTask,
    StrictModel,
    TCESConstraints,
    TCESOperator,
    TCESTask,
)


MANIFEST_SCHEMA_VERSION = "duraseed-dataset-manifest-v1"
TCES_PROMPT_TEMPLATE_ID = "tces_v1"
MAPS_PROMPT_TEMPLATE_ID = "maps_v1"
GENERATOR_VERSION = "1.0.0"
MAX_MANIFEST_BYTES = 256 * 1024 * 1024
_NONEMPTY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}\Z")
_TCES_OPERATOR_ORDER = ("+", "-", "*", "/")


class ManifestError(ValueError):
    """Base class for invalid or noncanonical manifest artifacts."""


class ManifestIntegrityError(ManifestError):
    """Raised when record or aggregate hashes do not match content."""


class NonCanonicalManifestError(ManifestError):
    """Raised when a manifest's bytes are not the canonical serialization."""


class DuplicateManifestKeyError(ManifestError):
    """Raised when JSON contains a duplicate object key."""


def _validated_json_mapping(
    value: dict[str, Any], *, field_name: str
) -> dict[str, Any]:
    """Validate and detach an ordinary JSON mapping without custom wrappers."""

    decoded = json.loads(canonical_json_bytes(value))
    if not isinstance(decoded, dict):  # Defensive: Pydantic declares a mapping.
        raise ValueError(f"{field_name} must be a JSON object")
    return decoded


def _validate_token(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _NONEMPTY_TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} is not a valid non-empty identifier")
    return value


class _ManifestModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _TaskRecordBase(_ManifestModel):
    task_id: str
    task_family: str
    split: str
    generator_version: str
    generator_seed: int = Field(ge=0, le=(1 << 63) - 1)
    item_index: int = Field(ge=0)
    accepted_attempt: int = Field(ge=0)
    prompt_template_id: str
    difficulty_features: dict[str, Any] = Field(default_factory=dict)
    content_hash: str

    @field_validator("task_id", "content_hash")
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)

    @field_validator("generator_seed", "item_index", "accepted_attempt", mode="before")
    @classmethod
    def provenance_numbers_are_integers(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("generator provenance numbers must be integers")
        return value

    @field_validator(
        "split",
        "generator_version",
        "prompt_template_id",
        mode="before",
    )
    @classmethod
    def identifiers_are_nonempty(cls, value: object, info: Any) -> object:
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be text")
        return _validate_token(value, info.field_name)

    @field_validator("difficulty_features", mode="after")
    @classmethod
    def difficulty_is_valid_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validated_json_mapping(value, field_name="difficulty_features")


def _tces_semantic_payload(
    *,
    operands: tuple[int, ...],
    target: ExactRational,
    allowed_ops: tuple[TCESOperator, ...],
    constraints: TCESConstraints,
    prompt_template_id: str,
) -> dict[str, Any]:
    allowed = frozenset(allowed_ops)
    return {
        "allowed_ops": [
            operator for operator in _TCES_OPERATOR_ORDER if operator in allowed
        ],
        "constraints": {
            "max_abs_intermediate": constraints.max_abs_intermediate,
            "max_answer_length": constraints.max_answer_length,
            "max_ast_nodes": constraints.max_ast_nodes,
            "max_denominator": constraints.max_denominator,
            "max_tree_depth": constraints.max_tree_depth,
            "use_each_once": constraints.use_each_once,
        },
        "operands": sorted(operands),
        "prompt_template_id": prompt_template_id,
        "target": {
            "denominator": target.denominator,
            "numerator": target.numerator,
        },
        "task_family": "tces",
        "version": 1,
    }


class TCESTaskManifestRecord(_TaskRecordBase):
    """The complete §6.12 TCES task record plus exact family provenance."""

    task_family: Literal["tces"] = "tces"
    operands: tuple[int, ...] = Field(min_length=1, max_length=16)
    target: ExactRational
    allowed_ops: tuple[TCESOperator, ...]
    constraints: TCESConstraints
    intended_family: str
    valid_family_ids: tuple[str, ...]
    valid_family_count: int = Field(ge=1)
    valid_expression_count: int = Field(ge=1)
    minimum_depth: int = Field(ge=1)

    @field_validator(
        "valid_family_count", "valid_expression_count", "minimum_depth", mode="before"
    )
    @classmethod
    def counts_are_integers(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("TCES count/depth fields must be integers")
        return value

    @field_validator("operands")
    @classmethod
    def operands_are_canonical(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if tuple(sorted(value)) != value:
            raise ValueError("manifest operands must use canonical sorted order")
        return value

    @field_validator("allowed_ops")
    @classmethod
    def operations_are_canonical(
        cls, value: tuple[TCESOperator, ...]
    ) -> tuple[TCESOperator, ...]:
        canonical = tuple(op for op in _TCES_OPERATOR_ORDER if op in set(value))
        if not value or len(value) != len(set(value)) or value != canonical:
            raise ValueError(
                "manifest allowed_ops must be unique and canonically ordered"
            )
        return value

    @field_validator("intended_family", mode="before")
    @classmethod
    def intended_family_is_nonempty(cls, value: object) -> object:
        if not isinstance(value, str) or not value:
            raise ValueError("intended_family must be non-empty text")
        return value

    @field_validator("valid_family_ids")
    @classmethod
    def family_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("valid_family_ids must be non-empty, unique, and sorted")
        return value

    @model_validator(mode="after")
    def hashes_and_counts_match_content(self) -> "TCESTaskManifestRecord":
        if self.valid_family_count != len(self.valid_family_ids):
            raise ValueError("valid_family_count does not match valid_family_ids")
        if self.intended_family not in self.valid_family_ids:
            raise ValueError("intended_family is absent from valid_family_ids")
        expected = canonical_json_hash(
            _tces_semantic_payload(
                operands=self.operands,
                target=self.target,
                allowed_ops=self.allowed_ops,
                constraints=self.constraints,
                prompt_template_id=self.prompt_template_id,
            )
        )
        if self.content_hash != expected or self.task_id != expected:
            raise ManifestIntegrityError(
                "TCES task_id/content_hash do not match semantic task content"
            )
        return self

    def to_task(self) -> TCESTask:
        return TCESTask(
            operands=self.operands,
            target=self.target,
            allowed_ops=self.allowed_ops,
            constraints=self.constraints,
            task_id=self.task_id,
            split=self.split,
        )


def _maps_semantic_payload(task: MAPSTask) -> dict[str, Any]:
    return {
        "allowed_instructions": sorted(
            instruction.canonical() for instruction in task.allowed_instructions
        ),
        "max_program_length": task.max_program_length,
        "modulus": task.modulus,
        "start": task.start,
        "target": task.target,
        "version": 1,
    }


def task_semantic_hash(task: TCESTask | MAPSTask) -> str:
    """Return the authenticated, split-independent manifest task identity."""

    if isinstance(task, TCESTask):
        return canonical_json_hash(
            _tces_semantic_payload(
                operands=tuple(sorted(task.operands)),
                target=task.target,
                allowed_ops=tuple(
                    operator
                    for operator in _TCES_OPERATOR_ORDER
                    if operator in set(task.allowed_ops)
                ),
                constraints=task.constraints,
                prompt_template_id=TCES_PROMPT_TEMPLATE_ID,
            )
        )
    if isinstance(task, MAPSTask):
        return canonical_json_hash(_maps_semantic_payload(task))
    raise TypeError(f"unsupported task type: {type(task).__name__}")


class MAPSTaskManifestRecord(_TaskRecordBase):
    """A solver-audited MAPS task record for Stage-B manifests."""

    task_family: Literal["maps"] = "maps"
    start: int
    modulus: int = Field(ge=2)
    target: int
    allowed_instructions: tuple[MAPSInstruction, ...] = Field(min_length=1)
    max_program_length: int = Field(ge=1)
    shortest_program_length: int = Field(ge=1)
    shortest_programs: tuple[str, ...]
    shortest_program_count: int = Field(ge=1)
    shortest_family_ids: tuple[str, ...]

    @field_validator(
        "start",
        "modulus",
        "target",
        "max_program_length",
        "shortest_program_length",
        "shortest_program_count",
        mode="before",
    )
    @classmethod
    def maps_numbers_are_integers(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("MAPS numeric manifest fields must be integers")
        return value

    @field_validator("shortest_programs", "shortest_family_ids")
    @classmethod
    def shortest_sets_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("shortest program/family sets must be unique and sorted")
        return value

    @field_validator("allowed_instructions")
    @classmethod
    def allowed_instructions_are_canonical(
        cls,
        value: tuple[MAPSInstruction, ...],
    ) -> tuple[MAPSInstruction, ...]:
        canonical = tuple(
            sorted(value, key=lambda instruction: instruction.canonical())
        )
        texts = [instruction.canonical() for instruction in value]
        if value != canonical or len(texts) != len(set(texts)):
            raise ValueError(
                "allowed_instructions must be unique and canonically ordered"
            )
        return value

    @model_validator(mode="after")
    def solver_metadata_and_hashes_match(self) -> "MAPSTaskManifestRecord":
        from duraseed.tasks.maps.interpreter import (  # Local: avoid import cycle.
            canonical_program,
            execute_program,
            parse_program,
            program_family_id,
        )
        from duraseed.tasks.maps.solver import solve_bfs

        if not (0 <= self.start < self.modulus and 0 <= self.target < self.modulus):
            raise ValueError("MAPS states must be normalized modulo the modulus")
        if self.shortest_program_length > self.max_program_length:
            raise ValueError("shortest program exceeds max_program_length")
        if self.shortest_program_count != len(self.shortest_programs):
            raise ValueError("shortest_program_count does not match programs")
        allowed = {instruction.canonical() for instruction in self.allowed_instructions}
        families: set[str] = set()
        for program_text in self.shortest_programs:
            program = parse_program(
                program_text,
                max_instructions=self.max_program_length,
            )
            if canonical_program(program) != program_text:
                raise ValueError("shortest program text is not canonical")
            if len(program) != self.shortest_program_length:
                raise ValueError("shortest program has the wrong length")
            if any(instruction.canonical() not in allowed for instruction in program):
                raise ValueError("shortest program uses an illegal instruction")
            if execute_program(self.start, self.modulus, program) != self.target:
                raise ValueError("shortest program does not execute to target")
            families.add(program_family_id(program))
        if tuple(sorted(families)) != self.shortest_family_ids:
            raise ValueError("shortest_family_ids do not match shortest programs")

        task = self.to_task()
        exact_solution = solve_bfs(
            task,
            max_programs=self.shortest_program_count,
        )
        if exact_solution.truncated:
            raise ValueError("declared MAPS shortest program set is incomplete")
        exact_programs = tuple(
            sorted(
                canonical_program(program)
                for program in exact_solution.shortest_programs
            )
        )
        if exact_solution.shortest_length != self.shortest_program_length:
            raise ValueError("declared MAPS shortest length is not globally shortest")
        if exact_programs != self.shortest_programs:
            raise ValueError("declared MAPS shortest program set is incomplete")
        if exact_solution.shortest_family_ids != self.shortest_family_ids:
            raise ValueError("declared MAPS shortest family set is incomplete")
        expected = canonical_json_hash(_maps_semantic_payload(task))
        if self.content_hash != expected or self.task_id != expected:
            raise ManifestIntegrityError(
                "MAPS task_id/content_hash do not match semantic task content"
            )
        return self

    def to_task(self) -> MAPSTask:
        return MAPSTask(
            start=self.start,
            modulus=self.modulus,
            target=self.target,
            allowed_instructions=self.allowed_instructions,
            max_program_length=self.max_program_length,
            task_id=self.task_id,
            split=self.split,
        )


TaskManifestRecord: TypeAlias = Annotated[
    TCESTaskManifestRecord | MAPSTaskManifestRecord,
    Field(discriminator="task_family"),
]


def _record_sort_key(record: _TaskRecordBase) -> tuple[str, str, int, int]:
    return (
        record.content_hash,
        record.task_id,
        record.item_index,
        record.accepted_attempt,
    )


def _records_hash(records: tuple[TaskManifestRecord, ...]) -> str:
    return canonical_json_hash(records)


def _manifest_identity_payload(manifest: "DatasetManifest") -> dict[str, Any]:
    """Return manifest content once, without re-embedding its aggregate hash."""

    return {
        "dataset_seed": manifest.dataset_seed,
        "generator_version": manifest.generator_version,
        "metadata": manifest.metadata,
        "name": manifest.name,
        "parent_manifest_id": manifest.parent_manifest_id,
        "record_count": manifest.record_count,
        "records": manifest.records,
        "root_seed": manifest.root_seed,
        "schema_version": manifest.schema_version,
        "split": manifest.split,
        "task_family": manifest.task_family,
    }


class DatasetManifest(_ManifestModel):
    """A homogeneous, canonically ordered immutable dataset manifest."""

    schema_version: Literal["duraseed-dataset-manifest-v1"] = MANIFEST_SCHEMA_VERSION
    manifest_id: str
    name: str
    task_family: Literal["tces", "maps"]
    split: str
    generator_version: str
    root_seed: int = Field(ge=0, le=(1 << 63) - 1)
    dataset_seed: int = Field(ge=0, le=(1 << 63) - 1)
    records: tuple[TaskManifestRecord, ...]
    record_count: int = Field(ge=0)
    records_hash: str
    parent_manifest_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("manifest_id", "records_hash")
    @classmethod
    def aggregate_hashes_are_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)

    @field_validator("parent_manifest_id")
    @classmethod
    def parent_hash_is_canonical(cls, value: str | None) -> str | None:
        return validate_sha256_id(value) if value is not None else None

    @field_validator("root_seed", "dataset_seed", "record_count", mode="before")
    @classmethod
    def aggregate_numbers_are_integers(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("manifest seeds/counts must be integers")
        return value

    @field_validator("name", "split", "generator_version", mode="before")
    @classmethod
    def aggregate_identifiers_are_valid(cls, value: object, info: Any) -> object:
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be text")
        return _validate_token(value, info.field_name)

    @field_validator("metadata", mode="after")
    @classmethod
    def metadata_is_valid_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validated_json_mapping(value, field_name="metadata")

    @model_validator(mode="after")
    def aggregate_contract_is_exact(self) -> "DatasetManifest":
        expected_dataset_seed = derive_namespaced_seed(
            self.root_seed,
            SeedNamespace.DATASET,
        )
        if self.dataset_seed != expected_dataset_seed:
            raise ManifestIntegrityError("dataset_seed does not match root_seed")
        if self.record_count != len(self.records):
            raise ManifestIntegrityError("record_count does not match records")
        if self.records != tuple(sorted(self.records, key=_record_sort_key)):
            raise NonCanonicalManifestError(
                "manifest records are not canonically ordered"
            )

        task_ids = [record.task_id for record in self.records]
        content_hashes = [record.content_hash for record in self.records]
        if len(task_ids) != len(set(task_ids)):
            raise ManifestIntegrityError("manifest contains duplicate task IDs")
        if len(content_hashes) != len(set(content_hashes)):
            raise ManifestIntegrityError("manifest contains duplicate task content")
        for record in self.records:
            if record.task_family != self.task_family:
                raise ManifestIntegrityError("record task_family differs from manifest")
            if record.split != self.split:
                raise ManifestIntegrityError("record split differs from manifest")
            if record.generator_version != self.generator_version:
                raise ManifestIntegrityError(
                    "record generator_version differs from manifest"
                )
        expected_generator_seed = derive_namespaced_seed(
            self.root_seed,
            f"dataset.{self.task_family}.split",
            self.split,
        )
        if any(
            record.generator_seed != expected_generator_seed for record in self.records
        ):
            raise ManifestIntegrityError(
                "record generator_seed does not match root/family/split namespace"
            )

        expected_records_hash = _records_hash(self.records)
        if self.records_hash != expected_records_hash:
            raise ManifestIntegrityError("records_hash does not match records")
        expected_manifest_id = canonical_json_hash(_manifest_identity_payload(self))
        if self.manifest_id != expected_manifest_id:
            raise ManifestIntegrityError("manifest_id does not match manifest content")
        return self


def build_tces_record(
    instance: Any,
    *,
    difficulty_features: Mapping[str, Any] | None = None,
) -> TCESTaskManifestRecord:
    """Build an authenticated §6.12 record from ``GeneratedTCESInstance``."""

    task: TCESTask = instance.task
    if task.split is None:
        raise ValueError("a manifest record requires an explicit split")
    if not instance.enumeration.complete:
        raise ValueError("cannot manifest an incomplete TCES enumeration")
    family_ids = tuple(sorted(instance.valid_family_ids))
    minimum_depth = instance.enumeration.shortest_depth
    if minimum_depth is None:
        raise ValueError("cannot manifest an unsolved TCES instance")
    semantic_hash = canonical_json_hash(
        _tces_semantic_payload(
            operands=tuple(sorted(task.operands)),
            target=task.target,
            allowed_ops=tuple(
                op for op in _TCES_OPERATOR_ORDER if op in set(task.allowed_ops)
            ),
            constraints=task.constraints,
            prompt_template_id=TCES_PROMPT_TEMPLATE_ID,
        )
    )
    if instance.content_hash != semantic_hash:
        raise ValueError("generated TCES instance content_hash is inconsistent")
    if task.task_id is not None and task.task_id not in {
        semantic_hash,
        "tces-" + semantic_hash.removeprefix("sha256:"),
    }:
        raise ValueError("generated TCES task_id is inconsistent")
    return TCESTaskManifestRecord(
        task_id=semantic_hash,
        task_family="tces",
        split=task.split,
        generator_version=GENERATOR_VERSION,
        generator_seed=instance.root_seed,
        item_index=instance.item_index,
        accepted_attempt=instance.accepted_attempt,
        operands=tuple(sorted(task.operands)),
        target=task.target,
        allowed_ops=tuple(
            op for op in _TCES_OPERATOR_ORDER if op in set(task.allowed_ops)
        ),
        constraints=task.constraints,
        intended_family=instance.intended_family,
        valid_family_ids=family_ids,
        valid_family_count=instance.valid_family_count,
        valid_expression_count=instance.valid_expression_count,
        minimum_depth=minimum_depth,
        difficulty_features=dict(difficulty_features or {}),
        prompt_template_id=TCES_PROMPT_TEMPLATE_ID,
        content_hash=semantic_hash,
    )


def build_maps_record(
    instance: Any,
    *,
    split: str,
    difficulty_features: Mapping[str, Any] | None = None,
) -> MAPSTaskManifestRecord:
    """Build a solver-audited record from ``GeneratedMAPSInstance``."""

    from duraseed.tasks.maps.interpreter import canonical_program

    task: MAPSTask = instance.task
    if task.split is not None and task.split != split:
        raise ValueError("split override does not match the generated MAPS task")
    if instance.solution.truncated:
        raise ValueError("cannot manifest a truncated MAPS shortest-program set")
    semantic_hash = canonical_json_hash(_maps_semantic_payload(task))
    if task.task_id is not None and task.task_id not in {
        semantic_hash,
        "maps-" + semantic_hash.removeprefix("sha256:"),
    }:
        raise ValueError("generated MAPS task_id is inconsistent")
    programs = tuple(
        sorted(
            canonical_program(program)
            for program in instance.solution.shortest_programs
        )
    )
    return MAPSTaskManifestRecord(
        task_id=semantic_hash,
        task_family="maps",
        split=split,
        generator_version=GENERATOR_VERSION,
        generator_seed=instance.root_seed,
        item_index=instance.item_index,
        accepted_attempt=instance.accepted_attempt,
        start=task.start,
        modulus=task.modulus,
        target=task.target,
        allowed_instructions=tuple(
            sorted(
                task.allowed_instructions,
                key=lambda instruction: instruction.canonical(),
            )
        ),
        max_program_length=task.max_program_length,
        shortest_program_length=instance.solution.shortest_length,
        shortest_programs=programs,
        shortest_program_count=len(programs),
        shortest_family_ids=tuple(sorted(instance.solution.shortest_family_ids)),
        difficulty_features=dict(difficulty_features or {}),
        prompt_template_id=MAPS_PROMPT_TEMPLATE_ID,
        content_hash=semantic_hash,
    )


def build_manifest(
    *,
    name: str,
    split: str,
    generator_version: str,
    root_seed: int,
    records: tuple[TaskManifestRecord, ...] | list[TaskManifestRecord],
    task_family: Literal["tces", "maps"] | None = None,
    parent_manifest_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetManifest:
    """Sort records canonically and compute aggregate and manifest hashes."""

    ordered = tuple(sorted(tuple(records), key=_record_sort_key))
    if task_family is None:
        if not ordered:
            raise ValueError("task_family is required for an empty manifest")
        task_family = ordered[0].task_family
    records_hash = _records_hash(ordered)
    values: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "sha256:" + "0" * 64,
        "name": name,
        "task_family": task_family,
        "split": split,
        "generator_version": generator_version,
        "root_seed": root_seed,
        "dataset_seed": derive_namespaced_seed(root_seed, SeedNamespace.DATASET),
        "records": ordered,
        "record_count": len(ordered),
        "records_hash": records_hash,
        "parent_manifest_id": parent_manifest_id,
        "metadata": dict(metadata or {}),
    }
    provisional = DatasetManifest.model_construct(**values)
    values["manifest_id"] = canonical_json_hash(_manifest_identity_payload(provisional))
    return DatasetManifest(**values)


def manifest_bytes(manifest: DatasetManifest) -> bytes:
    """Return the one canonical on-disk representation (with final newline)."""

    # Revalidate even model-constructed or otherwise untrusted instances.
    validated = DatasetManifest.model_validate(manifest.model_dump(mode="python"))
    return canonical_json_bytes(validated) + b"\n"


def write_manifest(path: str | os.PathLike[str], manifest: DatasetManifest) -> Path:
    """Atomically write the canonical manifest representation."""

    return atomic_write_bytes(path, manifest_bytes(manifest))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateManifestKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_manifest(
    path: str | os.PathLike[str],
    *,
    context: ExecutionContext | str,
    max_bytes: int = MAX_MANIFEST_BYTES,
) -> DatasetManifest:
    """Read canonical bytes and enforce embedded split access policy."""

    # Reject unknown roles before reading or validating any task content.
    guard_record_splits((), context)
    source = Path(path)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if source.stat().st_size > max_bytes:
        raise ManifestError(f"manifest exceeds {max_bytes} bytes")
    raw = source.read_bytes()
    if len(raw) > max_bytes:
        raise ManifestError(f"manifest exceeds {max_bytes} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ManifestError("manifest is not valid UTF-8") from error
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ManifestError("manifest is not valid JSON") from error
    # JSON-mode validation permits JSON arrays for immutable tuple fields while
    # retaining strict scalar types. Duplicate keys were already rejected.
    manifest = DatasetManifest.model_validate_json(canonical_json_bytes(value))
    expected = manifest_bytes(manifest)
    if raw != expected:
        raise NonCanonicalManifestError("manifest bytes are not canonical JSON v1")
    guard_record_splits(
        (manifest.split, *(record.split for record in manifest.records)),
        context,
    )
    return manifest


__all__ = [
    "DatasetManifest",
    "DuplicateManifestKeyError",
    "MANIFEST_SCHEMA_VERSION",
    "MAPSTaskManifestRecord",
    "ManifestError",
    "ManifestIntegrityError",
    "NonCanonicalManifestError",
    "TCESTaskManifestRecord",
    "TaskManifestRecord",
    "build_manifest",
    "build_maps_record",
    "build_tces_record",
    "manifest_bytes",
    "read_manifest",
    "task_semantic_hash",
    "write_manifest",
]
