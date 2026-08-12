"""Fail-closed leakage auditing for immutable task manifests.

The auditor deliberately works by attribute/key lookup rather than importing a
particular manifest schema.  It accepts generated TCES instances, Pydantic
manifest records, dictionaries, aggregate manifests exposing ``records``, or a
mapping from declared split name to any of those record collections.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
import re
from .splits import EVALUATION_SPLITS, TEACHER_SPLITS


class LeakageCode(StrEnum):
    """Stable machine-readable categories emitted by the leakage auditor."""

    INVALID_RECORD = "invalid_record"
    DUPLICATE_OPERANDS_TARGET = "duplicate_operands_target"
    DUPLICATE_CONTENT_HASH = "duplicate_content_hash"
    DUPLICATE_TASK_ID = "duplicate_task_id"
    TASK_ACROSS_SPLITS = "task_across_splits"
    TEACHER_EVALUATION_NUMERIC_OVERLAP = "teacher_evaluation_numeric_overlap"
    FORBIDDEN_FAMILY_OVERLAP = "forbidden_family_overlap"


@dataclass(frozen=True, slots=True, order=True)
class RecordLocation:
    """Stable location of one audited record."""

    split: str
    position: int
    task_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "position": self.position,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One exact leakage or malformed-record finding."""

    code: LeakageCode
    message: str
    key: str
    locations: tuple[RecordLocation, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": self.message,
            "key": self.key,
            "locations": [location.to_dict() for location in self.locations],
        }


class LeakageAuditError(RuntimeError):
    """Raised when a caller requires a clean report and findings exist."""

    def __init__(self, report: "LeakageAuditReport") -> None:
        self.report = report
        summary = "; ".join(
            f"{finding.code.value}: {finding.message}"
            for finding in report.findings[:5]
        )
        if len(report.findings) > 5:
            summary += f"; and {len(report.findings) - 5} more"
        super().__init__(
            f"leakage audit failed with {len(report.findings)} finding(s): {summary}"
        )


@dataclass(frozen=True, slots=True)
class LeakageAuditReport:
    """Complete deterministic result of one leakage audit."""

    record_count: int
    audited_splits: tuple[str, ...]
    findings: tuple[AuditFinding, ...]

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    def findings_for(self, code: LeakageCode) -> tuple[AuditFinding, ...]:
        return tuple(finding for finding in self.findings if finding.code is code)

    def assert_clean(self) -> "LeakageAuditReport":
        """Return this report when clean, otherwise raise with the full report."""

        if not self.clean:
            raise LeakageAuditError(self)
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "audited_splits": list(self.audited_splits),
            "clean": self.clean,
            "finding_count": self.finding_count,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True, slots=True)
class FamilyDisjointnessRule:
    """A declared pair of family sets that must have empty intersection."""

    name: str
    left_label: str
    left_family_ids: frozenset[str]
    right_label: str
    right_family_ids: frozenset[str]

    def __post_init__(self) -> None:
        for field_name in ("name", "left_label", "right_label"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        left = frozenset(self.left_family_ids)
        right = frozenset(self.right_family_ids)
        if any(not isinstance(value, str) or not value for value in (*left, *right)):
            raise ValueError("family IDs must be non-empty strings")
        object.__setattr__(self, "left_family_ids", left)
        object.__setattr__(self, "right_family_ids", right)

    @classmethod
    def from_records(
        cls,
        *,
        name: str,
        left_label: str,
        left_records: Iterable[object],
        right_label: str,
        right_records: Iterable[object],
    ) -> "FamilyDisjointnessRule":
        """Build a rule from complete family sets on generated/manifest records."""

        return cls(
            name=name,
            left_label=left_label,
            left_family_ids=collect_family_ids(left_records),
            right_label=right_label,
            right_family_ids=collect_family_ids(right_records),
        )


@dataclass(frozen=True, slots=True)
class _RecordView:
    location: RecordLocation
    numeric_key: tuple[object, ...]
    identifiers: tuple[tuple[str, str], ...]


_HASH_FIELD_CODES: Mapping[str, LeakageCode] = {
    "content_hash": LeakageCode.DUPLICATE_CONTENT_HASH,
    "task_id": LeakageCode.DUPLICATE_TASK_ID,
}
_RECORD_KEYS = frozenset(
    {
        "task_id",
        "content_hash",
        "operands",
        "start",
        "task",
    }
)
_HEX_DIGEST = re.compile(r"[0-9a-fA-F]{64}")


def _lookup(record: object, name: str, default: object = None) -> object:
    if isinstance(record, Mapping) and name in record:
        return record[name]
    value = getattr(record, name, default)
    if value is not default:
        return value
    task = (
        record.get("task")
        if isinstance(record, Mapping)
        else getattr(record, "task", None)
    )
    if isinstance(task, Mapping) and name in task:
        return task[name]
    return getattr(task, name, default)


def _looks_like_record(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(_RECORD_KEYS.intersection(value))
    return any(hasattr(value, key) for key in _RECORD_KEYS)


def _records_from_group(group: object) -> Iterable[object]:
    records = getattr(group, "records", None)
    if records is not None and not _looks_like_record(group):
        return records
    if _looks_like_record(group):
        return (group,)
    if isinstance(group, (str, bytes)):
        return (group,)
    try:
        return iter(group)  # type: ignore[arg-type]
    except TypeError:
        return (group,)


def _flatten_records(source: object) -> list[tuple[object, str | None]]:
    flattened: list[tuple[object, str | None]] = []
    if isinstance(source, Mapping) and not _looks_like_record(source):
        for declared_split in sorted(source, key=str):
            group = source[declared_split]
            for record in _records_from_group(group):
                flattened.append((record, str(declared_split)))
        return flattened

    records = getattr(source, "records", None)
    if records is not None and not _looks_like_record(source):
        declared_split = getattr(source, "split", None)
        for record in records:
            flattened.append((record, declared_split))
        return flattened

    for record in _records_from_group(source):
        nested_records = getattr(record, "records", None)
        if nested_records is not None and not _looks_like_record(record):
            declared_split = getattr(record, "split", None)
            for nested in nested_records:
                flattened.append((nested, declared_split))
        else:
            flattened.append((record, None))
    return flattened


def _normalized_fraction(target: object) -> tuple[int, int] | None:
    if isinstance(target, Mapping):
        numerator = target.get("numerator")
        denominator = target.get("denominator", 1)
    else:
        numerator = getattr(target, "numerator", None)
        denominator = getattr(target, "denominator", None)
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator == 0
    ):
        return None
    value = Fraction(numerator, denominator)
    return value.numerator, value.denominator


def _numeric_key(record: object) -> tuple[object, ...] | None:
    operands = _lookup(record, "operands")
    target = _lookup(record, "target")
    if isinstance(operands, (tuple, list)) and operands:
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in operands
        ):
            return None
        rational = _normalized_fraction(target)
        if rational is None:
            return None
        return ("tces", tuple(sorted(operands)), *rational)

    start = _lookup(record, "start")
    modulus = _lookup(record, "modulus")
    if (
        all(
            not isinstance(value, bool) and isinstance(value, int)
            for value in (start, modulus, target)
        )
        and modulus >= 2
    ):
        return ("maps", start % modulus, modulus, target % modulus)
    return None


def _identifiers(record: object) -> tuple[tuple[str, str], ...]:
    identifiers: list[tuple[str, str]] = []
    for field_name in _HASH_FIELD_CODES:
        value = _lookup(record, field_name)
        if isinstance(value, str) and value:
            identifiers.append((field_name, value))
    return tuple(identifiers)


def _identity_token(value: str) -> str:
    """Normalize common hash-prefixed task/content IDs to their digest."""

    if value.startswith("sha256:") and _HEX_DIGEST.fullmatch(value[7:]):
        return value[7:].lower()
    suffix = value.rsplit("-", maxsplit=1)[-1]
    if _HEX_DIGEST.fullmatch(suffix):
        return suffix.lower()
    return value


def _key_text(value: object) -> str:
    return repr(value)


def _ordered_locations(views: Iterable[_RecordView]) -> tuple[RecordLocation, ...]:
    return tuple(sorted((view.location for view in views)))


def record_family_ids(record: object) -> frozenset[str]:
    """Return a record's complete valid family set or fail closed."""

    values = _lookup(record, "valid_family_ids")
    if values is None:
        enumeration = _lookup(record, "enumeration")
        if enumeration is not None:
            if getattr(enumeration, "complete", None) is not True:
                raise ValueError("cannot use families from an incomplete enumeration")
            values = getattr(enumeration, "family_ids", None)
    if isinstance(values, (str, bytes)) or values is None:
        raise ValueError("record does not expose a complete valid_family_ids set")
    try:
        family_ids = frozenset(values)
    except TypeError as error:
        raise ValueError("valid_family_ids must be iterable") from error
    if any(not isinstance(value, str) or not value for value in family_ids):
        raise ValueError("valid_family_ids must contain non-empty strings")
    declared_count = _lookup(record, "valid_family_count")
    if declared_count is not None:
        if isinstance(declared_count, bool) or not isinstance(declared_count, int):
            raise ValueError("valid_family_count must be an integer")
        if declared_count != len(family_ids):
            raise ValueError("valid_family_count disagrees with valid_family_ids")
    return family_ids


def collect_family_ids(records: Iterable[object]) -> frozenset[str]:
    """Union complete family sets from records for a disjointness declaration."""

    collected: set[str] = set()
    for record in records:
        collected.update(record_family_ids(record))
    return frozenset(collected)


def audit_leakage(
    records: object,
    *,
    teacher_splits: Iterable[str] = TEACHER_SPLITS,
    evaluation_splits: Iterable[str] = EVALUATION_SPLITS,
    forbidden_family_rules: Iterable[FamilyDisjointnessRule] = (),
) -> LeakageAuditReport:
    """Audit records for every declared Vertical-Slice-2 leakage condition."""

    teacher = frozenset(teacher_splits)
    evaluation = frozenset(evaluation_splits)
    if any(
        not isinstance(value, str) or not value for value in (*teacher, *evaluation)
    ):
        raise ValueError("teacher/evaluation split names must be non-empty strings")
    if teacher.intersection(evaluation):
        raise ValueError("teacher and evaluation split declarations must be disjoint")

    flattened = _flatten_records(records)
    findings: list[AuditFinding] = []
    views: list[_RecordView] = []
    for position, (record, declared_split) in enumerate(flattened):
        record_split = _lookup(record, "split")
        if record_split is not None and (
            not isinstance(record_split, str) or not record_split.strip()
        ):
            findings.append(
                AuditFinding(
                    code=LeakageCode.INVALID_RECORD,
                    message="record split must be a non-empty string",
                    key=f"position:{position}",
                    locations=(
                        RecordLocation(declared_split or "<missing>", position, None),
                    ),
                )
            )
        if declared_split is not None and isinstance(record_split, str):
            if record_split != declared_split:
                findings.append(
                    AuditFinding(
                        code=LeakageCode.INVALID_RECORD,
                        message=(
                            f"record split {record_split!r} disagrees with declared "
                            f"collection split {declared_split!r}"
                        ),
                        key=f"position:{position}",
                        locations=(RecordLocation(declared_split, position, None),),
                    )
                )
        split = declared_split if declared_split is not None else record_split
        if not isinstance(split, str) or not split:
            split = "<missing>"
            findings.append(
                AuditFinding(
                    code=LeakageCode.INVALID_RECORD,
                    message="record has no non-empty split",
                    key=f"position:{position}",
                    locations=(RecordLocation(split, position, None),),
                )
            )

        task_id_value = _lookup(record, "task_id")
        task_id = task_id_value if isinstance(task_id_value, str) else None
        location = RecordLocation(split, position, task_id)
        numeric = _numeric_key(record)
        if numeric is None:
            findings.append(
                AuditFinding(
                    code=LeakageCode.INVALID_RECORD,
                    message="record has no valid exact numeric task identity",
                    key=f"position:{position}",
                    locations=(location,),
                )
            )
            # A malformed record is unsafe to compare and therefore cannot make
            # a report clean, but valid identifiers remain useful diagnostics.
            numeric = ("invalid", position)
        identifiers = _identifiers(record)
        if not identifiers:
            findings.append(
                AuditFinding(
                    code=LeakageCode.INVALID_RECORD,
                    message="record has no semantic/content/task identifier",
                    key=f"position:{position}",
                    locations=(location,),
                )
            )
        views.append(
            _RecordView(
                location=location,
                numeric_key=numeric,
                identifiers=identifiers,
            )
        )

    numeric_groups: dict[tuple[object, ...], list[_RecordView]] = defaultdict(list)
    for view in views:
        if view.numeric_key[0] != "invalid":
            numeric_groups[view.numeric_key].append(view)

    for key in sorted(numeric_groups, key=_key_text):
        group = numeric_groups[key]
        if len(group) < 2:
            continue
        locations = _ordered_locations(group)
        findings.append(
            AuditFinding(
                code=LeakageCode.DUPLICATE_OPERANDS_TARGET,
                message="identical numeric task appears more than once",
                key=_key_text(key),
                locations=locations,
            )
        )
        splits = {view.location.split for view in group}
        if len(splits) > 1:
            findings.append(
                AuditFinding(
                    code=LeakageCode.TASK_ACROSS_SPLITS,
                    message="identical numeric task appears across splits",
                    key="numeric:" + _key_text(key),
                    locations=locations,
                )
            )
        if splits.intersection(teacher) and splits.intersection(evaluation):
            findings.append(
                AuditFinding(
                    code=LeakageCode.TEACHER_EVALUATION_NUMERIC_OVERLAP,
                    message="teacher and evaluation splits share a numeric task",
                    key=_key_text(key),
                    locations=locations,
                )
            )

    for field_name, code in _HASH_FIELD_CODES.items():
        groups: dict[str, list[_RecordView]] = defaultdict(list)
        for view in views:
            for observed_name, value in view.identifiers:
                if observed_name == field_name:
                    groups[value].append(view)
        for value in sorted(groups):
            group = groups[value]
            if len(group) < 2:
                continue
            findings.append(
                AuditFinding(
                    code=code,
                    message=f"duplicate {field_name} appears more than once",
                    key=value,
                    locations=_ordered_locations(group),
                )
            )

    identity_groups: dict[str, dict[RecordLocation, _RecordView]] = defaultdict(dict)
    for view in views:
        for _, value in view.identifiers:
            identity_groups[_identity_token(value)][view.location] = view
    for token in sorted(identity_groups):
        group = tuple(identity_groups[token].values())
        if len(group) < 2:
            continue
        if len({view.location.split for view in group}) < 2:
            continue
        findings.append(
            AuditFinding(
                code=LeakageCode.TASK_ACROSS_SPLITS,
                message="stable task identity appears across splits",
                key="identifier:" + token,
                locations=_ordered_locations(group),
            )
        )

    for rule in forbidden_family_rules:
        if not isinstance(rule, FamilyDisjointnessRule):
            raise TypeError(
                "forbidden_family_rules must contain FamilyDisjointnessRule"
            )
        for family_id in sorted(
            rule.left_family_ids.intersection(rule.right_family_ids)
        ):
            findings.append(
                AuditFinding(
                    code=LeakageCode.FORBIDDEN_FAMILY_OVERLAP,
                    message=(
                        f"declared-disjoint family sets {rule.left_label!r} and "
                        f"{rule.right_label!r} overlap under rule {rule.name!r}"
                    ),
                    key=family_id,
                )
            )

    findings.sort(
        key=lambda finding: (
            finding.code.value,
            finding.key,
            tuple(finding.locations),
        )
    )
    return LeakageAuditReport(
        record_count=len(flattened),
        audited_splits=tuple(sorted({view.location.split for view in views})),
        findings=tuple(findings),
    )


__all__ = [
    "AuditFinding",
    "FamilyDisjointnessRule",
    "LeakageAuditError",
    "LeakageAuditReport",
    "LeakageCode",
    "RecordLocation",
    "audit_leakage",
    "collect_family_ids",
    "record_family_ids",
]
