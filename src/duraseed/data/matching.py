"""Deterministic pre-treatment matching for teacher-information controls."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json
import math
from typing import Sequence

from duraseed.provenance import MAX_ROOT_SEED


_OPERATOR_ORDER = {"+": 0, "-": 1, "*": 2, "/": 3}
_MAX_EXACT_MATCH_STATES = 500_000


@dataclass(frozen=True, slots=True)
class StructuralCovariates:
    """Pre-treatment variables permitted in the random-allocation strata."""

    tree_depth: int
    operand_count: int
    operator_multiset: tuple[str, ...]
    noncommutative_count: int
    fractional_intermediate: bool
    target_magnitude_bin: str
    valid_family_count_bin: str
    teacher_trace_token_bin: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.tree_depth, bool)
            or not isinstance(self.tree_depth, int)
            or self.tree_depth < 1
        ):
            raise ValueError("tree_depth must be a positive integer")
        if (
            isinstance(self.operand_count, bool)
            or not isinstance(self.operand_count, int)
            or self.operand_count < 2
        ):
            raise ValueError("operand_count must be an integer of at least two")
        if self.tree_depth > self.operand_count:
            raise ValueError("tree_depth cannot exceed operand_count for a full tree")
        if not isinstance(self.operator_multiset, tuple):
            raise ValueError("operator_multiset must be an immutable tuple")
        if len(self.operator_multiset) != self.operand_count - 1:
            raise ValueError("a full binary tree has operand_count - 1 operators")
        if any(operator not in _OPERATOR_ORDER for operator in self.operator_multiset):
            raise ValueError("operator_multiset contains an unsupported operator")
        normalized = tuple(
            sorted(self.operator_multiset, key=_OPERATOR_ORDER.__getitem__)
        )
        expected_noncommutative = sum(operator in {"-", "/"} for operator in normalized)
        if (
            isinstance(self.noncommutative_count, bool)
            or not isinstance(self.noncommutative_count, int)
            or self.noncommutative_count != expected_noncommutative
        ):
            raise ValueError(
                "noncommutative_count disagrees with the operator multiset"
            )
        if not isinstance(self.fractional_intermediate, bool):
            raise ValueError("fractional_intermediate must be boolean")
        for name, value in (
            ("target_magnitude_bin", self.target_magnitude_bin),
            ("valid_family_count_bin", self.valid_family_count_bin),
            ("teacher_trace_token_bin", self.teacher_trace_token_bin),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "operator_multiset", normalized)

    @property
    def stratum_key(self) -> tuple[object, ...]:
        return (
            self.tree_depth,
            self.operand_count,
            self.operator_multiset,
            self.noncommutative_count,
            self.fractional_intermediate,
            self.target_magnitude_bin,
            self.valid_family_count_bin,
            self.teacher_trace_token_bin,
        )


@dataclass(frozen=True, slots=True)
class TeacherExampleRecord:
    """One allocation candidate containing no post-treatment outcome fields."""

    record_id: str
    family_id: str
    allocation_group: str
    covariates: StructuralCovariates
    teacher_prompt_tokens: int
    teacher_target_tokens: int
    teacher_trace_format: str
    measurement_stage: str = "pre_treatment"

    def __post_init__(self) -> None:
        for name, value in (
            ("record_id", self.record_id),
            ("family_id", self.family_id),
            ("allocation_group", self.allocation_group),
            ("teacher_trace_format", self.teacher_trace_format),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.covariates, StructuralCovariates):
            raise ValueError("covariates must be StructuralCovariates")
        if self.measurement_stage != "pre_treatment":
            raise ValueError("matching records must contain pre-treatment data only")
        if (
            isinstance(self.teacher_prompt_tokens, bool)
            or not isinstance(self.teacher_prompt_tokens, int)
            or self.teacher_prompt_tokens < 1
        ):
            raise ValueError("teacher_prompt_tokens must be positive")
        if (
            isinstance(self.teacher_target_tokens, bool)
            or not isinstance(self.teacher_target_tokens, int)
            or self.teacher_target_tokens < 1
        ):
            raise ValueError("teacher_target_tokens must be positive")

    @property
    def stratum_key(self) -> tuple[object, ...]:
        return self.covariates.stratum_key + (self.teacher_trace_format,)


@dataclass(frozen=True, slots=True)
class TokenLedger:
    example_count: int
    prompt_tokens: int
    target_tokens: int

    @classmethod
    def from_records(cls, records: Sequence[TeacherExampleRecord]) -> "TokenLedger":
        return cls(
            example_count=len(records),
            prompt_tokens=sum(record.teacher_prompt_tokens for record in records),
            target_tokens=sum(record.teacher_target_tokens for record in records),
        )


@dataclass(frozen=True, slots=True)
class StratumMismatch:
    stratum: str
    required_count: int
    available_count: int
    selected_count: int


@dataclass(frozen=True, slots=True)
class StandardizedDifference:
    covariate: str
    level: str | None
    target_value: float | None
    matched_value: float | None
    standardized_difference: float | None
    undefined_due_to_zero_variance: bool = False

    @property
    def absolute(self) -> float:
        if self.standardized_difference is None:
            return math.inf
        return abs(self.standardized_difference)


@dataclass(frozen=True, slots=True)
class MatchingReport:
    allocation_seed: int
    target_record_ids: tuple[str, ...]
    matched_record_ids: tuple[str, ...]
    target_ledger: TokenLedger
    matched_ledger: TokenLedger
    exact_example_count: bool
    exact_prompt_token_budget: bool
    target_optimizer_updates: int
    matched_optimizer_updates: int
    exact_optimizer_updates: bool
    max_target_token_relative_difference: float
    target_token_relative_difference: float
    stratum_mismatches: tuple[StratumMismatch, ...]
    standardized_differences: tuple[StandardizedDifference, ...]
    failures: tuple[str, ...]
    passed: bool

    @property
    def maximum_absolute_standardized_difference(self) -> float:
        return max(
            (statistic.absolute for statistic in self.standardized_differences),
            default=0.0,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["maximum_absolute_standardized_difference"] = (
            None
            if math.isinf(self.maximum_absolute_standardized_difference)
            else self.maximum_absolute_standardized_difference
        )
        return payload


@dataclass(frozen=True, slots=True)
class MatchingAllocation:
    records: tuple[TeacherExampleRecord, ...]
    report: MatchingReport


@dataclass(frozen=True, slots=True)
class FamilyBlockRecord:
    """One pre-treatment row for the production family-block matcher."""

    record: TeacherExampleRecord
    fractional_profile: tuple[str, ...]
    absolute_target: float
    valid_family_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.record, TeacherExampleRecord):
            raise TypeError("record must be a TeacherExampleRecord")
        if (
            not isinstance(self.fractional_profile, tuple)
            or not self.fractional_profile
            or any(marker not in {"I", "F"} for marker in self.fractional_profile)
        ):
            raise ValueError("fractional_profile must be an immutable I/F tuple")
        if len(self.fractional_profile) != len(
            self.record.covariates.operator_multiset
        ):
            raise ValueError(
                "fractional_profile must align with the family operator count"
            )
        if self.record.covariates.fractional_intermediate != (
            "F" in self.fractional_profile
        ):
            raise ValueError(
                "fractional_profile disagrees with fractional_intermediate"
            )
        if (
            isinstance(self.absolute_target, bool)
            or not isinstance(self.absolute_target, (int, float))
            or not math.isfinite(self.absolute_target)
            or self.absolute_target < 0
        ):
            raise ValueError("absolute_target must be a finite non-negative number")
        if (
            isinstance(self.valid_family_count, bool)
            or not isinstance(self.valid_family_count, int)
            or self.valid_family_count < 1
        ):
            raise ValueError("valid_family_count must be a positive integer")
        object.__setattr__(self, "absolute_target", float(self.absolute_target))

    @property
    def structure_key(self) -> tuple[object, ...]:
        """Exact production stratum, excluding diagnostic-only quantities."""

        covariates = self.record.covariates
        return (
            covariates.tree_depth,
            covariates.operand_count,
            covariates.operator_multiset,
            covariates.noncommutative_count,
            self.fractional_profile,
            self.record.teacher_trace_format,
        )


@dataclass(frozen=True, slots=True)
class FamilyBlockMatchPolicy:
    """Frozen scientific constraints for the 12-family teacher control."""

    dose: int
    allocation_seed: int
    policy_id: str = "core_family_v1"
    target_family_count: int = 12
    max_target_token_relative_difference: float = 0.02
    max_search_states: int = _MAX_EXACT_MATCH_STATES
    target_allocation_group: str = "boundary"
    random_allocation_group: str = "random"

    def __post_init__(self) -> None:
        if self.policy_id != "core_family_v1":
            raise ValueError("production family-block policy_id must be core_family_v1")
        if (
            isinstance(self.dose, bool)
            or not isinstance(self.dose, int)
            or not 1 <= self.dose <= 16
        ):
            raise ValueError("dose must be an integer in [1, 16]")
        if self.target_family_count != 12:
            raise ValueError(
                "the production matcher requires exactly 12 target families"
            )
        _validate_match_options(
            self.allocation_seed,
            self.max_target_token_relative_difference,
            0,
            0,
        )
        if self.max_target_token_relative_difference > 0.02:
            raise ValueError("the production target-token tolerance cannot exceed 2%")
        if (
            isinstance(self.max_search_states, bool)
            or not isinstance(self.max_search_states, int)
            or self.max_search_states < 1
        ):
            raise ValueError("max_search_states must be a positive integer")
        for name, value in (
            ("target_allocation_group", self.target_allocation_group),
            ("random_allocation_group", self.random_allocation_group),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty text")
        if self.target_allocation_group == self.random_allocation_group:
            raise ValueError("target and random allocation groups must differ")


class FamilyBlockMatchStatus(StrEnum):
    SELECTED = "selected"
    INFEASIBLE = "infeasible"
    SEARCH_EXHAUSTED = "search_exhausted"


@dataclass(frozen=True, slots=True)
class FamilyBlockMatchReport:
    """Selection result with the complete family and budget contract exposed."""

    status: FamilyBlockMatchStatus
    policy: FamilyBlockMatchPolicy
    records: tuple[FamilyBlockRecord, ...]
    target_family_ids: tuple[str, ...]
    random_family_ids: tuple[str, ...]
    target_family_row_counts: tuple[tuple[str, int], ...]
    random_family_row_counts: tuple[tuple[str, int], ...]
    target_record_ids: tuple[str, ...]
    random_record_ids: tuple[str, ...]
    target_ledger: TokenLedger
    random_ledger: TokenLedger
    target_optimizer_updates: int
    random_optimizer_updates: int
    exact_optimizer_updates: bool
    exact_family_count: bool
    exact_rows_per_family: bool
    exact_family_structure: bool
    exact_prompt_token_budget: bool
    target_token_relative_difference: float | None
    diagnostics: tuple[StandardizedDifference, ...]
    search_states: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status is FamilyBlockMatchStatus.SELECTED

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MatchingFailure(RuntimeError):
    """Fail-closed allocation rejection that preserves its mismatch report."""

    def __init__(self, report: MatchingReport) -> None:
        super().__init__("teacher allocation failed matching requirements")
        self.report = report


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _stratum_label(key: tuple[object, ...]) -> str:
    return _canonical_json(key).decode("ascii")


def _seeded_order_key(seed: int, namespace: str, record_id: str) -> str:
    material = f"duraseed:matching:v1:{seed}:{namespace}:{record_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _validate_records(
    records: Sequence[TeacherExampleRecord], *, name: str
) -> tuple[TeacherExampleRecord, ...]:
    normalized = tuple(records)
    if any(not isinstance(record, TeacherExampleRecord) for record in normalized):
        raise TypeError(f"{name} must contain TeacherExampleRecord values")
    identifiers = [record.record_id for record in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} contains duplicate record IDs")
    return tuple(sorted(normalized, key=lambda record: record.record_id))


def _validate_match_options(
    allocation_seed: int,
    max_target_token_relative_difference: float,
    target_optimizer_updates: int,
    matched_optimizer_updates: int,
) -> None:
    if (
        isinstance(allocation_seed, bool)
        or not isinstance(allocation_seed, int)
        or not 0 <= allocation_seed <= MAX_ROOT_SEED
    ):
        raise ValueError(f"allocation_seed must be an integer in [0, {MAX_ROOT_SEED}]")
    if (
        isinstance(max_target_token_relative_difference, bool)
        or not isinstance(max_target_token_relative_difference, (int, float))
        or not math.isfinite(max_target_token_relative_difference)
        or not 0 <= max_target_token_relative_difference <= 1
    ):
        raise ValueError("target-token relative-difference limit must be in [0, 1]")
    for name, value in (
        ("target_optimizer_updates", target_optimizer_updates),
        ("matched_optimizer_updates", matched_optimizer_updates),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _improve_token_ledgers(
    selected: list[TeacherExampleRecord],
    remaining: list[TeacherExampleRecord],
    *,
    target_prompt_total: int,
    target_response_total: int,
    seed: int,
    namespace: str,
) -> None:
    """Apply deterministic swaps toward exact prompt and close target ledgers."""

    selected_prompt_total = sum(record.teacher_prompt_tokens for record in selected)
    selected_response_total = sum(record.teacher_target_tokens for record in selected)
    while selected and remaining:
        current_objective = (
            abs(selected_prompt_total - target_prompt_total),
            abs(selected_response_total - target_response_total),
        )
        best: (
            tuple[
                int,
                int,
                str,
                int,
                int,
                int,
                int,
            ]
            | None
        ) = None
        for selected_index, old in enumerate(selected):
            for remaining_index, new in enumerate(remaining):
                proposed_prompt_total = (
                    selected_prompt_total
                    - old.teacher_prompt_tokens
                    + new.teacher_prompt_tokens
                )
                proposed_response_total = (
                    selected_response_total
                    - old.teacher_target_tokens
                    + new.teacher_target_tokens
                )
                proposed_objective = (
                    abs(proposed_prompt_total - target_prompt_total),
                    abs(proposed_response_total - target_response_total),
                )
                if proposed_objective >= current_objective:
                    continue
                tie_break = _seeded_order_key(
                    seed,
                    namespace,
                    f"{old.record_id}->{new.record_id}",
                )
                candidate = (
                    proposed_objective[0],
                    proposed_objective[1],
                    tie_break,
                    selected_index,
                    remaining_index,
                    proposed_prompt_total,
                    proposed_response_total,
                )
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return
        (
            _,
            _,
            _,
            selected_index,
            remaining_index,
            selected_prompt_total,
            selected_response_total,
        ) = best
        selected[selected_index], remaining[remaining_index] = (
            remaining[remaining_index],
            selected[selected_index],
        )


def _select_within_stratum(
    targets: Sequence[TeacherExampleRecord],
    candidates: Sequence[TeacherExampleRecord],
    *,
    seed: int,
    stratum_label: str,
) -> tuple[TeacherExampleRecord, ...]:
    if len(candidates) <= len(targets):
        return tuple(
            sorted(
                candidates,
                key=lambda record: _seeded_order_key(
                    seed, stratum_label, record.record_id
                ),
            )
        )

    exact = _exact_ledger_subset(
        targets,
        candidates,
        seed=seed,
        stratum_label=stratum_label,
    )
    if exact is not None:
        return exact

    ordered_targets = sorted(
        targets,
        key=lambda record: (
            record.teacher_target_tokens,
            record.teacher_prompt_tokens,
            _seeded_order_key(seed, stratum_label, record.record_id),
        ),
    )
    remaining = list(candidates)
    selected: list[TeacherExampleRecord] = []
    for target in ordered_targets:
        candidate = min(
            remaining,
            key=lambda record: (
                abs(record.teacher_prompt_tokens - target.teacher_prompt_tokens),
                abs(record.teacher_target_tokens - target.teacher_target_tokens),
                _seeded_order_key(seed, stratum_label, record.record_id),
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)

    _improve_token_ledgers(
        selected,
        remaining,
        target_prompt_total=sum(record.teacher_prompt_tokens for record in targets),
        target_response_total=sum(record.teacher_target_tokens for record in targets),
        seed=seed,
        namespace=stratum_label,
    )
    return tuple(selected)


def _exact_ledger_subset(
    targets: Sequence[TeacherExampleRecord],
    candidates: Sequence[TeacherExampleRecord],
    *,
    seed: int,
    stratum_label: str,
) -> tuple[TeacherExampleRecord, ...] | None:
    """Find an exact prompt-token subset and closest target-token total.

    The dynamic program is exact up to a fail-closed state limit.  It retains
    target-token totals as a state dimension so a locally attractive record
    cannot hide a viable aggregate budget match.
    """

    required_count = len(targets)
    required_prompt_tokens = sum(record.teacher_prompt_tokens for record in targets)
    required_target_tokens = sum(record.teacher_target_tokens for record in targets)
    ordered = tuple(
        sorted(
            candidates,
            key=lambda record: _seeded_order_key(seed, stratum_label, record.record_id),
        )
    )
    # (count, prompt tokens, target tokens) -> ordered candidate indices.
    states: dict[tuple[int, int, int], tuple[int, ...]] = {(0, 0, 0): ()}
    for index, record in enumerate(ordered):
        additions: dict[tuple[int, int, int], tuple[int, ...]] = {}
        for (count, prompt_tokens, target_tokens), chosen in tuple(states.items()):
            if count >= required_count:
                continue
            next_prompt = prompt_tokens + record.teacher_prompt_tokens
            if next_prompt > required_prompt_tokens:
                continue
            key = (
                count + 1,
                next_prompt,
                target_tokens + record.teacher_target_tokens,
            )
            selection = chosen + (index,)
            previous = additions.get(key, states.get(key))
            if previous is None or selection < previous:
                additions[key] = selection
        states.update(additions)
        if len(states) > _MAX_EXACT_MATCH_STATES:
            return None

    viable = [
        (abs(target_tokens - required_target_tokens), chosen)
        for (count, prompt_tokens, target_tokens), chosen in states.items()
        if count == required_count and prompt_tokens == required_prompt_tokens
    ]
    if not viable:
        return None
    _, chosen = min(viable)
    return tuple(ordered[index] for index in chosen)


def _mean_and_variance(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance


def _standardized_numeric(
    covariate: str,
    target_values: Sequence[float],
    matched_values: Sequence[float],
) -> StandardizedDifference:
    if not target_values or not matched_values:
        return StandardizedDifference(
            covariate=covariate,
            level=None,
            target_value=(
                sum(target_values) / len(target_values) if target_values else None
            ),
            matched_value=(
                sum(matched_values) / len(matched_values) if matched_values else None
            ),
            standardized_difference=None,
            undefined_due_to_zero_variance=True,
        )
    target_mean, target_variance = _mean_and_variance(target_values)
    matched_mean, matched_variance = _mean_and_variance(matched_values)
    denominator = math.sqrt((target_variance + matched_variance) / 2)
    difference = target_mean - matched_mean
    if denominator == 0:
        return StandardizedDifference(
            covariate=covariate,
            level=None,
            target_value=target_mean,
            matched_value=matched_mean,
            standardized_difference=0.0 if difference == 0 else None,
            undefined_due_to_zero_variance=difference != 0,
        )
    return StandardizedDifference(
        covariate=covariate,
        level=None,
        target_value=target_mean,
        matched_value=matched_mean,
        standardized_difference=difference / denominator,
    )


def _standardized_categorical(
    covariate: str,
    target_values: Sequence[str],
    matched_values: Sequence[str],
) -> tuple[StandardizedDifference, ...]:
    levels = sorted(set(target_values) | set(matched_values))
    statistics: list[StandardizedDifference] = []
    for level in levels:
        target_probability = (
            sum(value == level for value in target_values) / len(target_values)
            if target_values
            else 0.0
        )
        matched_probability = (
            sum(value == level for value in matched_values) / len(matched_values)
            if matched_values
            else 0.0
        )
        denominator = math.sqrt(
            (
                target_probability * (1 - target_probability)
                + matched_probability * (1 - matched_probability)
            )
            / 2
        )
        difference = target_probability - matched_probability
        statistics.append(
            StandardizedDifference(
                covariate=covariate,
                level=level,
                target_value=target_probability,
                matched_value=matched_probability,
                standardized_difference=(
                    difference / denominator
                    if denominator != 0
                    else 0.0
                    if difference == 0
                    else None
                ),
                undefined_due_to_zero_variance=(denominator == 0 and difference != 0),
            )
        )
    return tuple(statistics)


def _balance_statistics(
    targets: Sequence[TeacherExampleRecord],
    matched: Sequence[TeacherExampleRecord],
) -> tuple[StandardizedDifference, ...]:
    statistics = [
        _standardized_numeric(
            "tree_depth",
            [float(record.covariates.tree_depth) for record in targets],
            [float(record.covariates.tree_depth) for record in matched],
        ),
        _standardized_numeric(
            "operand_count",
            [float(record.covariates.operand_count) for record in targets],
            [float(record.covariates.operand_count) for record in matched],
        ),
        _standardized_numeric(
            "noncommutative_count",
            [float(record.covariates.noncommutative_count) for record in targets],
            [float(record.covariates.noncommutative_count) for record in matched],
        ),
        _standardized_numeric(
            "fractional_intermediate",
            [float(record.covariates.fractional_intermediate) for record in targets],
            [float(record.covariates.fractional_intermediate) for record in matched],
        ),
        _standardized_numeric(
            "teacher_prompt_tokens",
            [float(record.teacher_prompt_tokens) for record in targets],
            [float(record.teacher_prompt_tokens) for record in matched],
        ),
        _standardized_numeric(
            "teacher_target_tokens",
            [float(record.teacher_target_tokens) for record in targets],
            [float(record.teacher_target_tokens) for record in matched],
        ),
    ]

    def operator_multiset(record: TeacherExampleRecord) -> str:
        return ",".join(record.covariates.operator_multiset)

    statistics.extend(
        _standardized_categorical(
            "operator_multiset",
            [operator_multiset(record) for record in targets],
            [operator_multiset(record) for record in matched],
        )
    )
    statistics.extend(
        _standardized_categorical(
            "target_magnitude_bin",
            [record.covariates.target_magnitude_bin for record in targets],
            [record.covariates.target_magnitude_bin for record in matched],
        )
    )
    statistics.extend(
        _standardized_categorical(
            "valid_family_count_bin",
            [record.covariates.valid_family_count_bin for record in targets],
            [record.covariates.valid_family_count_bin for record in matched],
        )
    )
    statistics.extend(
        _standardized_categorical(
            "teacher_trace_token_bin",
            [record.covariates.teacher_trace_token_bin for record in targets],
            [record.covariates.teacher_trace_token_bin for record in matched],
        )
    )
    statistics.extend(
        _standardized_categorical(
            "teacher_trace_format",
            [record.teacher_trace_format for record in targets],
            [record.teacher_trace_format for record in matched],
        )
    )
    return tuple(statistics)


def _stratum_mismatches(
    targets: Sequence[TeacherExampleRecord],
    matched: Sequence[TeacherExampleRecord],
    available_counts: dict[tuple[object, ...], int] | None = None,
) -> tuple[StratumMismatch, ...]:
    target_counts: dict[tuple[object, ...], int] = defaultdict(int)
    matched_counts: dict[tuple[object, ...], int] = defaultdict(int)
    for record in targets:
        target_counts[record.stratum_key] += 1
    for record in matched:
        matched_counts[record.stratum_key] += 1
    keys = sorted(set(target_counts) | set(matched_counts), key=_stratum_label)
    mismatches = []
    for key in keys:
        required = target_counts[key]
        selected = matched_counts[key]
        if required == selected:
            continue
        mismatches.append(
            StratumMismatch(
                stratum=_stratum_label(key),
                required_count=required,
                available_count=(
                    available_counts.get(key, selected)
                    if available_counts is not None
                    else selected
                ),
                selected_count=selected,
            )
        )
    return tuple(mismatches)


def _build_report(
    targets: Sequence[TeacherExampleRecord],
    matched: Sequence[TeacherExampleRecord],
    *,
    allocation_seed: int,
    max_target_token_relative_difference: float,
    target_optimizer_updates: int,
    matched_optimizer_updates: int,
    mismatches: tuple[StratumMismatch, ...],
) -> MatchingReport:
    target_ledger = TokenLedger.from_records(targets)
    matched_ledger = TokenLedger.from_records(matched)
    exact_count = target_ledger.example_count == matched_ledger.example_count
    exact_prompt_tokens = target_ledger.prompt_tokens == matched_ledger.prompt_tokens
    exact_optimizer_updates = target_optimizer_updates == matched_optimizer_updates
    token_difference = (
        abs(matched_ledger.target_tokens - target_ledger.target_tokens)
        / target_ledger.target_tokens
    )
    failures = [
        (
            "stratum_mismatch:"
            f"{mismatch.stratum}:required={mismatch.required_count}:"
            f"available={mismatch.available_count}:selected={mismatch.selected_count}"
        )
        for mismatch in mismatches
    ]
    if not exact_count:
        failures.append(
            "teacher_example_count_mismatch:"
            f"target={target_ledger.example_count}:"
            f"matched={matched_ledger.example_count}"
        )
    if not exact_prompt_tokens:
        failures.append(
            "teacher_prompt_token_mismatch:"
            f"target={target_ledger.prompt_tokens}:"
            f"matched={matched_ledger.prompt_tokens}"
        )
    if not exact_optimizer_updates:
        failures.append(
            "optimizer_update_mismatch:"
            f"target={target_optimizer_updates}:matched={matched_optimizer_updates}"
        )
    if token_difference > max_target_token_relative_difference + 1e-15:
        failures.append(
            "teacher_target_token_mismatch:"
            f"relative_difference={token_difference:.12f}:"
            f"limit={max_target_token_relative_difference:.12f}"
        )
    return MatchingReport(
        allocation_seed=allocation_seed,
        target_record_ids=tuple(record.record_id for record in targets),
        matched_record_ids=tuple(record.record_id for record in matched),
        target_ledger=target_ledger,
        matched_ledger=matched_ledger,
        exact_example_count=exact_count,
        exact_prompt_token_budget=exact_prompt_tokens,
        target_optimizer_updates=target_optimizer_updates,
        matched_optimizer_updates=matched_optimizer_updates,
        exact_optimizer_updates=exact_optimizer_updates,
        max_target_token_relative_difference=max_target_token_relative_difference,
        target_token_relative_difference=token_difference,
        stratum_mismatches=mismatches,
        standardized_differences=_balance_statistics(targets, matched),
        failures=tuple(failures),
        passed=not failures,
    )


def match_teacher_allocations(
    target_records: Sequence[TeacherExampleRecord],
    candidate_records: Sequence[TeacherExampleRecord],
    *,
    allocation_seed: int,
    target_optimizer_updates: int,
    matched_optimizer_updates: int,
    max_target_token_relative_difference: float = 0.02,
) -> MatchingAllocation:
    """Select an exact-count, structurally stratified deterministic control."""

    _validate_match_options(
        allocation_seed,
        max_target_token_relative_difference,
        target_optimizer_updates,
        matched_optimizer_updates,
    )
    targets = _validate_records(target_records, name="target_records")
    candidates = _validate_records(candidate_records, name="candidate_records")
    if not targets:
        raise ValueError("target_records must be non-empty")
    overlap = {record.record_id for record in targets}.intersection(
        record.record_id for record in candidates
    )
    if overlap:
        raise ValueError("target and candidate record IDs must be disjoint")

    targets_by_stratum: dict[tuple[object, ...], list[TeacherExampleRecord]] = (
        defaultdict(list)
    )
    candidates_by_stratum: dict[tuple[object, ...], list[TeacherExampleRecord]] = (
        defaultdict(list)
    )
    for record in targets:
        targets_by_stratum[record.stratum_key].append(record)
    for record in candidates:
        candidates_by_stratum[record.stratum_key].append(record)

    selected: list[TeacherExampleRecord] = []
    available_counts = {
        key: len(records) for key, records in candidates_by_stratum.items()
    }
    for key in sorted(targets_by_stratum, key=_stratum_label):
        stratum_targets = targets_by_stratum[key]
        stratum_candidates = candidates_by_stratum.get(key, [])
        selected.extend(
            _select_within_stratum(
                stratum_targets,
                stratum_candidates,
                seed=allocation_seed,
                stratum_label=_stratum_label(key),
            )
        )

    selected.sort(key=lambda record: record.record_id)
    mismatches = _stratum_mismatches(targets, selected, available_counts)
    report = _build_report(
        targets,
        selected,
        allocation_seed=allocation_seed,
        max_target_token_relative_difference=max_target_token_relative_difference,
        target_optimizer_updates=target_optimizer_updates,
        matched_optimizer_updates=matched_optimizer_updates,
        mismatches=mismatches,
    )
    allocation = MatchingAllocation(records=tuple(selected), report=report)
    if not report.passed:
        raise MatchingFailure(report)
    return allocation


def compare_teacher_allocations(
    target_records: Sequence[TeacherExampleRecord],
    matched_records: Sequence[TeacherExampleRecord],
    *,
    allocation_seed: int,
    target_optimizer_updates: int,
    matched_optimizer_updates: int,
    max_target_token_relative_difference: float = 0.02,
) -> MatchingReport:
    """Audit an already selected allocation with the same fail-closed rules."""

    _validate_match_options(
        allocation_seed,
        max_target_token_relative_difference,
        target_optimizer_updates,
        matched_optimizer_updates,
    )
    targets = _validate_records(target_records, name="target_records")
    matched = _validate_records(matched_records, name="matched_records")
    if not targets:
        raise ValueError("target_records must be non-empty")
    overlap = {record.record_id for record in targets}.intersection(
        record.record_id for record in matched
    )
    if overlap:
        raise ValueError("target and matched record IDs must be disjoint")
    mismatches = _stratum_mismatches(targets, matched)
    report = _build_report(
        targets,
        matched,
        allocation_seed=allocation_seed,
        max_target_token_relative_difference=max_target_token_relative_difference,
        target_optimizer_updates=target_optimizer_updates,
        matched_optimizer_updates=matched_optimizer_updates,
        mismatches=mismatches,
    )
    if not report.passed:
        raise MatchingFailure(report)
    return report


@dataclass(frozen=True, slots=True)
class _FamilyBlock:
    family_id: str
    records: tuple[FamilyBlockRecord, ...]
    structure_key: tuple[object, ...]
    prompt_tokens: int
    target_tokens: int


def _validate_family_block_records(
    records: Sequence[FamilyBlockRecord], *, name: str
) -> tuple[FamilyBlockRecord, ...]:
    normalized = tuple(records)
    if any(not isinstance(record, FamilyBlockRecord) for record in normalized):
        raise TypeError(f"{name} must contain FamilyBlockRecord values")
    identifiers = [record.record.record_id for record in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} contains duplicate record IDs")
    return tuple(sorted(normalized, key=lambda row: row.record.record_id))


def _build_family_blocks(
    records: Sequence[FamilyBlockRecord],
    *,
    name: str,
    required_rows: int,
    required_allocation_group: str,
) -> tuple[_FamilyBlock, ...]:
    grouped: dict[str, list[FamilyBlockRecord]] = defaultdict(list)
    for row in records:
        if row.record.allocation_group != required_allocation_group:
            raise ValueError(
                f"{name} must use allocation group {required_allocation_group!r}"
            )
        grouped[row.record.family_id].append(row)

    blocks: list[_FamilyBlock] = []
    for family_id in sorted(grouped):
        rows = tuple(sorted(grouped[family_id], key=lambda row: row.record.record_id))
        if len(rows) != required_rows:
            raise ValueError(
                f"{name} family {family_id!r} must contain exactly {required_rows} rows"
            )
        structure_keys = {row.structure_key for row in rows}
        if len(structure_keys) != 1:
            raise ValueError(
                f"{name} family {family_id!r} has non-immutable family structure"
            )
        blocks.append(
            _FamilyBlock(
                family_id=family_id,
                records=rows,
                structure_key=next(iter(structure_keys)),
                prompt_tokens=sum(row.record.teacher_prompt_tokens for row in rows),
                target_tokens=sum(row.record.teacher_target_tokens for row in rows),
            )
        )
    return tuple(blocks)


def _family_structure_counts(
    blocks: Sequence[_FamilyBlock],
) -> dict[tuple[object, ...], int]:
    counts: dict[tuple[object, ...], int] = defaultdict(int)
    for block in blocks:
        counts[block.structure_key] += 1
    return dict(counts)


def _family_block_options(
    block: _FamilyBlock,
    *,
    dose: int,
    max_search_states: int,
) -> tuple[tuple[_FamilyBlock, ...], int, bool]:
    """Build bounded d-row options, keeping the earliest IDs per token ledger."""

    states: dict[tuple[int, int, int], tuple[FamilyBlockRecord, ...]] = {(0, 0, 0): ()}
    for row in block.records:
        additions: dict[tuple[int, int, int], tuple[FamilyBlockRecord, ...]] = {}
        new_state_count = 0
        for (count, prompt_tokens, target_tokens), chosen in tuple(states.items()):
            if count >= dose:
                continue
            key = (
                count + 1,
                prompt_tokens + row.record.teacher_prompt_tokens,
                target_tokens + row.record.teacher_target_tokens,
            )
            selection = chosen + (row,)
            previous = additions.get(key, states.get(key))
            if previous is not None and tuple(
                item.record.record_id for item in previous
            ) <= tuple(item.record.record_id for item in selection):
                continue
            if key not in states and key not in additions:
                if len(states) + new_state_count >= max_search_states:
                    return (), max_search_states + 1, True
                new_state_count += 1
            additions[key] = selection
        states.update(additions)

    options = tuple(
        _FamilyBlock(
            family_id=block.family_id,
            records=records,
            structure_key=block.structure_key,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
        )
        for (count, prompt_tokens, target_tokens), records in sorted(
            states.items(),
            key=lambda item: tuple(row.record.record_id for row in item[1]),
        )
        if count == dose
    )
    return options, len(states), False


def _family_block_diagnostics(
    targets: Sequence[FamilyBlockRecord],
    random_records: Sequence[FamilyBlockRecord],
) -> tuple[StandardizedDifference, ...]:
    if not random_records:
        return ()
    return (
        _standardized_numeric(
            "absolute_target",
            [row.absolute_target for row in targets],
            [row.absolute_target for row in random_records],
        ),
        _standardized_numeric(
            "valid_family_count",
            [float(row.valid_family_count) for row in targets],
            [float(row.valid_family_count) for row in random_records],
        ),
        _standardized_numeric(
            "supervised_target_tokens",
            [float(row.record.teacher_target_tokens) for row in targets],
            [float(row.record.teacher_target_tokens) for row in random_records],
        ),
    )


def match_teacher_family_blocks(
    target_records: Sequence[FamilyBlockRecord],
    candidate_records: Sequence[FamilyBlockRecord],
    *,
    policy: FamilyBlockMatchPolicy,
    target_optimizer_updates: int,
    random_optimizer_updates: int,
) -> FamilyBlockMatchReport:
    """Select the random 12-family control under the frozen production policy.

    Malformed rows raise immediately.  A valid candidate universe instead
    returns ``infeasible`` when no allocation exists and ``search_exhausted``
    when the declared state bound prevents an exhaustive answer.
    """

    if not isinstance(policy, FamilyBlockMatchPolicy):
        raise TypeError("policy must be a FamilyBlockMatchPolicy")
    _validate_match_options(
        policy.allocation_seed,
        policy.max_target_token_relative_difference,
        target_optimizer_updates,
        random_optimizer_updates,
    )
    targets = _validate_family_block_records(target_records, name="target_records")
    candidates = _validate_family_block_records(
        candidate_records, name="candidate_records"
    )
    target_blocks = _build_family_blocks(
        targets,
        name="target_records",
        required_rows=policy.dose,
        required_allocation_group=policy.target_allocation_group,
    )
    candidate_blocks = _build_family_blocks(
        candidates,
        name="candidate_records",
        required_rows=16,
        required_allocation_group=policy.random_allocation_group,
    )
    if len(target_blocks) != policy.target_family_count:
        raise ValueError(
            "target_records must contain exactly 12 distinct complete families"
        )
    target_family_ids = tuple(block.family_id for block in target_blocks)
    candidate_family_ids = {block.family_id for block in candidate_blocks}
    overlap = set(target_family_ids).intersection(candidate_family_ids)
    if overlap:
        raise ValueError("target and random candidate family IDs must be disjoint")
    target_record_ids = {row.record.record_id for row in targets}
    if target_record_ids.intersection(row.record.record_id for row in candidates):
        raise ValueError("target and random candidate record IDs must be disjoint")

    target_rows = tuple(row.record for row in targets)
    target_ledger = TokenLedger.from_records(target_rows)
    target_structure_counts = _family_structure_counts(target_blocks)

    def report(
        status: FamilyBlockMatchStatus,
        selected_blocks: Sequence[_FamilyBlock],
        *,
        search_states: int,
        failures: Sequence[str],
    ) -> FamilyBlockMatchReport:
        selected = tuple(
            sorted(
                (row for block in selected_blocks for row in block.records),
                key=lambda row: (row.record.family_id, row.record.record_id),
            )
        )
        random_rows = tuple(row.record for row in selected)
        random_ledger = TokenLedger.from_records(random_rows)
        random_family_ids = tuple(sorted(block.family_id for block in selected_blocks))
        random_structure_counts = _family_structure_counts(selected_blocks)
        exact_family_count = (
            len(target_family_ids)
            == len(random_family_ids)
            == policy.target_family_count
        )
        exact_rows_per_family = (
            all(
                len(block.records) == policy.dose
                for block in (*target_blocks, *selected_blocks)
            )
            and len(random_rows) == policy.target_family_count * policy.dose
        )
        target_token_relative_difference = (
            abs(random_ledger.target_tokens - target_ledger.target_tokens)
            / target_ledger.target_tokens
            if random_rows
            else None
        )
        return FamilyBlockMatchReport(
            status=status,
            policy=policy,
            records=selected,
            target_family_ids=target_family_ids,
            random_family_ids=random_family_ids,
            target_family_row_counts=tuple(
                (block.family_id, len(block.records)) for block in target_blocks
            ),
            random_family_row_counts=tuple(
                (family_id, sum(row.record.family_id == family_id for row in selected))
                for family_id in random_family_ids
            ),
            target_record_ids=tuple(row.record_id for row in target_rows),
            random_record_ids=tuple(row.record.record_id for row in selected),
            target_ledger=target_ledger,
            random_ledger=random_ledger,
            target_optimizer_updates=target_optimizer_updates,
            random_optimizer_updates=random_optimizer_updates,
            exact_optimizer_updates=(
                target_optimizer_updates == random_optimizer_updates
            ),
            exact_family_count=exact_family_count,
            exact_rows_per_family=exact_rows_per_family,
            exact_family_structure=(
                exact_family_count
                and target_structure_counts == random_structure_counts
            ),
            exact_prompt_token_budget=(
                bool(random_rows)
                and target_ledger.prompt_tokens == random_ledger.prompt_tokens
            ),
            target_token_relative_difference=target_token_relative_difference,
            diagnostics=_family_block_diagnostics(targets, selected),
            search_states=search_states,
            failures=tuple(failures),
        )

    if target_optimizer_updates != random_optimizer_updates:
        return report(
            FamilyBlockMatchStatus.INFEASIBLE,
            (),
            search_states=0,
            failures=(
                "optimizer_update_mismatch:"
                f"target={target_optimizer_updates}:random={random_optimizer_updates}",
            ),
        )

    structure_keys = tuple(sorted(target_structure_counts, key=_stratum_label))
    structure_indices = {key: index for index, key in enumerate(structure_keys)}
    required_counts = tuple(target_structure_counts[key] for key in structure_keys)
    available_counts = _family_structure_counts(candidate_blocks)
    shortages = tuple(
        (
            key,
            required_counts[index],
            available_counts.get(key, 0),
        )
        for index, key in enumerate(structure_keys)
        if available_counts.get(key, 0) < required_counts[index]
    )
    if shortages:
        return report(
            FamilyBlockMatchStatus.INFEASIBLE,
            (),
            search_states=0,
            failures=tuple(
                "family_structure_shortage:"
                f"structure={_stratum_label(key)}:required={required}:available={available}"
                for key, required, available in shortages
            ),
        )

    ordered_candidate_families = tuple(
        sorted(
            (
                block
                for block in candidate_blocks
                if block.structure_key in structure_indices
            ),
            key=lambda block: _seeded_order_key(
                policy.allocation_seed,
                "family-block-v1",
                block.family_id,
            ),
        )
    )
    target_prompt_tokens = target_ledger.prompt_tokens
    target_target_tokens = target_ledger.target_tokens
    empty_counts = (0,) * len(structure_keys)
    states: dict[tuple[tuple[int, ...], int, int], tuple[tuple[int, int], ...]] = {
        (empty_counts, 0, 0): ()
    }
    family_options: list[tuple[_FamilyBlock, ...]] = []
    option_search_peak = 0
    for family in ordered_candidate_families:
        options, option_search_states, exhausted = _family_block_options(
            family,
            dose=policy.dose,
            max_search_states=policy.max_search_states,
        )
        option_search_peak = max(option_search_peak, option_search_states)
        if exhausted:
            return report(
                FamilyBlockMatchStatus.SEARCH_EXHAUSTED,
                (),
                search_states=option_search_states,
                failures=(
                    "family_block_option_search_exhausted:"
                    f"family={family.family_id}:states>{policy.max_search_states}:"
                    f"limit={policy.max_search_states}",
                ),
            )
        family_options.append(options)

    for family_index, family in enumerate(ordered_candidate_families):
        options = family_options[family_index]
        structure_index = structure_indices[family.structure_key]
        additions: dict[
            tuple[tuple[int, ...], int, int], tuple[tuple[int, int], ...]
        ] = {}
        new_state_count = 0
        for (counts, prompt_tokens, target_tokens), chosen in tuple(states.items()):
            if counts[structure_index] >= required_counts[structure_index]:
                continue
            for option_index, option in enumerate(options):
                next_prompt_tokens = prompt_tokens + option.prompt_tokens
                if next_prompt_tokens > target_prompt_tokens:
                    continue
                next_counts = list(counts)
                next_counts[structure_index] += 1
                key = (
                    tuple(next_counts),
                    next_prompt_tokens,
                    target_tokens + option.target_tokens,
                )
                selection = chosen + ((family_index, option_index),)
                previous = additions.get(key, states.get(key))
                if previous is None or selection < previous:
                    if key not in states and key not in additions:
                        if len(states) + new_state_count >= policy.max_search_states:
                            return report(
                                FamilyBlockMatchStatus.SEARCH_EXHAUSTED,
                                (),
                                search_states=policy.max_search_states + 1,
                                failures=(
                                    "family_block_search_exhausted:"
                                    f"states>{policy.max_search_states}:"
                                    f"limit={policy.max_search_states}",
                                ),
                            )
                        new_state_count += 1
                    additions[key] = selection
        states.update(additions)
        if len(states) > policy.max_search_states:
            return report(
                FamilyBlockMatchStatus.SEARCH_EXHAUSTED,
                (),
                search_states=len(states),
                failures=(
                    "family_block_search_exhausted:"
                    f"states={len(states)}:limit={policy.max_search_states}",
                ),
            )

    viable: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for (counts, prompt_tokens, target_tokens), chosen in states.items():
        if counts != required_counts or prompt_tokens != target_prompt_tokens:
            continue
        relative_difference = (
            abs(target_tokens - target_target_tokens) / target_target_tokens
        )
        if relative_difference <= (policy.max_target_token_relative_difference + 1e-15):
            viable.append((abs(target_tokens - target_target_tokens), chosen))
    if not viable:
        return report(
            FamilyBlockMatchStatus.INFEASIBLE,
            (),
            search_states=max(len(states), option_search_peak),
            failures=(
                "no_family_block_allocation_satisfies_prompt_and_target_token_budgets",
            ),
        )

    _, chosen = min(viable)
    selected_blocks = tuple(
        family_options[family_index][option_index]
        for family_index, option_index in chosen
    )
    return report(
        FamilyBlockMatchStatus.SELECTED,
        selected_blocks,
        search_states=max(len(states), option_search_peak),
        failures=(),
    )


__all__ = [
    "FamilyBlockMatchPolicy",
    "FamilyBlockMatchReport",
    "FamilyBlockMatchStatus",
    "FamilyBlockRecord",
    "MatchingAllocation",
    "MatchingFailure",
    "MatchingReport",
    "StandardizedDifference",
    "StructuralCovariates",
    "StratumMismatch",
    "TeacherExampleRecord",
    "TokenLedger",
    "compare_teacher_allocations",
    "match_teacher_family_blocks",
    "match_teacher_allocations",
]
