"""Deterministic pre-treatment matching for the crossed family panels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite
import random

from duraseed.data.panels import PanelLabel, SeedBlockPanelAssignment
from duraseed.data.boundary import BoundaryFamilySummary
from duraseed.data.manifests import TCESTaskManifestRecord
from duraseed.provenance import MAX_ROOT_SEED, derive_namespaced_seed


PANEL_MATCHING_COVARIATES = (
    "m0_posterior_mean_success",
    "informative_group_probability_i8",
    "tree_depth",
    "operator_multiset",
    "noncommutative_operation_count",
    "fractional_intermediate_profile",
    "target_magnitude",
    "valid_family_multiplicity",
    "teacher_trace_length",
    "available_disjoint_instances",
)
_NUMERIC_COVARIATES = (
    "m0_posterior_mean_success",
    "informative_group_probability_i8",
    "tree_depth",
    "noncommutative_operation_count",
    "target_magnitude",
    "valid_family_multiplicity",
    "teacher_trace_length",
    "available_disjoint_instances",
)
_CATEGORICAL_COVARIATES = (
    "operator_multiset",
    "fractional_intermediate_profile",
)
_OPERATOR_ORDER = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3}


class PanelMatchingError(ValueError):
    """Raised when a symmetric, authenticated panel match cannot be made."""


@dataclass(frozen=True, slots=True)
class TCESFamilyStructure:
    """Structural covariates decoded from one canonical TCES family ID."""

    tree_depth: int
    operator_multiset: tuple[str, ...]
    noncommutative_operation_count: int
    fractional_intermediate_profile: tuple[str, ...]


def parse_tces_family_structure(family_id: str) -> TCESFamilyStructure:
    """Parse the stable family identifier without reconstructing numeric operands."""

    if not isinstance(family_id, str) or not family_id.strip():
        raise PanelMatchingError("family_id must be nonempty text")
    parts = family_id.split("|")
    structure = parts[0]
    profile_parts = [part for part in parts[1:] if part.startswith("intermediates=")]
    if len(profile_parts) != 1:
        raise PanelMatchingError("family ID must contain one intermediate profile")
    profile_text = profile_parts[0].removeprefix("intermediates=")
    profile = tuple(profile_text.split(",")) if profile_text else ()

    def parse_node(position: int) -> tuple[int, tuple[str, ...], int]:
        if position >= len(structure):
            raise PanelMatchingError("family structure ended unexpectedly")
        if structure[position] == "r":
            end = position + 1
            while end < len(structure) and structure[end].isdigit():
                end += 1
            if end == position + 1:
                raise PanelMatchingError("family leaf is missing its rank")
            return 1, (), end
        operator = next(
            (
                name
                for name in _OPERATOR_ORDER
                if structure.startswith(f"{name}(", position)
            ),
            None,
        )
        if operator is None:
            raise PanelMatchingError("family structure has an unknown node")
        cursor = position + len(operator) + 1
        left_depth, left_operators, cursor = parse_node(cursor)
        if cursor >= len(structure) or structure[cursor] != ",":
            raise PanelMatchingError("family structure is missing a child separator")
        right_depth, right_operators, cursor = parse_node(cursor + 1)
        if cursor >= len(structure) or structure[cursor] != ")":
            raise PanelMatchingError(
                "family structure is missing a closing parenthesis"
            )
        return (
            1 + max(left_depth, right_depth),
            (operator, *left_operators, *right_operators),
            cursor + 1,
        )

    tree_depth, operators, end = parse_node(0)
    if end != len(structure):
        raise PanelMatchingError("family structure has trailing text")
    if not operators or len(operators) != len(profile):
        raise PanelMatchingError(
            "family operator and intermediate profiles must be nonempty and aligned"
        )
    return TCESFamilyStructure(
        tree_depth=tree_depth,
        operator_multiset=tuple(sorted(operators, key=_OPERATOR_ORDER.__getitem__)),
        noncommutative_operation_count=sum(
            operator in {"SUB", "DIV"} for operator in operators
        ),
        fractional_intermediate_profile=profile,
    )


@dataclass(frozen=True, slots=True)
class FamilyPanelCandidate:
    """One observation-eligible family represented only by pre-treatment data."""

    family_id: str
    m0_posterior_mean_success: float
    informative_group_probability_i8: float
    tree_depth: int
    operator_multiset: tuple[str, ...]
    noncommutative_operation_count: int
    fractional_intermediate_profile: tuple[str, ...]
    target_magnitude: float
    valid_family_multiplicity: float
    teacher_trace_length: float
    available_disjoint_instances: int

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or not self.family_id.strip():
            raise ValueError("family_id must be nonempty text")
        for name in (
            "m0_posterior_mean_success",
            "informative_group_probability_i8",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a finite probability")
        for name, minimum in (
            ("tree_depth", 1),
            ("noncommutative_operation_count", 0),
            ("available_disjoint_instances", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        for name, minimum in (
            ("target_magnitude", 0.0),
            ("valid_family_multiplicity", 1.0),
            ("teacher_trace_length", 1.0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < minimum
            ):
                raise ValueError(f"{name} must be finite and at least {minimum}")
        if (
            not isinstance(self.operator_multiset, tuple)
            or not self.operator_multiset
            or any(
                operator not in _OPERATOR_ORDER for operator in self.operator_multiset
            )
        ):
            raise ValueError("operator_multiset must use the canonical TCES operators")
        canonical_operators = tuple(
            sorted(self.operator_multiset, key=_OPERATOR_ORDER.__getitem__)
        )
        expected_noncommutative = sum(
            operator in {"SUB", "DIV"} for operator in canonical_operators
        )
        if self.noncommutative_operation_count != expected_noncommutative:
            raise ValueError(
                "noncommutative_operation_count disagrees with operator_multiset"
            )
        if (
            not isinstance(self.fractional_intermediate_profile, tuple)
            or not self.fractional_intermediate_profile
            or any(
                marker not in {"I", "F"}
                for marker in self.fractional_intermediate_profile
            )
        ):
            raise ValueError("fractional_intermediate_profile must contain I/F markers")
        if len(self.operator_multiset) != len(self.fractional_intermediate_profile):
            raise ValueError(
                "operator and fractional-intermediate profiles must have equal length"
            )
        object.__setattr__(self, "operator_multiset", canonical_operators)
        object.__setattr__(
            self,
            "m0_posterior_mean_success",
            float(self.m0_posterior_mean_success),
        )
        object.__setattr__(
            self,
            "informative_group_probability_i8",
            float(self.informative_group_probability_i8),
        )
        for name in (
            "target_magnitude",
            "valid_family_multiplicity",
            "teacher_trace_length",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))

    def numeric(self, name: str) -> float:
        return float(getattr(self, name))

    def categorical(self, name: str) -> tuple[str, ...]:
        return tuple(getattr(self, name))


def build_family_panel_candidate(
    summary: BoundaryFamilySummary,
    records: Sequence[TCESTaskManifestRecord],
    *,
    teacher_trace_token_counts: Mapping[str, int],
    available_disjoint_instances: int,
) -> FamilyPanelCandidate:
    """Build the ten-covariate row from authenticated equal-item evidence."""

    if not isinstance(summary, BoundaryFamilySummary):
        raise TypeError("summary must be BoundaryFamilySummary")
    values = tuple(records)
    if not values or any(
        not isinstance(record, TCESTaskManifestRecord) for record in values
    ):
        raise TypeError("records must contain TCESTaskManifestRecord values")
    task_ids = tuple(item.task_id for item in summary.items)
    record_by_task = {record.task_id: record for record in values}
    if len(record_by_task) != len(values) or set(record_by_task) != set(task_ids):
        raise PanelMatchingError(
            "manifest records must exactly match the summarized family items"
        )
    if any(record.intended_family != summary.intended_family_id for record in values):
        raise PanelMatchingError("manifest record has the wrong intended family")
    if set(teacher_trace_token_counts) != set(task_ids):
        raise PanelMatchingError(
            "teacher token counts must exactly cover the summarized family items"
        )
    token_counts = tuple(teacher_trace_token_counts[task_id] for task_id in task_ids)
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in token_counts
    ):
        raise PanelMatchingError("teacher trace token counts must be positive integers")
    structure = parse_tces_family_structure(summary.intended_family_id)
    ordered_records = tuple(record_by_task[task_id] for task_id in task_ids)
    target_magnitude = fsum(
        float(abs(record.target.as_fraction())) for record in ordered_records
    ) / len(ordered_records)
    valid_family_multiplicity = fsum(
        float(record.valid_family_count) for record in ordered_records
    ) / len(ordered_records)
    teacher_trace_length = fsum(float(count) for count in token_counts) / len(
        token_counts
    )
    return FamilyPanelCandidate(
        family_id=summary.intended_family_id,
        m0_posterior_mean_success=summary.equal_item_posterior_mean_success,
        informative_group_probability_i8=(
            summary.informative_probability_at_equal_item_mean
        ),
        tree_depth=structure.tree_depth,
        operator_multiset=structure.operator_multiset,
        noncommutative_operation_count=(structure.noncommutative_operation_count),
        fractional_intermediate_profile=(structure.fractional_intermediate_profile),
        target_magnitude=target_magnitude,
        valid_family_multiplicity=valid_family_multiplicity,
        teacher_trace_length=teacher_trace_length,
        available_disjoint_instances=available_disjoint_instances,
    )


@dataclass(frozen=True, slots=True)
class PanelCovariateBalance:
    covariate: str
    imbalance: float


@dataclass(frozen=True, slots=True)
class CrossPanelNearestDistance:
    family_id: str
    nearest_other_panel_family_id: str
    gower_distance: float


@dataclass(frozen=True, slots=True)
class FamilyPanelMatch:
    """Matched symmetric panels plus diagnostics fixed before method training."""

    allocation_seed: int
    panel_size: int
    selected_family_ids: tuple[str, ...]
    excluded_family_ids: tuple[str, ...]
    panel_a_family_ids: tuple[str, ...]
    panel_b_family_ids: tuple[str, ...]
    covariate_balance: tuple[PanelCovariateBalance, ...]
    maximum_covariate_imbalance: float
    mean_covariate_imbalance: float
    cross_panel_nearest_distances: tuple[CrossPanelNearestDistance, ...]


def _validated_candidates(
    candidates: Sequence[FamilyPanelCandidate],
) -> tuple[FamilyPanelCandidate, ...]:
    values = tuple(candidates)
    if any(not isinstance(value, FamilyPanelCandidate) for value in values):
        raise TypeError("candidates must contain FamilyPanelCandidate values")
    family_ids = tuple(value.family_id for value in values)
    if len(family_ids) != len(set(family_ids)):
        raise PanelMatchingError("candidate family IDs must be unique")
    return tuple(sorted(values, key=lambda value: value.family_id))


def _validate_options(panel_size: int, allocation_seed: int) -> None:
    if (
        isinstance(panel_size, bool)
        or not isinstance(panel_size, int)
        or panel_size < 1
    ):
        raise ValueError("panel_size must be a positive integer")
    if (
        isinstance(allocation_seed, bool)
        or not isinstance(allocation_seed, int)
        or not 0 <= allocation_seed <= MAX_ROOT_SEED
    ):
        raise ValueError(f"allocation_seed must be in [0, {MAX_ROOT_SEED}]")


def _selection_order(
    candidates: tuple[FamilyPanelCandidate, ...], allocation_seed: int
) -> tuple[FamilyPanelCandidate, ...]:
    """Prefer informative, well-supplied families; seed breaks exact ties only."""

    tie_seed = derive_namespaced_seed(
        allocation_seed,
        "family_selection.panel_candidate_ties",
    )
    shuffled_ids = [candidate.family_id for candidate in candidates]
    random.Random(tie_seed).shuffle(shuffled_ids)
    tie_rank = {family_id: rank for rank, family_id in enumerate(shuffled_ids)}
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.informative_group_probability_i8,
                -candidate.available_disjoint_instances,
                tie_rank[candidate.family_id],
                candidate.family_id,
            ),
        )
    )


def _numeric_scales(
    candidates: tuple[FamilyPanelCandidate, ...],
) -> dict[str, tuple[float, float]]:
    return {
        name: (
            min(candidate.numeric(name) for candidate in candidates),
            max(candidate.numeric(name) for candidate in candidates),
        )
        for name in _NUMERIC_COVARIATES
    }


def _covariate_imbalances(
    candidates: tuple[FamilyPanelCandidate, ...],
    panel_a_indices: tuple[int, ...],
) -> tuple[PanelCovariateBalance, ...]:
    panel_size = len(panel_a_indices)
    panel_a = set(panel_a_indices)
    scales = _numeric_scales(candidates)
    balances: list[PanelCovariateBalance] = []
    for name in _NUMERIC_COVARIATES:
        lower, upper = scales[name]
        if upper == lower:
            imbalance = 0.0
        else:
            total = fsum(candidate.numeric(name) for candidate in candidates)
            a_total = fsum(candidates[index].numeric(name) for index in panel_a_indices)
            imbalance = abs((2 * a_total - total) / panel_size) / (upper - lower)
        balances.append(PanelCovariateBalance(name, imbalance))
    for name in _CATEGORICAL_COVARIATES:
        all_counts = Counter(candidate.categorical(name) for candidate in candidates)
        a_counts = Counter(candidates[index].categorical(name) for index in panel_a)
        total_variation = 0.5 * fsum(
            abs((2 * a_counts[level] - count) / panel_size)
            for level, count in all_counts.items()
        )
        balances.append(PanelCovariateBalance(name, total_variation))
    by_name = {balance.covariate: balance for balance in balances}
    return tuple(by_name[name] for name in PANEL_MATCHING_COVARIATES)


def _partition_objective(
    balances: tuple[PanelCovariateBalance, ...],
) -> tuple[float, float]:
    values = tuple(balance.imbalance for balance in balances)
    return (round(max(values), 12), round(fsum(values) / len(values), 12))


def _best_symmetric_partition(
    candidates: tuple[FamilyPanelCandidate, ...],
    *,
    panel_size: int,
    allocation_seed: int,
) -> tuple[tuple[int, ...], tuple[PanelCovariateBalance, ...]]:
    scales = _numeric_scales(candidates)
    remaining = list(range(len(candidates)))
    pairs: list[tuple[int, int]] = []
    while remaining:
        left, right = min(
            (
                (left, right)
                for position, left in enumerate(remaining)
                for right in remaining[position + 1 :]
            ),
            key=lambda pair: (
                _gower_distance(candidates[pair[0]], candidates[pair[1]], scales),
                candidates[pair[0]].family_id,
                candidates[pair[1]].family_id,
            ),
        )
        pairs.append((left, right))
        remaining.remove(left)
        remaining.remove(right)
    if len(pairs) != panel_size:
        raise PanelMatchingError(
            "pair construction did not cover the selected families"
        )
    ordering_seed = derive_namespaced_seed(
        allocation_seed,
        "family_selection.panel_partition",
    )
    random.Random(ordering_seed).shuffle(pairs)
    best_indices: tuple[int, ...] | None = None
    best_balance: tuple[PanelCovariateBalance, ...] | None = None
    best_key: tuple[float, float, int] | None = None
    # Fix the first seeded pair orientation to avoid both label-swapped copies.
    for mask in range(1 << (panel_size - 1)):
        panel_a_indices = (pairs[0][0],) + tuple(
            pair[(mask >> (index - 1)) & 1]
            for index, pair in enumerate(pairs[1:], start=1)
        )
        balances = _covariate_imbalances(candidates, panel_a_indices)
        objective = (*_partition_objective(balances), mask)
        if best_key is None or objective < best_key:
            best_key = objective
            best_indices = tuple(sorted(panel_a_indices))
            best_balance = balances
    if best_indices is None or best_balance is None:
        raise PanelMatchingError("no symmetric panel partition could be constructed")
    return best_indices, best_balance


def _gower_distance(
    left: FamilyPanelCandidate,
    right: FamilyPanelCandidate,
    scales: dict[str, tuple[float, float]],
) -> float:
    contributions: list[float] = []
    for name in _NUMERIC_COVARIATES:
        lower, upper = scales[name]
        contributions.append(
            0.0
            if upper == lower
            else abs(left.numeric(name) - right.numeric(name)) / (upper - lower)
        )
    contributions.extend(
        0.0 if left.categorical(name) == right.categorical(name) else 1.0
        for name in _CATEGORICAL_COVARIATES
    )
    return fsum(contributions) / len(PANEL_MATCHING_COVARIATES)


def _nearest_cross_panel_distances(
    panel_a: tuple[FamilyPanelCandidate, ...],
    panel_b: tuple[FamilyPanelCandidate, ...],
) -> tuple[CrossPanelNearestDistance, ...]:
    all_candidates = panel_a + panel_b
    scales = _numeric_scales(all_candidates)
    rows: list[CrossPanelNearestDistance] = []
    for source, others in ((panel_a, panel_b), (panel_b, panel_a)):
        for candidate in source:
            distance, nearest_id = min(
                (
                    _gower_distance(candidate, other, scales),
                    other.family_id,
                )
                for other in others
            )
            rows.append(
                CrossPanelNearestDistance(
                    family_id=candidate.family_id,
                    nearest_other_panel_family_id=nearest_id,
                    gower_distance=distance,
                )
            )
    return tuple(sorted(rows, key=lambda row: row.family_id))


def match_family_panels(
    candidates: Sequence[FamilyPanelCandidate],
    *,
    panel_size: int,
    allocation_seed: int,
) -> FamilyPanelMatch:
    """Select and partition two equal panels using only frozen M0 covariates."""

    _validate_options(panel_size, allocation_seed)
    values = _validated_candidates(candidates)
    required = 2 * panel_size
    if len(values) < required:
        raise PanelMatchingError(
            f"symmetric {panel_size}/{panel_size} panels require at least {required} families"
        )
    ordered = _selection_order(values, allocation_seed)
    selected = tuple(sorted(ordered[:required], key=lambda value: value.family_id))
    selected_ids = tuple(candidate.family_id for candidate in selected)
    excluded_ids = tuple(
        candidate.family_id
        for candidate in values
        if candidate.family_id not in selected_ids
    )
    panel_a_indices, balances = _best_symmetric_partition(
        selected,
        panel_size=panel_size,
        allocation_seed=allocation_seed,
    )
    panel_a_index_set = set(panel_a_indices)
    panel_a = tuple(
        candidate
        for index, candidate in enumerate(selected)
        if index in panel_a_index_set
    )
    panel_b = tuple(
        candidate
        for index, candidate in enumerate(selected)
        if index not in panel_a_index_set
    )
    objective = _partition_objective(balances)
    return FamilyPanelMatch(
        allocation_seed=allocation_seed,
        panel_size=panel_size,
        selected_family_ids=selected_ids,
        excluded_family_ids=excluded_ids,
        panel_a_family_ids=tuple(candidate.family_id for candidate in panel_a),
        panel_b_family_ids=tuple(candidate.family_id for candidate in panel_b),
        covariate_balance=balances,
        maximum_covariate_imbalance=objective[0],
        mean_covariate_imbalance=objective[1],
        cross_panel_nearest_distances=_nearest_cross_panel_distances(panel_a, panel_b),
    )


def crossed_seed_assignments(
    training_seeds: Sequence[int], *, allocation_seed: int
) -> tuple[SeedBlockPanelAssignment, ...]:
    """Freeze a balanced A/B target-role schedule for an even seed set."""

    _validate_options(1, allocation_seed)
    seeds = tuple(training_seeds)
    if (
        not seeds
        or len(seeds) % 2
        or len(seeds) != len(set(seeds))
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= MAX_ROOT_SEED
            for seed in seeds
        )
    ):
        raise PanelMatchingError(
            "training seeds must be a nonempty, unique, even-sized valid seed set"
        )
    order = list(seeds)
    schedule_seed = derive_namespaced_seed(
        allocation_seed,
        "family_selection.panel_crossover",
    )
    random.Random(schedule_seed).shuffle(order)
    panel_a_targeted = set(order[: len(order) // 2])
    return tuple(
        SeedBlockPanelAssignment(
            training_seed=seed,
            targeted_panel=(PanelLabel.A if seed in panel_a_targeted else PanelLabel.B),
            sentinel_panel=(PanelLabel.B if seed in panel_a_targeted else PanelLabel.A),
        )
        for seed in sorted(seeds)
    )


__all__ = [
    "CrossPanelNearestDistance",
    "FamilyPanelCandidate",
    "FamilyPanelMatch",
    "PANEL_MATCHING_COVARIATES",
    "PanelCovariateBalance",
    "PanelMatchingError",
    "TCESFamilyStructure",
    "build_family_panel_candidate",
    "crossed_seed_assignments",
    "match_family_panels",
    "parse_tces_family_structure",
]
