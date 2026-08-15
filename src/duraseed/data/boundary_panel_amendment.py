"""Accepted algebraic de-duplication amendment for the boundary panel freeze."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from duraseed.data.panel_capacity import (
    PANEL_CAPACITY_PROBE_MULTIPLIER,
    PANEL_FILTERED_SPLIT_SCAN_MULTIPLIER,
    PANEL_SELECTED_TEST_SINGLE_MINIMUM,
    PANEL_SPLIT_MINIMUMS,
    FamilyCapacityAudit,
)
from duraseed.data.panel_matching import (
    PANEL_MATCHING_COVARIATES,
    FamilyPanelCandidate,
    FamilyPanelMatch,
    crossed_seed_assignments,
    match_family_panels,
    parse_tces_family_structure,
)
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.provenance import canonical_json_hash


_RANK_COUNT = 5
_ZERO_MONOMIAL = (0,) * _RANK_COUNT
_OPERATORS = ("ADD", "SUB", "MUL", "DIV")
_PANEL_REQUIREMENTS = {
    **dict(PANEL_SPLIT_MINIMUMS),
    "a_test_single": PANEL_SELECTED_TEST_SINGLE_MINIMUM,
}
Polynomial = dict[tuple[int, ...], int]
RationalPolynomial = tuple[Polynomial, Polynomial]


class BoundaryPanelAmendmentError(ValueError): ...


@dataclass(frozen=True, slots=True)
class AlgebraicFamilyClass:
    class_id: str
    member_family_ids: tuple[str, ...]
    representative_family_id: str


@dataclass(frozen=True, slots=True)
class BoundaryPanelAmendmentResult:
    algebraic_classes: tuple[AlgebraicFamilyClass, ...]
    representative_candidates: tuple[FamilyPanelCandidate, ...]
    panel_capacity_audits: tuple[FamilyCapacityAudit, ...]
    panel_eligible_family_ids: tuple[str, ...]
    match: FamilyPanelMatch | None
    intermediate_family_ids: tuple[str, ...]
    selected_capacity_audits: tuple[FamilyCapacityAudit, ...]
    candidate_payload: dict[str, object]
    matching_report_payload: dict[str, object] | None
    panel_artifact: FamilyPanelArtifact | None
    amendment_payload: dict[str, object]
    confirmation_summary: dict[str, object]


def _poly_add(left: Polynomial, right: Polynomial, sign: int = 1) -> Polynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, 0) + sign * coefficient
    return {monomial: value for monomial, value in result.items() if value}


def _poly_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_monomial, left_value in left.items():
        for right_monomial, right_value in right.items():
            powers = zip(left_monomial, right_monomial, strict=True)
            monomial = tuple(left + right for left, right in powers)
            result[monomial] = result.get(monomial, 0) + left_value * right_value
    return {monomial: value for monomial, value in result.items() if value}


def _family_rational(family_id: str) -> RationalPolynomial:
    structure = family_id.split("|", 1)[0]
    leaves: list[int] = []

    def parse(position: int) -> tuple[RationalPolynomial, int]:
        if position < len(structure) and structure[position] == "r":
            end = position + 1
            while end < len(structure) and structure[end].isdigit():
                end += 1
            rank = int(structure[position + 1 : end])
            if not 1 <= rank <= _RANK_COUNT:
                raise BoundaryPanelAmendmentError("family rank is outside r1..r5")
            leaves.append(rank)
            monomial = [0] * _RANK_COUNT
            monomial[rank - 1] = 1
            return ({tuple(monomial): 1}, {_ZERO_MONOMIAL: 1}), end
        operator = next(
            (name for name in _OPERATORS if structure.startswith(f"{name}(", position)),
            None,
        )
        if operator is None:
            raise BoundaryPanelAmendmentError("family structure is malformed")
        left, cursor = parse(position + len(operator) + 1)
        if cursor >= len(structure) or structure[cursor] != ",":
            raise BoundaryPanelAmendmentError("family structure is malformed")
        right, cursor = parse(cursor + 1)
        if cursor >= len(structure) or structure[cursor] != ")":
            raise BoundaryPanelAmendmentError("family structure is malformed")
        left_num, left_den = left
        right_num, right_den = right
        denominator = _poly_multiply(left_den, right_den)
        if operator == "ADD":
            numerator = _poly_add(
                _poly_multiply(left_num, right_den),
                _poly_multiply(right_num, left_den),
            )
        elif operator == "SUB":
            numerator = _poly_add(
                _poly_multiply(left_num, right_den),
                _poly_multiply(right_num, left_den),
                -1,
            )
        elif operator == "MUL":
            numerator = _poly_multiply(left_num, right_num)
        else:
            numerator = _poly_multiply(left_num, right_den)
            denominator = _poly_multiply(left_den, right_num)
        if not denominator:
            raise BoundaryPanelAmendmentError("family has a zero symbolic denominator")
        return (numerator, denominator), cursor + 1

    value, end = parse(0)
    if end != len(structure) or sorted(leaves) != list(range(1, _RANK_COUNT + 1)):
        raise BoundaryPanelAmendmentError("family must use each of r1..r5 exactly once")
    return value


def _algebraically_equal(left: RationalPolynomial, right: RationalPolynomial) -> bool:
    return _poly_multiply(left[0], right[1]) == _poly_multiply(right[0], left[1])


def _ranked_candidates(
    candidates: Sequence[FamilyPanelCandidate], allocation_seed: int
) -> tuple[FamilyPanelCandidate, ...]:
    from duraseed.data.panel_matching import _selection_order

    values = tuple(sorted(candidates, key=lambda row: row.family_id))
    family_ids = tuple(row.family_id for row in values)
    if not values or len(family_ids) != len(set(family_ids)):
        raise BoundaryPanelAmendmentError("candidate family IDs must be unique")
    return _selection_order(values, allocation_seed)


def algebraic_family_classes(
    candidates: Sequence[FamilyPanelCandidate], allocation_seed: int
) -> tuple[AlgebraicFamilyClass, ...]:
    """Collapse exactly equal rational functions and retain the frozen-rank winner."""

    ranked = _ranked_candidates(candidates, allocation_seed)
    rank = {row.family_id: index for index, row in enumerate(ranked)}
    for row in ranked:
        structure = parse_tces_family_structure(row.family_id)
        if (
            structure.operator_multiset != row.operator_multiset
            or structure.fractional_intermediate_profile
            != row.fractional_intermediate_profile
        ):
            raise BoundaryPanelAmendmentError("candidate structure covariates changed")
    values = {row.family_id: _family_rational(row.family_id) for row in ranked}
    groups: list[list[str]] = []
    for family_id in sorted(values):
        group = next(
            (
                members
                for members in groups
                if _algebraically_equal(values[family_id], values[members[0]])
            ),
            None,
        )
        if group is None:
            groups.append([family_id])
        else:
            group.append(family_id)
    classes = []
    for members in groups:
        ordered_members = tuple(sorted(members))
        representative = min(ordered_members, key=rank.__getitem__)
        classes.append(
            AlgebraicFamilyClass(
                class_id=canonical_json_hash(
                    {
                        "schema": "duraseed-tces-algebraic-family-class-v1",
                        "members": list(ordered_members),
                    }
                ),
                member_family_ids=ordered_members,
                representative_family_id=representative,
            )
        )
    by_representative = {row.representative_family_id: row for row in classes}
    representatives = tuple(row for row in ranked if row.family_id in by_representative)
    reranked = _ranked_candidates(representatives, allocation_seed)
    return tuple(by_representative[row.family_id] for row in reranked)


def _validate_panel_audits(
    audits: Sequence[FamilyCapacityAudit], representative_ids: tuple[str, ...]
) -> tuple[FamilyCapacityAudit, ...]:
    values = tuple(audits)
    if tuple(row.family_id for row in values) != representative_ids:
        raise BoundaryPanelAmendmentError(
            "panel-capacity evidence must follow the representative ranking"
        )
    for audit in values:
        rows = {row.split: row for row in audit.split_capacities}
        if set(rows) != set(_PANEL_REQUIREMENTS):
            raise BoundaryPanelAmendmentError("panel-capacity split set changed")
        for split, required in _PANEL_REQUIREMENTS.items():
            row = rows[split]
            multiplier = (
                PANEL_FILTERED_SPLIT_SCAN_MULTIPLIER
                if split == "a_test_single"
                else PANEL_CAPACITY_PROBE_MULTIPLIER
            )
            if (
                row.required_instances != required
                or row.probe_indices != required * multiplier
                or row.passed != (row.available_disjoint_instances >= required)
            ):
                raise BoundaryPanelAmendmentError(
                    "panel-capacity requirement or scan budget changed"
                )
        ordinary_passed = all(rows[name].passed for name, _ in PANEL_SPLIT_MINIMUMS)
        all_passed = all(row.passed for row in rows.values())
        if not ordinary_passed or audit.passed != all_passed:
            raise BoundaryPanelAmendmentError(
                "representative lost ordinary eligibility or audit disagrees"
            )
    return values


def reduce_panel_amendment(
    candidates: Sequence[FamilyPanelCandidate],
    panel_capacity_audits: Sequence[FamilyCapacityAudit],
    *,
    panel_size: int,
    intermediate_size: int,
    allocation_seed: int,
    training_seeds: Sequence[int],
    m0_checkpoint_path: str,
) -> BoundaryPanelAmendmentResult:
    """Apply the accepted panel and separate intermediate-pool rules."""

    classes = algebraic_family_classes(candidates, allocation_seed)
    source_candidates = tuple(sorted(candidates, key=lambda row: row.family_id))
    by_family = {row.family_id: row for row in candidates}
    representatives = tuple(by_family[row.representative_family_id] for row in classes)
    representative_ids = tuple(row.family_id for row in representatives)
    audits = _validate_panel_audits(panel_capacity_audits, representative_ids)
    panel_eligible_ids = tuple(row.family_id for row in audits if row.passed)
    panel_eligible_set = set(panel_eligible_ids)
    panel_candidates = tuple(
        row for row in representatives if row.family_id in panel_eligible_set
    )
    match = (
        match_family_panels(
            panel_candidates,
            panel_size=panel_size,
            allocation_seed=allocation_seed,
        )
        if len(panel_candidates) >= 2 * panel_size
        else None
    )
    selected_order = match.selected_family_ids if match is not None else ()
    selected = set(selected_order)
    intermediate = tuple(
        row.family_id for row in representatives if row.family_id not in selected
    )[:intermediate_size]
    audit_by_family = {row.family_id: row for row in audits}
    selected_audits = tuple(audit_by_family[family_id] for family_id in selected_order)
    candidate_payload: dict[str, object] = {
        "schema": "duraseed-panel-candidates-algebraic-dedup-v1",
        "source_family_count": len(tuple(candidates)),
        "algebraic_class_count": len(classes),
        "algebraic_classes": [asdict(row) for row in classes],
        "matching_covariates": list(PANEL_MATCHING_COVARIATES),
        "panel_eligible_family_ids": list(panel_eligible_ids),
        "source_candidates": [asdict(row) for row in source_candidates],
        "candidates": [
            asdict(row)
            for row in sorted(representatives, key=lambda row: row.family_id)
        ],
    }
    candidate_id = canonical_json_hash(candidate_payload)
    report = (
        {
            "schema": "duraseed-panel-match-algebraic-dedup-v1",
            "selection_rule": (
                "panel: unchanged frozen rank after 22/22 a_test_single; "
                "intermediate: full ordinary-eligible representative rank outside panels"
            ),
            "distance": "equal_weight_gower",
            "match": asdict(match),
            "intermediate_family_ids": list(intermediate),
        }
        if match is not None
        else None
    )
    report_id = canonical_json_hash(report) if report is not None else None
    resolved = match is not None and len(intermediate) == intermediate_size
    artifact = (
        FamilyPanelArtifact(
            m0_checkpoint_path=m0_checkpoint_path,
            candidate_family_table_manifest_id=candidate_id,
            panel_matching_report_id=report_id,
            allocation_seed=allocation_seed,
            panel_a_family_ids=match.panel_a_family_ids,
            panel_b_family_ids=match.panel_b_family_ids,
            seed_block_assignments=crossed_seed_assignments(
                training_seeds, allocation_seed=allocation_seed
            ),
        )
        if resolved and match is not None and report_id is not None
        else None
    )
    amendment = {
        "schema_version": "duraseed-boundary-panel-amendment-v1",
        "status": "resolved" if resolved else "unresolved",
        "source_population": "fixed_49_observation_eligible_families",
        "algebraic_equivalence": "exact expanded rational identity over r1 through r5",
        "representative_rule": (
            "highest confirmed I8, then ordinary split capacity; "
            "the frozen allocation seed resolves exact ties"
        ),
        "representative_family_ids": list(representative_ids),
        "panel_eligibility": {
            "requirements": dict(_PANEL_REQUIREMENTS),
            "protected_family_ids": list(representative_ids),
            "eligible_family_ids": list(panel_eligible_ids),
        },
        "intermediate_eligibility": {
            "requirements": dict(PANEL_SPLIT_MINIMUMS),
            "a_test_single_required": False,
            "excluded_panel_family_ids": sorted(selected),
            "selection": "first 12 full-representative-rank families outside panels",
            "selected_family_ids": list(intermediate),
        },
        "candidate_table_id": candidate_id,
        "panel_matching_report_id": report_id,
    }
    summary = {
        "status": "confirmed_panels_frozen"
        if resolved
        else "panel_amendment_unresolved",
        "source_kind": "three_cohort_boundary_freeze_v1",
        "selection_performed": resolved,
        "pilot_started": False,
        "source_observation_eligible_family_count": len(tuple(candidates)),
        "algebraic_class_count": len(classes),
        "representative_family_ids": list(representative_ids),
        "split_capacity_requirements": dict(PANEL_SPLIT_MINIMUMS),
        "split_capacity_eligible_family_ids": list(representative_ids),
        "panel_capacity_requirements": dict(_PANEL_REQUIREMENTS),
        "panel_capacity_eligible_family_ids": list(panel_eligible_ids),
        "selected_panel_split_capacity_passed": bool(selected_audits)
        and all(row.passed for row in selected_audits),
        "panel_a_family_ids": list(match.panel_a_family_ids)
        if resolved and match
        else [],
        "panel_b_family_ids": list(match.panel_b_family_ids)
        if resolved and match
        else [],
        "intermediate_family_ids": list(intermediate) if resolved else [],
        "candidate_table_id": candidate_id if resolved else None,
        "panel_matching_report_id": report_id if resolved else None,
    }
    return BoundaryPanelAmendmentResult(
        classes,
        representatives,
        audits,
        panel_eligible_ids,
        match,
        intermediate if resolved else (),
        selected_audits,
        candidate_payload,
        report,
        artifact,
        amendment,
        summary,
    )


__all__ = [
    "AlgebraicFamilyClass",
    "BoundaryPanelAmendmentError",
    "BoundaryPanelAmendmentResult",
    "algebraic_family_classes",
    "reduce_panel_amendment",
]
