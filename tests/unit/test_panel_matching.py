from __future__ import annotations

from dataclasses import replace

import pytest

from duraseed.data import (
    BoundaryFamilySummary,
    BoundaryItemSummary,
    FamilyPanelCandidate,
    PanelLabel,
    PanelMatchingError,
    build_family_panel_candidate,
    crossed_seed_assignments,
    match_family_panels,
    parse_tces_family_structure,
)
from duraseed.data.manifests import DatasetManifest, build_manifest, build_tces_record
from duraseed.data.splits import TCESSplitBuilder, derive_tces_split_seed
from duraseed.tasks.tces import TCESFamilyGenerator, TCESGeneratorConfig


def _build_broad_manifest(
    generator_config: TCESGeneratorConfig,
    *,
    root_seed: int = 5,
    family_count: int,
    items_per_family: int,
    template_scan_ceiling: int,
    variant_index_offset: int,
) -> DatasetManifest:
    """Construct the initial-cohort fixture through carried instrument APIs."""

    split = "a_candidate"
    candidates = TCESSplitBuilder(root_seed, generator_config).lazy_split(
        split, size=template_scan_ceiling
    )
    distinct_templates = []
    seen_families: set[str] = set()
    for accepted_index in range(template_scan_ceiling):
        template = candidates[accepted_index]
        if template.intended_family in seen_families:
            continue
        distinct_templates.append(template)
        seen_families.add(template.intended_family)
        if len(distinct_templates) == family_count:
            break
    if len(distinct_templates) != family_count:
        raise ValueError("candidate prefix lacks the requested distinct families")

    split_seed = derive_tces_split_seed(root_seed, split)
    family_config = replace(generator_config, split=split)
    records = []
    template_metadata = []
    for family_ordinal, template in enumerate(distinct_templates):
        start_index = variant_index_offset + family_ordinal * items_per_family
        variants = TCESFamilyGenerator(
            split_seed, template, family_config
        ).generate_many(items_per_family, start_index=start_index)
        records.extend(build_tces_record(variant) for variant in variants)
        template_metadata.append(
            {
                "family_id": template.intended_family,
                "template_content_hash": template.content_hash,
                "template_item_index": template.item_index,
                "variant_item_indices": [variant.item_index for variant in variants],
            }
        )
    return build_manifest(
        name="m0-boundary-broad-a-candidate",
        split=split,
        generator_version="1.0.0",
        root_seed=root_seed,
        records=records,
        parent_manifest_id=None,
        metadata={
            "scope": "m0_boundary_broad_screen",
            "scientific_manifest": False,
            "generation_mode": "exact_family_numeric_variants_v1",
            "family_count": family_count,
            "items_per_family": items_per_family,
            "templates": template_metadata,
        },
    )


def _candidate(
    family_id: str,
    *,
    level: int,
    informative_probability: float,
    available: int = 64,
) -> FamilyPanelCandidate:
    fractional = level % 2 == 1
    operators = ("SUB", "DIV") if fractional else ("ADD", "MUL")
    return FamilyPanelCandidate(
        family_id=family_id,
        m0_posterior_mean_success=0.05 + 0.05 * level,
        informative_group_probability_i8=informative_probability,
        tree_depth=2 + level,
        operator_multiset=operators,
        noncommutative_operation_count=2 if fractional else 0,
        fractional_intermediate_profile=("F", "I") if fractional else ("I", "I"),
        target_magnitude=10.0 + 10.0 * level,
        valid_family_multiplicity=2.0 + level,
        teacher_trace_length=20.0 + 5.0 * level,
        available_disjoint_instances=available,
    )


def test_matching_is_symmetric_balanced_and_input_order_invariant() -> None:
    candidates = (
        _candidate("low-a", level=0, informative_probability=0.30),
        _candidate("low-b", level=0, informative_probability=0.30),
        _candidate("high-a", level=1, informative_probability=0.40),
        _candidate("high-b", level=1, informative_probability=0.40),
    )

    first = match_family_panels(candidates, panel_size=2, allocation_seed=101)
    second = match_family_panels(
        tuple(reversed(candidates)), panel_size=2, allocation_seed=101
    )

    assert first == second
    assert first.panel_size == 2
    assert not first.excluded_family_ids
    assert set(first.panel_a_family_ids).isdisjoint(first.panel_b_family_ids)
    assert set(first.panel_a_family_ids + first.panel_b_family_ids) == {
        candidate.family_id for candidate in candidates
    }
    assert (
        sum(family_id.startswith("low") for family_id in first.panel_a_family_ids) == 1
    )
    assert (
        sum(family_id.startswith("low") for family_id in first.panel_b_family_ids) == 1
    )
    assert first.maximum_covariate_imbalance == 0.0
    assert first.mean_covariate_imbalance == 0.0
    assert len(first.cross_panel_nearest_distances) == 4
    assert {row.gower_distance for row in first.cross_panel_nearest_distances} == {0.0}


def test_extra_candidates_are_selected_by_predeclared_informativeness_then_capacity() -> (
    None
):
    candidates = tuple(
        _candidate(
            f"family-{index}",
            level=index % 2,
            informative_probability=0.25 + 0.05 * index,
            available=100 - index,
        )
        for index in range(5)
    )

    result = match_family_panels(candidates, panel_size=2, allocation_seed=11)

    assert result.excluded_family_ids == ("family-0",)
    assert result.selected_family_ids == (
        "family-1",
        "family-2",
        "family-3",
        "family-4",
    )


def test_core_twelve_by_twelve_match_is_bounded_and_complete() -> None:
    candidates = tuple(
        _candidate(
            f"family-{index:02d}",
            level=index // 2,
            informative_probability=0.30 + 0.001 * (index // 2),
        )
        for index in range(24)
    )

    result = match_family_panels(candidates, panel_size=12, allocation_seed=211)

    assert len(result.panel_a_family_ids) == 12
    assert len(result.panel_b_family_ids) == 12
    assert len(result.covariate_balance) == 10
    assert len(result.cross_panel_nearest_distances) == 24


def test_core_match_fails_closed_when_fewer_than_twenty_four_families_exist() -> None:
    candidates = tuple(
        _candidate(
            f"family-{index}",
            level=index % 2,
            informative_probability=0.30,
        )
        for index in range(23)
    )

    with pytest.raises(PanelMatchingError, match="require at least 24"):
        match_family_panels(candidates, panel_size=12, allocation_seed=101)


def test_crossed_seed_schedule_is_deterministic_balanced_and_canonical() -> None:
    first = crossed_seed_assignments((71, 11, 47, 29), allocation_seed=101)
    second = crossed_seed_assignments((29, 47, 11, 71), allocation_seed=101)

    assert first == second
    assert tuple(row.training_seed for row in first) == (11, 29, 47, 71)
    assert sum(row.targeted_panel is PanelLabel.A for row in first) == 2
    assert sum(row.targeted_panel is PanelLabel.B for row in first) == 2
    assert all(row.targeted_panel is not row.sentinel_panel for row in first)


@pytest.mark.parametrize("seeds", ((), (11,), (11, 11), (11, -1)))
def test_crossed_schedule_rejects_unbalanced_or_invalid_seed_sets(
    seeds: tuple[int, ...],
) -> None:
    with pytest.raises(PanelMatchingError, match="even-sized valid seed set"):
        crossed_seed_assignments(seeds, allocation_seed=101)


def test_candidate_covariates_fail_closed_on_internal_disagreement() -> None:
    with pytest.raises(ValueError, match="disagrees with operator_multiset"):
        FamilyPanelCandidate(
            family_id="bad-family",
            m0_posterior_mean_success=0.1,
            informative_group_probability_i8=0.3,
            tree_depth=2,
            operator_multiset=("ADD", "SUB"),
            noncommutative_operation_count=0,
            fractional_intermediate_profile=("I", "I"),
            target_magnitude=10.0,
            valid_family_multiplicity=2.0,
            teacher_trace_length=20.0,
            available_disjoint_instances=64,
        )


def test_canonical_family_structure_is_decoded_without_numeric_instances() -> None:
    structure = parse_tces_family_structure(
        "ADD(MUL(r1,r2),SUB(r3,r4))|intermediates=I,F,I"
    )

    assert structure.tree_depth == 3
    assert structure.operator_multiset == ("ADD", "SUB", "MUL")
    assert structure.noncommutative_operation_count == 1
    assert structure.fractional_intermediate_profile == ("I", "F", "I")

    with pytest.raises(PanelMatchingError, match="aligned"):
        parse_tces_family_structure("ADD(r1,r2)|intermediates=I,I")


def test_candidate_builder_uses_equal_item_records_and_renderer_token_counts() -> None:
    manifest = _build_broad_manifest(
        TCESGeneratorConfig(
            n_operands=3,
            operand_min=2,
            operand_max=20,
            max_tree_depth=3,
            max_ast_nodes=5,
            max_attempts=256,
        ),
        family_count=1,
        items_per_family=2,
        template_scan_ceiling=8,
        variant_index_offset=16,
    )
    records = tuple(manifest.records)
    family_id = records[0].intended_family
    items = tuple(
        BoundaryItemSummary(
            task_id=record.task_id,
            item_index=record.item_index,
            intended_family_id=family_id,
            trials=16,
            successes=2,
            posterior_mean_success=2.5 / 17,
            informative_group_probability=0.7,
            format_compliance=1.0,
            parser_valid_rate=1.0,
            truncation_rate=0.0,
        )
        for record in records
    )
    summary = BoundaryFamilySummary(
        intended_family_id=family_id,
        sampler_checkpoint_path="tinker://m0/sampler",
        group_size=8,
        items=items,
        equal_item_posterior_mean_success=2.5 / 17,
        pooled_posterior_mean_success=4.5 / 33,
        median_item_posterior_mean_success=2.5 / 17,
        item_posterior_mean_dispersion=0.0,
        informative_probability_at_equal_item_mean=0.7,
        mean_item_informative_probability=0.7,
        successful_item_count=2,
        total_successes=4,
        maximum_item_success_share=0.5,
        format_compliance=1.0,
        parser_valid_rate=1.0,
        truncation_rate=0.0,
        observed_strategy_family_ids=(family_id,),
    )

    candidate = build_family_panel_candidate(
        summary,
        records,
        teacher_trace_token_counts={
            records[0].task_id: 20,
            records[1].task_id: 24,
        },
        available_disjoint_instances=64,
    )

    assert candidate.family_id == family_id
    assert candidate.teacher_trace_length == 22.0
    assert candidate.valid_family_multiplicity == pytest.approx(
        sum(record.valid_family_count for record in records) / 2
    )
    assert candidate.available_disjoint_instances == 64

    with pytest.raises(PanelMatchingError, match="exactly cover"):
        build_family_panel_candidate(
            summary,
            records,
            teacher_trace_token_counts={records[0].task_id: 20},
            available_disjoint_instances=64,
        )
