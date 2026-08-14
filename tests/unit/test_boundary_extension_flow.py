from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from duraseed.config import load_pilot_config
from duraseed.data.boundary import BoundaryFamilySummary, BoundaryItemSummary
from duraseed.data.boundary_confirmation import (
    build_confirmation_manifest,
    choose_refinement_family_ids,
)
from duraseed.data.boundary_protocol import (
    BOUNDARY_BROAD_EXTENSION_1_COHORT,
    BOUNDARY_BROAD_EXTENSION_2_COHORT,
    build_broad_manifest,
)
from duraseed.data.manifests import DatasetManifest, build_manifest
from duraseed.runners import (
    RunnerGateError,
    authenticate_extension1_source,
    authorize_launch,
)
from duraseed.runners.boundary_extension import (
    BoundaryBlockInputs,
    build_plan as build_boundary_plan,
    reduce_block,
    run_mock as run_boundary_mock,
)
from duraseed.tasks.tces import TCESGeneratorConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_pilot_config(ROOT / "duraseed_pilot_config.yaml")


def _summary(family: str, item_count: int, *, passing: bool) -> BoundaryFamilySummary:
    successes = 1 if passing else 0
    items = tuple(
        BoundaryItemSummary(
            task_id=f"{family}:item:{index}",
            item_index=index,
            intended_family_id=family,
            trials=16,
            successes=successes,
            posterior_mean_success=0.1 if passing else 0.01,
            informative_group_probability=0.5 if passing else 0.05,
            format_compliance=1.0,
            parser_valid_rate=1.0,
            truncation_rate=0.0,
        )
        for index in range(item_count)
    )
    return BoundaryFamilySummary(
        intended_family_id=family,
        sampler_checkpoint_path="mock://m0",
        group_size=8,
        items=items,
        equal_item_posterior_mean_success=0.1 if passing else 0.01,
        pooled_posterior_mean_success=0.1 if passing else 0.01,
        median_item_posterior_mean_success=0.1 if passing else 0.01,
        item_posterior_mean_dispersion=0.0,
        informative_probability_at_equal_item_mean=0.5 if passing else 0.05,
        mean_item_informative_probability=0.5 if passing else 0.05,
        successful_item_count=item_count if passing else 0,
        total_successes=item_count if passing else 0,
        maximum_item_success_share=1 / item_count if passing else None,
        format_compliance=1.0,
        parser_valid_rate=1.0,
        truncation_rate=0.0,
        observed_strategy_family_ids=(family,) if passing else (),
    )


def _block(cohort: str, broad, prior_broad, prior_confirm) -> BoundaryBlockInputs:
    families = tuple(sorted({row.intended_family for row in broad.records}))
    selected = families[0]
    successes = {family: int(family == selected) for family in families}
    candidates, audit = choose_refinement_family_ids(successes)
    assert candidates == (selected,)
    refined = tuple(sorted((*candidates, *audit)))
    return BoundaryBlockInputs(
        cohort_id=cohort,
        broad_manifest=broad,
        family_successes=successes,
        refinement_summaries=tuple(
            _summary(family, 4, passing=family == selected) for family in refined
        ),
        confirmation_summaries=(_summary(selected, 8, passing=True),),
        prior_broad_manifests=prior_broad,
        prior_confirmation_manifests=prior_confirm,
    )


@pytest.fixture(scope="module")
def boundary_inputs():
    generator = TCESGeneratorConfig(
        n_operands=3,
        operand_min=2,
        operand_max=20,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_attempts=256,
    )
    source = build_broad_manifest(
        generator,
        family_count=9,
        items_per_family=2,
        template_scan_ceiling=32,
        variant_index_offset=64,
    )
    templates = source.metadata["templates"]

    def cohort(index: int, parent: DatasetManifest | None) -> DatasetManifest:
        selected_templates = templates[index * 3 : (index + 1) * 3]
        family_ids = {row["family_id"] for row in selected_templates}
        metadata = {
            **source.metadata,
            "family_count": 3,
            "templates": selected_templates,
        }
        name = "mock-boundary-initial"
        if index:
            cohort_id = (
                BOUNDARY_BROAD_EXTENSION_1_COHORT
                if index == 1
                else BOUNDARY_BROAD_EXTENSION_2_COHORT
            )
            start = 64 * index
            name = f"mock-boundary-{cohort_id}"
            metadata["cohort_provenance"] = {
                "cohort_id": cohort_id,
                "distinct_family_ordinal_start": start,
                "distinct_family_ordinal_stop": start + 64,
            }
        return build_manifest(
            name=name,
            split="a_candidate",
            generator_version="1.0.0",
            root_seed=5,
            records=[
                row for row in source.records if row.intended_family in family_ids
            ],
            parent_manifest_id=parent.manifest_id if parent else None,
            metadata=metadata,
        )

    initial = cohort(0, None)
    initial_family = sorted({row.intended_family for row in initial.records})[0]
    initial_confirmation = build_confirmation_manifest(
        generator, initial, (initial_family,)
    )
    extension1 = cohort(1, initial)
    first = _block(
        BOUNDARY_BROAD_EXTENSION_1_COHORT,
        extension1,
        (initial,),
        (initial_confirmation,),
    )
    selected1 = next(
        family for family, successes in first.family_successes.items() if successes
    )
    extension1_confirmation = build_confirmation_manifest(
        generator, extension1, (selected1,)
    )
    extension2 = cohort(2, extension1)
    second = _block(
        BOUNDARY_BROAD_EXTENSION_2_COHORT,
        extension2,
        (initial, extension1),
        (initial_confirmation, extension1_confirmation),
    )
    return generator, first, second


def test_boundary_real_mock_flow_runs_both_blocks_and_fails_freeze_closed(
    boundary_inputs, monkeypatch
) -> None:
    import duraseed.runners.boundary_extension as boundary_runner
    from duraseed.data.panel_capacity import FamilyCapacityAudit

    generator, first, second = boundary_inputs
    order = []
    monkeypatch.setattr(
        boundary_runner,
        "build_extension2_manifest",
        lambda _config: order.append("freeze") or second.broad_manifest,
    )
    monkeypatch.setattr(
        boundary_runner,
        "audit_family_split_capacities",
        lambda _templates, family_ids, *_args, **_kwargs: tuple(
            order.append("capacity") or FamilyCapacityAudit(family_id, (), 1, True)
            for family_id in family_ids
        ),
    )
    result = run_boundary_mock(generator, first, second)

    assert order[0] == "freeze"
    assert order.count("capacity") == 2
    assert result.extension1.capacity_audits
    assert result.extension2.capacity_audits
    assert result.extension1.evidence.final_eligible_family_ids
    assert result.extension2.evidence.final_eligible_family_ids
    assert result.composite_status == "blocked_pending_three_cohort_equivalence"
    assert tuple(action.name for action in build_boundary_plan().actions) == (
        "freeze-extension2-manifest",
        "extension1-confirm",
        "extension2-broad",
        "extension2-refine",
        "extension2-confirm",
        "three-cohort-composite",
    )


def test_boundary_mock_rejects_scientific_output_path(boundary_inputs) -> None:
    generator, first, second = boundary_inputs
    with pytest.raises(RunnerGateError, match="mock output root"):
        run_boundary_mock(
            generator,
            first,
            second,
            output_root=ROOT / "runs/mock",
        )


def test_boundary_reducer_rejects_incomplete_family_coverage(boundary_inputs) -> None:
    generator, first, _ = boundary_inputs
    incomplete = dict(first.family_successes)
    incomplete.pop(next(key for key, value in incomplete.items() if value == 0))
    with pytest.raises(RunnerGateError, match="family-success coverage"):
        reduce_block(generator, replace(first, family_successes=incomplete))


def test_boundary_reducer_rejects_corrupt_carried_capacity_order(
    boundary_inputs,
) -> None:
    from duraseed.data.panel_capacity import FamilyCapacityAudit

    generator, first, _ = boundary_inputs
    corrupt = (FamilyCapacityAudit("wrong-family", (), 1, True),)
    with pytest.raises(RunnerGateError, match="carried capacity audits"):
        reduce_block(generator, replace(first, capacity_audits=corrupt))


def test_boundary_reducer_reuses_valid_carried_capacity_audits(
    boundary_inputs, monkeypatch
) -> None:
    import duraseed.runners.boundary_extension as boundary_runner
    from duraseed.data.panel_capacity import FamilyCapacityAudit

    generator, first, _ = boundary_inputs
    finalist = first.confirmation_summaries[0].intended_family_id
    carried = (FamilyCapacityAudit(finalist, (), 1, True),)
    monkeypatch.setattr(
        boundary_runner,
        "audit_family_split_capacities",
        lambda *_args, **_kwargs: pytest.fail("capacity was recomputed"),
    )

    result = reduce_block(generator, replace(first, capacity_audits=carried))

    assert result.capacity_audits == carried


def test_completed_extension1_handoff_authenticates() -> None:
    boundary = ROOT / "frozen/v0/runs/tinker-calibration/boundary"
    initial = DatasetManifest.model_validate_json(
        (
            boundary / "boundary-broad-20260809T214400Z/a_candidate_manifest.json"
        ).read_text()
    )
    contract = authenticate_extension1_source(
        CONFIG,
        boundary / "boundary-broad-extension1-20260810T223000Z",
        boundary / "boundary-refine-extension1-20260811T104725Z",
        expected_parent_manifest_id=initial.manifest_id,
    )
    assert contract.cohort_id == BOUNDARY_BROAD_EXTENSION_1_COHORT


def test_boundary_authorization_is_explicit_and_exact() -> None:
    ready = dict(
        live_smoke_passed=True,
        boundary_extension_human_approval=True,
        extension1_source_authenticated=True,
        remaining_balance_verified=True,
    )
    with pytest.raises(RunnerGateError, match="--authorize"):
        authorize_launch(
            build_boundary_plan(),
            execute=False,
            authorized_cost_usd="120",
            preconditions=ready,
        )
    with pytest.raises(RunnerGateError, match=r"\$120"):
        authorize_launch(
            build_boundary_plan(),
            execute=True,
            authorized_cost_usd="119.99",
            preconditions=ready,
        )
    assert authorize_launch(
        build_boundary_plan(),
        execute=True,
        authorized_cost_usd="120.00",
        preconditions=ready,
    ).authorized_cost_usd == Decimal("120.00")
