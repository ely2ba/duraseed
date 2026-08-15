from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from duraseed.config import load_pilot_config
from duraseed.data.boundary import BoundaryFamilySummary, BoundaryItemSummary
from duraseed.data.boundary_freeze import (
    BoundaryFreezeReductionError,
    _reduce_three_cohort_panels,
)
from duraseed.data.boundary_freeze_contracts import (
    BoundaryFreezeCohort,
    freeze_settings_from_config,
)
from duraseed.data.manifests import build_manifest
from duraseed.data.panel_capacity import FamilyCapacityAudit
from duraseed.data.panel_matching import FamilyPanelCandidate
from duraseed.provenance import canonical_json_bytes, canonical_json_hash


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_pilot_config(ROOT / "duraseed_pilot_config.yaml")


def _manifest(name: str, marker: str, *, parent: str | None = None, confirmation=False):
    metadata = {"marker": marker, "families" if confirmation else "templates": []}
    return build_manifest(
        name=name,
        split="a_candidate",
        generator_version="1.0.0",
        root_seed=5,
        records=(),
        parent_manifest_id=parent,
        task_family="tces",
        metadata=metadata,
    )


def _summary(family_id: str) -> BoundaryFamilySummary:
    items = tuple(
        BoundaryItemSummary(
            task_id=f"{family_id}:item:{index}",
            item_index=index,
            intended_family_id=family_id,
            trials=16,
            successes=1,
            posterior_mean_success=0.1,
            informative_group_probability=0.5,
            format_compliance=1.0,
            parser_valid_rate=1.0,
            truncation_rate=0.0,
        )
        for index in range(8)
    )
    return BoundaryFamilySummary(
        intended_family_id=family_id,
        sampler_checkpoint_path="tinker://m0/sampler",
        group_size=8,
        items=items,
        equal_item_posterior_mean_success=0.1,
        pooled_posterior_mean_success=0.1,
        median_item_posterior_mean_success=0.1,
        item_posterior_mean_dispersion=0.0,
        informative_probability_at_equal_item_mean=0.5,
        mean_item_informative_probability=0.5,
        successful_item_count=8,
        total_successes=8,
        maximum_item_success_share=0.125,
        format_compliance=1.0,
        parser_valid_rate=1.0,
        truncation_rate=0.0,
        observed_strategy_family_ids=(family_id,),
    )


def _cohorts(count: int = 36) -> tuple[BoundaryFreezeCohort, ...]:
    family_ids = tuple(f"family-{index:03d}" for index in range(count))
    initial_ids = family_ids[: min(15, count)]
    remaining = family_ids[len(initial_ids) :]
    split = (len(remaining) + 1) // 2
    groups = (initial_ids, remaining[:split], remaining[split:])
    broad0 = _manifest("broad-initial", "b0")
    broad1 = _manifest("broad-extension-1", "b1", parent=broad0.manifest_id)
    broad2 = _manifest("broad-extension-2", "b2", parent=broad1.manifest_id)
    broads = (broad0, broad1, broad2)
    confirmations = tuple(
        _manifest(f"confirm-{index}", f"c{index}", confirmation=True)
        for index in range(3)
    )
    return tuple(
        BoundaryFreezeCohort(
            cohort_id=cohort_id,
            broad_manifest=broad,
            confirmation_manifest=confirmation,
            finalist_summaries=tuple(_summary(family_id) for family_id in ids),
            sampler_checkpoint_path="tinker://m0/sampler",
            locked_eligible_family_ids=initial_ids if index == 0 else (),
        )
        for index, (cohort_id, broad, confirmation, ids) in enumerate(
            zip(
                ("initial", "extension_1", "extension_2"),
                broads,
                confirmations,
                groups,
                strict=True,
            )
        )
    )


def _candidate(family_id: str, capacity: int) -> FamilyPanelCandidate:
    index = int(family_id.rsplit("-", 1)[1])
    return FamilyPanelCandidate(
        family_id=family_id,
        m0_posterior_mean_success=0.1 + index / 10_000,
        informative_group_probability_i8=0.7 + index / 10_000,
        tree_depth=3,
        operator_multiset=("ADD", "SUB"),
        noncommutative_operation_count=1,
        fractional_intermediate_profile=("I", "I"),
        target_magnitude=10 + index,
        valid_family_multiplicity=1,
        teacher_trace_length=20 + index,
        available_disjoint_instances=capacity,
    )


def _patch_primitives(monkeypatch, *, fail_combined=(), fail_selected=False):
    import duraseed.data.boundary_freeze as freeze

    regenerated = []
    calls = []

    def regenerate(_config, broad, family_ids):
        regenerated.append((broad, tuple(family_ids)))
        return {family_id: family_id for family_id in family_ids}

    def audit_many(
        templates,
        family_ids,
        _config,
        *,
        root_seed,
        requirements=(),
        **_kwargs,
    ):
        selected = "a_test_single" in dict(requirements)
        audits = []
        for family_id in family_ids:
            template = templates[family_id]
            calls.append((template, root_seed, selected))
            passed = template not in fail_combined and not (selected and fail_selected)
            audits.append(FamilyCapacityAudit(template, (), 200 + len(calls), passed))
        return tuple(audits)

    monkeypatch.setattr(freeze, "regenerate_family_templates", regenerate)
    monkeypatch.setattr(freeze, "audit_family_split_capacities", audit_many)
    monkeypatch.setattr(
        freeze,
        "build_family_panel_candidate",
        lambda summary, _records, *, available_disjoint_instances, **_kwargs: (
            _candidate(summary.intended_family_id, available_disjoint_instances)
        ),
    )
    return regenerated, calls


def _reduce(monkeypatch, cohorts, **patch_options):
    regenerated, calls = _patch_primitives(monkeypatch, **patch_options)
    eligible = {
        summary.intended_family_id
        for cohort in cohorts
        for summary in cohort.finalist_summaries
    }.difference(patch_options.get("fail_combined", ()))
    result = _reduce_three_cohort_panels(
        cohorts,
        settings=freeze_settings_from_config(CONFIG),
        teacher_trace_token_counts={family_id: {} for family_id in eligible},
    )
    return result, regenerated, calls


def test_freeze_config_projection_is_exact() -> None:
    settings = freeze_settings_from_config(CONFIG)
    payload = canonical_json_bytes(settings.projection())

    assert len(payload) == 1014
    assert settings.projection_sha256 == (
        "sha256:b1ddb88ff9ef31a0f0396b31c87e6d089c7b973705ae42615589c21a66fe6361"
    )
    assert settings.capacity_root_seed == 11
    assert settings.training_seeds == (
        17,
        37,
        11,
        29,
        47,
        71,
        97,
        131,
        197,
        251,
        307,
        401,
    )


def test_private_reducer_regenerates_templates_and_resolves(monkeypatch) -> None:
    import duraseed.data.boundary_freeze as freeze

    cohorts = _cohorts()
    seen_seeds = []
    original = freeze.crossed_seed_assignments

    def crossed(seeds, *, allocation_seed):
        seen_seeds.append(tuple(seeds))
        return original(seeds, allocation_seed=allocation_seed)

    monkeypatch.setattr(freeze, "crossed_seed_assignments", crossed)
    result, regenerated, calls = _reduce(monkeypatch, cohorts)

    assert result.selection_performed
    assert regenerated == [
        (
            result.combined_broad_manifest,
            tuple(row.intended_family_id for row in result.finalist_summaries),
        ),
        (result.combined_broad_manifest, result.match.selected_family_ids),
    ]
    assert len(calls) == 60
    assert {root_seed for _, root_seed, _ in calls} == {11}
    assert sum(selected for _, _, selected in calls) == 24
    assert len(result.ranked_family_ids) == 36
    assert result.intermediate_family_ids == tuple(
        sorted(result.ranked_family_ids[24:36])
    )
    assert result.candidate_table_id == canonical_json_hash(result.candidate_payload)
    assert result.matching_report_id == canonical_json_hash(
        result.matching_report_payload
    )
    assert result.confirmation_summary["status"] == "confirmed_panels_frozen"
    assert seen_seeds == [freeze_settings_from_config(CONFIG).training_seeds]


def test_extension_capacity_drop_is_allowed_and_stays_unresolved(monkeypatch) -> None:
    cohorts = _cohorts(16)
    dropped = cohorts[1].finalist_summaries[0].intended_family_id
    result, _, _ = _reduce(monkeypatch, cohorts, fail_combined=(dropped,))

    assert result.eligible_family_ids == cohorts[0].locked_eligible_family_ids
    assert len(result.candidates) == 15
    assert not result.selection_performed
    assert result.candidate_payload is result.matching_report_payload is None
    assert result.confirmation_summary["candidate_table_id"] is None
    assert result.intermediate_family_ids == ()


def test_locked_capacity_failure_is_fatal(monkeypatch) -> None:
    cohorts = _cohorts(16)
    locked = cohorts[0].locked_eligible_family_ids[0]
    _patch_primitives(monkeypatch, fail_combined=(locked,))
    counts = {
        summary.intended_family_id: {}
        for cohort in cohorts
        for summary in cohort.finalist_summaries
        if summary.intended_family_id != locked
    }
    with pytest.raises(BoundaryFreezeReductionError, match="locked passer"):
        _reduce_three_cohort_panels(
            cohorts,
            settings=freeze_settings_from_config(CONFIG),
            teacher_trace_token_counts=counts,
        )


def test_selected_capacity_failure_returns_unresolved_surface(monkeypatch) -> None:
    result, _, calls = _reduce(monkeypatch, _cohorts(), fail_selected=True)

    assert len(result.selected_capacity_audits) == 24
    assert any(selected for _, _, selected in calls)
    assert not result.selection_performed
    assert result.match is None
    assert result.candidate_payload is result.matching_report_payload is None
    assert result.intermediate_family_ids == ()
    assert result.confirmation_summary["selected_panel_split_capacity_passed"] is False


def test_private_reducer_rejects_wrong_cohort_order_and_teacher_keys(
    monkeypatch,
) -> None:
    cohorts = _cohorts()
    settings = freeze_settings_from_config(CONFIG)
    with pytest.raises(BoundaryFreezeReductionError, match="three distinct cohorts"):
        _reduce_three_cohort_panels(
            tuple(reversed(cohorts)),
            settings=settings,
            teacher_trace_token_counts={},
        )

    _patch_primitives(monkeypatch)
    counts = {
        summary.intended_family_id: {}
        for cohort in cohorts
        for summary in cohort.finalist_summaries
    }
    counts["extra-family"] = {}
    with pytest.raises(BoundaryFreezeReductionError, match="teacher token counts"):
        _reduce_three_cohort_panels(
            cohorts,
            settings=settings,
            teacher_trace_token_counts=counts,
        )


def test_locked_ids_must_be_fifteen_unique(monkeypatch) -> None:
    cohorts = _cohorts(16)
    duplicated = (
        *cohorts[0].locked_eligible_family_ids[:-1],
        cohorts[0].locked_eligible_family_ids[0],
    )
    cohorts = (replace(cohorts[0], locked_eligible_family_ids=duplicated), *cohorts[1:])
    _patch_primitives(monkeypatch)
    counts = {
        summary.intended_family_id: {}
        for cohort in cohorts
        for summary in cohort.finalist_summaries
    }
    with pytest.raises(BoundaryFreezeReductionError, match="locked passer"):
        _reduce_three_cohort_panels(
            cohorts,
            settings=freeze_settings_from_config(CONFIG),
            teacher_trace_token_counts=counts,
        )


def test_private_reducer_rejects_corrupt_capacity_audit_identity(monkeypatch) -> None:
    import duraseed.data.boundary_freeze as freeze

    cohorts = _cohorts()
    _patch_primitives(monkeypatch)
    monkeypatch.setattr(
        freeze,
        "audit_family_split_capacities",
        lambda _templates, family_ids, *_args, **_kwargs: tuple(
            FamilyCapacityAudit("wrong-family", (), 200, True) for _ in family_ids
        ),
    )
    counts = {
        summary.intended_family_id: {}
        for cohort in cohorts
        for summary in cohort.finalist_summaries
    }
    with pytest.raises(BoundaryFreezeReductionError, match="audit order"):
        _reduce_three_cohort_panels(
            cohorts,
            settings=freeze_settings_from_config(CONFIG),
            teacher_trace_token_counts=counts,
        )


def test_private_reducer_counts_tokens_after_capacity_selection(monkeypatch) -> None:
    cohorts = _cohorts()
    _patch_primitives(monkeypatch)
    seen = []

    def count(summary, records):
        seen.append((summary.intended_family_id, tuple(records)))
        return {}

    result = _reduce_three_cohort_panels(
        cohorts,
        settings=freeze_settings_from_config(CONFIG),
        teacher_trace_token_counter=count,
    )

    assert tuple(family_id for family_id, _ in seen) == result.eligible_family_ids
    assert result.teacher_trace_token_counts == {
        family_id: {} for family_id in result.eligible_family_ids
    }
