from __future__ import annotations

import pytest
from types import SimpleNamespace

from duraseed.data.boundary_confirmation import (
    BoundaryConfirmationError,
    build_confirmation_manifest,
    choose_refinement_family_ids,
    reduce_confirmation_evidence,
)
from duraseed.data.boundary_freeze import (
    BOUNDARY_PANEL_FREEZE_EQUIVALENCE_STATUS,
    BoundaryFreezeUnverifiedError,
    freeze_three_cohort_panels,
)
from duraseed.data.boundary_protocol import (
    BoundaryProtocolError,
    build_broad_manifest,
    validate_broad_cohort_provenance,
)
from duraseed.data.panel_capacity import FamilyCapacityAudit
from duraseed.data.boundary_sources import (
    BoundarySourceError,
    validate_confirmation_source_contract,
)
from duraseed.run_records import RunStatus
from duraseed.tasks.tces import TCESGeneratorConfig


def _generator_config() -> TCESGeneratorConfig:
    return TCESGeneratorConfig(
        n_operands=3,
        operand_min=2,
        operand_max=20,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_attempts=256,
    )


def test_broad_and_confirmation_manifests_are_deterministic_and_disjoint() -> None:
    config = _generator_config()
    broad = build_broad_manifest(
        config,
        family_count=3,
        items_per_family=2,
        template_scan_ceiling=16,
        variant_index_offset=32,
    )
    repeated = build_broad_manifest(
        config,
        family_count=3,
        items_per_family=2,
        template_scan_ceiling=16,
        variant_index_offset=32,
    )
    finalists = tuple(sorted({row.intended_family for row in broad.records})[:2])
    confirmation = build_confirmation_manifest(config, broad, finalists)

    assert broad == repeated
    assert (
        broad.manifest_id
        == "sha256:211333bcb26564892f28f6258191faa466e1fd904b6a43e34b0ec521f8e51361"
    )
    assert len(confirmation.records) == 8
    assert confirmation.manifest_id == (
        "sha256:5304936d7229ac999caf8898a9129e53f313926874528681f3e99b43fa9811a4"
    )
    assert not {row.task_id for row in broad.records}.intersection(
        row.task_id for row in confirmation.records
    )


def test_extension_cohorts_are_chained_and_disjoint(monkeypatch) -> None:
    import duraseed.data.boundary_protocol as protocol

    monkeypatch.setattr(protocol, "BOUNDARY_BROAD_FAMILY_COUNT", 2)
    monkeypatch.setattr(protocol, "BOUNDARY_BROAD_ITEMS_PER_FAMILY", 2)
    monkeypatch.setattr(protocol, "BOUNDARY_BROAD_VARIANT_INDEX_OFFSET", 16)
    kwargs = {
        "family_count": 2,
        "items_per_family": 2,
        "template_scan_ceiling": 16,
    }
    manifests = tuple(
        build_broad_manifest(_generator_config(), cohort=cohort, **kwargs)
        for cohort in ("initial", "extension_1", "extension_2")
    )

    assert manifests[1].parent_manifest_id == manifests[0].manifest_id
    assert manifests[2].parent_manifest_id == manifests[1].manifest_id
    families = [
        {row.intended_family for row in manifest.records} for manifest in manifests
    ]
    assert all(
        families[left].isdisjoint(families[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


def test_refinement_selection_is_order_invariant() -> None:
    successes = {
        **{f"positive-{index:02d}": index % 4 + 1 for index in range(35)},
        **{f"zero-{index:02d}": 0 for index in range(29)},
    }
    first = choose_refinement_family_ids(successes)
    second = choose_refinement_family_ids(dict(reversed(tuple(successes.items()))))

    assert first == second
    assert len(first[0]) == 35
    assert len(first[1]) == 12


def test_extension_provenance_is_bound_to_manifest_and_measurement(monkeypatch) -> None:
    import duraseed.data.boundary_protocol as protocol

    monkeypatch.setattr(protocol, "BOUNDARY_BROAD_FAMILY_COUNT", 2)
    monkeypatch.setattr(protocol, "BOUNDARY_BROAD_ITEMS_PER_FAMILY", 2)
    monkeypatch.setattr(protocol, "BOUNDARY_BROAD_VARIANT_INDEX_OFFSET", 16)
    kwargs = {
        "family_count": 2,
        "items_per_family": 2,
        "template_scan_ceiling": 16,
    }
    initial = build_broad_manifest(_generator_config(), cohort="initial", **kwargs)
    extension = build_broad_manifest(
        _generator_config(), cohort="extension_1", **kwargs
    )
    recorded = dict(extension.metadata["cohort_provenance"])
    recorded["parent_manifest_id"] = initial.manifest_id
    plan = {
        "measurement": {
            "manifest_id": extension.manifest_id,
            "cohort_provenance": recorded,
        }
    }

    validate_broad_cohort_provenance(
        extension,
        plan,
        expected_cohort="extension_1",
        expected_parent_manifest_id=initial.manifest_id,
    )
    plan["measurement"]["manifest_id"] = "sha256:" + "0" * 64
    with pytest.raises(BoundaryProtocolError, match="differs from its manifest"):
        validate_broad_cohort_provenance(
            extension,
            plan,
            expected_cohort="extension_1",
            expected_parent_manifest_id=initial.manifest_id,
        )
    plan["measurement"]["manifest_id"] = extension.manifest_id
    plan["measurement"]["cohort_provenance"]["parent_manifest_id"] = "wrong"
    with pytest.raises(BoundaryProtocolError, match="provenance changed"):
        validate_broad_cohort_provenance(
            extension,
            plan,
            expected_cohort="extension_1",
            expected_parent_manifest_id=initial.manifest_id,
        )


def test_three_cohort_freeze_fails_closed_before_phase7_equivalence() -> None:
    assert BOUNDARY_PANEL_FREEZE_EQUIVALENCE_STATUS == (
        "pending_phase7_three_cohort_check"
    )
    with pytest.raises(BoundaryFreezeUnverifiedError, match="pending the Phase-7"):
        freeze_three_cohort_panels(
            (),
            (),
            panel_size=12,
            allocation_seed=11,
            training_seeds=(1, 2),
            m0_checkpoint_path="tinker://unused",
        )


def test_confirmation_reducer_rejects_empty_or_misordered_capacity(monkeypatch) -> None:
    import duraseed.data.boundary_confirmation as confirmation

    with pytest.raises(BoundaryConfirmationError, match="nonempty and unique"):
        reduce_confirmation_evidence((), (), (), ())

    summary = type(
        "Summary",
        (),
        {
            "intended_family_id": "family-a",
            "item_count": 4,
            "items": tuple(type("Item", (), {"trials": 16})() for _ in range(4)),
        },
    )()
    audit = FamilyCapacityAudit(
        family_id="family-b",
        split_capacities=(),
        available_disjoint_instances=1,
        passed=True,
    )
    monkeypatch.setattr(
        confirmation,
        "assess_refinement_finalist_gate",
        lambda value: type("Gate", (), {"eligible": True})(),
    )
    with pytest.raises(BoundaryConfirmationError, match="finalist order"):
        reduce_confirmation_evidence((summary,), (summary,), ("family-a",), (audit,))


def test_confirmation_uses_global_extension_ordinal() -> None:
    config = _generator_config()
    broad = build_broad_manifest(
        config,
        family_count=3,
        items_per_family=2,
        template_scan_ceiling=16,
        variant_index_offset=32,
    )
    extension = broad.model_copy(
        update={
            "metadata": {
                **broad.metadata,
                "cohort_provenance": {
                    "cohort_id": "extension_1",
                    "distinct_family_ordinal_start": 64,
                    "distinct_family_ordinal_stop": 128,
                },
            }
        }
    )
    family_id = str(extension.metadata["templates"][0]["family_id"])
    manifest = build_confirmation_manifest(config, extension, (family_id,))

    assert manifest.metadata["families"][0]["held_out_item_indices"] == list(
        range(1024 + 64 * 4, 1024 + 65 * 4)
    )


def test_confirmation_source_contract_binds_checkpoint_and_sampling(
    monkeypatch,
) -> None:
    import duraseed.data.boundary_sources as sources

    manifest = build_broad_manifest(
        _generator_config(),
        family_count=3,
        items_per_family=2,
        template_scan_ceiling=16,
        variant_index_offset=32,
    )
    monkeypatch.setattr(sources, "broad_cohort", lambda value: ("initial", 0))
    monkeypatch.setattr(
        sources, "validate_broad_cohort_provenance", lambda *args, **kwargs: None
    )
    common = dict(
        status=RunStatus.COMPLETED,
        run_kind="m0_calibration",
        method=None,
        protocol_version="v",
        resolved_config_hash="hash",
        model_id="model",
        renderer="role_colon",
        lora_rank=32,
        parent_tinker_checkpoint_path="state",
        project_id="project",
    )
    run = SimpleNamespace(
        **common,
        run_id="broad",
        task_manifest_ids={"boundary_broad_a_candidate": manifest.manifest_id},
    )
    refinement = SimpleNamespace(
        **common,
        run_id="refine",
        task_manifest_ids={"boundary_refinement_a_candidate": manifest.manifest_id},
    )
    config = SimpleNamespace(
        tinker=SimpleNamespace(
            model_id="model",
            renderer_name="role_colon",
            lora_rank=32,
            max_sampled_tokens=4096,
            group_size=8,
        ),
        evaluation={"temperature": 1.0, "top_p": 0.95},
    )
    row = SimpleNamespace(
        run_id="broad",
        sampling_temperature=1.0,
        sampling_top_p=0.95,
        sampling_max_tokens=4096,
        sampler_checkpoint_path="sampler",
        origin_sampler_checkpoint_path="sampler",
        training_step=24,
        seed=5,
    )
    refine_row = SimpleNamespace(**vars(row))
    refine_row.run_id = "refine"
    measurement = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096}
    broad_plan = {
        "run_id": "broad",
        "project_id": "project",
        "source": {
            "sampler_checkpoint_path": "sampler",
            "state_checkpoint_path": "state",
            "training_step": 24,
        },
        "measurement": {**measurement, "group_size_for_i8": 8},
    }
    refinement_plan = {
        "run_id": "refine",
        "project_id": "project",
        "source": {
            "broad_run_id": "broad",
            "manifest_id": manifest.manifest_id,
            "sampler_checkpoint_path": "sampler",
            "training_step": 24,
        },
        "measurement": measurement,
    }
    contract = validate_confirmation_source_contract(
        config=config,
        broad_run=run,
        refinement_run=refinement,
        broad_plan=broad_plan,
        refinement_plan=refinement_plan,
        broad_manifest=manifest,
        refinement_manifest=manifest,
        broad_generations=(row,),
        refinement_generations=(refine_row,),
        expected_parent_manifest_id=None,
    )
    assert contract.sampler_checkpoint_path == "sampler"
    refinement_plan["source"]["training_step"] = 25
    with pytest.raises(BoundarySourceError, match="observed checkpoint"):
        validate_confirmation_source_contract(
            config=config,
            broad_run=run,
            refinement_run=refinement,
            broad_plan=broad_plan,
            refinement_plan=refinement_plan,
            broad_manifest=manifest,
            refinement_manifest=manifest,
            broad_generations=(row,),
            refinement_generations=(refine_row,),
            expected_parent_manifest_id=None,
        )
