from __future__ import annotations

from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest

import duraseed.data.boundary_panel_amendment_source as source_auth
from duraseed.boundary_panel_amendment_finalizer import _candidate, _write_completed
from duraseed.data.boundary_panel_amendment_source import AuthenticatedUnresolvedFreeze
from duraseed.data.boundary_panel_amendment_source import (
    BoundaryPanelAmendmentSourceError,
    validate_source_metadata,
)
from duraseed.data.panel_capacity import FamilyCapacityAudit
from duraseed.data.panel_matching import FamilyPanelCandidate
from duraseed.data.panels import (
    FamilyPanelArtifact,
    PanelLabel,
    SeedBlockPanelAssignment,
)
from duraseed.provenance import canonical_json_bytes, canonical_json_hash, sha256_bytes
from duraseed.run_records import RunRecord, RunStatus, read_run_record


def _authenticated_metadata(monkeypatch):
    source_hashes = {
        "preflight.json": "sha256:preflight",
        "run.json": "sha256:run",
        "scientific_outputs.json": "sha256:scientific",
        "three_cohort_equivalence.json": "sha256:equivalence",
    }
    monkeypatch.setattr(source_auth, "SOURCE_RAW_SHA256", source_hashes)
    scientific = {
        "candidate_payload": None,
        "candidates": [{} for _ in range(49)],
        "confirmation_summary": {
            "status": "confirmation_complete_panel_selection_unresolved"
        },
        "matching_report_payload": None,
        "panel_artifact": None,
        "teacher_trace_token_counts": {"family": {"task": 1}},
    }
    equivalence = {
        "schema_version": "duraseed-three-cohort-freeze-equivalence-v1",
        "status": "passed",
        "old_new_identical": True,
        "token_count_map_identical": True,
        "scientific_outputs_sha256": sha256_bytes(canonical_json_bytes(scientific)),
        "teacher_trace_token_counts_sha256": sha256_bytes(
            canonical_json_bytes(scientific["teacher_trace_token_counts"])
        ),
        "compared_fields": sorted(scientific),
    }
    preflight = {
        "status": "completed_local_freeze",
        "source_kind": "three_cohort_boundary_freeze_v1",
        "remote_execution": False,
        "source": {
            "source_kind": "three_cohort_boundary_freeze_v1",
            "equivalence_sha256": source_hashes["three_cohort_equivalence.json"],
        },
    }
    run = SimpleNamespace(
        status=RunStatus.COMPLETED,
        run_kind="m0_calibration",
        cost_usd=0.0,
    )
    return scientific, preflight, equivalence, run, source_hashes


def test_published_unresolved_source_requires_passed_raw_equivalence(
    monkeypatch,
) -> None:
    scientific, preflight, equivalence, run, hashes = _authenticated_metadata(
        monkeypatch
    )

    validate_source_metadata(
        scientific=scientific,
        preflight=preflight,
        equivalence=equivalence,
        run=run,
        source_hashes=hashes,
    )

    equivalence["old_new_identical"] = False
    with pytest.raises(BoundaryPanelAmendmentSourceError, match="authentication"):
        validate_source_metadata(
            scientific=scientific,
            preflight=preflight,
            equivalence=equivalence,
            run=run,
            source_hashes=hashes,
        )


def test_raw_scientific_source_hash_is_pinned(monkeypatch) -> None:
    scientific, preflight, equivalence, run, hashes = _authenticated_metadata(
        monkeypatch
    )
    hashes = {**hashes, "scientific_outputs.json": "sha256:changed"}

    with pytest.raises(BoundaryPanelAmendmentSourceError, match="authentication"):
        validate_source_metadata(
            scientific=scientific,
            preflight=preflight,
            equivalence=equivalence,
            run=run,
            source_hashes=hashes,
        )


def test_candidate_loader_restores_tuple_covariates() -> None:
    candidate = _candidate(
        {
            "family_id": "ADD(r1,ADD(r2,ADD(r3,ADD(r4,r5))))|intermediates=I,I,I,I",
            "m0_posterior_mean_success": 0.1,
            "informative_group_probability_i8": 0.5,
            "tree_depth": 5,
            "operator_multiset": ["ADD", "ADD", "ADD", "ADD"],
            "noncommutative_operation_count": 0,
            "fractional_intermediate_profile": ["I", "I", "I", "I"],
            "target_magnitude": 10,
            "valid_family_multiplicity": 1,
            "teacher_trace_length": 20,
            "available_disjoint_instances": 100,
        }
    )

    assert candidate.operator_multiset == ("ADD",) * 4
    assert candidate.fractional_intermediate_profile == ("I",) * 4


def test_completed_publication_serializes_bound_two_pool_result(
    tmp_path, monkeypatch
) -> None:
    candidates = tuple(
        FamilyPanelCandidate(
            family_id=f"family-{index:02d}",
            m0_posterior_mean_success=0.1,
            informative_group_probability_i8=1.0 - index / 100,
            tree_depth=5,
            operator_multiset=("ADD",) * 4,
            noncommutative_operation_count=0,
            fractional_intermediate_profile=("I",) * 4,
            target_magnitude=10,
            valid_family_multiplicity=1,
            teacher_trace_length=20,
            available_disjoint_instances=100,
        )
        for index in range(37)
    )
    audits = tuple(
        FamilyCapacityAudit(row.family_id, (), 100, True) for row in candidates
    )
    candidate_payload = {
        "schema": "duraseed-panel-candidates-algebraic-dedup-v1",
        "candidates": [row.family_id for row in candidates],
    }
    matching_payload = {
        "schema": "duraseed-panel-match-algebraic-dedup-v1",
        "intermediate_family_ids": [row.family_id for row in candidates[24:36]],
    }
    panel = FamilyPanelArtifact(
        m0_checkpoint_path="tinker://m0/sampler",
        candidate_family_table_manifest_id=canonical_json_hash(candidate_payload),
        panel_matching_report_id=canonical_json_hash(matching_payload),
        allocation_seed=17,
        panel_a_family_ids=tuple(row.family_id for row in candidates[:12]),
        panel_b_family_ids=tuple(row.family_id for row in candidates[12:24]),
        seed_block_assignments=(
            SeedBlockPanelAssignment(
                training_seed=17,
                targeted_panel=PanelLabel.A,
                sentinel_panel=PanelLabel.B,
            ),
            SeedBlockPanelAssignment(
                training_seed=37,
                targeted_panel=PanelLabel.B,
                sentinel_panel=PanelLabel.A,
            ),
        ),
    )
    result = SimpleNamespace(
        representative_candidates=candidates,
        panel_capacity_audits=audits,
        panel_eligible_family_ids=tuple(row.family_id for row in candidates),
        match={"selected_family_ids": [row.family_id for row in candidates[:24]]},
        intermediate_family_ids=tuple(row.family_id for row in candidates[24:36]),
        selected_capacity_audits=audits[:24],
        candidate_payload=candidate_payload,
        matching_report_payload=matching_payload,
        panel_artifact=panel,
        amendment_payload={
            "schema_version": "duraseed-boundary-panel-amendment-v1",
            "status": "resolved",
        },
        confirmation_summary={"status": "confirmed_panels_frozen"},
    )
    source_dir = tmp_path / "boundary-panel-freeze-20260815T032009Z"
    source_dir.mkdir()
    for name, content in (
        ("a_candidate_manifest.json", "{}\n"),
        ("a_candidate_confirmation_manifest.json", "{}\n"),
        ("confirmation_family_table.jsonl", ""),
        ("three_cohort_equivalence.json", "{}\n"),
    ):
        (source_dir / name).write_text(content)
    now = datetime.now(UTC)
    run = RunRecord(
        protocol_version="test",
        git_commit="source",
        resolved_config_hash="config-hash",
        run_kind="m0_calibration",
        method=None,
        seed=11,
        model_id="model",
        renderer="role_colon",
        lora_rank=32,
        task_manifest_ids={"boundary": "sha256:" + "a" * 64},
        parent_tinker_checkpoint_path="tinker://m0/state",
        status=RunStatus.COMPLETED,
        started_at=now,
        updated_at=now,
        finished_at=now,
        project_id="project",
    )
    scientific = {
        "capacity_audits": [
            {"family_id": row.family_id, "passed": True} for row in candidates
        ],
        "confirmation_summary": {},
        "teacher_trace_token_counts": {
            **{row.family_id: {"task": 1} for row in candidates},
            "excluded-family": {"task": 1},
        },
    }
    source = AuthenticatedUnresolvedFreeze(
        source_dir,
        scientific,
        {
            "source": {
                "source_kind": "three_cohort_boundary_freeze_v1",
                "sampler_checkpoint_path": "tinker://m0/sampler",
            }
        },
        {},
        run,
        None,
        None,
        {"scientific_outputs.json": "sha256:" + "b" * 64},
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(
        "duraseed.boundary_panel_amendment_finalizer._git_commit",
        lambda: "output-commit",
    )

    _write_completed(
        staging,
        source,
        result,
        SimpleNamespace(resolved_config_hash=lambda: "config-hash"),
        "amended-freeze",
    )

    counts = json.loads((staging / "teacher_trace_token_counts.json").read_text())
    amendment = json.loads((staging / "panel_amendment.json").read_text())
    published_science = json.loads((staging / "scientific_outputs.json").read_text())
    assert set(counts) == {row.family_id for row in candidates}
    assert len(published_science["panel_capacity_audits"]) == 37
    assert amendment["source_run_id"] == source_dir.name
    assert amendment["source_scientific_outputs_sha256"] == "sha256:" + "b" * 64
    assert read_run_record(staging).status is RunStatus.COMPLETED
