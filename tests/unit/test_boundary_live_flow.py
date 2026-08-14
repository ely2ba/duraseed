from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.boundary_live_artifacts import BoundaryLiveArtifacts
from duraseed.boundary_live_sampling import (
    action_limits,
    collect_groups,
)
from duraseed.boundary_live_sources import BoundaryLiveSource
from duraseed.config import load_pilot_config
from duraseed.data.boundary_confirmation import ConfirmationEvidence
from duraseed.data.boundary_sources import BoundarySourceContract
from duraseed.data.manifests import build_manifest
from duraseed.provenance import derive_namespaced_seed
from duraseed.runners import RunnerGateError
from duraseed.runners.boundary_extension import BoundaryBlockResult
from duraseed.runners.boundary_live import execute_boundary_live
from duraseed.runtime import TokenLedger
from duraseed.runtime import TokenBudget
from duraseed.run_records import RunRecord, RunStatus
from tests.unit.test_boundary_extension_flow import _summary
from tests.unit.test_boundary_scan import _shared_family_manifest
from tests.unit.test_runtime_sampling import Sampler, _runtime


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_pilot_config(ROOT / "duraseed_pilot_config.yaml")


def _run(manifest_id: str) -> RunRecord:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    return RunRecord(
        protocol_version="duraseed-prepilot-v5",
        git_commit="abc123",
        resolved_config_hash=CONFIG.resolved_config_hash(),
        run_kind="m0_calibration",
        method=None,
        seed=5,
        model_id="Qwen/Qwen3.5-9B-Base",
        renderer="role_colon",
        lora_rank=32,
        task_manifest_ids={"boundary": manifest_id},
        parent_tinker_checkpoint_path="tinker://m0/state",
        status=RunStatus.RUNNING,
        started_at=now,
        updated_at=now,
        project_id="project",
        authorized_cost_usd=120.0,
        reserved_cost_usd=120.0,
    )


def _source_contract(manifest_id: str) -> BoundarySourceContract:
    return BoundarySourceContract(
        cohort_id="extension_1",
        cohort_ordinal_start=64,
        prior_run_ids=("broad", "refine"),
        sampler_checkpoint_path="tinker://m0/sampler",
        training_step=0,
        model_id="Qwen/Qwen3.5-9B-Base",
        renderer="role_colon",
        lora_rank=32,
        state_checkpoint_path="tinker://m0/state",
        project_id="project",
        protocol_version="duraseed-prepilot-v5",
        resolved_config_hash=CONFIG.resolved_config_hash(),
        broad_manifest_id=manifest_id,
    )


def _artifacts(tmp_path: Path, manifest_id: str) -> BoundaryLiveArtifacts:
    return BoundaryLiveArtifacts(
        tmp_path / "live",
        preflight={"gate_name": "test"},
        new_run=_run(manifest_id),
    )


def test_group_journal_preserves_seed_contract_and_join_resumes(tmp_path: Path) -> None:
    manifest = _shared_family_manifest(2)
    artifacts = _artifacts(tmp_path, manifest.manifest_id)
    limits = action_limits("extension2-broad", manifest, 4, 16)
    ledger = TokenLedger(limits, 10.0)
    sampler = Sampler(ledger)
    source = SimpleNamespace(contract=_source_contract(manifest.manifest_id))

    first = asyncio.run(
        collect_groups(
            artifacts,
            _runtime(),
            sampler,
            manifest,
            action="extension2-broad",
            run_id="boundary-live",
            source=source,
            samples=4,
            sample_start=0,
            config=CONFIG.model_copy(
                update={
                    "tinker": CONFIG.tinker.model_copy(
                        update={"max_sampled_tokens": 16}
                    )
                }
            ),
            ledger=ledger,
        )
    )
    expected = tuple(
        derive_namespaced_seed(
            5,
            "tinker.tces_boundary_broad",
            "tces",
            record.task_id,
            record.item_index,
            index,
        )
        for record in manifest.records
        for index in range(4)
    )
    assert tuple(sampler.seeds) == expected
    assert len(first[0]) == len(first[1]) == 8
    assert {row.sample_id for row in first[0]} == {row.sample_id for row in first[1]}

    restarted = _artifacts(tmp_path, manifest.manifest_id)
    resumed_ledger = restarted.restore_ledger(
        "extension2-broad", limits, authorized_usd=10
    )
    assert resumed_ledger.committed == ledger.committed
    assert resumed_ledger.observed == ledger.observed
    unused = Sampler(resumed_ledger)
    second = asyncio.run(
        collect_groups(
            restarted,
            _runtime(),
            unused,
            manifest,
            action="extension2-broad",
            run_id="boundary-live",
            source=source,
            samples=4,
            sample_start=0,
            config=CONFIG.model_copy(
                update={
                    "tinker": CONFIG.tinker.model_copy(
                        update={"max_sampled_tokens": 16}
                    )
                }
            ),
            ledger=resumed_ledger,
        )
    )
    assert second == first
    assert unused.seeds == []


def test_refinement_reservation_counts_only_filtered_tasks() -> None:
    manifest = _shared_family_manifest(2)
    limits = action_limits("extension2-refine", manifest, 12, 4096, task_count=1)
    assert limits.sample == 12 * 4096


def test_complete_action_budget_is_checked_before_sampling(tmp_path: Path) -> None:
    manifest = _shared_family_manifest(1)
    artifacts = _artifacts(tmp_path, manifest.manifest_id)
    ledger = TokenLedger(TokenBudget(0, 4 * 16, 0), 10.0)
    sampler = Sampler(ledger)
    source = SimpleNamespace(contract=_source_contract(manifest.manifest_id))
    config = CONFIG.model_copy(
        update={"tinker": CONFIG.tinker.model_copy(update={"max_sampled_tokens": 16})}
    )

    with pytest.raises(RunnerGateError, match="complete action grid"):
        asyncio.run(
            collect_groups(
                artifacts,
                _runtime(),
                sampler,
                manifest,
                action="extension2-broad",
                run_id="boundary-live",
                source=source,
                samples=4,
                sample_start=0,
                config=config,
                ledger=ledger,
            )
        )
    assert sampler.seeds == []
    assert not artifacts.pending.exists()


def test_empty_capacity_cleared_confirmation_is_a_valid_no_call(
    tmp_path: Path,
) -> None:
    empty = build_manifest(
        name="empty-extension-confirmation",
        split="a_candidate",
        generator_version="1.0.0",
        root_seed=5,
        records=(),
        task_family="tces",
    )
    artifacts = _artifacts(tmp_path, empty.manifest_id)
    ledger = TokenLedger(action_limits("extension2-confirm", empty, 16, 4096), 40)
    sampler = Sampler(ledger)
    source = SimpleNamespace(contract=_source_contract(empty.manifest_id))

    rows = asyncio.run(
        collect_groups(
            artifacts,
            _runtime(),
            sampler,
            empty,
            action="extension2-confirm",
            run_id="boundary-live",
            source=source,
            samples=16,
            sample_start=0,
            config=CONFIG,
            ledger=ledger,
            allow_empty=True,
        )
    )
    assert rows == ((), ())
    assert sampler.seeds == []


def test_unresolved_pending_group_fails_restart_closed(tmp_path: Path) -> None:
    manifest = _shared_family_manifest(1)
    artifacts = _artifacts(tmp_path, manifest.manifest_id)
    artifacts.begin_group(
        "extension2-broad",
        manifest.records[0].task_id,
        manifest_id=manifest.manifest_id,
        run_id="boundary-live:extension2-broad",
        sample_indices=(0, 1, 2, 3),
        reservation=TokenBudget(8, 64, 0),
    )
    pending = json.loads((tmp_path / "live/pending_group.json").read_text())
    assert pending["reserved_tokens"] == {"prefill": 8, "sample": 64, "train": 0}

    with pytest.raises(RunnerGateError, match="ambiguous in-flight"):
        _artifacts(tmp_path, manifest.manifest_id)


def test_live_orchestrator_runs_fixed_actions_and_leaves_freeze_closed(
    tmp_path: Path, monkeypatch
) -> None:
    import duraseed.runners.boundary_live as live

    manifest = _shared_family_manifest(2)
    family_id = manifest.records[0].intended_family
    summary = _summary(family_id, 4, passing=True)
    contract = _source_contract(manifest.manifest_id)
    source = BoundaryLiveSource(
        contract,
        manifest,
        manifest,
        manifest,
        (),
        (),
        (),
        (),
        {family_id: 1},
        (summary,),
    )
    actions: list[str] = []

    async def fake_collect(*_args, action, **_kwargs):
        actions.append(action)
        return (), ()

    def fake_reduce(_generator, inputs):
        return BoundaryBlockResult(
            inputs.cohort_id,
            (),
            inputs.broad_manifest,
            ConfirmationEvidence((), (), (), ()),
            (),
        )

    monkeypatch.setattr(live, "build_extension2_manifest", lambda _generator: manifest)
    monkeypatch.setattr(live, "audit_new_broad_cohort", lambda *_args: {})
    monkeypatch.setattr(
        live, "capacity_cleared_confirmation", lambda *_args: (manifest, ())
    )
    monkeypatch.setattr(live, "collect_groups", fake_collect)
    monkeypatch.setattr(live, "summarize", lambda *_args, **_kwargs: (summary,))
    monkeypatch.setattr(
        live, "confirmed_family_summaries", lambda *_args, **_kwargs: (summary,)
    )
    monkeypatch.setattr(
        live, "choose_refinement_family_ids", lambda _successes: ((family_id,), ())
    )
    monkeypatch.setattr(live, "reduce_block", fake_reduce)
    result = asyncio.run(
        execute_boundary_live(
            _runtime(),
            object(),
            source=source,
            config=CONFIG,
            output_root=tmp_path,
            run_id="boundary-live",
            git_commit="abc123",
        )
    )

    assert actions == [
        "extension1-confirm",
        "extension2-broad",
        "extension2-refine",
        "extension2-confirm",
    ]
    assert result.composite_status == "blocked_pending_three_cohort_equivalence"
    assert (tmp_path / "boundary-live/result.json").is_file()
