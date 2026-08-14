from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import duraseed.boundary_live_fresh_resume as fresh
import duraseed.boundary_live_retry as retry
from duraseed.boundary_live_artifacts import BoundaryLiveArtifacts
from duraseed.boundary_live_fresh_resume import (
    BoundaryFreshResumeArtifacts,
    FRESH_RESUME_MARKER,
)
from duraseed.boundary_live_retry import BoundaryRetryArtifacts, RETRY_MARKER
from duraseed.boundary_live_sampling import action_limits, collect_groups
from duraseed.provenance import derive_namespaced_seed
from duraseed.runners import RunnerGateError
from duraseed.runtime import TokenBudget, TokenLedger
from duraseed.run_records import RunStatus
from tests.unit.test_boundary_live_flow import CONFIG, _run, _source_contract
from tests.unit.test_boundary_live_retry import (
    APIConnectionError,
    LongRenderer,
    RetrySampler,
)
from tests.unit.test_boundary_scan import _shared_family_manifest
from tests.unit.test_runtime_sampling import _runtime


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = _shared_family_manifest(1)
    record = manifest.records[0]
    directory = tmp_path / "boundary-live"
    preflight = {
        "gate_name": "boundary-extension",
        "run_id": directory.name,
        "actions": {"extension2-refine": "30"},
        "extension2_manifest_id": manifest.manifest_id,
    }
    artifacts = BoundaryLiveArtifacts(
        directory, preflight=preflight, new_run=_run(manifest.manifest_id)
    )
    artifacts.write_manifest("extension2_broad_manifest.json", manifest)
    runtime = _runtime()
    runtime = runtime.__class__(
        runtime.sdk,
        runtime.service,
        runtime.model,
        LongRenderer(),
        runtime.tokenizer,
    )
    config = CONFIG.model_copy(
        update={"tinker": CONFIG.tinker.model_copy(update={"max_sampled_tokens": 4096})}
    )
    broad_ledger = TokenLedger(action_limits("extension2-broad", manifest, 4, 4096), 10)
    asyncio.run(
        collect_groups(
            artifacts,
            runtime,
            RetrySampler(broad_ledger),
            manifest,
            action="extension2-broad",
            run_id=directory.name,
            source=SimpleNamespace(contract=_source_contract(manifest.manifest_id)),
            samples=4,
            sample_start=0,
            config=config,
            ledger=broad_ledger,
        )
    )
    reservation = TokenBudget(1584, 49152, 0)
    artifacts.begin_group(
        "extension2-refine",
        record.task_id,
        manifest_id=manifest.manifest_id,
        run_id=f"{directory.name}:extension2-refine",
        sample_indices=tuple(range(4, 16)),
        reservation=reservation,
    )
    refine_ledger = TokenLedger(
        action_limits("extension2-refine", manifest, 12, 4096), 30
    )
    refine_ledger.reserve_call(reservation)
    refine_ledger.abort_call()
    artifacts.record_error(
        "extension2-refine",
        APIConnectionError("No progress made in 7200s. Requests appear to be stuck."),
    )
    artifacts.finish(
        RunStatus.FAILED,
        {"extension2-broad": broad_ledger, "extension2-refine": refine_ledger},
    )
    error = json.loads((directory / "errors.jsonl").read_text())
    pending = json.loads((directory / "pending_group.json").read_text())
    first_trace = tmp_path / "first.pftrace"
    first_trace.write_bytes(b"first trace")
    incident = {
        "run_id": directory.name,
        "project_id": "project",
        "original_git_commit": "abc123",
        "pending": pending,
        "error": error,
        "remote_proof": {
            "tinker_session_id": "first",
            "future_checks": [],
            "post_failure_trace_sha256": _hash(first_trace),
            "post_failure_trace_max_future_id": 3,
            "absent_future_ids": [],
        },
    }
    monkeypatch.setattr(retry, "INCIDENT", incident)
    monkeypatch.setattr(fresh, "INCIDENT", incident)
    retry_run = _run(manifest.manifest_id).model_copy(
        update={"git_commit": "retry-commit"}
    )
    retry_artifacts = BoundaryRetryArtifacts(
        directory, preflight=preflight, new_run=retry_run, trace_path=first_trace
    )
    retry_artifacts.begin_group(
        "extension2-refine",
        record.task_id,
        manifest_id=manifest.manifest_id,
        run_id=f"{directory.name}:extension2-refine",
        sample_indices=tuple(range(4, 16)),
        reservation=reservation,
    )
    stale_trace = tmp_path / "stale.pftrace"
    stale_trace.write_bytes(b"descriptor-only stale trace")
    stale = {
        "run_id": directory.name,
        "stale_session_id": "stale-session",
        "retry_started_at": json.loads((directory / RETRY_MARKER).read_text())[
            "started_at"
        ],
        "stale_trace_sha256": _hash(stale_trace),
        "stale_trace_packet_count": 1,
        "stale_trace_descriptor_only": True,
        "retry_git_commit": retry_run.git_commit,
        "snapshot_sha256": {
            name: _hash(directory / name)
            for name in (
                "run.json",
                "preflight.json",
                "observation_groups.jsonl",
                "errors.jsonl",
                "pending_group.json",
                "billing.json",
                RETRY_MARKER,
                "extension2_broad_manifest.json",
            )
        },
        "journal_groups": {"extension2-broad": 1},
        "journal_samples": 4,
        "refinement": {"positive_families": 1, "audit_families": 0, "tasks": 1},
    }
    monkeypatch.setattr(fresh, "STALE_RUNTIME_INCIDENT", stale)
    fresh_run = _run(manifest.manifest_id).model_copy(
        update={"git_commit": "fresh-commit"}
    )
    return (
        directory,
        preflight,
        manifest,
        record,
        fresh_run,
        stale_trace,
        runtime,
        config,
    )


def test_fresh_client_preserves_coordinates_seeds_and_single_failed_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path, monkeypatch)
    directory, preflight, manifest, record, run, trace, runtime, config = values
    artifacts = BoundaryFreshResumeArtifacts(
        directory, preflight=preflight, new_run=run, trace_path=trace
    )
    limits = action_limits("extension2-refine", manifest, 12, 4096)
    ledger = artifacts.restore_ledger("extension2-refine", limits, 30)
    reservation = TokenBudget(1584, 49152, 0)
    assert ledger.limits == limits.plus(reservation)
    assert ledger.committed == ledger.observed == reservation
    sampler = RetrySampler(ledger)
    rows = asyncio.run(
        collect_groups(
            artifacts,
            runtime,
            sampler,
            manifest,
            action="extension2-refine",
            run_id=directory.name,
            source=SimpleNamespace(contract=_source_contract(manifest.manifest_id)),
            samples=12,
            sample_start=4,
            config=config,
            ledger=ledger,
        )
    )
    assert len(rows[0]) == len(rows[1]) == 12
    assert tuple(sampler.seeds) == tuple(
        derive_namespaced_seed(
            5,
            "tinker.tces_boundary_broad",
            "tces",
            record.task_id,
            record.item_index,
            index,
        )
        for index in range(4, 16)
    )
    assert ledger.committed == TokenBudget(3168, 98304, 0)
    assert json.loads((directory / FRESH_RESUME_MARKER).read_text())["status"] == (
        "completed"
    )
    assert not (directory / "pending_group.json").exists()
    resumed = BoundaryFreshResumeArtifacts(
        directory, preflight=preflight, new_run=run, trace_path=None
    )
    assert resumed.restore_ledger("extension2-refine", limits, 30).committed == (
        TokenBudget(3168, 98304, 0)
    )
    assert record.task_id == retry.INCIDENT["pending"]["task_id"]


def test_fresh_client_rejects_tamper_and_any_later_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path, monkeypatch)
    directory, preflight, manifest, record, run, trace, _, _ = values
    billing = directory / "billing.json"
    billing.write_bytes(billing.read_bytes() + b" ")
    with pytest.raises(RunnerGateError, match="snapshot changed: billing"):
        BoundaryFreshResumeArtifacts(
            directory, preflight=preflight, new_run=run, trace_path=trace
        )
    billing.write_bytes(billing.read_bytes()[:-1])
    artifacts = BoundaryFreshResumeArtifacts(
        directory, preflight=preflight, new_run=run, trace_path=trace
    )
    artifacts.begin_group(
        "extension2-refine",
        record.task_id,
        manifest_id=manifest.manifest_id,
        run_id=f"{directory.name}:extension2-refine",
        sample_indices=tuple(range(4, 16)),
        reservation=TokenBudget(1584, 49152, 0),
    )
    with pytest.raises(RunnerGateError, match="no further retry"):
        BoundaryFreshResumeArtifacts(
            directory, preflight=preflight, new_run=run, trace_path=None
        )


def test_refinement_only_runner_stops_cleanly_without_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duraseed.runners.boundary_refine_resume as runner

    manifest = _shared_family_manifest(1)
    record = manifest.records[0]
    finished: list[tuple[RunStatus, set[str]]] = []

    class Artifacts:
        groups = {("extension2-broad", record.task_id): ()}
        pending = tmp_path / "no-pending"

        def write_manifest(self, _name, _manifest):  # type: ignore[no-untyped-def]
            return None

        def restore_ledger(self, _action, limits, cap):  # type: ignore[no-untyped-def]
            return TokenLedger(limits, float(cap))

        def finish(self, status, ledgers):  # type: ignore[no-untyped-def]
            finished.append((status, set(ledgers)))

        def record_error(self, _action, error):  # type: ignore[no-untyped-def]
            raise AssertionError("clean stop recorded an error") from error

    artifacts = Artifacts()

    async def fake_collect(*_args, action, **_kwargs):  # type: ignore[no-untyped-def]
        if action == "extension2-refine":
            artifacts.groups[(action, record.task_id)] = ()
        return (), ()

    monkeypatch.setattr(runner, "open_boundary_artifacts", lambda *_a, **_k: artifacts)
    monkeypatch.setattr(
        runner, "load_frozen_extension1_confirmation", lambda *_a: manifest
    )
    monkeypatch.setattr(runner, "collect_groups", fake_collect)
    monkeypatch.setattr(
        runner,
        "summarize",
        lambda *_a: (
            SimpleNamespace(
                intended_family_id=record.intended_family, total_successes=1
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "choose_refinement_family_ids",
        lambda _successes: ((record.intended_family,), ()),
    )
    monkeypatch.setattr(
        runner,
        "STALE_RUNTIME_INCIDENT",
        {
            "journal_groups": {"extension2-broad": 1},
            "refinement": {"positive_families": 1, "audit_families": 0, "tasks": 1},
        },
    )
    monkeypatch.setattr(runner, "INCIDENT", {"pending": {"task_id": record.task_id}})
    source = SimpleNamespace(
        contract=_source_contract(manifest.manifest_id),
        extension1_broad_manifest=manifest,
    )
    asyncio.run(
        runner.execute_boundary_refine_resume(
            _runtime(),
            object(),
            source=source,
            config=CONFIG,
            output_root=tmp_path,
            run_id="boundary-live",
            git_commit="fresh",
            extension1_confirmation_path=tmp_path / "extension1.json",
            refine_retry_trace=tmp_path / "trace.pftrace",
            extension2=manifest,
        )
    )
    assert finished == [
        (
            RunStatus.INTERRUPTED,
            {"extension1-confirm", "extension2-broad", "extension2-refine"},
        )
    ]
