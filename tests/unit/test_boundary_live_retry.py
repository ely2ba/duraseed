from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import duraseed.boundary_live_retry as retry
from duraseed.boundary_live_artifacts import BoundaryLiveArtifacts
from duraseed.boundary_live_retry import BoundaryRetryArtifacts, RETRY_MARKER
from duraseed.boundary_live_sampling import action_limits, collect_groups
from duraseed.provenance import derive_namespaced_seed
from duraseed.runners import RunnerGateError
from duraseed.runtime import RuntimeBundle, TokenBudget, TokenLedger
from duraseed.run_records import RunStatus
from tests.unit.test_boundary_live_flow import CONFIG, _run, _source_contract
from tests.unit.test_boundary_scan import _shared_family_manifest
from tests.unit.test_runtime_sampling import Input, Renderer, _runtime


class APIConnectionError(RuntimeError):
    pass


class LongRenderer(Renderer):
    def build_generation_prompt(self, messages, *, role):  # type: ignore[no-untyped-def]
        assert messages[0]["role"] == "user" and role == "assistant"
        return Input(list(range(132)))


class RetrySampler:
    def __init__(self, ledger: TokenLedger) -> None:
        self.ledger = ledger
        self.seeds: list[int] = []

    async def sample_async(self, *, prompt, num_samples, sampling_params):  # type: ignore[no-untyped-def]
        assert self.ledger.has_pending_call
        assert prompt.length == 132 and num_samples == 1
        self.seeds.append(sampling_params.values["seed"])
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=[7, 8], logprobs=[-0.2, -0.3], stop_reason="stop"
                )
            ]
        )


def _failed_refine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = _shared_family_manifest(1)
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
    record = manifest.records[0]
    reservation = TokenBudget(1584, 49152, 0)
    artifacts.begin_group(
        "extension2-refine",
        record.task_id,
        manifest_id=manifest.manifest_id,
        run_id="boundary-live:extension2-refine",
        sample_indices=tuple(range(4, 16)),
        reservation=reservation,
    )
    ledger = TokenLedger(action_limits("extension2-refine", manifest, 12, 4096), 30)
    ledger.reserve_call(reservation)
    ledger.abort_call()
    artifacts.record_error(
        "extension2-refine",
        APIConnectionError("No progress made in 7200s. Requests appear to be stuck."),
    )
    artifacts.finish(RunStatus.FAILED, {"extension2-refine": ledger})
    trace = tmp_path / "session.pftrace"
    trace.write_bytes(b"exact test trace")
    error = json.loads((directory / "errors.jsonl").read_text())
    pending = json.loads((directory / "pending_group.json").read_text())
    incident = {
        "run_id": directory.name,
        "project_id": "project",
        "original_git_commit": "abc123",
        "pending": pending,
        "error": error,
        "remote_proof": {
            "tinker_session_id": "test-session",
            "future_checks": [
                {"future_id": future_id, "http_status": 404}
                for future_id in range(3200, 3212)
            ],
            "post_failure_trace_sha256": "sha256:"
            + hashlib.sha256(trace.read_bytes()).hexdigest(),
            "post_failure_trace_max_future_id": 3199,
            "absent_future_ids": list(range(3200, 3212)),
        },
    }
    monkeypatch.setattr(retry, "INCIDENT", incident)
    recovery_run = _run(manifest.manifest_id).model_copy(
        update={"git_commit": "recovery123"}
    )
    return directory, preflight, manifest, record, reservation, recovery_run, trace


def _long_runtime() -> RuntimeBundle:
    runtime = _runtime()
    return RuntimeBundle(
        runtime.sdk,
        runtime.service,
        runtime.model,
        LongRenderer(),
        runtime.tokenizer,
    )


def _collect(
    artifacts: BoundaryRetryArtifacts,
    manifest,  # type: ignore[no-untyped-def]
    ledger: TokenLedger,
) -> tuple[RetrySampler, tuple]:
    sampler = RetrySampler(ledger)
    config = CONFIG.model_copy(
        update={"tinker": CONFIG.tinker.model_copy(update={"max_sampled_tokens": 4096})}
    )
    rows = asyncio.run(
        collect_groups(
            artifacts,
            _long_runtime(),
            sampler,
            manifest,
            action="extension2-refine",
            run_id="boundary-live",
            source=SimpleNamespace(contract=_source_contract(manifest.manifest_id)),
            samples=12,
            sample_start=4,
            config=config,
            ledger=ledger,
        )
    )
    return sampler, rows


def test_recovery_is_locked_to_the_observed_incident() -> None:
    assert retry.INCIDENT["run_id"] == "boundary-live-20260813T130013Z"
    assert retry.INCIDENT["project_id"] == "7727a6e3-fadb-4b07-9801-721221235e1e"
    assert retry.INCIDENT["original_git_commit"] == (
        "60f36a3d5ac475f0531111f3a80fc35f98322b80"
    )
    pending = retry.INCIDENT["pending"]
    assert pending["run_id"] == "boundary-live-20260813T130013Z:extension2-refine"
    assert pending["manifest_id"] == (
        "sha256:683bf6485b42755dfe4f0210b63d5a2a70975be7801ef3f0d3454b427921a00e"
    )
    assert pending["task_id"] == (
        "sha256:01a2cb69e51eed898834340422f7383dc4c7e09e19bac6daf9a16de17b4cb5e8"
    )
    assert pending["sample_indices"] == list(range(4, 16))
    assert pending["reserved_tokens"] == {
        "prefill": 1584,
        "sample": 49152,
        "train": 0,
    }
    proof = retry.INCIDENT["remote_proof"]
    assert proof["tinker_session_id"] == "ca14e9cf-61aa-57a3-8657-26f018c26710"
    assert proof["future_checks"] == [
        {"future_id": future_id, "http_status": 404} for future_id in range(3200, 3212)
    ]
    assert proof["post_failure_trace_sha256"] == (
        "sha256:b417f8c985f64c297868c31cb0067a1a11e8dbc955398ebe530265d2495e4c43"
    )


def test_one_exact_refine_retry_preserves_reservation_and_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _failed_refine(tmp_path, monkeypatch)
    directory, preflight, manifest, record, reservation, recovery_run, trace = values
    errors_before = (directory / "errors.jsonl").read_bytes()
    artifacts = BoundaryRetryArtifacts(
        directory, preflight=preflight, new_run=recovery_run, trace_path=trace
    )
    limits = action_limits("extension2-refine", manifest, 12, 4096)
    ledger = artifacts.restore_ledger("extension2-refine", limits, 30)
    assert ledger.limits == limits.plus(reservation)
    assert ledger.authorized_usd == 30
    assert ledger.committed == ledger.observed == reservation
    broad_limits = TokenBudget(500_000, 16_384, 0)
    assert artifacts.restore_ledger("extension2-broad", broad_limits, 10).limits == (
        broad_limits
    )
    sampler, rows = _collect(artifacts, manifest, ledger)
    expected_seeds = tuple(
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
    assert tuple(sampler.seeds) == expected_seeds
    assert len(rows[0]) == len(rows[1]) == 12
    assert ledger.committed == TokenBudget(3168, 98304, 0)
    marker = json.loads((directory / RETRY_MARKER).read_text())
    assert marker["status"] == "completed"
    assert marker["incident"] == retry.INCIDENT
    assert not (directory / "pending_group.json").exists()
    assert (directory / "errors.jsonl").read_bytes() == errors_before


def test_completed_marker_restores_failed_plus_journal_floor_after_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _failed_refine(tmp_path, monkeypatch)
    directory, preflight, manifest, _, _, recovery_run, trace = values
    artifacts = BoundaryRetryArtifacts(
        directory, preflight=preflight, new_run=recovery_run, trace_path=trace
    )
    limits = action_limits("extension2-refine", manifest, 12, 4096)
    ledger = artifacts.restore_ledger("extension2-refine", limits, 30)

    def crash_before_billing(_ledgers):  # type: ignore[no-untyped-def]
        raise RuntimeError("crash before billing")

    monkeypatch.setattr(artifacts, "write_billing", crash_before_billing)
    with pytest.raises(RuntimeError, match="crash before billing"):
        _collect(artifacts, manifest, ledger)
    assert json.loads((directory / RETRY_MARKER).read_text())["status"] == "completed"
    assert json.loads((directory / "billing.json").read_text())["actions"][
        "extension2-refine"
    ]["committed_tokens"] == {
        "prefill": 1584,
        "sample": 49152,
        "train": 0,
    }
    restarted = BoundaryRetryArtifacts(
        directory, preflight=preflight, new_run=recovery_run, trace_path=None
    )
    resumed = restarted.restore_ledger("extension2-refine", limits, 30)
    assert resumed.committed == TokenBudget(3168, 98304, 0)
    assert resumed.observed == TokenBudget(3168, 49176, 0)


def test_retry_mismatch_and_second_ambiguity_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _failed_refine(tmp_path, monkeypatch)
    directory, preflight, manifest, record, reservation, recovery_run, trace = values
    artifacts = BoundaryRetryArtifacts(
        directory, preflight=preflight, new_run=recovery_run, trace_path=trace
    )
    coordinates = {
        "manifest_id": manifest.manifest_id,
        "run_id": "boundary-live:extension2-refine",
        "sample_indices": tuple(range(4, 16)),
        "reservation": reservation,
    }
    with pytest.raises(RunnerGateError, match="coordinates differ"):
        artifacts.begin_group("extension2-refine", "wrong-task", **coordinates)
    assert not (directory / RETRY_MARKER).exists()
    artifacts.begin_group("extension2-refine", record.task_id, **coordinates)
    with pytest.raises(RunnerGateError, match="no second retry"):
        BoundaryRetryArtifacts(
            directory, preflight=preflight, new_run=recovery_run, trace_path=trace
        )


def test_retry_rejects_wrong_trace_and_tampered_completed_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _failed_refine(tmp_path, monkeypatch)
    directory, preflight, manifest, _, _, recovery_run, trace = values
    trace.write_bytes(b"different trace")
    with pytest.raises(RunnerGateError, match="not exactly retryable"):
        BoundaryRetryArtifacts(
            directory, preflight=preflight, new_run=recovery_run, trace_path=trace
        )
    trace.write_bytes(b"exact test trace")
    artifacts = BoundaryRetryArtifacts(
        directory, preflight=preflight, new_run=recovery_run, trace_path=trace
    )
    ledger = artifacts.restore_ledger(
        "extension2-refine", action_limits("extension2-refine", manifest, 12, 4096), 30
    )
    _collect(artifacts, manifest, ledger)
    marker = json.loads((directory / RETRY_MARKER).read_text())
    marker["incident"]["original_git_commit"] = "wrong"
    (directory / RETRY_MARKER).write_text(json.dumps(marker))
    with pytest.raises(RunnerGateError, match="differs from the incident"):
        BoundaryRetryArtifacts(
            directory, preflight=preflight, new_run=recovery_run, trace_path=None
        )
