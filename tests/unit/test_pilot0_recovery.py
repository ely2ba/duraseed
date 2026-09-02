from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from duraseed.cli import app
from duraseed.pilot0_recovery import (
    prepare_pilot0_recovery,
    restore_pilot0_recovery_ledger,
)
from duraseed.provenance import canonical_json_bytes
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners import pilot0_stage_a
from duraseed.runtime import PRICE_SNAPSHOT, TokenBudget, TokenLedger, UsageQuantities


def _cost(tokens: TokenBudget) -> float:
    return (
        PRICE_SNAPSHOT.cost(
            UsageQuantities(
                prefill_tokens=tokens.prefill,
                sample_tokens=tokens.sample,
                train_tokens=tokens.train,
            )
        )
        + 0.1
    )


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def test_infrastructure_recovery_reuses_checkpoint_and_reconciles_one_call(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    segment = root / "seed-11/B-S/steps-170-180"
    evaluation = segment / "a-monitor"
    committed = TokenBudget(10, 20, 30)
    observed = TokenBudget(9, 14, 30)
    _write(
        root / "run.json",
        {
            "run_id": root.name,
            "status": "interrupted",
            "error": "RequestFailedError: self.request_id='session:sample:17:2'",
            "ledger": {
                "committed_tokens": {
                    "prefill": committed.prefill,
                    "sample": committed.sample,
                    "train": committed.train,
                },
                "observed_tokens": {
                    "prefill": observed.prefill,
                    "sample": observed.sample,
                    "train": observed.train,
                },
                "committed_cost_usd": _cost(committed),
                "observed_cost_usd": _cost(observed),
            },
        },
    )
    _write(
        root / "preflight.json",
        {"run_id": root.name, "lineage": {"session_id": "failed-session"}},
    )
    (evaluation / "generations.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (evaluation / "generations.jsonl").write_text("{}\n{}\n")
    (evaluation / "rewards.jsonl").write_text("{}\n{}\n")
    _write(
        evaluation / "remote-call-state.json",
        {
            "completed_count": 2,
            "attempt_started_at_utc": "2026-08-25T00:00:00+00:00",
            "reserved_floor": {
                "prefill_tokens": 2,
                "sample_tokens": 4,
                "train_tokens": 0,
                "fixed_usd": 0.0,
            },
            "pending": {
                "sequence": 2,
                "operation": "pilot0-validation-group",
                "coordinate": {"task_id": "task-2"},
                "reservation": {
                    "prefill_tokens": 1,
                    "sample_tokens": 2,
                    "train_tokens": 0,
                    "fixed_usd": 0.0,
                },
            },
        },
    )
    (segment / "remote-calls.jsonl").write_text(
        json.dumps(
            {
                "sequence": 11,
                "status": "completed",
                "operation": "pilot0-save-checkpoint-pair",
                "sampler_path": "tinker://session/sampler_weights/step-180",
                "state_path": "tinker://session/weights/step-180",
            }
        )
        + "\n"
    )

    recovery = prepare_pilot0_recovery(
        root, recovery_session_id="resume-session", recovery_git_commit="commit"
    )
    assert recovery["sampler_path"].endswith("step-180")
    assert recovery["method"] == "B-S"
    assert (
        json.loads((evaluation / "remote-call-state.json").read_text())["pending"]
        is None
    )
    assert json.loads((evaluation / "remote-calls.jsonl").read_text())["status"] == (
        "failed_infrastructure"
    )

    inputs = SimpleNamespace(ledger=TokenLedger(TokenBudget(100, 100, 100), 10.0))
    assert restore_pilot0_recovery_ledger(inputs, root) is True
    assert inputs.ledger.committed == TokenBudget(9, 18, 30)
    assert inputs.ledger.observed == TokenBudget(8, 12, 30)
    resumed = RemoteJournal(evaluation, reconciled_resume=True)
    resumed.begin("replacement", {}, {"prefill_tokens": 1, "sample_tokens": 2})
    resumed.complete({"operation": "replacement"})
    assert (
        json.loads((evaluation / "remote-call-state.json").read_text())[
            "completed_count"
        ]
        == 3
    )


def test_cli_does_not_render_exception_locals() -> None:
    assert app.pretty_exceptions_enable is False


def test_stage_b_grouped_evaluation_timeout_is_reconciled(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    segment = root / "seed-29/B-S/stage-b/steps-1-2"
    evaluation = segment / "a-retention"
    committed = TokenBudget(100, 200, 30)
    observed = TokenBudget(90, 160, 30)
    _write(
        root / "run.json",
        {
            "run_id": root.name,
            "status": "interrupted",
            "error": (
                "APIConnectionError: No progress made in 7200s. "
                "Requests appear to be stuck."
            ),
            "ledger": {
                "committed_tokens": {
                    "prefill": committed.prefill,
                    "sample": committed.sample,
                    "train": committed.train,
                },
                "observed_tokens": {
                    "prefill": observed.prefill,
                    "sample": observed.sample,
                    "train": observed.train,
                },
                "committed_cost_usd": _cost(committed),
                "observed_cost_usd": _cost(observed),
            },
        },
    )
    _write(
        root / "preflight.json",
        {"run_id": root.name, "lineage": {"session_id": "failed-session"}},
    )
    evaluation.mkdir(parents=True)
    rows = "{}\n" * 8
    (evaluation / "generations.jsonl").write_text(rows)
    (evaluation / "rewards.jsonl").write_text(rows)
    _write(
        evaluation / "remote-call-state.json",
        {
            "completed_count": 2,
            "attempt_started_at_utc": "2026-09-02T00:00:00+00:00",
            "reserved_floor": {
                "prefill_tokens": 8,
                "sample_tokens": 16,
                "train_tokens": 0,
                "fixed_usd": 0.0,
            },
            "pending": {
                "sequence": 2,
                "operation": "pilot0-validation-group",
                "coordinate": {"task_id": "task-2"},
                "reservation": {
                    "prefill_tokens": 4,
                    "sample_tokens": 8,
                    "train_tokens": 0,
                    "fixed_usd": 0.0,
                },
            },
        },
    )
    (segment / "remote-calls.jsonl").write_text(
        json.dumps(
            {
                "sequence": 4,
                "status": "completed",
                "operation": "pilot0-save-checkpoint-pair",
                "sampler_path": "tinker://session/sampler_weights/stage-b-step-2",
                "state_path": "tinker://session/weights/stage-b-step-2",
            }
        )
        + "\n"
    )

    recovery = prepare_pilot0_recovery(
        root, recovery_session_id="resume-session", recovery_git_commit="commit"
    )
    assert recovery["phase"] == "stage_b"
    assert recovery["method"] == "B-S"
    assert recovery["samples_per_completed_call"] == 4
    assert recovery["failed_request_id"] is None
    assert recovery["failed_sequence"] == 2
    assert (
        json.loads((evaluation / "remote-call-state.json").read_text())["pending"]
        is None
    )


def test_recovery_segment_skips_retraining_and_reuses_saved_pair(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "pilot"
    output = root / "seed-11/B-S/steps-170-180"
    _write(
        root / "infrastructure-recovery.json",
        {
            "schema_version": "duraseed-pilot0-infrastructure-recovery-v1",
            "status": "authorized_resume",
            "run_id": root.name,
            "recovery_session_id": "resume",
            "segment": output.relative_to(root).as_posix(),
            "evaluation": (output / "a-monitor").relative_to(root).as_posix(),
            "method": "B-S",
            "start": 170,
            "stop": 180,
            "sampler_path": "tinker://session/sampler_weights/step-180",
            "state_path": "tinker://session/weights/step-180",
        },
    )
    _write(
        output / "remote-call-state.json",
        {
            "completed_count": 1,
            "attempt_started_at_utc": "2026-08-25T00:00:00+00:00",
            "reserved_floor": {
                "prefill_tokens": 0,
                "sample_tokens": 0,
                "train_tokens": 0,
                "fixed_usd": 0.0,
            },
            "pending": None,
        },
    )
    (output / "remote-calls.jsonl").write_text(
        '{"operation":"saved","sequence":0,"status":"completed"}\n'
    )
    inputs = SimpleNamespace(
        output_root=tmp_path,
        run_id=root.name,
        project_id="project",
        source_authentication=SimpleNamespace(bundle_sha256="sha256:" + "a" * 64),
        acquisition=SimpleNamespace(
            learning_rates={"static_sft": 0.0001}, selected_max_tokens=4096
        ),
        ledger=TokenLedger(TokenBudget(100, 100, 100), 10),
    )
    source = SimpleNamespace(seed=11, a_cadence=object())
    monkeypatch.setattr(
        pilot0_stage_a, "segment_coordinates", lambda *args, **kwargs: kwargs
    )

    async def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("recovery retrained or resaved the interrupted segment")

    async def sampler(*args, **kwargs):  # type: ignore[no-untyped-def]
        return object()

    async def evaluate(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {"generation_sha256": "sha256:" + "b" * 64}

    monkeypatch.setattr(pilot0_stage_a, "restore_runtime", forbidden)
    monkeypatch.setattr(pilot0_stage_a, "supervised_update", forbidden)
    monkeypatch.setattr(pilot0_stage_a, "save_pair", forbidden)
    monkeypatch.setattr(pilot0_stage_a, "sampler_for_path", sampler)
    monkeypatch.setattr(pilot0_stage_a, "evaluate_manifest", evaluate)
    monkeypatch.setattr(
        pilot0_stage_a, "write_segment", lambda directory, value, **kwargs: value
    )
    result = asyncio.run(
        pilot0_stage_a._branch_segment(
            inputs,
            source,
            {"sampler_path": "m0-sampler", "state_path": "m0-state"},
            {"sampler_path": "step-170-sampler", "state_path": "step-170-state"},
            method="B-S",
            start=170,
            stop=180,
            pools={},
            sources={},
            output=output,
            preflight_sha256="sha256:" + "c" * 64,
        )
    )
    assert result["sampler_path"].endswith("step-180")
    assert result["state_path"].endswith("step-180")
