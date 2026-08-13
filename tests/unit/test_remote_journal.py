from __future__ import annotations

import json
from pathlib import Path

import pytest

from duraseed.calibration_attempts import (
    ArmAttempts,
    ReconciledRestart,
    load_reconciled_restart,
)
from duraseed.provenance import sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import TokenBudget, TokenLedger


SHA = "sha256:" + "a" * 64
CONTEXT = {
    "run_id": "calibration-run",
    "action": "teacher-dose",
    "project_id": "project",
    "failed_tinker_session_id": "session",
    "preflight_sha256": SHA,
}


def _ledger() -> TokenLedger:
    return TokenLedger(TokenBudget(10_000, 10_000, 10_000), 150)


def _attempts(root: Path, ledger: TokenLedger, *rows: ReconciledRestart) -> ArmAttempts:
    context = {
        key: value
        for key, value in CONTEXT.items()
        if key != "failed_tinker_session_id"
    }
    return ArmAttempts(root, ledger, reconciliations=rows, **context)


def test_remote_journal_rejects_ambiguous_restart(tmp_path: Path) -> None:
    journal = RemoteJournal(tmp_path)
    journal.begin("paid", {"arm": 1}, {"sample_tokens": 8})
    with pytest.raises(RunnerGateError, match="ambiguous pending remote call"):
        RemoteJournal(tmp_path)


def test_completed_arm_skips_exactly_and_hydrates_spend_floor(tmp_path: Path) -> None:
    first = _ledger()
    attempts = _attempts(tmp_path, first)
    arm = attempts.open("dose-1")
    assert arm.journal is not None
    arm.journal.begin(
        "paid",
        {"arm": 1},
        {"prefill_tokens": 3, "sample_tokens": 8, "train_tokens": 13},
    )
    arm.journal.complete({"rows": 1})
    attempts.complete(arm, {"selected": True})

    resumed = _ledger()
    completed = _attempts(tmp_path, resumed).open("dose-1")
    assert completed.completed_payload == {"selected": True}
    assert resumed.committed == TokenBudget(3, 8, 13)


def test_incomplete_arm_requires_exact_reconciliation_then_fresh_attempt(
    tmp_path: Path,
) -> None:
    first = _attempts(tmp_path, _ledger()).open("dose-1")
    assert first.journal is not None
    first.journal.begin("paid", {"arm": 1}, {"sample_tokens": 8})
    with pytest.raises(RunnerGateError, match="reconcile billing"):
        _attempts(tmp_path, _ledger()).open("dose-1")

    reconciliation = ReconciledRestart(
        **CONTEXT,
        arm_id="dose-1",
        failed_attempt=1,
        raw_billing_sha256=SHA,
        raw_billing_entry_count=1,
        raw_usage_cutoff_utc="2026-08-14T00:00:00+00:00",
        cumulative_billed_usd=1.25,
        aggregate_billed_usd=1.25,
        reconciled_at_utc="2026-08-14T01:00:00+00:00",
        authorizer="Ely",
        authorized_at_utc="2026-08-14T01:01:00+00:00",
        artifact_sha256=SHA,
    )
    resumed_ledger = _ledger()
    resumed = _attempts(tmp_path, resumed_ledger, reconciliation).open("dose-1")
    assert resumed.number == 2
    assert resumed.directory.name == "attempt-0002"
    assert resumed_ledger.committed.sample == 8


def test_reconciliation_binds_raw_billing_and_launch_identity(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text('{"data":[{"session_id":"session","usage":1}]}')
    raw_hash = sha256_bytes(raw.read_bytes())
    artifact = tmp_path / "reconciliation.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "duraseed-calibration-reconciliation-v1",
                "status": "billing_reconciled",
                "authorized_restart": True,
                **CONTEXT,
                "arm_id": "dose-1",
                "failed_attempt": 1,
                "raw_billing_sha256": raw_hash,
                "raw_billing_entry_count": 1,
                "raw_usage_cutoff_utc": "2026-08-13T00:00:00+00:00",
                "console_cumulative_billed_usd": 1.25,
                "console_aggregate_billed_usd": 1.25,
                "reconciled_at_utc": "2026-08-13T01:00:00+00:00",
                "authorizer": "Ely",
                "authorized_at_utc": "2026-08-13T01:01:00+00:00",
            }
        )
    )
    loaded = load_reconciled_restart(artifact, raw)
    assert loaded.project_id == "project"
    raw.write_text('{"data":[{"session_id":"session","usage":2}]}')
    with pytest.raises(RunnerGateError, match="hash differs"):
        load_reconciled_restart(artifact, raw)


def test_reconciliation_cannot_replay_in_another_run(tmp_path: Path) -> None:
    reconciliation = ReconciledRestart(
        **CONTEXT,
        arm_id="dose-1",
        failed_attempt=1,
        raw_billing_sha256=SHA,
        raw_billing_entry_count=1,
        raw_usage_cutoff_utc="2026-08-13T00:00:00+00:00",
        cumulative_billed_usd=1.0,
        aggregate_billed_usd=1.0,
        reconciled_at_utc="2026-08-13T01:00:00+00:00",
        authorizer="Ely",
        authorized_at_utc="2026-08-13T01:01:00+00:00",
        artifact_sha256=SHA,
    )
    with pytest.raises(RunnerGateError, match="another launch"):
        ArmAttempts(
            tmp_path,
            _ledger(),
            run_id="other-run",
            action="teacher-dose",
            project_id="project",
            preflight_sha256=SHA,
            reconciliations=(reconciliation,),
        )
