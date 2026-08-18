from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from duraseed import calibration_prior_stage_a as prior
from duraseed.calibration_billing import reconcile_calibration_billing
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import RunRecord, RunStatus, write_run_record
from duraseed.runners import RunnerGateError
from duraseed.teacher_exposure_spec import (
    AMENDED_AGGREGATE_CAP_USD,
    AMENDED_STAGE_A_CAP_USD,
    PRIOR_DIRECT_STAGE_A_CHARGE_USD,
)
from tests.unit.test_teacher_exposure_repair import (
    M1_LINEAGE,
    PRIOR_LINEAGE,
    SHA_A,
    _required,
    _run,
)


SHA = "sha256:" + "a" * 64


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, pending: bool) -> tuple:
    parent = root / "parent"
    run = root / prior.PRIOR_DIRECT_STAGE_A_RUN_ID
    attempt = run / ("stage-a-arms/complete-bounded-stage-a/attempt-0001")
    attempt.mkdir(parents=True)
    parent_lineage = {"run_id": "parent"}
    repair = {"run_id": "repair"}
    m1 = {"run_id": "m1"}
    preflight = {
        "run_id": run.name,
        "project_id": "project",
        "parent_calibration": parent_lineage,
        "prior_repair": repair,
        "interrupted_m1": m1,
        "cost_caps_usd": {
            "teacher-dose": 0.0,
            "teacher-allocation": 0,
            "stage-a": prior.DIRECT_M0_STAGE_A_CAP_USD,
            "total": prior.DIRECT_M0_AGGREGATE_CAP_USD,
        },
        "lifetime_calibration_cap_usd": prior.LIFETIME_CALIBRATION_CAP_USD,
    }
    preflight_raw = canonical_json_bytes(preflight)
    (run / "preflight.json").write_bytes(preflight_raw)
    terminal = {
        "status": "no_eligible_learning_rate",
        "preflight_sha256": sha256_bytes(preflight_raw),
        "usage": {
            "committed_tokens": {
                "prefill": prior.PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[0],
                "sample": prior.PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[1],
                "train": prior.PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[2],
            },
            "committed_cost_usd": prior.PRIOR_DIRECT_STAGE_A_CHARGE_USD,
        },
    }
    terminal_raw = canonical_json_bytes(terminal)
    (run / "stage-a-terminal.json").write_bytes(terminal_raw)
    journal = {
        "completed_count": 1_933,
        "pending": {"operation": "sample"} if pending else None,
        "reserved_floor": {
            "prefill_tokens": prior.PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[0],
            "sample_tokens": prior.PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[1],
            "train_tokens": prior.PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[2],
            "fixed_usd": prior.PRIOR_DIRECT_STAGE_A_COMMITTED_FIXED_USD,
        },
    }
    journal_raw = canonical_json_bytes(journal)
    (attempt / "remote-call-state.json").write_bytes(journal_raw)
    now = datetime(2026, 8, 18, 12, tzinfo=UTC)
    write_run_record(
        run,
        RunRecord(
            protocol_version="v1",
            git_commit="commit",
            resolved_config_hash=SHA,
            run_kind="stage_a_calibration",
            method=None,
            seed=17,
            model_id="Qwen/Qwen3.5-9B-Base",
            renderer="qwen3",
            lora_rank=32,
            task_manifest_ids={"a_monitor": SHA, "a_rl_train": SHA},
            parent_tinker_checkpoint_path="tinker://m0/state",
            tinker_session_id=prior.PRIOR_DIRECT_STAGE_A_SESSION_ID,
            status=RunStatus.FAILED,
            started_at=now,
            updated_at=now,
            finished_at=now,
            project_id="project",
            authorized_cost_usd=prior.DIRECT_M0_STAGE_A_CAP_USD,
            reserved_cost_usd=prior.DIRECT_M0_STAGE_A_CAP_USD,
        ),
    )
    monkeypatch.setattr(
        prior, "PRIOR_DIRECT_STAGE_A_PREFLIGHT_SHA256", sha256_bytes(preflight_raw)
    )
    monkeypatch.setattr(
        prior, "PRIOR_DIRECT_STAGE_A_TERMINAL_SHA256", sha256_bytes(terminal_raw)
    )
    monkeypatch.setattr(
        prior, "PRIOR_DIRECT_STAGE_A_JOURNAL_SHA256", sha256_bytes(journal_raw)
    )
    monkeypatch.setattr(
        prior,
        "existing_stage_a_terminal",
        lambda *_args: {"terminal": terminal},
    )
    return parent, parent_lineage, repair, m1


def test_prior_stage_a_binds_clear_journal_and_full_committed_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, parent_lineage, repair, m1 = _fixture(tmp_path, monkeypatch, pending=False)

    lineage = prior.load_prior_stage_a(
        parent,
        project_id="project",
        parent_lineage=parent_lineage,
        prior_repair_lineage=repair,
        m1_lineage=m1,
    )

    assert lineage["pending_remote_calls"] == 0
    assert lineage["charged_stage_a_usd"] == 51.876308513
    assert lineage["billing_basis"] == "conservative_local_committed_upper_bound"


def test_prior_stage_a_rejects_an_ambiguous_pending_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, parent_lineage, repair, m1 = _fixture(tmp_path, monkeypatch, pending=True)

    with pytest.raises(RunnerGateError, match="terminal or spend bound"):
        prior.load_prior_stage_a(
            parent,
            project_id="project",
            parent_lineage=parent_lineage,
            prior_repair_lineage=repair,
            m1_lineage=m1,
        )


def test_amended_reconciler_charges_prior_direct_stage_a_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "child"
    root.mkdir()
    run = _run(root, RunStatus.COMPLETED)
    write_run_record(
        root,
        run.model_copy(
            update={
                "authorized_cost_usd": AMENDED_STAGE_A_CAP_USD,
                "reserved_cost_usd": AMENDED_STAGE_A_CAP_USD,
            }
        ),
    )
    sessions = {"schema_version": "lineage", "session_ids": ["session-child"]}
    (root / "session-lineage.json").write_bytes(canonical_json_bytes(sessions))
    prior_stage_a = {
        "run_id": "prior-direct",
        "pending_remote_calls": 0,
        "charged_stage_a_usd": PRIOR_DIRECT_STAGE_A_CHARGE_USD,
    }
    required = _required(
        "completed",
        action_caps_usd={"stage-a": AMENDED_STAGE_A_CAP_USD},
        aggregate_cap_usd=AMENDED_AGGREGATE_CAP_USD,
        prior_direct_stage_a=prior_stage_a,
        prior_direct_stage_a_charge_usd=PRIOR_DIRECT_STAGE_A_CHARGE_USD,
    )
    (root / "billing-reconciliation-required.json").write_bytes(
        canonical_json_bytes(required)
    )
    (root / "preflight.json").write_bytes(
        canonical_json_bytes(
            {
                "parent_calibration": {"billing_sha256": SHA_A},
                "prior_repair": PRIOR_LINEAGE,
                "interrupted_m1": M1_LINEAGE,
                "prior_direct_stage_a": prior_stage_a,
            }
        )
    )
    raw = tmp_path / "raw.json"
    raw.write_bytes(canonical_json_bytes({"data": [{"session_id": "session-child"}]}))
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "duraseed-calibration-final-reconciliation-v1",
                "status": "billing_reconciled",
                "run_id": "child",
                "project_id": "project",
                "session_ids": ["session-child"],
                "raw_billing_sha256": sha256_bytes(raw.read_bytes()),
                "raw_billing_entry_count": 1,
                "action_billed_usd": {"stage-a": 3},
                "aggregate_billed_usd": 3,
                "remaining_balance_usd": 4000,
                "protected_reserve_usd": 984.46,
                "protected_reserve_survives": True,
                "raw_usage_cutoff_utc": "2026-08-17T13:00:00Z",
                "reconciled_at_utc": "2026-08-17T14:00:00Z",
            }
        )
    )

    result = reconcile_calibration_billing(root, reconciliation, raw)

    assert result["aggregate_billed_usd"] == 3
