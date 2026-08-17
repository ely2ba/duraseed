from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.calibration_billing import reconcile_calibration_billing
from duraseed.calibration_billing_requirement import calibration_billing_requirement
from duraseed.calibration_parent import PARENT_BILLED_USD, PARENT_RUN_ID
from duraseed.calibration_stage_a_terminal import (
    StageAScientificFailure,
    existing_stage_a_terminal,
    finish_stage_a_terminal,
)
from duraseed.config import load_pilot_config
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import RunRecord, RunStatus, write_run_record
from duraseed.runtime import TokenBudget, TokenLedger
from duraseed.training.acquisition_freeze import StageALiveEvidence
from duraseed.training.stage_a_calibration import (
    StageALearningRateDecision,
    StageALearningRateDecisionStatus,
)


def _decision(method: str, selected: bool) -> StageALearningRateDecision:
    status = (
        StageALearningRateDecisionStatus.SELECTED
        if selected
        else StageALearningRateDecisionStatus.NO_ELIGIBLE_CANDIDATE
    )
    rate = 1e-4 if selected else None
    return StageALearningRateDecision(
        method, status, rate, (rate,) if rate else (), (), "complete grid decision"
    )


def test_negative_stage_a_terminal_is_durable_and_billable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from duraseed import calibration_stage_a_terminal as terminal_module

    root = tmp_path / "direct-negative"
    arm = root / "stage-a-arms/complete-bounded-stage-a"
    arm.mkdir(parents=True)
    evidence = StageALiveEvidence((), (), ())
    (arm / "completed.json").write_bytes(
        canonical_json_bytes({"evidence": (evidence, None)})
    )
    ttl = root / "stage-a-arms/checkpoint-ttl-audit.json"
    ttl.write_bytes(canonical_json_bytes({"rows": [{"path": "candidate"}]}))
    now = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    write_run_record(
        root,
        RunRecord(
            protocol_version="v6",
            git_commit="commit",
            resolved_config_hash="sha256:" + "a" * 64,
            run_kind="stage_a_calibration",
            method=None,
            seed=17,
            model_id="Qwen/Qwen3.5-9B-Base",
            renderer="qwen3",
            lora_rank=32,
            task_manifest_ids={
                "a_monitor": "sha256:" + "e" * 64,
                "a_rl_train": "sha256:" + "f" * 64,
            },
            parent_tinker_checkpoint_path="tinker://m0/state",
            tinker_session_id="session",
            status=RunStatus.FAILED,
            started_at=now,
            updated_at=now,
            finished_at=now,
            project_id="project",
            authorized_cost_usd=153.32,
            reserved_cost_usd=153.32,
        ),
    )
    failure = StageAScientificFailure(
        "no_eligible_learning_rate",
        evidence,
        (_decision("B-S", True), _decision("B-G", False)),
        None,
    )
    integrity = {"artifact_sha256": "sha256:" + "b" * 64}
    monkeypatch.setattr(
        terminal_module, "seal_calibration_action", lambda *_args, **_kwargs: integrity
    )
    monkeypatch.setattr(
        terminal_module,
        "validate_action_ttl_audit",
        lambda *_args, **_kwargs: {"rows": [{"path": "candidate"}]},
    )
    monkeypatch.setattr(
        terminal_module, "finish_calibration_run", lambda *_args, **_kwargs: None
    )
    prior = {"run_id": "repair", "charged_teacher_cap_usd": 44.27}
    m1 = {"run_id": "m1", "charged_teacher_cap_usd": 53.35}
    parent = SimpleNamespace(
        parent_run_id=PARENT_RUN_ID,
        parent_billing_sha256="sha256:" + "c" * 64,
        parent_billed_usd=PARENT_BILLED_USD,
        prior_repair_lineage=prior,
        prior_repair_teacher_cap_usd=44.27,
        m1_lineage=m1,
        m1_teacher_cap_usd=53.35,
        protected_reserve_usd=984.46,
    )
    inputs = SimpleNamespace(
        run_id=root.name,
        project_id="project",
        stage_a_ledger=TokenLedger(TokenBudget(0, 0, 0), 153.32),
        parent_teacher_evidence=parent,
    )
    preflight_value = {
        "parent_calibration": {"billing_sha256": parent.parent_billing_sha256},
        "prior_repair": prior,
        "interrupted_m1": m1,
    }
    (root / "preflight.json").write_bytes(canonical_json_bytes(preflight_value))
    preflight = sha256_bytes((root / "preflight.json").read_bytes())
    result = finish_stage_a_terminal(
        inputs,
        root,
        preflight_sha256=preflight,
        failure=failure,
        integrity=integrity,
        ttl_audit_sha256=sha256_bytes(ttl.read_bytes()),
    )

    assert result["state"]["status"] == "failed"
    assert existing_stage_a_terminal(root, preflight) is not None
    (root / "session-lineage.json").write_bytes(
        canonical_json_bytes({"session_ids": ["session"]})
    )
    requirement = calibration_billing_requirement(
        inputs, root, RunStatus.FAILED, now, 1.0
    )
    assert requirement is not None
    assert requirement["terminal_status"] == "no_eligible_learning_rate"
    (root / "billing-reconciliation-required.json").write_bytes(
        canonical_json_bytes(requirement)
    )
    raw = tmp_path / "raw.json"
    raw.write_bytes(canonical_json_bytes({"data": [{"session_id": "session"}]}))
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "duraseed-calibration-final-reconciliation-v1",
                "status": "billing_reconciled",
                "run_id": root.name,
                "project_id": "project",
                "session_ids": ["session"],
                "raw_billing_sha256": sha256_bytes(raw.read_bytes()),
                "raw_billing_entry_count": 1,
                "action_billed_usd": {"stage-a": 1},
                "aggregate_billed_usd": 1,
                "remaining_balance_usd": 4000,
                "protected_reserve_usd": 984.46,
                "protected_reserve_survives": True,
                "raw_usage_cutoff_utc": "2026-08-17T13:00:00Z",
                "reconciled_at_utc": "2026-08-17T14:00:00Z",
            }
        )
    )
    assert (
        reconcile_calibration_billing(root, reconciliation, raw)["aggregate_billed_usd"]
        == 1
    )


def test_negative_terminal_resume_does_not_construct_remote_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from duraseed.calibration_input_loader import (
        ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256,
    )
    from duraseed.runners import calibration_launch as launch

    config = load_pilot_config(
        Path(__file__).resolve().parents[2] / "duraseed_pilot_config.yaml"
    )
    parent = SimpleNamespace(
        remaining_balance_usd=4922.30,
        protected_reserve_usd=984.46,
        parent_billed_usd=PARENT_BILLED_USD,
        lifetime_sunk_usd=PARENT_BILLED_USD + 44.27 + 53.35,
        parent_billing_sha256="sha256:" + "a" * 64,
        parent_raw_billing_sha256="sha256:" + "b" * 64,
    )
    loaded = SimpleNamespace(
        teacher=object(),
        prompts=object(),
        authorization_sha256=ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256,
        equivalence_sha256="sha256:" + "c" * 64,
    )
    source = SimpleNamespace(
        smoke=SimpleNamespace(runtime_diagnostic_passed=True),
        m0_sampler_path="tinker://m0/sampler",
        m0_state_path="tinker://m0/state",
        m0_training_step=2,
    )
    allocation = SimpleNamespace(
        teacher_tokens=TokenBudget(0, 0, 0),
        stage_a_tokens=TokenBudget(1, 1, 1),
        teacher_cap_usd=0,
        stage_a_cap_usd=153.32,
    )
    modules = {
        "tinker_cookbook.tokenizer_utils": SimpleNamespace(
            get_tokenizer=lambda _model: object()
        ),
        "tinker_cookbook.renderers": SimpleNamespace(
            get_renderer=lambda *_args, **_kwargs: object(), TrainOnWhat=object()
        ),
    }
    monkeypatch.setattr(launch, "load_pilot_config", lambda _path: config)
    monkeypatch.setattr(launch, "load_calibration_parent", lambda *_a, **_k: parent)
    monkeypatch.setattr(
        launch, "load_calibration_source_objects", lambda **_kwargs: loaded
    )
    monkeypatch.setattr(
        launch, "authenticate_calibration_sources", lambda **_kwargs: source
    )
    monkeypatch.setattr(
        launch,
        "load_max_token_evidence",
        lambda *_args: SimpleNamespace(selected_max_tokens=4096),
    )
    monkeypatch.setattr(launch.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(launch, "calibration_allocation", lambda _inputs: allocation)
    monkeypatch.setattr(launch, "_git_commit", lambda: "commit")
    monkeypatch.setattr(
        launch, "calibration_preflight", lambda *_args: {"run": "direct-negative"}
    )
    monkeypatch.setattr(launch, "validate_restart_reconciliations", lambda *_a: None)
    monkeypatch.setattr(
        launch,
        "existing_stage_a_terminal",
        lambda *_a: {"terminal": {"status": "no_eligible_learning_rate"}},
    )
    monkeypatch.setattr(
        launch,
        "completed_calibration",
        lambda *_a: pytest.fail("completed path should not run after a terminal"),
    )
    monkeypatch.setattr(
        launch, "load_sdk", lambda: pytest.fail("resume constructed a remote SDK")
    )
    run_id = "direct-negative"
    result = asyncio.run(
        launch.run_remote_calibration(
            run_id=run_id,
            output_root=tmp_path,
            config_path="config",
            boundary_directory="boundary",
            source_directory="sources",
            smoke_acceptance_path="smoke",
            m0_selection_path="selection",
            m0_ttl_path="ttl",
            panel_split_authorization_path="panel-auth",
            panel_split_equivalence_path="panel-equivalence",
            max_token_specification_path="max-spec",
            max_token_authorization_path="max-auth",
            max_token_evidence_path="max-evidence",
            billing_reconciliation_path="billing",
            raw_billing_path="raw-billing",
            project_id="project",
            authorized_cost_usd="153.32",
            human_approval=True,
        )
    )

    assert result == tmp_path / run_id
