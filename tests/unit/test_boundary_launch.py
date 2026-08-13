from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from duraseed.runners import RunnerGateError
from duraseed.runners.boundary_launch import (
    authenticate_live_smoke,
    authorize_boundary,
)
from duraseed.boundary_live_sources import load_frozen_extension1_confirmation
from duraseed.config import load_pilot_config
from duraseed.boundary_live_sources import load_boundary_live_source
from duraseed.run_records import RunRecord, RunStatus


def _write_smoke(root: Path, *, real: bool = True, project_id: str = "project") -> Path:
    path = root / "smoke-id" / "acceptance.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "phase_label": "live-smoke-gate",
                "status": "passed",
                "real_data": real,
                "online_offline_reward_parity": True,
                "stop_contract_verified": True,
                "full_state_resume": True,
                "weights_only_branch": True,
                "max_tokens": {
                    "protocol_value": 4096,
                    "sample_count": 16,
                    "runtime_diagnostic_passed": True,
                },
                "checkpoint_lineage": {
                    "stage_a_state_path": "tinker://stage-a",
                    "resumed_roundtrip_state_path": "tinker://roundtrip",
                    "stage_b_sampler_path": "tinker://stage-b",
                },
                "observed_cost_usd": 1.25,
            }
        )
    )
    started = datetime(2026, 8, 13, 0, 0, tzinfo=UTC)
    finished = datetime(2026, 8, 13, 1, 0, tzinfo=UTC)
    run = RunRecord(
        protocol_version="duraseed-prepilot-v5",
        git_commit="abc123",
        resolved_config_hash="resolved",
        run_kind="engineering_smoke",
        method=None,
        seed=5,
        model_id="Qwen/Qwen3.5-9B-Base",
        renderer="role_colon",
        lora_rank=32,
        task_manifest_ids={"smoke": "sha256:" + "0" * 64},
        final_sampler_checkpoint_path="tinker://stage-b",
        status=RunStatus.COMPLETED,
        started_at=started,
        updated_at=finished,
        finished_at=finished,
        cost_usd=1.25,
        project_id=project_id,
        authorized_cost_usd=25,
        reserved_cost_usd=25,
    )
    (path.parent / "run.json").write_text(run.model_dump_json())
    ttl = [
        {
            "path": checkpoint,
            "training_run_id": f"training-{index}",
            "expires_at": "2026-08-20T01:00:00Z",
            "checkpoint_type": "weights",
            "ttl_seconds": 604800,
        }
        for index, checkpoint in enumerate(
            ("tinker://stage-a", "tinker://roundtrip", "tinker://stage-b")
        )
    ]
    (path.parent / "checkpoint_ttl_audit.json").write_text(json.dumps(ttl))
    return path


def _write_billing(root: Path, smoke: Path, *, project_id: str = "project") -> Path:
    raw = root / "raw-usage.csv"
    raw.write_text("session_id,cost\nsmoke-id,1.25\n")
    path = root / "billing.json"
    path.write_text(
        json.dumps(
            {
                "status": "reconciled",
                "source_run_id": "smoke-id",
                "source_acceptance_sha256": "sha256:"
                + hashlib.sha256(smoke.read_bytes()).hexdigest(),
                "project_id": project_id,
                "raw_usage_cutoff_utc": "2026-08-13T02:00:00Z",
                "raw_usage_path": raw.name,
                "raw_usage_sha256": "sha256:"
                + hashlib.sha256(raw.read_bytes()).hexdigest(),
                "remaining_balance_usd": "4955.66",
                "protected_reserve_usd": "991.132",
                "boundary_authorization_usd": "120",
                "remaining_balance_verified": True,
                "protected_reserve_survives": True,
            }
        )
    )
    return path


def test_boundary_authorization_derives_gates_from_artifacts(tmp_path: Path) -> None:
    smoke = _write_smoke(tmp_path)
    billing = _write_billing(tmp_path, smoke)
    authorization, digest = authorize_boundary(
        authorized_cost_usd="120",
        smoke_acceptance=smoke,
        billing_reconciliation=billing,
        human_approval=True,
        project_id="project",
    )
    assert authorization.plan_name == "boundary-extension"
    assert authorization.authorized_cost_usd == 120
    assert digest.startswith("sha256:") and len(digest) == 71


def test_boundary_authorization_rejects_mock_smoke(tmp_path: Path) -> None:
    with pytest.raises(RunnerGateError, match="acceptance gates"):
        authenticate_live_smoke(
            _write_smoke(tmp_path, real=False), project_id="project"
        )


def test_boundary_authorization_rejects_unreconciled_billing(tmp_path: Path) -> None:
    smoke = _write_smoke(tmp_path)
    billing = _write_billing(tmp_path, smoke)
    value = json.loads(billing.read_text())
    value["remaining_balance_verified"] = False
    billing.write_text(json.dumps(value))
    with pytest.raises(RunnerGateError, match="billing reconciliation"):
        authorize_boundary(
            authorized_cost_usd="120",
            smoke_acceptance=smoke,
            billing_reconciliation=billing,
            human_approval=True,
            project_id="project",
        )


def test_boundary_authorization_rejects_missing_ttl_and_stale_usage(
    tmp_path: Path,
) -> None:
    smoke = _write_smoke(tmp_path)
    (smoke.parent / "checkpoint_ttl_audit.json").unlink()
    with pytest.raises(RunnerGateError, match="TTL audit"):
        authenticate_live_smoke(smoke, project_id="project")

    smoke = _write_smoke(tmp_path / "second")
    billing = _write_billing(tmp_path / "second", smoke)
    value = json.loads(billing.read_text())
    value["raw_usage_cutoff_utc"] = "2026-08-13T00:30:00Z"
    billing.write_text(json.dumps(value))
    with pytest.raises(RunnerGateError, match="billing reconciliation"):
        authorize_boundary(
            authorized_cost_usd="120",
            smoke_acceptance=smoke,
            billing_reconciliation=billing,
            human_approval=True,
            project_id="project",
        )


def test_frozen_extension1_confirmation_matches_equivalence_anchor() -> None:
    config = load_pilot_config("duraseed_pilot_config.yaml")
    source = load_boundary_live_source(
        config, "frozen/v0/runs/tinker-calibration/boundary"
    )
    manifest = load_frozen_extension1_confirmation(
        "frozen/v0/derived/boundary/extension1_confirmation_manifest.json",
        source.extension1_broad_manifest,
    )
    assert manifest.record_count == 136
