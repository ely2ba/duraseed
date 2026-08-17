"""Read-only authentication for a completed calibration launch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from duraseed.calibration_attempts import hydrate_attempt_ledger
from duraseed.calibration_integrity import validate_committed_action
from duraseed.calibration_preflight import calibration_preflight
from duraseed.calibration_provenance import (
    read_calibration_session_ids,
    validate_action_ttl_audit,
)
from duraseed.calibration_state import SCHEMA_VERSION, existing, read
from duraseed.provenance import sha256_bytes
from duraseed.run_records import RunStatus, read_run_record
from duraseed.runners import RunnerGateError


def _validate_run(inputs: Any, root: Path) -> None:
    run = read_run_record(root)
    sessions = read_calibration_session_ids(root)
    manifests = {
        "a_monitor": inputs.prompt_pools.a_monitor_manifest.manifest_id,
        "a_rl_train": inputs.prompt_pools.a_rl_train_manifest.manifest_id,
        "a_seed_gate": inputs.teacher_sources.gate_manifest.manifest_id,
        "a_seed_train": inputs.teacher_sources.target_train_manifest.manifest_id,
    }
    if (
        run.status is not RunStatus.COMPLETED
        or run.git_commit != inputs.git_commit
        or run.resolved_config_hash != inputs.config.resolved_config_hash()
        or run.model_id != inputs.config.tinker.model_id
        or run.renderer != inputs.config.tinker.renderer_name
        or run.lora_rank != inputs.config.tinker.lora_rank
        or run.project_id != inputs.project_id
        or run.parent_tinker_checkpoint_path != inputs.m0_state_path
        or run.task_manifest_ids != manifests
        or not sessions
        or run.tinker_session_id != sessions[0]
        or not run.tinker_training_run_ids
        or run.authorized_cost_usd
        != inputs.teacher_ledger.authorized_usd + inputs.stage_a_ledger.authorized_usd
        or run.reserved_cost_usd != run.authorized_cost_usd
        or run.cost_usd > run.authorized_cost_usd
    ):
        raise RunnerGateError("completed calibration RunRecord identity differs")
    required = read(root / "billing-reconciliation-required.json")
    if (
        required.get("status") != "pending"
        or required.get("run_id") != inputs.run_id
        or required.get("project_id") != inputs.project_id
        or required.get("session_ids") != list(sessions)
        or required.get("aggregate_cap_usd") != run.authorized_cost_usd
        or required.get("parent_run_id") != inputs.parent_teacher_evidence.parent_run_id
        or required.get("parent_billing_sha256")
        != inputs.parent_teacher_evidence.parent_billing_sha256
        or required.get("parent_billed_usd")
        != inputs.parent_teacher_evidence.parent_billed_usd
        or required.get("lifetime_calibration_cap_usd") != 300
        or required.get("action_caps_usd")
        != {
            "teacher-dose": inputs.teacher_ledger.authorized_usd,
            "stage-a": inputs.stage_a_ledger.authorized_usd,
        }
        or float(required.get("local_observed_cost_usd", -1)) < run.cost_usd
    ):
        raise RunnerGateError("completed calibration billing handoff differs")


def completed_calibration(inputs: Any, root: Path) -> bool:
    """Return true only after fully revalidating an existing completed run."""

    if not (root / "state.json").exists():
        return False
    expected = calibration_preflight(inputs, SCHEMA_VERSION)
    preflight_path = root / "preflight.json"
    if not preflight_path.exists() or read(preflight_path) != expected:
        raise RunnerGateError("completed calibration preflight changed")
    preflight_sha256 = sha256_bytes(preflight_path.read_bytes())
    state, artifacts = existing(root, preflight_sha256)
    if state["status"] != "completed":
        return False
    teacher_updates = int(artifacts["teacher-dose"]["recipe"]["selected_updates"])
    for action, directory, ledger in (
        ("teacher-dose", root / "teacher-dose-arms", inputs.teacher_ledger),
        ("stage-a", root / "stage-a-arms", inputs.stage_a_ledger),
    ):
        validate_action_ttl_audit(action, directory, inputs)
        validate_committed_action(
            action,
            directory,
            artifacts[action],
            teacher_updates=teacher_updates,
        )
        hydrate_attempt_ledger(directory, ledger)
    _validate_run(inputs, root)
    return True


__all__ = ["completed_calibration"]
