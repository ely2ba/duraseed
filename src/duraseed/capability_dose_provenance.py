"""Terminal, run record, and billing handoff for the capability dose."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from duraseed.capability_dose_integrity import validate_capability_dose_attempt
from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import RunStatus, read_run_record, write_run_record
from duraseed.runners import RunnerGateError
from duraseed.training.capability_dose_evidence import CapabilityDoseLiveEvidence


TERMINAL_FILE = "capability-dose-terminal.json"
_EVIDENCE = TypeAdapter(CapabilityDoseLiveEvidence)


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(
            f"invalid capability-dose artifact: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"capability-dose artifact is not an object: {path.name}")
    return value


def _write_exact(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise RunnerGateError(f"capability-dose artifact changed: {path.name}")
    atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def _completed(
    root: Path, preflight_sha256: str
) -> tuple[CapabilityDoseLiveEvidence, Path]:
    arm = root / "capability-dose-arms/b-s-capability-dose"
    completed = _object(arm / "completed.json")
    attempt_number = completed.get("attempt")
    if (
        completed.get("arm_id") != "b-s-capability-dose"
        or completed.get("preflight_sha256") != preflight_sha256
        or type(attempt_number) is not int
        or attempt_number < 1
    ):
        raise RunnerGateError("capability-dose completion omits its attempt")
    evidence = _EVIDENCE.validate_json(canonical_json_bytes(completed.get("evidence")))
    return evidence, arm / f"attempt-{attempt_number:04d}"


def existing_capability_dose_terminal(
    root: Path, preflight_sha256: str
) -> dict[str, Any] | None:
    """Revalidate a completed terminal locally, before any service is created."""

    path = root / TERMINAL_FILE
    if not path.exists():
        return None
    terminal = _object(path)
    evidence, attempt = _completed(root, preflight_sha256)
    integrity = validate_capability_dose_attempt(attempt, evidence)
    state = _object(root / "state.json")
    run = read_run_record(root)
    expected_status = (
        RunStatus.COMPLETED
        if evidence.decision.action == "proceed_to_pilot"
        else RunStatus.FAILED
    )
    if (
        terminal.get("schema_version") != "duraseed-capability-dose-terminal-v1"
        or terminal.get("run_id") != root.name
        or terminal.get("preflight_sha256") != preflight_sha256
        or canonical_json_bytes(terminal.get("decision"))
        != canonical_json_bytes(evidence.decision)
        or terminal.get("status") != evidence.decision.action
    ):
        raise RunnerGateError("capability-dose terminal identity differs")
    if (
        canonical_json_bytes(terminal.get("evidence")) != canonical_json_bytes(evidence)
        or terminal.get("integrity") != integrity
        or state.get("status") != expected_status.value
        or state.get("terminal_sha256") != sha256_bytes(path.read_bytes())
        or run.status is not expected_status
        or not (root / "billing-reconciliation-required.json").exists()
    ):
        raise RunnerGateError("capability-dose terminal lineage differs")
    return terminal


def finish_capability_dose(
    inputs: Any,
    root: Path,
    *,
    preflight_sha256: str,
) -> dict[str, Any]:
    evidence, attempt = _completed(root, preflight_sha256)
    integrity = validate_capability_dose_attempt(attempt, evidence)
    terminal = {
        "schema_version": "duraseed-capability-dose-terminal-v1",
        "run_id": inputs.run_id,
        "preflight_sha256": preflight_sha256,
        "status": evidence.decision.action,
        "decision": evidence.decision,
        "evidence": evidence,
        "integrity": integrity,
        "usage": {
            "committed_tokens": inputs.stage_a_ledger.committed,
            "observed_tokens": inputs.stage_a_ledger.observed,
            "committed_cost_usd": inputs.stage_a_ledger.committed_cost_usd,
            "observed_cost_usd": inputs.stage_a_ledger.observed_cost_usd,
        },
    }
    terminal_sha256 = _write_exact(root / TERMINAL_FILE, terminal)
    run_status = (
        RunStatus.COMPLETED
        if evidence.decision.action == "proceed_to_pilot"
        else RunStatus.FAILED
    )
    _write_exact(
        root / "state.json",
        {
            "schema_version": "duraseed-capability-dose-state-v1",
            "status": run_status.value,
            "preflight_sha256": preflight_sha256,
            "terminal_sha256": terminal_sha256,
            "decision": evidence.decision.action,
        },
    )
    ttl = _object(attempt / "checkpoint-ttl-audit.json")["rows"]
    _write_exact(
        root / "checkpoint-lineage.json",
        {
            "schema_version": "duraseed-capability-dose-checkpoint-lineage-v1",
            "parent_m0_sampler_path": inputs.m0_sampler_path,
            "parent_m0_state_path": inputs.m0_state_path,
            "retained_checkpoints": ttl,
            "weights_only_restore_validated": True,
        },
    )
    finished = datetime.now(UTC)
    run = read_run_record(root)
    training_ids = tuple(sorted({row["training_run_id"] for row in ttl}))
    write_run_record(
        root,
        run.model_copy(
            update={
                "status": run_status,
                "updated_at": finished,
                "finished_at": finished,
                "prompt_tokens": inputs.stage_a_ledger.observed.prefill,
                "sampled_tokens": inputs.stage_a_ledger.observed.sample,
                "train_tokens": inputs.stage_a_ledger.observed.train,
                "cost_usd": inputs.stage_a_ledger.observed_cost_usd,
                "deviations": ["billing reconciliation required after completion"],
                "tinker_training_run_ids": training_ids,
                "final_sampler_checkpoint_path": evidence.retained_sampler_checkpoint_path,
                "final_state_checkpoint_path": evidence.retained_state_checkpoint_path,
            }
        ),
    )
    sessions = _object(root / "session-lineage.json")["session_ids"]
    _write_exact(
        root / "billing-reconciliation-required.json",
        {
            "schema_version": "duraseed-capability-dose-billing-required-v1",
            "status": "pending",
            "run_id": inputs.run_id,
            "project_id": inputs.project_id,
            "run_status": run_status.value,
            "terminal_sha256": terminal_sha256,
            "session_ids": sessions,
            "execution_finished_at_utc": finished.isoformat(),
            "action_cap_usd": inputs.stage_a_ledger.authorized_usd,
            "local_observed_cost_usd": inputs.stage_a_ledger.observed_cost_usd,
            "prelaunch_actual_lifetime_spend_usd": (
                inputs.actual_lifetime_billing.actual_lifetime_spend_usd
            ),
            "lifetime_calibration_cap_usd": 300,
            "prelaunch_billing_lineage": inputs.actual_lifetime_billing.lineage,
        },
    )
    return terminal


def interrupt_capability_dose(inputs: Any, root: Path, error: Exception) -> None:
    if not (root / "run.json").exists():
        return
    run = read_run_record(root)
    finished = datetime.now(UTC)
    write_run_record(
        root,
        run.model_copy(
            update={
                "status": RunStatus.INTERRUPTED,
                "updated_at": finished,
                "finished_at": finished,
                "prompt_tokens": inputs.stage_a_ledger.observed.prefill,
                "sampled_tokens": inputs.stage_a_ledger.observed.sample,
                "train_tokens": inputs.stage_a_ledger.observed.train,
                "cost_usd": inputs.stage_a_ledger.observed_cost_usd,
                "deviations": [f"interrupted: {type(error).__name__}: {error}"],
            }
        ),
    )


__all__ = [
    "existing_capability_dose_terminal",
    "finish_capability_dose",
    "interrupt_capability_dose",
]
