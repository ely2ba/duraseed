"""Idempotent terminal state for a completed no-stable-exposure result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duraseed.calibration_provenance import finish_calibration_run
from duraseed.calibration_state import SCHEMA_VERSION
from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import RunStatus, read_run_record
from duraseed.runners import RunnerGateError


TERMINAL_FILE = "teacher-dose-terminal.json"
TERMINAL_SCHEMA = "duraseed-teacher-exposure-terminal-v2"


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("invalid teacher-exposure terminal artifact") from error
    if not isinstance(value, dict):
        raise RunnerGateError("teacher-exposure terminal is not an object")
    return value


def existing_teacher_exposure_terminal(
    root: Path, preflight_sha256: str, inputs: Any | None = None
) -> dict[str, Any] | None:
    path = root / TERMINAL_FILE
    if not path.exists():
        return None
    terminal = _object(path)
    state = _object(root / "state.json")
    run = read_run_record(root)
    digest = sha256_bytes(path.read_bytes())
    integrity_path = root / "teacher-dose-arms/integrity.json"
    ttl_path = root / "teacher-dose-arms/checkpoint-ttl-audit.json"
    if (
        terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("status") != "no_stable_checkpoint"
        or terminal.get("preflight_sha256") != preflight_sha256
        or terminal.get("run_id") != root.name
        or state.get("status") != "failed"
        or state.get("completed_actions") != []
        or state.get("artifact_sha256") != {}
        or state.get("preflight_sha256") != preflight_sha256
        or state.get("terminal_decision", {}).get("artifact_sha256") != digest
        or terminal.get("integrity", {}).get("artifact_sha256")
        != sha256_bytes(integrity_path.read_bytes())
        or terminal.get("checkpoint_ttl_audit_sha256")
        != sha256_bytes(ttl_path.read_bytes())
        or run.status is not RunStatus.FAILED
    ):
        raise RunnerGateError("teacher-exposure terminal lineage differs")
    if (
        inputs is not None
        and not (root / "billing-reconciliation-required.json").exists()
    ):
        finish_calibration_run(
            inputs,
            root,
            RunStatus.FAILED,
            error="teacher exposure found no stable checkpoint through update 12",
        )
    return {"state": state, "terminal": terminal}


def finish_teacher_exposure_terminal(
    inputs: Any,
    root: Path,
    *,
    preflight_sha256: str,
    evidence: Any,
    selection: Any,
    integrity: dict[str, Any],
    ttl_audit_sha256: str,
) -> dict[str, Any]:
    if selection.status != "no_stable_checkpoint" or selection.recipe is not None:
        raise ValueError("only a completed no-stable result is terminal")
    terminal = {
        "schema_version": TERMINAL_SCHEMA,
        "action": "teacher-dose",
        "status": selection.status,
        "run_id": inputs.run_id,
        "preflight_sha256": preflight_sha256,
        "decision": selection,
        "evidence": evidence,
        "integrity": integrity,
        "checkpoint_ttl_audit_sha256": ttl_audit_sha256,
        "usage": {
            "committed_tokens": inputs.teacher_ledger.committed,
            "observed_tokens": inputs.teacher_ledger.observed,
            "committed_cost_usd": inputs.teacher_ledger.committed_cost_usd,
            "observed_cost_usd": inputs.teacher_ledger.observed_cost_usd,
        },
    }
    payload = canonical_json_bytes(terminal)
    path = root / TERMINAL_FILE
    if path.exists() and path.read_bytes() != payload:
        raise RunnerGateError("teacher-exposure terminal artifact changed")
    atomic_write_bytes(path, payload)
    terminal_hash = sha256_bytes(payload)
    error = "teacher exposure found no stable checkpoint through update 12"
    atomic_write_bytes(
        root / "state.json",
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "completed_actions": [],
                "artifact_sha256": {},
                "preflight_sha256": preflight_sha256,
                "terminal_decision": {
                    "action": "teacher-dose",
                    "status": selection.status,
                    "artifact_sha256": terminal_hash,
                },
                "error": error,
            }
        ),
    )
    finish_calibration_run(inputs, root, RunStatus.FAILED, error=error)
    return {"state": _object(root / "state.json"), "terminal": terminal}


__all__ = [
    "existing_teacher_exposure_terminal",
    "finish_teacher_exposure_terminal",
]
