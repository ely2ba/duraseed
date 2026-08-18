"""Durable terminal state for a complete negative direct-M0 Stage-A result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from duraseed.calibration_attempts import hydrate_attempt_ledger
from duraseed.calibration_integrity import seal_calibration_action
from duraseed.calibration_provenance import (
    finish_calibration_run,
    validate_action_ttl_audit,
)
from duraseed.calibration_state import SCHEMA_VERSION
from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import RunStatus, read_run_record
from duraseed.runners import RunnerGateError
from duraseed.runtime import ZERO_TOKENS
from duraseed.training.stage_a_calibration import (
    StageADurationDecisionStatus,
    StageALearningRateDecisionStatus,
)
from duraseed.training.stage_a_update_health import (
    StageAUpdateHealthFailureEvidence,
    parse_stage_a_update_health_failure,
)


TerminalStatus = Literal[
    "no_eligible_learning_rate",
    "duration_gate_failed",
    "common_rl_gate_failed",
    "update_health_failed",
]
TERMINAL_FILE = "stage-a-terminal.json"
TERMINAL_SCHEMA = "duraseed-stage-a-terminal-v1"
_STATUSES = {
    "no_eligible_learning_rate",
    "duration_gate_failed",
    "common_rl_gate_failed",
    "update_health_failed",
}


class StageAScientificFailure(Exception):
    """A complete negative gate result, never an ambiguous remote failure."""

    def __init__(
        self,
        status: TerminalStatus,
        evidence: Any,
        decisions: tuple[Any, ...],
        duration: Any | None,
    ) -> None:
        super().__init__(status)
        self.status = status
        self.evidence = evidence
        self.decisions = decisions
        self.duration = duration

    @property
    def screen_only(self) -> bool:
        return self.status == "no_eligible_learning_rate"

    @property
    def update_health_failure(self) -> StageAUpdateHealthFailureEvidence | None:
        return (
            parse_stage_a_update_health_failure(self.evidence)
            if self.status == "update_health_failed"
            else None
        )


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("invalid Stage-A terminal artifact") from error
    if not isinstance(value, dict):
        raise RunnerGateError("Stage-A terminal is not an object")
    return value


def _validate_failure(failure: StageAScientificFailure) -> None:
    if failure.status == "update_health_failed":
        evidence = parse_stage_a_update_health_failure(failure.evidence)
        if (
            not isinstance(evidence, StageAUpdateHealthFailureEvidence)
            or failure.decisions
            or failure.duration is not None
        ):
            raise ValueError("invalid Stage-A update-health terminal")
        return
    methods = tuple(row.method for row in failure.decisions)
    selected = all(
        row.status is StageALearningRateDecisionStatus.SELECTED
        for row in failure.decisions
    )
    finals = len(failure.evidence.final_evidence)
    duration_frozen = (
        failure.duration is not None
        and failure.duration.status is StageADurationDecisionStatus.FROZEN
    )
    valid = {
        "no_eligible_learning_rate": not selected and finals == 0,
        "duration_gate_failed": selected and finals == 2 and not duration_frozen,
        "common_rl_gate_failed": selected and finals == 2 and duration_frozen,
    }
    if (
        len(methods) != 2
        or set(methods) != {"B-S", "B-G"}
        or failure.status not in _STATUSES
        or valid.get(failure.status) is not True
    ):
        raise ValueError("only a complete negative Stage-A gate is terminal")


def _completed_evidence(root: Path) -> Any:
    completed = _object(root / "stage-a-arms/complete-bounded-stage-a/completed.json")
    payload = completed.get("evidence")
    if not isinstance(payload, list) or len(payload) != 2 or payload[1] is not None:
        raise RunnerGateError("Stage-A terminal lacks its completed negative attempt")
    return payload[0]


def existing_stage_a_terminal(
    root: Path, preflight_sha256: str, inputs: Any | None = None
) -> dict[str, Any] | None:
    """Revalidate an exact negative terminal without constructing a service."""

    path = root / TERMINAL_FILE
    if not path.exists():
        return None
    terminal = _object(path)
    state = _object(root / "state.json")
    run = read_run_record(root)
    status = terminal.get("status")
    screen_only = status == "no_eligible_learning_rate"
    update_health = (
        parse_stage_a_update_health_failure(terminal.get("evidence"))
        if status == "update_health_failed"
        else None
    )
    integrity = seal_calibration_action(
        "stage-a",
        root / "stage-a-arms",
        teacher_updates=0,
        stage_a_screen_only=screen_only,
        stage_a_update_health_failure=update_health,
    )
    ttl = validate_action_ttl_audit(
        "stage-a",
        root / "stage-a-arms",
        inputs,
        stage_a_screen_only=screen_only,
        stage_a_update_health_failure=update_health,
    )
    digest = sha256_bytes(path.read_bytes())
    if (
        terminal.get("schema_version") != TERMINAL_SCHEMA
        or status not in _STATUSES
        or terminal.get("action") != "stage-a"
        or terminal.get("preflight_sha256") != preflight_sha256
        or terminal.get("run_id") != root.name
        or terminal.get("evidence") != _completed_evidence(root)
        or terminal.get("integrity") != integrity
        or terminal.get("checkpoint_ttl_audit_sha256")
        != sha256_bytes((root / "stage-a-arms/checkpoint-ttl-audit.json").read_bytes())
        or not ttl.get("rows")
        or state.get("status") != "failed"
        or state.get("completed_actions") != []
        or state.get("artifact_sha256") != {}
        or state.get("preflight_sha256") != preflight_sha256
        or state.get("terminal_decision", {}).get("artifact_sha256") != digest
        or run.status is not RunStatus.FAILED
    ):
        raise RunnerGateError("Stage-A terminal lineage differs")
    if (
        inputs is not None
        and not (root / "billing-reconciliation-required.json").exists()
    ):
        ledger = inputs.stage_a_ledger
        if (
            ledger.committed == ZERO_TOKENS
            and ledger.observed == ZERO_TOKENS
            and not ledger.committed_fixed_usd
            and not ledger.observed_fixed_usd
        ):
            hydrate_attempt_ledger(root / "stage-a-arms", ledger)
        finish_calibration_run(
            inputs, root, RunStatus.FAILED, error=f"Stage-A {status}"
        )
    return {"state": state, "terminal": terminal}


def finish_stage_a_terminal(
    inputs: Any,
    root: Path,
    *,
    preflight_sha256: str,
    failure: StageAScientificFailure,
    integrity: dict[str, Any],
    ttl_audit_sha256: str,
) -> dict[str, Any]:
    """Commit one complete negative decision and its exact raw-evidence seal."""

    _validate_failure(failure)
    terminal = {
        "schema_version": TERMINAL_SCHEMA,
        "action": "stage-a",
        "status": failure.status,
        "run_id": inputs.run_id,
        "preflight_sha256": preflight_sha256,
        "learning_rate_decisions": failure.decisions,
        "duration_decision": failure.duration,
        "evidence": failure.evidence,
        "integrity": integrity,
        "checkpoint_ttl_audit_sha256": ttl_audit_sha256,
        "usage": {
            "committed_tokens": inputs.stage_a_ledger.committed,
            "observed_tokens": inputs.stage_a_ledger.observed,
            "committed_cost_usd": inputs.stage_a_ledger.committed_cost_usd,
            "observed_cost_usd": inputs.stage_a_ledger.observed_cost_usd,
        },
    }
    payload = canonical_json_bytes(terminal)
    if canonical_json_bytes(terminal["evidence"]) != canonical_json_bytes(
        _completed_evidence(root)
    ):
        raise RunnerGateError("Stage-A terminal evidence differs from its attempt")
    path = root / TERMINAL_FILE
    if path.exists() and path.read_bytes() != payload:
        raise RunnerGateError("Stage-A terminal artifact changed")
    atomic_write_bytes(path, payload)
    terminal_hash = sha256_bytes(payload)
    error = f"Stage-A {failure.status}"
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
                    "action": "stage-a",
                    "status": failure.status,
                    "artifact_sha256": terminal_hash,
                },
                "error": error,
            }
        ),
    )
    finish_calibration_run(inputs, root, RunStatus.FAILED, error=error)
    return {"state": _object(root / "state.json"), "terminal": terminal}


__all__ = [
    "existing_stage_a_terminal",
    "finish_stage_a_terminal",
    "StageAScientificFailure",
]
