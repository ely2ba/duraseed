"""One explicit resume record for an interrupted Pilot evaluation call."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes
from duraseed.run_records import append_jsonl
from duraseed.runners import RunnerGateError
from duraseed.runtime import PRICE_SNAPSHOT, TokenBudget, UsageQuantities


SCHEMA_VERSION = "duraseed-pilot0-infrastructure-recovery-v1"
_REQUEST_ID = re.compile(r"self\.request_id='([^']+)'")
_STUCK_REQUEST = re.compile(
    r"APIConnectionError: No progress made in \d+s\. Requests appear to be stuck\."
)


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"Pilot recovery {label} is unreadable") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"Pilot recovery {label} is not an object")
    return value


def _tokens(value: Any, label: str) -> TokenBudget:
    try:
        return TokenBudget(
            int(value["prefill"]), int(value["sample"]), int(value["train"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError(f"Pilot recovery {label} tokens are invalid") from error


def _token_value(value: TokenBudget) -> dict[str, int]:
    return {"prefill": value.prefill, "sample": value.sample, "train": value.train}


def _minus(left: TokenBudget, right: TokenBudget) -> TokenBudget:
    values = (
        left.prefill - right.prefill,
        left.sample - right.sample,
        left.train - right.train,
    )
    if any(value < 0 for value in values):
        raise RunnerGateError("Pilot recovery reservation exceeds persisted usage")
    return TokenBudget(*values)


def _fixed_usd(cost: float, tokens: TokenBudget) -> float:
    token_cost = PRICE_SNAPSHOT.cost(
        UsageQuantities(
            prefill_tokens=tokens.prefill,
            sample_tokens=tokens.sample,
            train_tokens=tokens.train,
        )
    )
    fixed = float(cost) - token_cost
    if fixed < -1e-9:
        raise RunnerGateError("Pilot recovery fixed spend is negative")
    return max(0.0, fixed)


def _checkpoint(events: Path) -> tuple[str, str]:
    try:
        rows = [json.loads(line) for line in events.read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("Pilot recovery segment journal is unreadable") from error
    saves = [
        row
        for row in rows
        if row.get("status") == "completed"
        and row.get("operation") == "pilot0-save-checkpoint-pair"
    ]
    if len(saves) != 1:
        raise RunnerGateError("Pilot recovery requires one saved checkpoint pair")
    sampler = saves[0].get("sampler_path")
    state = saves[0].get("state_path")
    if not isinstance(sampler, str) or not isinstance(state, str):
        raise RunnerGateError("Pilot recovery checkpoint paths are missing")
    return sampler, state


def prepare_pilot0_recovery(
    root: Path,
    *,
    recovery_session_id: str,
    recovery_git_commit: str,
) -> dict[str, Any]:
    """Bind and clear one known failed call so its seeded coordinate can resume."""

    run = _object(root / "run.json", "run record")
    preflight = _object(root / "preflight.json", "preflight")
    error = run.get("error")
    match = _REQUEST_ID.search(error) if isinstance(error, str) else None
    stuck_request = _STUCK_REQUEST.fullmatch(error) if isinstance(error, str) else None
    if (
        run.get("status") != "interrupted"
        or run.get("run_id") != root.name
        or preflight.get("run_id") != root.name
        or not recovery_session_id.strip()
        or not recovery_git_commit.strip()
        or (match is None and stuck_request is None)
    ):
        raise RunnerGateError("Pilot recovery is not the interrupted run incident")

    candidate_paths = (
        *root.glob("seed-*/B-*/steps-*/a-monitor/remote-call-state.json"),
        *root.glob("seed-*/B-*/stage-b/steps-*/*/remote-call-state.json"),
    )
    candidates = [
        path
        for path in candidate_paths
        if _object(path, "call state").get("pending") is not None
    ]
    if len(candidates) != 1:
        raise RunnerGateError("Pilot recovery requires exactly one pending call")
    call_state_path = candidates[0]
    evaluation = call_state_path.parent
    segment = evaluation.parent
    call_state = _object(call_state_path, "call state")
    pending = call_state.get("pending")
    if not isinstance(pending, dict) or pending.get("operation") != (
        "pilot0-validation-group"
    ):
        raise RunnerGateError("Pilot recovery pending call is not evaluation sampling")
    reservation = pending.get("reservation")
    if not isinstance(reservation, dict) or float(reservation.get("fixed_usd", 0)):
        raise RunnerGateError("Pilot recovery pending reservation is invalid")
    failed_tokens = TokenBudget(
        int(reservation.get("prefill_tokens", -1)),
        int(reservation.get("sample_tokens", -1)),
        int(reservation.get("train_tokens", -1)),
    )
    sequence = pending.get("sequence")
    if type(sequence) is not int or sequence != call_state.get("completed_count"):
        raise RunnerGateError("Pilot recovery pending sequence is inconsistent")
    if (evaluation / "result.json").exists() or (segment / "segment.json").exists():
        raise RunnerGateError("Pilot recovery cannot alter completed evidence")
    generation_rows = (evaluation / "generations.jsonl").read_text().splitlines()
    reward_rows = (evaluation / "rewards.jsonl").read_text().splitlines()
    if (
        len(generation_rows) != len(reward_rows)
        or (sequence == 0 and generation_rows)
        or (sequence > 0 and len(generation_rows) % sequence)
    ):
        raise RunnerGateError("Pilot recovery durable evaluation prefix is incomplete")
    samples_per_completed_call = len(generation_rows) // sequence if sequence else None
    sampler_path, state_path = _checkpoint(segment / "remote-calls.jsonl")
    try:
        start_text, stop_text = segment.name.removeprefix("steps-").split("-")
        if segment.parent.name == "stage-b":
            phase = "stage_b"
            method = segment.parent.parent.name
        else:
            phase = "stage_a"
            method = segment.parent.name
        start, stop = int(start_text), int(stop_text)
    except (ValueError, AttributeError) as error:
        raise RunnerGateError(
            "Pilot recovery segment coordinates are invalid"
        ) from error

    ledger = run.get("ledger")
    if not isinstance(ledger, dict):
        raise RunnerGateError("Pilot recovery run ledger is missing")
    committed = _tokens(ledger.get("committed_tokens"), "committed")
    observed = _tokens(ledger.get("observed_tokens"), "observed")
    resume_committed = _minus(committed, failed_tokens)
    resume_observed = _minus(observed, failed_tokens)
    committed_fixed = _fixed_usd(float(ledger["committed_cost_usd"]), committed)
    observed_fixed = _fixed_usd(float(ledger["observed_cost_usd"]), observed)
    if abs(committed_fixed - observed_fixed) > 1e-9:
        raise RunnerGateError("Pilot recovery fixed spend is inconsistent")

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_resume",
        "run_id": root.name,
        "created_at": datetime.now(UTC).isoformat(),
        "reason": "tinker_infrastructure_request_failure",
        "failed_error": error,
        "failed_request_id": match.group(1) if match is not None else None,
        "failed_session_id": preflight["lineage"]["session_id"],
        "recovery_session_id": recovery_session_id,
        "recovery_git_commit": recovery_git_commit,
        "segment": segment.relative_to(root).as_posix(),
        "evaluation": evaluation.relative_to(root).as_posix(),
        "phase": phase,
        "method": method,
        "start": start,
        "stop": stop,
        "failed_sequence": sequence,
        "samples_per_completed_call": samples_per_completed_call,
        "failed_coordinate": pending.get("coordinate"),
        "failed_reservation": _token_value(failed_tokens),
        "sampler_path": sampler_path,
        "state_path": state_path,
        "resume_ledger": {
            "committed_tokens": _token_value(resume_committed),
            "observed_tokens": _token_value(resume_observed),
            "fixed_usd": committed_fixed,
        },
    }
    recovery_path = root / "infrastructure-recovery.json"
    if recovery_path.exists():
        existing = _object(recovery_path, "artifact")
        comparable = {
            key: value for key, value in existing.items() if key != "created_at"
        }
        expected = {
            key: value for key, value in artifact.items() if key != "created_at"
        }
        if comparable != expected:
            raise RunnerGateError("Pilot recovery artifact changed")
    else:
        atomic_write_bytes(recovery_path, canonical_json_bytes(artifact))
        append_jsonl(
            evaluation / "remote-calls.jsonl",
            {
                "sequence": sequence,
                "status": "failed_infrastructure",
                "operation": pending["operation"],
                "request_id": match.group(1) if match is not None else None,
                "coordinate": pending.get("coordinate"),
                "error": error,
            },
        )
        call_state["pending"] = None
        atomic_write_bytes(call_state_path, canonical_json_bytes(call_state))
    return artifact


def prepare_pilot0_resume(
    root: Path,
    *,
    recovery_session_id: str,
    recovery_git_commit: str,
) -> dict[str, Any]:
    """Prepare either the first failed-call recovery or a clean checkpoint resume."""

    from duraseed.pilot0_clean_resume import (
        clean_pause_candidate,
        prepare_clean_resume,
    )

    candidate = clean_pause_candidate(root)
    if candidate is not None:
        return prepare_clean_resume(
            root,
            recovery_session_id=recovery_session_id,
            recovery_git_commit=recovery_git_commit,
            candidate=candidate,
        )
    if (root / "infrastructure-recovery.json").exists():
        raise RunnerGateError("Pilot resume has no clean pause marker")
    return prepare_pilot0_recovery(
        root,
        recovery_session_id=recovery_session_id,
        recovery_git_commit=recovery_git_commit,
    )


def load_pilot0_recovery(root: Path) -> dict[str, Any] | None:
    path = root / "infrastructure-recovery.json"
    if not path.exists():
        return None
    value = _object(path, "artifact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "authorized_resume"
        or value.get("run_id") != root.name
        or not isinstance(value.get("recovery_session_id"), str)
    ):
        raise RunnerGateError("Pilot recovery artifact is invalid")
    return value


def restore_pilot0_recovery_ledger(inputs: Any, root: Path) -> bool:
    recovery = load_pilot0_recovery(root)
    if recovery is None:
        return False
    ledger = recovery.get("resume_ledger")
    if not isinstance(ledger, dict):
        raise RunnerGateError("Pilot recovery ledger is missing")
    committed = _tokens(ledger.get("committed_tokens"), "resume committed")
    observed = _tokens(ledger.get("observed_tokens"), "resume observed")
    if inputs.ledger.committed != TokenBudget(0, 0, 0):
        raise RunnerGateError("Pilot recovery ledger is not fresh")
    inputs.ledger.reserve_call(committed, fixed_usd=float(ledger["fixed_usd"]))
    inputs.ledger.settle_call(observed)
    return True


def pilot0_resume_ledgers(root: Path) -> list[dict[str, Any]]:
    from duraseed.pilot0_clean_resume import clean_resumes

    ledgers = []
    recovery = load_pilot0_recovery(root)
    if recovery is not None:
        ledgers.append(recovery["resume_ledger"])
    ledgers.extend(entry["resume_ledger"] for entry in clean_resumes(root))
    return ledgers


def recovery_segment(inputs: Any, output: Path) -> dict[str, Any] | None:
    from duraseed.pilot0_clean_resume import clean_resumes

    root = inputs.output_root / inputs.run_id
    relative = output.relative_to(root).as_posix()
    for entry in reversed(clean_resumes(root)):
        if relative == entry.get("segment") and not (output / "segment.json").exists():
            return entry
    recovery = load_pilot0_recovery(root)
    if recovery is None or relative != recovery.get("segment"):
        return None
    return recovery


def reconciled_evaluation(inputs: Any, output: Path) -> bool:
    from duraseed.pilot0_clean_resume import clean_resumes

    root = inputs.output_root / inputs.run_id
    relative = output.relative_to(root).as_posix()
    if any(relative == entry.get("evaluation") for entry in clean_resumes(root)):
        return True
    recovery = load_pilot0_recovery(root)
    return recovery is not None and relative == recovery.get("evaluation")


def pilot0_session_ids(root: Path, primary_session_id: str) -> list[str]:
    from duraseed.pilot0_clean_resume import clean_resumes

    recovery = load_pilot0_recovery(root)
    sessions = [primary_session_id]
    if recovery is not None and recovery["recovery_session_id"] not in sessions:
        sessions.append(recovery["recovery_session_id"])
    for entry in clean_resumes(root):
        session_id = entry.get("recovery_session_id")
        if isinstance(session_id, str) and session_id not in sessions:
            sessions.append(session_id)
    return sessions


__all__ = [
    "load_pilot0_recovery",
    "pilot0_session_ids",
    "pilot0_resume_ledgers",
    "prepare_pilot0_recovery",
    "prepare_pilot0_resume",
    "reconciled_evaluation",
    "recovery_segment",
    "restore_pilot0_recovery_ledger",
]
