"""Clean checkpoint-boundary resume records for Pilot 0."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.pilot0_recovery import (
    _checkpoint,
    _fixed_usd,
    _object,
    _token_value,
    _tokens,
)
from duraseed.provenance import canonical_json_bytes
from duraseed.runners import RunnerGateError


SCHEMA_VERSION = "duraseed-pilot0-clean-resume-lineage-v1"
FILE_NAME = "clean-resume-lineage.json"
LOCAL_PAUSE_OPERATION = "pilot0-local-pause"


def clean_resumes(root: Path) -> list[dict[str, Any]]:
    path = root / FILE_NAME
    if not path.exists():
        return []
    value = _object(path, "clean resume lineage")
    entries = value.get("entries")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("run_id") != root.name
        or not isinstance(entries, list)
        or any(not isinstance(entry, dict) for entry in entries)
    ):
        raise RunnerGateError("Pilot clean resume lineage is invalid")
    return entries


def _write(root: Path, entries: list[dict[str, Any]]) -> None:
    atomic_write_bytes(
        root / FILE_NAME,
        canonical_json_bytes(
            {"schema_version": SCHEMA_VERSION, "run_id": root.name, "entries": entries}
        ),
    )


def clean_pause_candidate(root: Path) -> tuple[Path, dict[str, Any]] | None:
    candidates = []
    for path in root.glob("seed-*/B-*/stage-b/steps-*/*/remote-call-state.json"):
        state = _object(path, "call state")
        pending = state.get("pending")
        if (
            isinstance(pending, dict)
            and pending.get("operation") == LOCAL_PAUSE_OPERATION
        ):
            candidates.append((path, state))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RunnerGateError("Pilot clean resume requires one local pause marker")
    return candidates[0]


def prepare_clean_resume(
    root: Path,
    *,
    recovery_session_id: str,
    recovery_git_commit: str,
    candidate: tuple[Path, dict[str, Any]],
) -> dict[str, Any]:
    from duraseed.pilot0_recovery import pilot0_session_ids

    run = _object(root / "run.json", "run record")
    preflight = _object(root / "preflight.json", "preflight")
    call_state_path, call_state = candidate
    pending = call_state.get("pending")
    reservation = pending.get("reservation") if isinstance(pending, dict) else None
    zero_reservation = {
        "prefill_tokens": 0,
        "sample_tokens": 0,
        "train_tokens": 0,
        "fixed_usd": 0.0,
    }
    if (
        run.get("status") != "interrupted"
        or run.get("run_id") != root.name
        or preflight.get("run_id") != root.name
        or not recovery_session_id.strip()
        or not recovery_git_commit.strip()
        or call_state.get("local_pause") is not True
        or call_state.get("completed_count") != 0
        or not isinstance(pending, dict)
        or pending.get("sequence") != 0
        or pending.get("operation") != LOCAL_PAUSE_OPERATION
        or reservation != zero_reservation
    ):
        raise RunnerGateError("Pilot clean resume marker is invalid")
    evaluation = call_state_path.parent
    segment = evaluation.parent
    if (
        (evaluation / "result.json").exists()
        or (evaluation / "generations.jsonl").exists()
        or (evaluation / "rewards.jsonl").exists()
        or (segment / "segment.json").exists()
    ):
        raise RunnerGateError("Pilot clean resume marker crossed durable evidence")
    sampler_path, state_path = _checkpoint(segment / "remote-calls.jsonl")
    try:
        start_text, stop_text = segment.name.removeprefix("steps-").split("-")
        start, stop = int(start_text), int(stop_text)
        method = segment.parent.parent.name
    except (ValueError, AttributeError) as error:
        raise RunnerGateError("Pilot clean resume coordinates are invalid") from error
    ledger = run.get("ledger")
    if not isinstance(ledger, dict):
        raise RunnerGateError("Pilot clean resume ledger is missing")
    committed = _tokens(ledger.get("committed_tokens"), "clean resume committed")
    observed = _tokens(ledger.get("observed_tokens"), "clean resume observed")
    committed_fixed = _fixed_usd(float(ledger["committed_cost_usd"]), committed)
    observed_fixed = _fixed_usd(float(ledger["observed_cost_usd"]), observed)
    if abs(committed_fixed - observed_fixed) > 1e-9:
        raise RunnerGateError("Pilot clean resume fixed spend is inconsistent")
    primary_session_id = str(preflight["lineage"]["session_id"])
    entry = {
        "kind": "clean_checkpoint_pause",
        "status": "authorized_resume",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": root.name,
        "paused_session_id": pilot0_session_ids(root, primary_session_id)[-1],
        "recovery_session_id": recovery_session_id,
        "recovery_git_commit": recovery_git_commit,
        "segment": segment.relative_to(root).as_posix(),
        "evaluation": evaluation.relative_to(root).as_posix(),
        "phase": "stage_b",
        "method": method,
        "start": start,
        "stop": stop,
        "sampler_path": sampler_path,
        "state_path": state_path,
        "resume_ledger": {
            "committed_tokens": _token_value(committed),
            "observed_tokens": _token_value(observed),
            "fixed_usd": committed_fixed,
        },
    }
    entries = clean_resumes(root)
    entries.append(entry)
    _write(root, entries)
    call_state["pending"] = None
    call_state["local_pause"] = False
    atomic_write_bytes(call_state_path, canonical_json_bytes(call_state))
    return entry


__all__ = ["clean_pause_candidate", "clean_resumes", "prepare_clean_resume"]
