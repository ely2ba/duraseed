"""Authenticate the outcome-blind stopped M0 repair used by amendment M1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duraseed.provenance import sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.teacher_exposure_spec import (
    PRIOR_REPAIR_CHILD_AUTHORIZATION_USD,
    PRIOR_REPAIR_PREFLIGHT_SHA256,
    PRIOR_REPAIR_RUN_ID,
    PRIOR_REPAIR_SESSION_ID,
    PRIOR_REPAIR_TEACHER_CAP_USD,
)


_HASHES = {
    "run": "sha256:2ad2af930cb555e855f608d6e0d1067f494736b7b5a92b0e99e78d44e3af9be9",
    "state": "sha256:cbea11d2ec3a38d310876f2be262988fe7a99b00282bfa754eb4528eef08d632",
    "sessions": "sha256:56f108bb2a9ada40e4522b2888e3d741c98ee3d674c16937d90c1810347425d6",
    "journal_seed_17": "sha256:33d7a05e344dfc976b63926271342bcdf64621f97b50473e148faaca98be781d",
    "journal_seed_37": "sha256:f2d02329e46bad1cb16977772d52d0c7206b162ae22381fd4d47fcfad03d4027",
}
_M0_REPAIR_SPEC = {
    "seeds": [17, 37],
    "dose": 2,
    "learning_rate": 1e-4,
    "checkpoint_updates": [4, 8, 12],
    "selection_rule": "earliest_checkpoint_passing_all_orientations",
}


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid prior repair {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"prior repair {label} is not an object")
    return value, raw


def load_prior_repair(
    parent_root: Path, *, project_id: str, parent_lineage: dict[str, object]
) -> dict[str, object]:
    """Bind M1 to the exact interrupted M0 repair and its clear journals."""

    from duraseed.run_records import RunStatus, read_run_record

    root = parent_root.parent / PRIOR_REPAIR_RUN_ID
    preflight, preflight_raw = _object(root / "preflight.json", "preflight")
    state, state_raw = _object(root / "state.json", "state")
    sessions, sessions_raw = _object(root / "session-lineage.json", "sessions")
    journal_hashes = {}
    for seed in (17, 37):
        path = (
            root
            / "teacher-dose-arms"
            / f"trajectory-seed-{seed}"
            / "attempt-0001/remote-call-state.json"
        )
        journal, raw = _object(path, f"seed-{seed} journal")
        digest = sha256_bytes(raw)
        if (
            journal.get("pending") is not None
            or digest != _HASHES[f"journal_seed_{seed}"]
        ):
            raise RunnerGateError("prior repair journal is pending or changed")
        journal_hashes[str(seed)] = digest
    run = read_run_record(root)
    if (
        sha256_bytes(preflight_raw) != PRIOR_REPAIR_PREFLIGHT_SHA256
        or sha256_bytes(state_raw) != _HASHES["state"]
        or sha256_bytes((root / "run.json").read_bytes()) != _HASHES["run"]
        or sha256_bytes(sessions_raw) != _HASHES["sessions"]
        or preflight.get("run_id") != PRIOR_REPAIR_RUN_ID
        or preflight.get("project_id") != project_id
        or preflight.get("parent_calibration") != parent_lineage
        or preflight.get("cost_caps_usd")
        != {
            "teacher-dose": PRIOR_REPAIR_TEACHER_CAP_USD,
            "teacher-allocation": 0,
            "stage-a": 155.09,
            "total": PRIOR_REPAIR_CHILD_AUTHORIZATION_USD,
        }
        or preflight.get("teacher_exposure_repair") != _M0_REPAIR_SPEC
        or state.get("schema_version") != "duraseed-acquisition-calibration-v1"
        or state.get("status") != "interrupted"
        or state.get("completed_actions") != []
        or state.get("artifact_sha256") != {}
        or state.get("preflight_sha256") != PRIOR_REPAIR_PREFLIGHT_SHA256
        or "terminal_decision" in state
        or sessions.get("session_ids") != [PRIOR_REPAIR_SESSION_ID]
        or run.status is not RunStatus.INTERRUPTED
        or run.project_id != project_id
        or run.tinker_session_id != PRIOR_REPAIR_SESSION_ID
        or run.authorized_cost_usd != PRIOR_REPAIR_CHILD_AUTHORIZATION_USD
        or run.reserved_cost_usd != PRIOR_REPAIR_CHILD_AUTHORIZATION_USD
        or run.finished_at is None
        or (root / "teacher-dose-terminal.json").exists()
    ):
        raise RunnerGateError("prior repair interruption lineage differs")
    return {
        "run_id": PRIOR_REPAIR_RUN_ID,
        "status": "interrupted",
        "preflight_sha256": PRIOR_REPAIR_PREFLIGHT_SHA256,
        "run_sha256": _HASHES["run"],
        "state_sha256": _HASHES["state"],
        "session_lineage_sha256": _HASHES["sessions"],
        "session_id": PRIOR_REPAIR_SESSION_ID,
        "journal_state_sha256": journal_hashes,
        "pending_remote_calls": 0,
        "charged_teacher_cap_usd": PRIOR_REPAIR_TEACHER_CAP_USD,
    }


__all__ = ["load_prior_repair"]
