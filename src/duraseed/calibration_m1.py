"""Authenticate the safely interrupted M1 warm-start attempt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duraseed.provenance import canonical_json_value, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.teacher_exposure_spec import (
    M1_PREFLIGHT_SHA256,
    M1_RUN_ID,
    M1_SESSION_ID,
    M1_TEACHER_CAP_USD,
    REPAIR_AGGREGATE_CAP_USD,
    REPAIR_SPEC,
    REPAIR_STAGE_A_CAP_USD,
)


_HASHES = {
    "run": "sha256:056b61148f35d2f940534de7493e2ca9b8a5740b7de1069c567846915abe4760",
    "state": "sha256:ae2e152ca1ea3d2b12b5990842002afa8d56bda2625bfb9a9331128844dd956b",
    "sessions": "sha256:7c46d30d72c3e8b83d622ded2b186b679b160a01a1cc46cd5d958b799130569b",
    "journal_seed_17": "sha256:7bbaac75c68df5ff4cd3e28226cb67721392ba819c618abe36f7c8b12d058c00",
    "journal_seed_37": "sha256:570eb1845f49a791e9103d5917cd21a4b8cbfa192a10addb8fa037c5666d44d5",
}


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid M1 interruption {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"M1 interruption {label} is not an object")
    return value, raw


def load_m1_interruption(
    parent_root: Path,
    *,
    project_id: str,
    parent_lineage: dict[str, object],
    prior_repair_lineage: dict[str, object],
) -> dict[str, object]:
    """Bind a new launch to M1's exact stopped state and full action cap."""

    from duraseed.run_records import RunStatus, read_run_record

    root = parent_root.parent / M1_RUN_ID
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
            raise RunnerGateError("M1 interruption journal is pending or changed")
        journal_hashes[str(seed)] = digest
    run = read_run_record(root)
    if (
        sha256_bytes(preflight_raw) != M1_PREFLIGHT_SHA256
        or sha256_bytes(state_raw) != _HASHES["state"]
        or sha256_bytes((root / "run.json").read_bytes()) != _HASHES["run"]
        or sha256_bytes(sessions_raw) != _HASHES["sessions"]
        or preflight.get("run_id") != M1_RUN_ID
        or preflight.get("project_id") != project_id
        or preflight.get("parent_calibration") != parent_lineage
        or preflight.get("prior_repair") != prior_repair_lineage
        or preflight.get("cost_caps_usd")
        != {
            "teacher-dose": M1_TEACHER_CAP_USD,
            "teacher-allocation": 0,
            "stage-a": REPAIR_STAGE_A_CAP_USD,
            "total": REPAIR_AGGREGATE_CAP_USD,
        }
        or preflight.get("teacher_exposure_repair") != canonical_json_value(REPAIR_SPEC)
        or state.get("status") != "interrupted"
        or state.get("completed_actions") != []
        or state.get("artifact_sha256") != {}
        or state.get("preflight_sha256") != M1_PREFLIGHT_SHA256
        or "terminal_decision" in state
        or sessions.get("session_ids") != [M1_SESSION_ID]
        or run.status is not RunStatus.INTERRUPTED
        or run.project_id != project_id
        or run.tinker_session_id != M1_SESSION_ID
        or run.authorized_cost_usd != REPAIR_AGGREGATE_CAP_USD
        or run.reserved_cost_usd != REPAIR_AGGREGATE_CAP_USD
        or run.finished_at is None
        or (root / "teacher-dose-terminal.json").exists()
        or (root / "stage-a-arms").exists()
    ):
        raise RunnerGateError("M1 interruption lineage differs")
    return {
        "run_id": M1_RUN_ID,
        "status": "interrupted",
        "preflight_sha256": M1_PREFLIGHT_SHA256,
        "run_sha256": _HASHES["run"],
        "state_sha256": _HASHES["state"],
        "session_lineage_sha256": _HASHES["sessions"],
        "session_id": M1_SESSION_ID,
        "journal_state_sha256": journal_hashes,
        "pending_remote_calls": 0,
        "charged_teacher_cap_usd": M1_TEACHER_CAP_USD,
    }


__all__ = ["load_m1_interruption"]
