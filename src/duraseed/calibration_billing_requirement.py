"""Exact post-run billing handoff for the bounded calibration repair."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from duraseed.provenance import sha256_bytes

if TYPE_CHECKING:
    from duraseed.run_records import RunStatus


def calibration_billing_requirement(
    inputs: Any,
    root: Path,
    status: RunStatus,
    finished: datetime,
    cost: float,
) -> dict[str, Any] | None:
    """Return a handoff for success or the exact scientific no-stable terminal."""

    terminal_status = terminal_sha256 = None
    terminal_path = root / "teacher-dose-terminal.json"
    if status.value == "failed" and terminal_path.exists():
        try:
            terminal = json.loads(terminal_path.read_bytes())
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(terminal, dict)
            or terminal.get("schema_version") != "duraseed-teacher-exposure-terminal-v2"
            or terminal.get("status") != "no_stable_checkpoint"
            or terminal.get("run_id") != inputs.run_id
        ):
            return None
        terminal_status = terminal["status"]
        terminal_sha256 = sha256_bytes(terminal_path.read_bytes())
    elif status.value != "completed":
        return None
    sessions = json.loads((root / "session-lineage.json").read_bytes())["session_ids"]
    parent = inputs.parent_teacher_evidence
    child_cap = (
        inputs.teacher_ledger.authorized_usd + inputs.stage_a_ledger.authorized_usd
    )
    return {
        "schema_version": "duraseed-calibration-billing-required-v1",
        "status": "pending",
        "run_id": inputs.run_id,
        "project_id": inputs.project_id,
        "session_ids": sessions,
        "execution_finished_at_utc": finished.isoformat(),
        "run_status": status.value,
        "terminal_status": terminal_status,
        "terminal_sha256": terminal_sha256,
        "local_observed_cost_usd": cost,
        "action_caps_usd": {
            "teacher-dose": inputs.teacher_ledger.authorized_usd,
            "stage-a": inputs.stage_a_ledger.authorized_usd,
        },
        "aggregate_cap_usd": child_cap,
        "parent_run_id": parent.parent_run_id,
        "parent_billing_sha256": parent.parent_billing_sha256,
        "parent_billed_usd": parent.parent_billed_usd,
        "protected_reserve_usd": parent.protected_reserve_usd,
        "lifetime_calibration_cap_usd": 300,
    }


__all__ = ["calibration_billing_requirement"]
