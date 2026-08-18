"""Authenticate and conservatively charge the completed direct-M0 screen."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from duraseed.calibration_stage_a_terminal import existing_stage_a_terminal
from duraseed.provenance import sha256_bytes
from duraseed.run_records import RunStatus, read_run_record
from duraseed.runners import RunnerGateError
from duraseed.runtime import PRICE_SNAPSHOT, UsageQuantities
from duraseed.teacher_exposure_spec import (
    DIRECT_M0_AGGREGATE_CAP_USD,
    DIRECT_M0_STAGE_A_CAP_USD,
    LIFETIME_CALIBRATION_CAP_USD,
    PRIOR_DIRECT_STAGE_A_CHARGE_USD,
    PRIOR_DIRECT_STAGE_A_COMMITTED_FIXED_USD,
    PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS,
    PRIOR_DIRECT_STAGE_A_JOURNAL_SHA256,
    PRIOR_DIRECT_STAGE_A_PREFLIGHT_SHA256,
    PRIOR_DIRECT_STAGE_A_RUN_ID,
    PRIOR_DIRECT_STAGE_A_SESSION_ID,
    PRIOR_DIRECT_STAGE_A_TERMINAL_SHA256,
)


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid prior Stage-A {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"prior Stage-A {label} is not an object")
    return value, raw


def load_prior_stage_a(
    parent_root: Path,
    *,
    project_id: str,
    parent_lineage: dict[str, object],
    prior_repair_lineage: dict[str, object],
    m1_lineage: dict[str, object],
) -> dict[str, object]:
    """Bind the amended launch to the exact clear-journal screen terminal."""

    root = parent_root.parent / PRIOR_DIRECT_STAGE_A_RUN_ID
    preflight, preflight_raw = _object(root / "preflight.json", "preflight")
    terminal, terminal_raw = _object(root / "stage-a-terminal.json", "terminal")
    journal_path = (
        root
        / "stage-a-arms/complete-bounded-stage-a/attempt-0001/remote-call-state.json"
    )
    journal, journal_raw = _object(journal_path, "remote journal")
    run = read_run_record(root)
    usage = terminal.get("usage")
    if not isinstance(usage, dict):
        raise RunnerGateError("prior direct Stage-A terminal omits usage")
    committed = usage.get("committed_tokens")
    committed_cost = usage.get("committed_cost_usd")
    expected_cost = (
        PRICE_SNAPSHOT.cost(
            UsageQuantities(
                prefill_tokens=PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[0],
                sample_tokens=PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[1],
                train_tokens=PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[2],
            )
        )
        + PRIOR_DIRECT_STAGE_A_COMMITTED_FIXED_USD
    )
    if (
        sha256_bytes(preflight_raw) != PRIOR_DIRECT_STAGE_A_PREFLIGHT_SHA256
        or sha256_bytes(terminal_raw) != PRIOR_DIRECT_STAGE_A_TERMINAL_SHA256
        or sha256_bytes(journal_raw) != PRIOR_DIRECT_STAGE_A_JOURNAL_SHA256
        or preflight.get("run_id") != PRIOR_DIRECT_STAGE_A_RUN_ID
        or preflight.get("project_id") != project_id
        or preflight.get("parent_calibration") != parent_lineage
        or preflight.get("prior_repair") != prior_repair_lineage
        or preflight.get("interrupted_m1") != m1_lineage
        or preflight.get("cost_caps_usd")
        != {
            "teacher-dose": 0.0,
            "teacher-allocation": 0,
            "stage-a": DIRECT_M0_STAGE_A_CAP_USD,
            "total": DIRECT_M0_AGGREGATE_CAP_USD,
        }
        or preflight.get("lifetime_calibration_cap_usd") != LIFETIME_CALIBRATION_CAP_USD
        or terminal.get("status") != "no_eligible_learning_rate"
        or terminal.get("preflight_sha256") != PRIOR_DIRECT_STAGE_A_PREFLIGHT_SHA256
        or committed
        != {
            "prefill": PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[0],
            "sample": PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[1],
            "train": PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[2],
        }
        or not isinstance(committed_cost, (int, float))
        or not math.isclose(
            float(committed_cost),
            PRIOR_DIRECT_STAGE_A_CHARGE_USD,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            expected_cost, PRIOR_DIRECT_STAGE_A_CHARGE_USD, rel_tol=0, abs_tol=1e-12
        )
        or journal.get("pending") is not None
        or journal.get("completed_count") != 1_933
        or journal.get("reserved_floor")
        != {
            "prefill_tokens": PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[0],
            "sample_tokens": PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[1],
            "train_tokens": PRIOR_DIRECT_STAGE_A_COMMITTED_TOKENS[2],
            "fixed_usd": PRIOR_DIRECT_STAGE_A_COMMITTED_FIXED_USD,
        }
        or run.status is not RunStatus.FAILED
        or run.project_id != project_id
        or run.tinker_session_id != PRIOR_DIRECT_STAGE_A_SESSION_ID
        or run.finished_at is None
        or run.authorized_cost_usd != DIRECT_M0_STAGE_A_CAP_USD
        or run.reserved_cost_usd != DIRECT_M0_STAGE_A_CAP_USD
    ):
        raise RunnerGateError("prior direct Stage-A terminal or spend bound differs")
    existing_stage_a_terminal(root, PRIOR_DIRECT_STAGE_A_PREFLIGHT_SHA256)
    return {
        "run_id": PRIOR_DIRECT_STAGE_A_RUN_ID,
        "status": "failed",
        "preflight_sha256": PRIOR_DIRECT_STAGE_A_PREFLIGHT_SHA256,
        "terminal_sha256": PRIOR_DIRECT_STAGE_A_TERMINAL_SHA256,
        "terminal_status": "no_eligible_learning_rate",
        "session_id": PRIOR_DIRECT_STAGE_A_SESSION_ID,
        "journal_state_sha256": PRIOR_DIRECT_STAGE_A_JOURNAL_SHA256,
        "completed_remote_calls": 1_933,
        "pending_remote_calls": 0,
        "committed_tokens": committed,
        "committed_fixed_usd": PRIOR_DIRECT_STAGE_A_COMMITTED_FIXED_USD,
        "billing_basis": "conservative_local_committed_upper_bound",
        "charged_stage_a_usd": PRIOR_DIRECT_STAGE_A_CHARGE_USD,
    }


__all__ = ["load_prior_stage_a"]
