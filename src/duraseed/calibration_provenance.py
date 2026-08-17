"""Run identity, retained-checkpoint TTLs, and billing handoff for calibration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from duraseed.calibration_billing_requirement import (
    calibration_billing_requirement,
)
from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import (
    RunRecord,
    RunStatus,
    read_run_record,
    write_run_record,
)
from duraseed.runners import RunnerGateError
from duraseed.runtime import PRICE_SNAPSHOT, set_and_verify_ttl


CANDIDATE_TTL_SECONDS = 7 * 24 * 60 * 60


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid calibration artifact: {path.name}") from error


def _write_exact(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise RunnerGateError(f"calibration artifact changed: {path.name}")
    atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def _session_lineage(root: Path, session_id: str) -> tuple[str, ...]:
    path = root / "session-lineage.json"
    value = (
        _json(path)
        if path.exists()
        else {
            "schema_version": "duraseed-calibration-session-lineage-v1",
            "session_ids": [],
        }
    )
    sessions = value.get("session_ids") if isinstance(value, dict) else None
    if not isinstance(sessions, list) or any(
        not isinstance(item, str) or not item.strip() for item in sessions
    ):
        raise RunnerGateError("calibration session lineage is malformed")
    if session_id not in sessions:
        sessions.append(session_id)
        atomic_write_bytes(path, canonical_json_bytes(value))
    return tuple(sessions)


def read_calibration_session_ids(root: Path) -> tuple[str, ...]:
    """Read the exact persisted session lineage without mutating it."""

    path = root / "session-lineage.json"
    if not path.exists():
        return ()
    value = _json(path)
    sessions = value.get("session_ids") if isinstance(value, dict) else None
    if (
        not isinstance(sessions, list)
        or any(not isinstance(item, str) or not item.strip() for item in sessions)
        or len(sessions) != len(set(sessions))
    ):
        raise RunnerGateError("calibration session lineage is malformed")
    return tuple(sessions)


def start_calibration_run(inputs: Any, root: Path) -> RunRecord:
    """Create or resume the immutable calibration RunRecord identity."""

    sessions = _session_lineage(root, inputs.tinker_session_id)
    now = datetime.now(UTC)
    expected = RunRecord(
        protocol_version=str(inputs.config.protocol["version"]),
        git_commit=inputs.git_commit,
        resolved_config_hash=inputs.config.resolved_config_hash(),
        run_kind="stage_a_calibration",
        method=None,
        seed=17,
        model_id=inputs.config.tinker.model_id,
        renderer=inputs.config.tinker.renderer_name,
        lora_rank=inputs.config.tinker.lora_rank,
        task_manifest_ids={
            "a_monitor": inputs.prompt_pools.a_monitor_manifest.manifest_id,
            "a_rl_train": inputs.prompt_pools.a_rl_train_manifest.manifest_id,
            "a_seed_gate": inputs.teacher_sources.gate_manifest.manifest_id,
            "a_seed_train": inputs.teacher_sources.target_train_manifest.manifest_id,
        },
        parent_tinker_checkpoint_path=inputs.m0_state_path,
        tinker_session_id=sessions[0],
        status=RunStatus.RUNNING,
        started_at=now,
        updated_at=now,
        project_id=inputs.project_id,
        authorized_cost_usd=(
            inputs.teacher_ledger.authorized_usd + inputs.stage_a_ledger.authorized_usd
        ),
        reserved_cost_usd=(
            inputs.teacher_ledger.authorized_usd + inputs.stage_a_ledger.authorized_usd
        ),
        price_snapshot_id=PRICE_SNAPSHOT.snapshot_id,
        tinker_sdk_version=inputs.runtime.sdk.sdk_version,
        tinker_cookbook_version=inputs.runtime.sdk.cookbook_version,
        deviations=["billing reconciliation required after remote completion"],
    )
    path = root / "run.json"
    if not path.exists():
        write_run_record(root, expected)
        return expected
    observed = read_run_record(root)
    immutable = (
        "protocol_version",
        "git_commit",
        "resolved_config_hash",
        "run_kind",
        "method",
        "seed",
        "model_id",
        "renderer",
        "lora_rank",
        "task_manifest_ids",
        "parent_tinker_checkpoint_path",
        "project_id",
        "authorized_cost_usd",
        "reserved_cost_usd",
        "price_snapshot_id",
        "tinker_sdk_version",
        "tinker_cookbook_version",
    )
    if any(getattr(observed, name) != getattr(expected, name) for name in immutable):
        raise RunnerGateError("calibration RunRecord identity changed on restart")
    if observed.status is RunStatus.COMPLETED:
        return observed
    resumed = observed.model_copy(
        update={"status": RunStatus.RUNNING, "updated_at": now}
    )
    write_run_record(root, resumed)
    return resumed


def _attempt(arm: Path) -> Path:
    value = _json(arm / "completed.json")
    if not isinstance(value, dict) or type(value.get("attempt")) is not int:
        raise RunnerGateError("completed arm omits its accepted attempt")
    return arm / f"attempt-{value['attempt']:04d}"


def _ttl_paths(
    action: str, root: Path, inputs: Any
) -> dict[int | None, tuple[str, ...]]:
    if action == "teacher-dose":
        candidates = []
        for completed in sorted(root.glob("*/completed.json")):
            value = _json(completed)
            evidence = value.get("evidence", {}) if isinstance(value, dict) else {}
            path = evidence.get("sampler_path") if isinstance(evidence, dict) else None
            if isinstance(path, str):
                candidates.append(path)
            points = evidence.get("points") if isinstance(evidence, dict) else None
            if isinstance(points, list):
                candidates.extend(
                    row["sampler_path"]
                    for row in points
                    if isinstance(row, dict)
                    and isinstance(row.get("sampler_path"), str)
                )
        return {
            None: (inputs.m0_sampler_path, inputs.m0_state_path),
            CANDIDATE_TTL_SECONDS: tuple(candidates),
        }
    attempt = _attempt(root / "complete-bounded-stage-a")
    rows = [
        json.loads(line)
        for line in (attempt / "checkpoints.jsonl").read_text().splitlines()
    ]
    boundary = next((row for row in rows if row.get("kind") == "boundary-seed"), None)
    if not isinstance(boundary, dict):
        raise RunnerGateError("Stage-A checkpoint lineage omits the boundary pair")
    candidates = tuple(
        row["sampler"]
        for row in rows
        if "method" in row and isinstance(row.get("sampler"), str)
    )
    return {
        None: (boundary["sampler"], boundary["state"]),
        CANDIDATE_TTL_SECONDS: candidates,
    }


async def verify_action_ttls(action: str, root: Path, inputs: Any) -> dict[str, Any]:
    """Set and verify every retained source/candidate TTL before action commit."""

    path = root / "checkpoint-ttl-audit.json"
    if path.exists():
        return validate_action_ttl_audit(action, root, inputs)
    marker = root / "ttl-verification-pending.json"
    if marker.exists():
        raise RunnerGateError(
            "ambiguous calibration TTL call requires billing/metadata reconciliation"
        )
    atomic_write_bytes(
        marker,
        canonical_json_bytes({"action": action, "operation": "set-and-verify-ttl"}),
    )
    ledger = (
        inputs.teacher_ledger if action == "teacher-dose" else inputs.stage_a_ledger
    )
    rows = []
    for ttl_seconds, paths in _ttl_paths(action, root, inputs).items():
        if not paths:
            continue
        result = await set_and_verify_ttl(
            inputs.runtime,
            inputs.rest_client,
            paths,
            ttl_seconds=ttl_seconds,
            ledger=ledger,
        )
        rows.extend(
            {
                **asdict(row),
                "expires_at": (
                    row.expires_at.isoformat()
                    if hasattr(row.expires_at, "isoformat")
                    else row.expires_at
                ),
                "checkpoint_type": str(row.checkpoint_type),
            }
            for row in result
        )
    value = {
        "schema_version": "duraseed-calibration-ttl-audit-v1",
        "action": action,
        "rows": rows,
    }
    _write_exact(path, value)
    marker.unlink(missing_ok=True)
    return value


def validate_action_ttl_audit(action: str, root: Path, inputs: Any) -> dict[str, Any]:
    """Revalidate a retained TTL audit without making a remote call."""

    value = _json(root / "checkpoint-ttl-audit.json")
    rows = value.get("rows") if isinstance(value, dict) else None
    expected = {
        checkpoint: ttl
        for ttl, paths in _ttl_paths(action, root, inputs).items()
        for checkpoint in paths
    }
    observed = {
        row.get("path"): row.get("ttl_seconds")
        for row in rows or ()
        if isinstance(row, dict)
    }
    if (
        not isinstance(value, dict)
        or value.get("action") != action
        or not isinstance(rows, list)
        or len(observed) != len(rows)
        or observed != expected
    ):
        raise RunnerGateError("calibration TTL audit identity differs")
    return value


def finish_calibration_run(
    inputs: Any, root: Path, status: RunStatus, *, error: str | None = None
) -> None:
    record = read_run_record(root)
    finished = datetime.now(UTC)
    observed = inputs.teacher_ledger.observed.plus(inputs.stage_a_ledger.observed)
    cost = max(
        record.cost_usd,
        inputs.teacher_ledger.observed_cost_usd
        + inputs.stage_a_ledger.observed_cost_usd,
        max(
            (row.aggregate_billed_usd for row in inputs.reconciled_restarts),
            default=0.0,
        ),
    )
    child_cap = (
        inputs.teacher_ledger.authorized_usd + inputs.stage_a_ledger.authorized_usd
    )
    if (
        cost > child_cap
        or cost + inputs.parent_teacher_evidence.lifetime_sunk_usd > 300
    ):
        raise RunnerGateError("child or lifetime calibration spend exceeds its cap")
    deviations = ["billing reconciliation required after remote completion"]
    if error:
        deviations.append(error)
    checkpoint_rows = []
    for path in (
        root / "teacher-dose-arms/checkpoint-ttl-audit.json",
        root / "stage-a-arms/checkpoint-ttl-audit.json",
    ):
        if path.exists():
            value = _json(path)
            rows = value.get("rows") if isinstance(value, dict) else None
            if not isinstance(rows, list):
                raise RunnerGateError("calibration TTL audit rows are malformed")
            checkpoint_rows.extend(rows)
    training_run_ids = tuple(
        sorted(
            {
                row["training_run_id"]
                for row in checkpoint_rows
                if isinstance(row, dict)
                and isinstance(row.get("training_run_id"), str)
                and row["training_run_id"].strip()
            }
        )
    )
    if status is RunStatus.COMPLETED and not training_run_ids:
        raise RunnerGateError("completed calibration omits checkpoint lineage")
    if checkpoint_rows:
        _write_exact(
            root / "checkpoint-lineage.json",
            {
                "schema_version": "duraseed-calibration-checkpoint-lineage-v1",
                "parent_m0_sampler_path": inputs.m0_sampler_path,
                "parent_m0_state_path": inputs.m0_state_path,
                "retained_checkpoints": checkpoint_rows,
            },
        )
    write_run_record(
        root,
        record.model_copy(
            update={
                "status": status,
                "updated_at": finished,
                "finished_at": finished,
                "prompt_tokens": max(record.prompt_tokens, observed.prefill),
                "sampled_tokens": max(record.sampled_tokens, observed.sample),
                "train_tokens": max(record.train_tokens, observed.train),
                "cost_usd": cost,
                "deviations": deviations,
                "tinker_training_run_ids": training_run_ids,
            }
        ),
    )
    billing = calibration_billing_requirement(inputs, root, status, finished, cost)
    if billing is not None:
        _write_exact(
            root / "billing-reconciliation-required.json",
            billing,
        )


__all__ = [
    "finish_calibration_run",
    "read_calibration_session_ids",
    "start_calibration_run",
    "validate_action_ttl_audit",
    "verify_action_ttls",
]
