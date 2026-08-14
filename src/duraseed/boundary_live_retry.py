"""One-shot recovery for the reconciled Extension-2 refinement incident."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from duraseed.boundary_live_artifacts import BoundaryLiveArtifacts, _read_json
from duraseed.data.io import atomic_write_bytes
from duraseed.run_records import RunRecord, RunStatus
from duraseed.runners import RunnerGateError
from duraseed.runtime import TokenBudget, TokenLedger


RETRY_ACTION = "extension2-refine"
RETRY_MARKER = "extension2_refine_retry.json"
INCIDENT = {
    "run_id": "boundary-live-20260813T130013Z",
    "project_id": "7727a6e3-fadb-4b07-9801-721221235e1e",
    "original_git_commit": "60f36a3d5ac475f0531111f3a80fc35f98322b80",
    "pending": {
        "action": RETRY_ACTION,
        "task_id": (
            "sha256:01a2cb69e51eed898834340422f7383dc4c7e09e19bac6daf9a16de17b4cb5e8"
        ),
        "manifest_id": (
            "sha256:683bf6485b42755dfe4f0210b63d5a2a70975be7801ef3f0d3454b427921a00e"
        ),
        "run_id": "boundary-live-20260813T130013Z:extension2-refine",
        "sample_indices": list(range(4, 16)),
        "reserved_tokens": {"prefill": 1584, "sample": 49152, "train": 0},
    },
    "error": {
        "action": RETRY_ACTION,
        "error_type": "APIConnectionError",
        "message": "No progress made in 7200s. Requests appear to be stuck.",
        "time": "2026-08-14T01:36:39.323406+00:00",
    },
    "remote_proof": {
        "tinker_session_id": "ca14e9cf-61aa-57a3-8657-26f018c26710",
        "future_checks": [
            {"future_id": future_id, "http_status": 404}
            for future_id in range(3200, 3212)
        ],
        "post_failure_trace_sha256": (
            "sha256:b417f8c985f64c297868c31cb0067a1a11e8dbc955398ebe530265d2495e4c43"
        ),
        "post_failure_trace_max_future_id": 3199,
        "absent_future_ids": list(range(3200, 3212)),
    },
}


def _write(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    atomic_write_bytes(path, payload + b"\n")


def _errors(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        rows = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("boundary retry error evidence is invalid") from error
    if not all(isinstance(row, dict) for row in rows):
        raise RunnerGateError("boundary retry error evidence is invalid")
    return rows


def _budget(value: Any) -> TokenBudget:
    try:
        return TokenBudget(**value)
    except (TypeError, ValueError) as error:
        raise RunnerGateError("boundary retry reservation is invalid") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RunnerGateError("boundary retry trace is invalid") from error
    return "sha256:" + digest.hexdigest()


def _has_refinement(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return any(
            json.loads(line).get("action") == RETRY_ACTION
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (AttributeError, OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("boundary retry group journal is invalid") from error


class BoundaryRetryArtifacts(BoundaryLiveArtifacts):
    """Permit exactly one retry of the authenticated failed task group."""

    def __init__(
        self,
        directory: Path,
        *,
        preflight: dict[str, Any],
        new_run: RunRecord,
        trace_path: Path | None,
    ) -> None:
        self.retry_marker = directory / RETRY_MARKER
        self._retry_pending: dict[str, Any] | None = None
        self._retry_evidence: dict[str, Any] | None = None
        self._validate(directory, preflight, new_run, trace_path)
        super().__init__(
            directory,
            preflight=preflight,
            new_run=new_run,
            _allow_pending=True,
            _allow_git_change=True,
        )

    def _validate(
        self,
        directory: Path,
        preflight: dict[str, Any],
        new_run: RunRecord,
        trace_path: Path | None,
    ) -> None:
        if not directory.is_dir():
            raise RunnerGateError("boundary retry requires the exact existing run")
        pending_path = directory / "pending_group.json"
        pending = _read_json(pending_path) if pending_path.exists() else None
        marker = _read_json(self.retry_marker) if self.retry_marker.exists() else None
        try:
            run = RunRecord.model_validate_json((directory / "run.json").read_bytes())
        except (OSError, ValueError) as error:
            raise RunnerGateError("boundary retry run record is invalid") from error
        errors = _errors(directory / "errors.jsonl")
        common = (
            directory.name == INCIDENT["run_id"]
            and run.git_commit == INCIDENT["original_git_commit"]
            and run.project_id == INCIDENT["project_id"]
            and INCIDENT["error"] in errors
            and preflight.get("run_id") == INCIDENT["run_id"]
            and preflight.get("extension2_manifest_id")
            == INCIDENT["pending"]["manifest_id"]
            and preflight.get("actions", {}).get(RETRY_ACTION) == "30"
        )
        if marker is not None:
            valid_marker = (
                common
                and marker.get("schema_version") == 1
                and marker.get("action") == RETRY_ACTION
                and marker.get("incident") == INCIDENT
                and marker.get("recovery_git_commit") == new_run.git_commit
            )
            if not valid_marker:
                raise RunnerGateError("boundary retry marker differs from the incident")
            if pending is not None or marker.get("status") != "completed":
                raise RunnerGateError(
                    "refinement retry is already ambiguous; no second retry"
                )
            if trace_path is not None:
                raise RunnerGateError("refinement retry was already consumed")
            self._retry_evidence = marker
            return
        billing = _read_json(directory / "billing.json")
        action_row = (
            billing.get("actions", {}).get(RETRY_ACTION, {})
            if isinstance(billing, dict)
            else {}
        )
        valid_initial = (
            common
            and trace_path is not None
            and pending == INCIDENT["pending"]
            and run.status is RunStatus.FAILED
            and errors[-1] == INCIDENT["error"]
            and _sha256(trace_path)
            == INCIDENT["remote_proof"]["post_failure_trace_sha256"]
            and action_row.get("committed_tokens")
            == INCIDENT["pending"]["reserved_tokens"]
            and action_row.get("observed_tokens")
            == INCIDENT["pending"]["reserved_tokens"]
            and Decimal(str(action_row.get("authorized_cost_usd"))) == Decimal("30")
            and not _has_refinement(directory / "observation_groups.jsonl")
        )
        if not valid_initial:
            raise RunnerGateError("failed refinement incident is not exactly retryable")
        self._retry_pending = pending
        self._retry_evidence = {
            "schema_version": 1,
            "status": "ready",
            "action": RETRY_ACTION,
            "incident": INCIDENT,
            "recovery_git_commit": new_run.git_commit,
        }

    def restore_ledger(
        self, action: str, limits: TokenBudget, authorized_usd: Decimal
    ) -> TokenLedger:
        failed = _budget(INCIDENT["pending"]["reserved_tokens"])
        ledger = super().restore_ledger(
            action,
            limits.plus(failed) if action == RETRY_ACTION else limits,
            authorized_usd,
        )
        if action != RETRY_ACTION or self._retry_evidence is None:
            return ledger
        generations = tuple(
            row
            for (group_action, _), rows in self.groups.items()
            if group_action == RETRY_ACTION
            for row in rows[: len(rows) // 2]
        )
        completed = TokenBudget(
            sum(row.prompt_tokens for row in generations),
            sum(int(row.sampling_max_tokens or 0) for row in generations),
            0,
        )
        observed = TokenBudget(
            completed.prefill, sum(row.sampled_tokens for row in generations), 0
        )
        committed_floor, observed_floor = failed.plus(completed), failed.plus(observed)
        ledger.committed = TokenBudget(
            max(ledger.committed.prefill, committed_floor.prefill),
            max(ledger.committed.sample, committed_floor.sample),
            0,
        )
        ledger.observed = TokenBudget(
            max(ledger.observed.prefill, observed_floor.prefill),
            max(ledger.observed.sample, observed_floor.sample),
            0,
        )
        if (
            ledger.committed.prefill > ledger.limits.prefill
            or ledger.committed.sample > ledger.limits.sample
            or ledger.committed_cost_usd > ledger.authorized_usd
        ):
            raise RunnerGateError("resumed refinement retry exceeds its fixed cap")
        return ledger

    def begin_group(self, action: str, task_id: str, **values: Any) -> None:
        if not self.pending.exists():
            return super().begin_group(action, task_id, **values)
        expected = {"action": action, "task_id": task_id, **values}
        reservation = values.get("reservation")
        expected["reserved_tokens"] = {
            "prefill": reservation.prefill,
            "sample": reservation.sample,
            "train": reservation.train,
        }
        expected.pop("reservation")
        expected["sample_indices"] = list(expected["sample_indices"])
        if (
            expected != INCIDENT["pending"]
            or self._retry_pending != INCIDENT["pending"]
            or self.retry_marker.exists()
        ):
            raise RunnerGateError("refinement retry coordinates differ from incident")
        assert self._retry_evidence is not None
        self._retry_evidence.update(
            {"status": "started", "started_at": datetime.now(UTC).isoformat()}
        )
        _write(self.retry_marker, self._retry_evidence)

    def append_group(self, action: str, task_id: str, rows: tuple[Any, ...]) -> None:
        super().append_group(action, task_id, rows)
        if action == RETRY_ACTION and self._retry_pending is not None:
            assert self._retry_evidence is not None
            self._retry_evidence.update(
                {"status": "completed", "completed_at": datetime.now(UTC).isoformat()}
            )
            _write(self.retry_marker, self._retry_evidence)
            self._retry_pending = None

    def finish(self, status: RunStatus, ledgers: dict[str, Any]) -> None:
        if status is RunStatus.COMPLETED and self.pending.exists():
            raise RunnerGateError("cannot finish boundary run with a pending retry")
        super().finish(status, ledgers)


__all__ = ["BoundaryRetryArtifacts", "INCIDENT", "RETRY_MARKER"]
