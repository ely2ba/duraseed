"""Exact fresh-client continuation of the reconciled refinement retry."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from duraseed.boundary_live_artifacts import BoundaryLiveArtifacts, _read_json
from duraseed.boundary_live_retry import INCIDENT, RETRY_ACTION, RETRY_MARKER
from duraseed.data.io import atomic_write_bytes
from duraseed.data.manifests import DatasetManifest
from duraseed.run_records import RunRecord
from duraseed.runners import RunnerGateError
from duraseed.runtime import TokenBudget, TokenLedger


FRESH_RESUME_MARKER = "extension2_refine_fresh_resume.json"
STALE_RUNTIME_INCIDENT = {
    "run_id": "boundary-live-20260813T130013Z",
    "stale_session_id": "bade00bf-d79d-5ac2-b958-1b5992c900b2",
    "retry_started_at": "2026-08-14T14:30:47.132269+00:00",
    "stale_trace_sha256": (
        "sha256:62e6a75c0a5865459aed6f6a6d63d48fc7b10901c1b213437864cf19266f9528"
    ),
    "stale_trace_packet_count": 28,
    "stale_trace_descriptor_only": True,
    "retry_git_commit": "83689e7da1fbf09e72345e34bea82a5ba7417f6a",
    "snapshot_sha256": {
        "run.json": (
            "sha256:e21c456ec0bfc47124ec74c02eb3c281b004cffc3edbcb232dee2bd16ab19081"
        ),
        "preflight.json": (
            "sha256:3b2baa28ed5660ffd6d99b75d3e1d3141b1c8669b9a0171d3926ea011e0d7925"
        ),
        "observation_groups.jsonl": (
            "sha256:fe86c516f39753455715f3d9ebc274e9c39469d2a78939ce574d13a8491666b3"
        ),
        "errors.jsonl": (
            "sha256:a1a8754efb5bfb0b14f68da61840b98537236dbb8107b74553fea1afed653360"
        ),
        "pending_group.json": (
            "sha256:bf2bc2000ad5842bfc98abf86bcb18139550fb4cf231ad3b1a9053ad7076779a"
        ),
        "billing.json": (
            "sha256:0af29c7bed16f77997ceccfaf6c2b849a69f55f9447da44f90437cddbbe2a219"
        ),
        RETRY_MARKER: (
            "sha256:3d1b38081b522e5ab020a7c1024ec9f81d3bc20d401796aa1088932441a8c6ed"
        ),
        "extension2_broad_manifest.json": (
            "sha256:8705dae1fc32c5a108931886857ab330e363e37bbf7cfaaf184c737703cd299d"
        ),
    },
    "journal_groups": {"extension1-confirm": 136, "extension2-broad": 256},
    "journal_samples": 3200,
    "refinement": {"positive_families": 42, "audit_families": 12, "tasks": 216},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RunnerGateError("fresh refinement resume artifact is missing") from error
    return "sha256:" + digest.hexdigest()


def is_fresh_resume_trace(path: str | Path | None) -> bool:
    return (
        path is not None
        and _sha256(Path(path).resolve())
        == STALE_RUNTIME_INCIDENT["stale_trace_sha256"]
    )


def _write(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    atomic_write_bytes(path, payload + b"\n")


def _validate_initial_snapshot(directory: Path, trace_path: Path) -> None:
    expected = STALE_RUNTIME_INCIDENT
    if directory.name != expected["run_id"] or not directory.is_dir():
        raise RunnerGateError("fresh refinement resume requires the exact run")
    if _sha256(trace_path) != expected["stale_trace_sha256"]:
        raise RunnerGateError(
            "fresh refinement resume trace differs from stale session"
        )
    for name, digest in expected["snapshot_sha256"].items():
        if _sha256(directory / name) != digest:
            raise RunnerGateError(f"fresh refinement resume snapshot changed: {name}")
    retry_marker = _read_json(directory / RETRY_MARKER)
    if (
        retry_marker.get("schema_version") != 1
        or retry_marker.get("status") != "started"
        or retry_marker.get("started_at") != expected["retry_started_at"]
        or retry_marker.get("recovery_git_commit") != expected["retry_git_commit"]
        or retry_marker.get("incident") != INCIDENT
        or _read_json(directory / "pending_group.json") != INCIDENT["pending"]
    ):
        raise RunnerGateError("fresh refinement resume marker differs from incident")
    try:
        rows = [
            json.loads(line)
            for line in (directory / "observation_groups.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("fresh refinement resume journal is invalid") from error
    counts = Counter(row.get("action") for row in rows)
    samples = sum(len(row.get("generations", ())) for row in rows)
    broad_tasks = {
        row.get("task_id") for row in rows if row.get("action") == "extension2-broad"
    }
    manifest = load_saved_extension2_manifest(directory)
    if (
        counts != expected["journal_groups"]
        or samples != expected["journal_samples"]
        or broad_tasks != {row.task_id for row in manifest.records}
        or any(
            len(row.get("generations", ())) != len(row.get("rewards", ()))
            for row in rows
        )
        or any(row.get("action") == RETRY_ACTION for row in rows)
    ):
        raise RunnerGateError("fresh refinement resume journal grid changed")


def load_saved_extension2_manifest(directory: Path) -> DatasetManifest:
    try:
        manifest = DatasetManifest.model_validate_json(
            (directory / "extension2_broad_manifest.json").read_bytes()
        )
    except (OSError, ValueError) as error:
        raise RunnerGateError("saved Extension-2 manifest is invalid") from error
    if manifest.manifest_id != INCIDENT["pending"]["manifest_id"]:
        raise RunnerGateError("saved Extension-2 manifest identity changed")
    return manifest


def validate_fresh_resume(directory: Path, trace_path: Path) -> DatasetManifest:
    if (directory / FRESH_RESUME_MARKER).exists():
        raise RunnerGateError("fresh refinement resume was already consumed")
    _validate_initial_snapshot(directory, trace_path)
    return load_saved_extension2_manifest(directory)


class BoundaryFreshResumeArtifacts(BoundaryLiveArtifacts):
    """Allow one fresh-client call after the proven request-free stale client."""

    def __init__(
        self,
        directory: Path,
        *,
        preflight: dict[str, Any],
        new_run: RunRecord,
        trace_path: Path | None,
    ) -> None:
        self.fresh_marker = directory / FRESH_RESUME_MARKER
        self._fresh_pending: dict[str, Any] | None = None
        marker = _read_json(self.fresh_marker) if self.fresh_marker.exists() else None
        if marker is None:
            if trace_path is None:
                raise RunnerGateError("fresh refinement resume requires its trace")
            _validate_initial_snapshot(directory, trace_path)
            self._fresh_pending = INCIDENT["pending"]
            self._fresh_evidence = {
                "schema_version": 1,
                "status": "ready",
                "action": RETRY_ACTION,
                "incident": STALE_RUNTIME_INCIDENT,
                "recovery_git_commit": new_run.git_commit,
            }
        else:
            valid = (
                marker.get("schema_version") == 1
                and marker.get("status") == "completed"
                and marker.get("action") == RETRY_ACTION
                and marker.get("incident") == STALE_RUNTIME_INCIDENT
                and marker.get("recovery_git_commit") == new_run.git_commit
            )
            if not valid or (directory / "pending_group.json").exists():
                raise RunnerGateError(
                    "fresh refinement resume is ambiguous; no further retry"
                )
            if trace_path is not None:
                raise RunnerGateError("fresh refinement resume was already consumed")
            self._fresh_evidence = marker
        super().__init__(
            directory,
            preflight=preflight,
            new_run=new_run,
            _allow_pending=True,
            _allow_git_change=True,
        )

    def restore_ledger(
        self, action: str, limits: TokenBudget, authorized_usd: Decimal
    ) -> TokenLedger:
        failed = TokenBudget(**INCIDENT["pending"]["reserved_tokens"])
        ledger = super().restore_ledger(
            action,
            limits.plus(failed) if action == RETRY_ACTION else limits,
            authorized_usd,
        )
        if action != RETRY_ACTION:
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
            or ledger.committed.train > ledger.limits.train
            or ledger.committed_cost_usd > ledger.authorized_usd
        ):
            raise RunnerGateError("fresh refinement resume exceeds its fixed cap")
        return ledger

    def begin_group(self, action: str, task_id: str, **values: Any) -> None:
        if not self.pending.exists():
            return super().begin_group(action, task_id, **values)
        reservation = values.pop("reservation")
        expected = {
            "action": action,
            "task_id": task_id,
            **values,
            "sample_indices": list(values["sample_indices"]),
            "reserved_tokens": {
                "prefill": reservation.prefill,
                "sample": reservation.sample,
                "train": reservation.train,
            },
        }
        if (
            expected != INCIDENT["pending"]
            or self._fresh_pending != INCIDENT["pending"]
            or self.fresh_marker.exists()
        ):
            raise RunnerGateError("fresh refinement coordinates differ from incident")
        self._fresh_evidence.update(
            {"status": "started", "started_at": datetime.now(UTC).isoformat()}
        )
        _write(self.fresh_marker, self._fresh_evidence)

    def append_group(self, action: str, task_id: str, rows: tuple[Any, ...]) -> None:
        super().append_group(action, task_id, rows)
        if action == RETRY_ACTION and self._fresh_pending is not None:
            self._fresh_evidence.update(
                {"status": "completed", "completed_at": datetime.now(UTC).isoformat()}
            )
            _write(self.fresh_marker, self._fresh_evidence)
            self._fresh_pending = None


__all__ = [
    "BoundaryFreshResumeArtifacts",
    "FRESH_RESUME_MARKER",
    "STALE_RUNTIME_INCIDENT",
    "is_fresh_resume_trace",
    "load_saved_extension2_manifest",
    "validate_fresh_resume",
]
