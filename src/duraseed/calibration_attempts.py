"""Bounded per-arm restart and conservative spend restoration for calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import json
import math
from pathlib import Path
import re
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.calibration_billing_events import validate_restart_billing_rows
from duraseed.provenance import canonical_json_bytes, sha256_bytes, validate_sha256_id
from duraseed.runners import RunnerGateError
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import TokenBudget, TokenLedger, ZERO_TOKENS


_ATTEMPT = re.compile(r"attempt-(\d{4})\Z")
_ARM_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")
_TOKEN_KEYS = ("prefill_tokens", "sample_tokens", "train_tokens")


@dataclass(frozen=True, slots=True)
class ReconciledRestart:
    run_id: str
    action: str
    project_id: str
    failed_tinker_session_id: str
    preflight_sha256: str
    arm_id: str
    failed_attempt: int
    raw_billing_sha256: str
    raw_billing_entry_count: int
    raw_usage_cutoff_utc: str
    cumulative_billed_usd: float
    aggregate_billed_usd: float
    reconciled_at_utc: str
    authorizer: str
    authorized_at_utc: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if (
            not all(
                value.strip()
                for value in (
                    self.run_id,
                    self.action,
                    self.project_id,
                    self.failed_tinker_session_id,
                    self.arm_id,
                    self.authorizer,
                )
            )
            or _ARM_ID.fullmatch(self.arm_id) is None
            or "/" in self.arm_id
            or "\\" in self.arm_id
            or self.failed_attempt < 1
            or self.raw_billing_entry_count < 1
            or not math.isfinite(self.cumulative_billed_usd)
            or not 0 <= self.cumulative_billed_usd <= 300
            or not math.isfinite(self.aggregate_billed_usd)
            or not self.cumulative_billed_usd <= self.aggregate_billed_usd <= 300
        ):
            raise ValueError("invalid reconciled restart coordinate")
        validate_sha256_id(self.preflight_sha256)
        validate_sha256_id(self.raw_billing_sha256)
        validate_sha256_id(self.artifact_sha256)
        try:
            cutoff = datetime.fromisoformat(self.raw_usage_cutoff_utc)
            reconciled = datetime.fromisoformat(self.reconciled_at_utc)
            authorized = datetime.fromisoformat(self.authorized_at_utc)
        except ValueError as error:
            raise ValueError("invalid reconciliation usage cutoff") from error
        if (
            any(
                value.tzinfo is None or value.utcoffset() is None
                for value in (cutoff, reconciled, authorized)
            )
            or reconciled < cutoff
            or authorized < reconciled
        ):
            raise ValueError("reconciliation timestamps must be ordered and aware")


@dataclass(slots=True)
class ArmAttempt:
    arm_id: str
    number: int
    directory: Path
    journal: RemoteJournal | None
    completed_payload: Any | None = None

    @property
    def completed(self) -> bool:
        return self.journal is None


def _read(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid calibration {label}") from error


def load_reconciled_restart(
    path: str | Path, raw_billing_path: str | Path
) -> ReconciledRestart:
    """Load one explicit billing reconciliation for one failed arm attempt."""

    source = Path(path)
    value = _read(source, "restart reconciliation")
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "duraseed-calibration-reconciliation-v1"
        or value.get("status") != "billing_reconciled"
        or value.get("authorized_restart") is not True
        or type(value.get("failed_attempt")) is not int
        or type(value.get("raw_billing_entry_count")) is not int
        or not isinstance(value.get("console_cumulative_billed_usd"), (int, float))
        or not isinstance(value.get("raw_usage_cutoff_utc"), str)
        or not isinstance(value.get("reconciled_at_utc"), str)
        or not isinstance(value.get("authorized_at_utc"), str)
        or not isinstance(value.get("authorizer"), str)
        or not isinstance(value.get("console_aggregate_billed_usd"), (int, float))
    ):
        raise RunnerGateError("restart reconciliation is not an accepted artifact")
    raw_billing = Path(raw_billing_path)
    try:
        raw_bytes = raw_billing.read_bytes()
        raw_hash = sha256_bytes(raw_bytes)
        raw_value = json.loads(raw_bytes)
    except OSError as error:
        raise RunnerGateError("restart raw billing evidence is unreadable") from error
    except json.JSONDecodeError as error:
        raise RunnerGateError("restart raw billing evidence is malformed") from error
    events = raw_value.get("data") if isinstance(raw_value, dict) else None
    if (
        value.get("raw_billing_sha256") != raw_hash
        or not isinstance(events, list)
        or value.get("raw_billing_entry_count") != len(events)
        or not any(
            isinstance(row, dict)
            and row.get("session_id") == value.get("failed_tinker_session_id")
            for row in events
        )
    ):
        raise RunnerGateError("restart raw billing evidence hash differs")
    validate_restart_billing_rows(events, value)
    try:
        return ReconciledRestart(
            str(value.get("run_id", "")),
            str(value.get("action", "")),
            str(value.get("project_id", "")),
            str(value.get("failed_tinker_session_id", "")),
            str(value.get("preflight_sha256", "")),
            str(value.get("arm_id", "")),
            value["failed_attempt"],
            raw_hash,
            value["raw_billing_entry_count"],
            value["raw_usage_cutoff_utc"],
            float(value["console_cumulative_billed_usd"]),
            float(value["console_aggregate_billed_usd"]),
            value["reconciled_at_utc"],
            value["authorizer"],
            value["authorized_at_utc"],
            sha256_bytes(source.read_bytes()),
        )
    except (TypeError, ValueError) as error:
        raise RunnerGateError("restart reconciliation identity is malformed") from error


def _reservation(value: Any) -> tuple[TokenBudget, float]:
    if not isinstance(value, dict):
        raise RunnerGateError("calibration journal reservation is malformed")
    try:
        tokens = TokenBudget(*(value.get(key, 0) for key in _TOKEN_KEYS))
        fixed = float(value.get("fixed_usd", 0.0))
    except (TypeError, ValueError) as error:
        raise RunnerGateError("calibration journal reservation is malformed") from error
    if fixed < 0:
        raise RunnerGateError("calibration journal fixed reservation is negative")
    return tokens, fixed


def hydrate_attempt_ledger(root: Path, ledger: TokenLedger) -> None:
    """Restore every prior reservation, including ambiguous pending-call ceilings."""

    states = sorted(root.glob("*/attempt-*/remote-call-state.json"))
    if not states:
        return
    if (
        ledger.committed != ZERO_TOKENS
        or ledger.observed != ZERO_TOKENS
        or ledger.committed_fixed_usd
        or ledger.observed_fixed_usd
        or ledger.has_pending_call
    ):
        raise RunnerGateError("calibration resume ledger is not fresh")
    tokens = ZERO_TOKENS
    fixed = 0.0
    for path in states:
        value = _read(path, "remote-call state")
        if not isinstance(value, dict):
            raise RunnerGateError("calibration remote-call state is not an object")
        floor_tokens, floor_fixed = _reservation(value.get("reserved_floor", {}))
        tokens = tokens.plus(floor_tokens)
        fixed = float(Decimal(str(fixed)) + Decimal(str(floor_fixed)))
        pending = value.get("pending")
        if pending is not None:
            if not isinstance(pending, dict):
                raise RunnerGateError("calibration pending call is malformed")
            pending_tokens, pending_fixed = _reservation(pending.get("reservation"))
            tokens = tokens.plus(pending_tokens)
            fixed = float(Decimal(str(fixed)) + Decimal(str(pending_fixed)))
    ledger.reserve_call(tokens, fixed_usd=fixed)
    ledger.settle_call(tokens)


class ArmAttempts:
    """Open one complete bounded arm or a fresh post-reconciliation attempt."""

    def __init__(
        self,
        root: Path,
        ledger: TokenLedger,
        *,
        run_id: str,
        action: str,
        project_id: str,
        preflight_sha256: str,
        reconciliations: tuple[ReconciledRestart, ...] = (),
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        hydrate_attempt_ledger(root, ledger)
        context = (run_id, action, project_id, preflight_sha256)
        if any(
            (
                row.run_id,
                row.action,
                row.project_id,
                row.preflight_sha256,
            )
            != context
            for row in reconciliations
        ):
            raise RunnerGateError("restart reconciliation belongs to another launch")
        coordinates = tuple((row.arm_id, row.failed_attempt) for row in reconciliations)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("duplicate calibration restart reconciliation")
        self.root = root
        self.reconciliations = {
            (row.arm_id, row.failed_attempt): row for row in reconciliations
        }
        self.preflight_sha256 = preflight_sha256

    @property
    def completed_arm_ids(self) -> frozenset[str]:
        return frozenset(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / "completed.json").is_file()
        )

    @property
    def prior_billed_usd(self) -> float:
        return max(
            (row.cumulative_billed_usd for row in self.reconciliations),
            default=0.0,
        )

    def assert_no_unused_reconciliations(self) -> None:
        used = {
            (path.parent.name, int(path.stem.removeprefix("restart-after-")))
            for path in self.root.glob("*/restart-after-*.json")
        }
        if set(self.reconciliations).difference(used):
            raise RunnerGateError("restart reconciliation was not consumed exactly")

    def _attempts(self, arm: Path) -> list[tuple[int, Path]]:
        values = []
        for path in arm.glob("attempt-*"):
            match = _ATTEMPT.fullmatch(path.name)
            if match is None or not path.is_dir():
                raise RunnerGateError("calibration arm has a malformed attempt path")
            values.append((int(match.group(1)), path))
        values.sort()
        if tuple(number for number, _ in values) != tuple(range(1, len(values) + 1)):
            raise RunnerGateError("calibration arm attempts are not contiguous")
        return values

    def open(self, arm_id: str) -> ArmAttempt:
        if _ARM_ID.fullmatch(arm_id) is None:
            raise ValueError("arm_id must be one safe path token")
        arm = self.root / arm_id
        arm.mkdir(exist_ok=True)
        identity = arm / "coordinate.json"
        expected = canonical_json_bytes(
            {"arm_id": arm_id, "preflight_sha256": self.preflight_sha256}
        )
        if identity.exists() and identity.read_bytes() != expected:
            raise RunnerGateError("calibration arm coordinate changed")
        atomic_write_bytes(identity, expected)
        completed = arm / "completed.json"
        attempts = self._attempts(arm)
        if completed.exists():
            value = _read(completed, "completed arm")
            if (
                not isinstance(value, dict)
                or value.get("arm_id") != arm_id
                or value.get("preflight_sha256") != self.preflight_sha256
                or type(value.get("attempt")) is not int
                or value["attempt"] < 1
                or value["attempt"] != len(attempts)
                or "evidence" not in value
            ):
                raise RunnerGateError("completed calibration arm identity changed")
            state = _read(
                attempts[value["attempt"] - 1][1] / "remote-call-state.json",
                "completed arm journal",
            )
            if state.get("pending") is not None:
                raise RunnerGateError(
                    "completed calibration arm retains a pending call"
                )
            return ArmAttempt(arm_id, value["attempt"], arm, None, value["evidence"])
        if not attempts:
            directory = arm / "attempt-0001"
            return ArmAttempt(arm_id, 1, directory, RemoteJournal(directory))
        number, directory = attempts[-1]
        state_path = directory / "remote-call-state.json"
        state = _read(state_path, "incomplete arm journal")
        has_spend = (
            state.get("pending") is not None or state.get("completed_count") != 0
        )
        if not has_spend:
            return ArmAttempt(arm_id, number, directory, RemoteJournal(directory))
        reconciliation = self.reconciliations.get((arm_id, number))
        if reconciliation is None:
            raise RunnerGateError(
                f"arm {arm_id} attempt {number} is incomplete; reconcile billing "
                "before a whole-arm restart"
            )
        try:
            started = datetime.fromisoformat(state["attempt_started_at_utc"])
            cutoff = datetime.fromisoformat(reconciliation.raw_usage_cutoff_utc)
        except (KeyError, TypeError, ValueError) as error:
            raise RunnerGateError(
                "restart reconciliation chronology is malformed"
            ) from error
        if cutoff < started:
            raise RunnerGateError("restart billing cutoff predates the failed attempt")
        next_number = number + 1
        next_directory = arm / f"attempt-{next_number:04d}"
        atomic_write_bytes(
            arm / f"restart-after-{number:04d}.json",
            canonical_json_bytes(
                {
                    "arm_id": arm_id,
                    "failed_attempt": number,
                    "preflight_sha256": self.preflight_sha256,
                    "reconciliation_sha256": reconciliation.artifact_sha256,
                    "restart_attempt": next_number,
                }
            ),
        )
        return ArmAttempt(
            arm_id,
            next_number,
            next_directory,
            RemoteJournal(next_directory),
        )

    def complete(self, attempt: ArmAttempt, evidence: Any) -> None:
        if attempt.journal is None or attempt.journal.pending:
            raise RunnerGateError("cannot complete an arm with a pending remote call")
        atomic_write_bytes(
            self.root / attempt.arm_id / "completed.json",
            canonical_json_bytes(
                {
                    "arm_id": attempt.arm_id,
                    "attempt": attempt.number,
                    "preflight_sha256": self.preflight_sha256,
                    "evidence": evidence,
                }
            ),
        )


__all__ = [
    "ArmAttempt",
    "ArmAttempts",
    "ReconciledRestart",
    "hydrate_attempt_ledger",
    "load_reconciled_restart",
]
