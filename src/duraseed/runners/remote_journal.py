"""Fail-closed pending-call journal for paid remote operations."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes
from duraseed.run_records import append_jsonl
from duraseed.runners import RunnerGateError
from duraseed.runtime.ledger import TokenBudget, TokenLedger, ZERO_TOKENS


_TOKEN_KEYS = ("prefill_tokens", "sample_tokens", "train_tokens")


def _reservation(value: dict[str, int | float]) -> dict[str, int | float]:
    tokens = tuple(value.get(name, 0) for name in _TOKEN_KEYS)
    fixed = value.get("fixed_usd", 0.0)
    if (
        any(type(item) is not int or item < 0 for item in tokens)
        or isinstance(fixed, bool)
        or not isinstance(fixed, (int, float))
        or not math.isfinite(fixed)
        or fixed < 0
    ):
        raise ValueError("remote reservation must contain nonnegative usage")
    return {
        **dict(zip(_TOKEN_KEYS, tokens, strict=True)),
        "fixed_usd": float(fixed),
    }


class RemoteJournal:
    """Persist one pending coordinate; an ambiguous restart always stops."""

    def __init__(
        self,
        directory: Path,
        ledger: TokenLedger | None = None,
        *,
        reconciled_resume: bool = False,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.state_path = directory / "remote-call-state.json"
        self.events_path = directory / "remote-calls.jsonl"
        self.sequence = 0
        self.pending = False
        self.floor = _reservation({})
        self.current = _reservation({})
        self.started_at_utc = datetime.now(UTC).isoformat()
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_bytes())
            except (OSError, json.JSONDecodeError) as error:
                raise RunnerGateError("remote-call journal is unreadable") from error
            if state.get("pending") is not None:
                raise RunnerGateError(
                    "ambiguous pending remote call; reconcile spend/evidence before resume"
                )
            self.sequence = int(state.get("completed_count", 0))
            started = state.get("attempt_started_at_utc")
            try:
                parsed = datetime.fromisoformat(started)
            except (TypeError, ValueError) as error:
                raise RunnerGateError(
                    "remote-call journal omitted attempt time"
                ) from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise RunnerGateError("remote-call journal attempt time is not aware")
            self.started_at_utc = started
            floor = state.get("reserved_floor")
            if self.sequence and not isinstance(floor, dict):
                raise RunnerGateError("remote-call journal omitted its spend floor")
            self.floor = _reservation(floor or {})
            if ledger is None and self.sequence and not reconciled_resume:
                raise RunnerGateError(
                    "partial remote action requires ledger reconciliation"
                )
            if ledger is not None and self.sequence:
                if ledger.committed != ZERO_TOKENS or ledger.committed_fixed_usd:
                    raise RunnerGateError("resume ledger is not fresh")
                tokens = TokenBudget(*(int(self.floor[key]) for key in _TOKEN_KEYS))
                ledger.reserve_call(tokens, fixed_usd=float(self.floor["fixed_usd"]))
                ledger.settle_call(tokens)
        else:
            self._state(None)

    def _state(self, pending: dict[str, Any] | None) -> None:
        atomic_write_bytes(
            self.state_path,
            canonical_json_bytes(
                {
                    "completed_count": self.sequence,
                    "attempt_started_at_utc": self.started_at_utc,
                    "pending": pending,
                    "reserved_floor": self.floor,
                }
            ),
        )

    def begin(
        self,
        operation: str,
        coordinate: dict[str, Any],
        reservation: dict[str, int | float],
    ) -> None:
        if not operation.strip():
            raise ValueError("remote operation must be named")
        if self.pending:
            raise RunnerGateError("a remote call is already pending")
        self.current = _reservation(reservation)
        self._state(
            {
                "sequence": self.sequence,
                "operation": operation,
                "coordinate": coordinate,
                "reservation": self.current,
            }
        )
        self.pending = True

    def complete(self, evidence: dict[str, Any]) -> None:
        """Call only after the operation's scientific evidence is durable."""

        if not self.pending:
            raise RunnerGateError("no remote call is pending")
        append_jsonl(
            self.events_path,
            {"sequence": self.sequence, "status": "completed", **evidence},
        )
        self.floor = {
            key: self.floor[key] + self.current[key]
            for key in (*_TOKEN_KEYS, "fixed_usd")
        }
        self.sequence += 1
        self._state(None)
        self.pending = False


__all__ = ["RemoteJournal"]
