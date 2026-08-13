"""Validate raw vendor rows referenced by a restart reconciliation."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from duraseed.runners import RunnerGateError


def validate_restart_billing_rows(events: list[Any], artifact: dict[str, Any]) -> None:
    session_id = artifact.get("failed_tinker_session_id")
    rows = [
        row
        for row in events
        if isinstance(row, dict) and row.get("session_id") == session_id
    ]
    if not rows:
        raise RunnerGateError("restart raw billing omits the failed session")
    cost_rows = [row for row in rows if "billed_cost_usd" in row]
    if cost_rows and len(cost_rows) != len(rows):
        raise RunnerGateError("restart raw billing exposes only partial vendor costs")
    if not cost_rows:
        return
    try:
        total = sum(
            (Decimal(str(row["billed_cost_usd"])) for row in cost_rows),
            start=Decimal("0"),
        )
        expected = Decimal(str(artifact["console_cumulative_billed_usd"]))
    except (InvalidOperation, KeyError, ValueError) as error:
        raise RunnerGateError("restart vendor cost rows are malformed") from error
    if (
        any(row.get("currency") != "USD" for row in cost_rows)
        or not total.is_finite()
        or total < 0
        or total != expected
    ):
        raise RunnerGateError("restart vendor costs differ from console reconciliation")


__all__ = ["validate_restart_billing_rows"]
