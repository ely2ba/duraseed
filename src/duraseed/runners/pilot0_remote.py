"""Paid-call wrappers and checkpoint-boundary resume for Pilot 0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.pilot0_contract import EPHEMERAL_SAMPLER_FIXED_USD, Pilot0Inputs
from duraseed.pilot0_integrity import bind_segment_evidence, verify_segment_evidence
from duraseed.provenance import canonical_json_bytes, canonical_json_hash, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import (
    CheckpointPair,
    RuntimeBundle,
    ZERO_TOKENS,
    TokenBudget,
    bind_model,
    create_sampler,
    restore_checkpoint,
    save_checkpoint,
)


def read_segment(directory: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    path = directory / "segment.json"
    if not path.exists():
        if (directory / "remote-calls.jsonl").exists():
            RemoteJournal(directory)
            raise RunnerGateError("incomplete Pilot-0 segment requires reconciliation")
        return None
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError("Pilot-0 segment artifact is unreadable") from error
    if any(
        value.get(name) != expected_value for name, expected_value in expected.items()
    ):
        raise RunnerGateError("completed Pilot-0 segment changed coordinates")
    try:
        call_state = json.loads((directory / "remote-call-state.json").read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(
            "completed Pilot-0 segment omitted its call state"
        ) from error
    if call_state.get("pending") is not None:
        raise RunnerGateError("completed Pilot-0 segment has a pending remote call")
    payload = dict(value)
    declared = payload.pop("artifact_sha256", None)
    if declared != sha256_bytes(canonical_json_bytes(payload)):
        raise RunnerGateError("Pilot-0 segment artifact hash mismatch")
    verify_segment_evidence(directory, payload)
    return value


def _tokens(value: TokenBudget) -> dict[str, int]:
    return {
        "prefill": value.prefill,
        "sample": value.sample,
        "train": value.train,
    }


def write_segment(
    directory: Path,
    value: dict[str, Any],
    *,
    ledger: Any | None = None,
) -> dict[str, Any]:
    payload = dict(value)
    if ledger is not None:
        payload["ledger_snapshot"] = {
            "committed": _tokens(ledger.committed),
            "observed": _tokens(ledger.observed),
            "committed_fixed_usd": ledger.committed_fixed_usd,
            "observed_fixed_usd": ledger.observed_fixed_usd,
        }
    payload = bind_segment_evidence(directory, payload)
    payload["artifact_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    atomic_write_bytes(directory / "segment.json", canonical_json_bytes(payload))
    return payload


def _budget(value: dict[str, Any]) -> TokenBudget:
    try:
        return TokenBudget(
            int(value["prefill"]), int(value["sample"]), int(value["train"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError("Pilot-0 ledger snapshot is malformed") from error


def hydrate_ledger(inputs: Pilot0Inputs, root: Path) -> None:
    """Restore cumulative reservations from the latest completed segment."""

    snapshots = []
    for path in root.rglob("segment.json"):
        segment = read_segment(path.parent, {})
        assert segment is not None
        snapshot = segment.get("ledger_snapshot")
        if not isinstance(snapshot, dict):
            raise RunnerGateError(
                "completed Pilot-0 segment omitted its ledger snapshot"
            )
        snapshots.append(snapshot)
    if not snapshots:
        return
    latest = max(
        snapshots,
        key=lambda row: (
            _budget(row["committed"]).prefill
            + _budget(row["committed"]).sample
            + _budget(row["committed"]).train,
            float(row["committed_fixed_usd"]),
        ),
    )
    committed = _budget(latest["committed"])
    observed = _budget(latest["observed"])
    fixed = float(latest["committed_fixed_usd"])
    if float(latest["observed_fixed_usd"]) != fixed:
        raise RunnerGateError("Pilot-0 completed-call fixed spend is inconsistent")
    current = (
        inputs.ledger.committed,
        inputs.ledger.observed,
        inputs.ledger.committed_fixed_usd,
        inputs.ledger.observed_fixed_usd,
    )
    expected = (committed, observed, fixed, fixed)
    if current == expected:
        return
    if current != (ZERO_TOKENS, ZERO_TOKENS, 0.0, 0.0):
        raise RunnerGateError("Pilot-0 resume ledger differs from persisted spend")
    inputs.ledger.reserve_call(committed, fixed_usd=fixed)
    inputs.ledger.settle_call(observed)


async def restore_runtime(
    inputs: Pilot0Inputs,
    journal: RemoteJournal,
    *,
    path: str,
    full_state: bool,
    coordinate: dict[str, Any],
) -> RuntimeBundle:
    journal.begin(
        "pilot0-restore",
        coordinate,
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    client = await restore_checkpoint(
        inputs.runtime,
        path,
        full_state=full_state,
        ledger=inputs.ledger,
        user_metadata={
            "gate": "pilot0",
            **{key: str(value) for key, value in coordinate.items()},
        },
    )
    runtime = bind_model(inputs.runtime.sdk, inputs.runtime.service, client)
    journal.complete(
        {"operation": "pilot0-restore", "path": path, "full_state": full_state}
    )
    return runtime


async def sampler_for_path(
    inputs: Pilot0Inputs,
    journal: RemoteJournal,
    *,
    path: str,
    coordinate: dict[str, Any],
) -> object:
    journal.begin(
        "pilot0-create-sampler",
        coordinate,
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    sampler = await create_sampler(
        inputs.runtime, ledger=inputs.ledger, checkpoint_path=path
    )
    journal.complete({"operation": "pilot0-create-sampler", "path": path})
    return sampler


async def ephemeral_sampler(
    inputs: Pilot0Inputs,
    runtime: RuntimeBundle,
    journal: RemoteJournal,
    *,
    coordinate: dict[str, Any],
) -> tuple[object, str]:
    journal.begin(
        "pilot0-ephemeral-sampler",
        coordinate,
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": 0,
            "fixed_usd": EPHEMERAL_SAMPLER_FIXED_USD,
        },
    )
    inputs.ledger.reserve_call(ZERO_TOKENS, fixed_usd=EPHEMERAL_SAMPLER_FIXED_USD)
    try:
        sampler = await runtime.model.save_weights_and_get_sampling_client_async()
    except Exception:
        inputs.ledger.abort_call()
        raise
    inputs.ledger.settle_call(ZERO_TOKENS)
    coordinate_hash = canonical_json_hash(coordinate).removeprefix("sha256:")
    path = f"ephemeral:{inputs.run_id}:{coordinate_hash}"
    journal.complete({"operation": "pilot0-ephemeral-sampler", "path": path})
    return sampler, path


async def save_pair(
    inputs: Pilot0Inputs,
    runtime: RuntimeBundle,
    journal: RemoteJournal,
    *,
    name: str,
    ttl_seconds: int | None,
    coordinate: dict[str, Any],
) -> CheckpointPair:
    storage_usd = 1.0 if ttl_seconds is None else 0.1
    journal.begin(
        "pilot0-save-checkpoint-pair",
        coordinate,
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": 0,
            "fixed_usd": storage_usd,
        },
    )
    pair = await save_checkpoint(
        runtime,
        name=name,
        ttl_seconds=ttl_seconds,
        ledger=inputs.ledger,
        reserved_storage_usd=storage_usd,
    )
    journal.complete(
        {
            "operation": "pilot0-save-checkpoint-pair",
            "sampler_path": pair.sampler_path,
            "state_path": pair.state_path,
            "ttl_seconds": ttl_seconds,
        }
    )
    return pair


__all__ = [
    "ephemeral_sampler",
    "hydrate_ledger",
    "read_segment",
    "restore_runtime",
    "sampler_for_path",
    "save_pair",
    "write_segment",
]
