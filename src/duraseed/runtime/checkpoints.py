"""Checkpoint save, restore, and retained-TTL verification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence

from duraseed.runtime.client import RuntimeBundle
from duraseed.runtime.ledger import TokenLedger, ZERO_TOKENS


@dataclass(frozen=True, slots=True)
class CheckpointPair:
    sampler_path: str
    state_path: str
    ttl_seconds: int | None


@dataclass(frozen=True, slots=True)
class CheckpointTTL:
    path: str
    training_run_id: str
    expires_at: Any
    checkpoint_type: Any
    ttl_seconds: int | None


def _path(value: Any) -> str:
    path = value if isinstance(value, str) else getattr(value, "path", None)
    if not isinstance(path, str) or not path.strip():
        raise RuntimeError("Tinker returned an empty checkpoint path")
    return path


async def save_checkpoint(
    runtime: RuntimeBundle,
    *,
    name: str,
    ttl_seconds: int | None,
    ledger: TokenLedger,
    reserved_storage_usd: float,
) -> CheckpointPair:
    """Save sampler weights and full optimizer state as one bounded operation."""

    if not name.strip() or (ttl_seconds is not None and ttl_seconds < 1):
        raise ValueError("checkpoint name and positive-or-null TTL are required")
    ledger.reserve_call(ZERO_TOKENS, fixed_usd=reserved_storage_usd)
    try:
        sampler_future = await runtime.model.save_weights_for_sampler_async(
            f"{name}-sampler", ttl_seconds=ttl_seconds
        )
        state_future = await runtime.model.save_state_async(
            f"{name}-state", ttl_seconds=ttl_seconds, overwrite=False
        )
        sampler, state = await asyncio.gather(
            sampler_future.result_async(), state_future.result_async()
        )
        pair = CheckpointPair(_path(sampler), _path(state), ttl_seconds)
    except Exception:
        ledger.abort_call()
        raise
    ledger.settle_call(ZERO_TOKENS)
    return pair


async def save_sampler_checkpoint(
    runtime: RuntimeBundle,
    *,
    name: str,
    ttl_seconds: int | None,
    ledger: TokenLedger,
    reserved_storage_usd: float,
) -> str:
    """Save a sampler-only checkpoint for a calibration evaluation point."""

    if not name.strip() or (ttl_seconds is not None and ttl_seconds < 1):
        raise ValueError("checkpoint name and positive-or-null TTL are required")
    ledger.reserve_call(ZERO_TOKENS, fixed_usd=reserved_storage_usd)
    try:
        future = await runtime.model.save_weights_for_sampler_async(
            name, ttl_seconds=ttl_seconds
        )
        path = _path(await future.result_async())
    except Exception:
        ledger.abort_call()
        raise
    ledger.settle_call(ZERO_TOKENS)
    return path


async def restore_checkpoint(
    runtime: RuntimeBundle,
    path: str,
    *,
    full_state: bool,
    ledger: TokenLedger,
    user_metadata: dict[str, str] | None = None,
) -> Any:
    """Create a training client from full state or weights-only state."""

    if not path.strip():
        raise ValueError("checkpoint path must be nonempty")
    ledger.reserve_call(ZERO_TOKENS)
    try:
        operation = (
            runtime.service.create_training_client_from_state_with_optimizer_async
            if full_state
            else runtime.service.create_training_client_from_state_async
        )
        result = await operation(path, user_metadata=user_metadata)
    except Exception:
        ledger.abort_call()
        raise
    ledger.settle_call(ZERO_TOKENS)
    return result


async def set_and_verify_ttl(
    runtime: RuntimeBundle,
    rest_client: Any,
    paths: Sequence[str],
    *,
    ttl_seconds: int | None,
    ledger: TokenLedger,
) -> tuple[CheckpointTTL, ...]:
    if not paths or (ttl_seconds is not None and ttl_seconds < 1):
        raise ValueError("checkpoint paths and positive-or-null TTL are required")
    ledger.reserve_call(ZERO_TOKENS)
    by_run: dict[str, set[str]] = {}
    try:
        for path in paths:
            parsed = runtime.sdk.tinker.ParsedCheckpointTinkerPath.from_tinker_path(
                path
            )
            by_run.setdefault(parsed.training_run_id, set()).add(path)
            await rest_client.set_checkpoint_ttl_from_tinker_path_async(
                path, ttl_seconds
            )
        rows: list[CheckpointTTL] = []
        for run_id, expected in by_run.items():
            listing = await rest_client.list_checkpoints_async(run_id)
            available = {value.tinker_path: value for value in listing.checkpoints}
            for path in sorted(expected):
                value = available.get(path)
                expiry = None if value is None else value.expires_at
                expiry_mismatch = ttl_seconds is None and expiry is not None
                if ttl_seconds is not None:
                    expiry_mismatch = not isinstance(expiry, datetime)
                    if isinstance(expiry, datetime):
                        if expiry.tzinfo is None:
                            expiry_mismatch = True
                        else:
                            remaining = (expiry - datetime.now(UTC)).total_seconds()
                            tolerance = max(60.0, ttl_seconds * 0.05)
                            expiry_mismatch = abs(remaining - ttl_seconds) > tolerance
                if value is None or expiry_mismatch:
                    raise RuntimeError(f"checkpoint TTL could not be verified: {path}")
                rows.append(
                    CheckpointTTL(
                        path,
                        run_id,
                        value.expires_at,
                        value.checkpoint_type,
                        ttl_seconds,
                    )
                )
    except Exception:
        ledger.abort_call()
        raise
    ledger.settle_call(ZERO_TOKENS)
    return tuple(rows)


__all__ = [
    "CheckpointPair",
    "CheckpointTTL",
    "restore_checkpoint",
    "save_checkpoint",
    "save_sampler_checkpoint",
    "set_and_verify_ttl",
]
