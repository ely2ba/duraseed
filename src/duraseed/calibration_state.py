"""Durable action-prefix state for acquisition calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, canonical_json_value, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runtime import TokenLedger


SCHEMA_VERSION = "duraseed-acquisition-calibration-v2"
ACTION_ORDER = ("stage-a",)
ACTION_FILES = {"stage-a": "acquisition-freeze.json"}


def artifact(
    action: str, cost: int | float, *, preflight_sha256: str, **payload: Any
) -> dict[str, Any]:
    return canonical_json_value(
        {
            "schema_version": SCHEMA_VERSION,
            "action": action,
            "cost_cap_usd": cost,
            "preflight_sha256": preflight_sha256,
            **payload,
        }
    )


def usage(ledger: TokenLedger) -> dict[str, Any]:
    return {
        "authorized_usd": ledger.authorized_usd,
        "committed_tokens": ledger.committed,
        "observed_tokens": ledger.observed,
        "committed_fixed_usd": ledger.committed_fixed_usd,
        "observed_fixed_usd": ledger.observed_fixed_usd,
        "committed_cost_usd": ledger.committed_cost_usd,
        "observed_cost_usd": ledger.observed_cost_usd,
    }


def write(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(
            f"cannot read calibration artifact: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"calibration artifact is not an object: {path.name}")
    return value


def existing(
    root: Path, preflight_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifacts: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    missing_seen = False
    state_path = root / "state.json"
    if not state_path.exists() and any(
        (root / filename).exists() for filename in ACTION_FILES.values()
    ):
        raise RunnerGateError("calibration action exists without its committed state")
    for action in ACTION_ORDER:
        path = root / ACTION_FILES[action]
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise RunnerGateError("calibration artifacts are not a dependency prefix")
        value = read(path)
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("action") != action
            or value.get("preflight_sha256") != preflight_sha256
        ):
            raise RunnerGateError(
                f"calibration artifact identity mismatch: {path.name}"
            )
        artifacts[action] = value
        hashes[action] = sha256_bytes(path.read_bytes())
    if state_path.exists():
        prior_state = read(state_path)
        prior_hashes = prior_state.get("artifact_sha256", {})
        if prior_state.get("preflight_sha256") != preflight_sha256:
            raise RunnerGateError("persisted calibration state differs from artifacts")
        for action, digest in prior_hashes.items():
            if hashes.get(action) != digest:
                raise RunnerGateError(f"persisted calibration hash mismatch: {action}")
        appended = tuple(action for action in artifacts if action not in prior_hashes)
        next_index = len(prior_hashes)
        if appended:
            if (
                len(appended) != 1
                or next_index >= len(ACTION_ORDER)
                or appended[0] != ACTION_ORDER[next_index]
            ):
                raise RunnerGateError(
                    "persisted calibration state differs from artifacts"
                )
            intent = read(root / f"commit-intent-{appended[0]}.json")
            if intent != {
                "action": appended[0],
                "artifact_sha256": hashes[appended[0]],
                "preflight_sha256": preflight_sha256,
            }:
                raise RunnerGateError(
                    "calibration action lacks its exact commit intent"
                )
    state = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if len(artifacts) == len(ACTION_ORDER) else "running",
        "completed_actions": list(artifacts),
        "artifact_sha256": hashes,
        "preflight_sha256": preflight_sha256,
    }
    return state, artifacts


def checkpoint(
    root: Path, preflight_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    state, artifacts = existing(root, preflight_sha256)
    write(root / "state.json", state)
    return state, artifacts


def commit_action(
    root: Path,
    action: str,
    value: Any,
    preflight_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Commit one prefix action with a recoverable exact-hash intent."""

    if action not in ACTION_FILES:
        raise ValueError("unknown calibration action")
    payload = canonical_json_bytes(value)
    digest = sha256_bytes(payload)
    write(
        root / f"commit-intent-{action}.json",
        {
            "action": action,
            "artifact_sha256": digest,
            "preflight_sha256": preflight_sha256,
        },
    )
    atomic_write_bytes(root / ACTION_FILES[action], payload)
    return checkpoint(root, preflight_sha256)


__all__ = [
    "ACTION_FILES",
    "SCHEMA_VERSION",
    "artifact",
    "checkpoint",
    "commit_action",
    "existing",
    "read",
    "usage",
    "write",
]
