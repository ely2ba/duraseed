"""Authenticate the historical M0 source for the accepted max-token decision."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

from duraseed.provenance import sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.training.acquisition_freeze import (
    FROZEN_PROTOCOL_MAX_TOKENS,
    MAX_TOKEN_GENERATIONS_SHA256,
    MAX_TOKEN_REFERENCE_RUN_ID,
    MAX_TOKEN_REWARDS_SHA256,
    MAX_TOKEN_SELECTION_RULE,
    MAX_TOKEN_SUMMARY_SHA256,
    MaxTokenCapObservation,
    MaxTokenFreezeEvidence,
    MaxTokenReferenceEvidence,
)


REFERENCE_TIME_UTC = "2026-08-09T20:43:37Z"
REFERENCE_DESCRIPTOR = {
    "run_id": MAX_TOKEN_REFERENCE_RUN_ID,
    "reference_time_utc": REFERENCE_TIME_UTC,
    "source_split": "a_validation",
    "seed": 5,
    "training_step": 2,
    "reference_max_tokens": FROZEN_PROTOCOL_MAX_TOKENS,
    "sample_count": 128,
    "summary_sha256": MAX_TOKEN_SUMMARY_SHA256,
    "generations_sha256": MAX_TOKEN_GENERATIONS_SHA256,
    "rewards_sha256": MAX_TOKEN_REWARDS_SHA256,
}


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"{label} is not an object")
    return value, raw


def _jsonl(path: Path, label: str) -> tuple[tuple[dict[str, Any], ...], bytes]:
    try:
        raw = path.read_bytes()
        values = tuple(json.loads(line) for line in raw.splitlines())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid {label}") from error
    if any(not isinstance(value, dict) for value in values):
        raise RunnerGateError(f"{label} contains a non-object row")
    return values, raw


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RunnerGateError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RunnerGateError(f"{label} is malformed") from error
    if parsed.utcoffset() != timedelta(0):
        raise RunnerGateError(f"{label} is not UTC")
    return parsed


def _reference(summary_path: Path) -> MaxTokenReferenceEvidence:
    if (
        summary_path.name != "tces_cap_summary.json"
        or summary_path.parent.name != MAX_TOKEN_REFERENCE_RUN_ID
    ):
        raise RunnerGateError("max-token source is not the frozen M0 reference run")
    summary, summary_raw = _object(summary_path, "M0 max-token summary")
    generations, generations_raw = _jsonl(
        summary_path.with_name("generations.jsonl"), "M0 max-token generations"
    )
    rewards, rewards_raw = _jsonl(
        summary_path.with_name("rewards.jsonl"), "M0 max-token rewards"
    )
    hashes = (
        sha256_bytes(summary_raw),
        sha256_bytes(generations_raw),
        sha256_bytes(rewards_raw),
    )
    if hashes != (
        MAX_TOKEN_SUMMARY_SHA256,
        MAX_TOKEN_GENERATIONS_SHA256,
        MAX_TOKEN_REWARDS_SHA256,
    ):
        raise RunnerGateError("M0 max-token source hash differs")
    sample_ids = tuple(row.get("sample_id") for row in generations)
    if (
        len(generations) != 128
        or len(set(sample_ids)) != 128
        or {row.get("item_index") for row in generations} != set(range(128))
        or len({row.get("sampler_checkpoint_path") for row in generations}) != 1
        or any(
            row.get("run_id") != MAX_TOKEN_REFERENCE_RUN_ID
            or row.get("source_split") != "a_validation"
            or row.get("seed") != 5
            or row.get("training_step") != 2
            or row.get("checkpoint_stage") != "m0"
            or row.get("sampling_max_tokens") != FROZEN_PROTOCOL_MAX_TOKENS
            or row.get("origin_sampler_checkpoint_path")
            != row.get("sampler_checkpoint_path")
            for row in generations
        )
    ):
        raise RunnerGateError("M0 max-token generations are not one 4096 reference")
    if (
        len(rewards) != 128
        or {row.get("sample_id") for row in rewards} != set(sample_ids)
        or sum(row.get("reward") == 1.0 for row in rewards) != 8
    ):
        raise RunnerGateError("M0 max-token rewards do not join to the reference")
    try:
        observations = tuple(
            MaxTokenCapObservation(
                row["max_tokens"],
                row["censored_answer_count"],
                row["censored_exact_success_count"],
                row["passes"],
            )
            for row in summary["caps"]
        )
        reference = MaxTokenReferenceEvidence(
            *hashes,
            MAX_TOKEN_REFERENCE_RUN_ID,
            len(generations),
            "a_validation",
            5,
            2,
            FROZEN_PROTOCOL_MAX_TOKENS,
            summary["answer_termination_count"],
            summary["exact_success_count"],
            observations,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError("M0 max-token summary is malformed") from error
    if (
        summary.get("maximum_observed_censor_rate") != 0.05
        or summary.get("minimum_answer_terminations") != 64
        or summary.get("selection_rule") != MAX_TOKEN_SELECTION_RULE
        or summary.get("selected_common_max_tokens") != FROZEN_PROTOCOL_MAX_TOKENS
        or summary.get("status") != "selected"
        or summary.get("success_only_guards_against_censoring_observed_exact_success")
        is not True
    ):
        raise RunnerGateError("M0 max-token summary rule differs")
    return reference


def load_ratification(
    specification_path: str | Path,
    authorization_path: str | Path,
    summary_path: str | Path,
    *,
    accepted_specification_sha256: str,
    accepted_authorization_sha256: str,
) -> MaxTokenFreezeEvidence:
    """Load an accepted ratification while keeping source and apply-to distinct."""

    specification, specification_raw = _object(
        Path(specification_path), "max-token ratification specification"
    )
    authorization, authorization_raw = _object(
        Path(authorization_path), "max-token ratification authorization"
    )
    specification_hash = sha256_bytes(specification_raw)
    authorization_hash = sha256_bytes(authorization_raw)
    expected_spec_keys = {
        "schema_version",
        "status",
        "specified_at_utc",
        "reference",
        "candidate_max_tokens",
        "maximum_reference_censor_rate",
        "minimum_answer_terminations",
        "require_zero_censored_exact_successes",
        "provisional_failure_inferred_from_max_tokens",
        "selected_max_tokens",
        "common_apply_to_methods",
    }
    if (
        specification_hash != accepted_specification_sha256
        or authorization_hash != accepted_authorization_sha256
        or set(specification) != expected_spec_keys
        or specification.get("schema_version")
        != "duraseed-acquisition-max-token-ratification-v1"
        or specification.get("status") != "ratified_from_preexisting_m0_evidence"
        or specification.get("reference") != REFERENCE_DESCRIPTOR
        or authorization.get("schema_version")
        != "duraseed-acquisition-max-token-authorization-v1"
        or authorization.get("status") != "accepted"
        or authorization.get("specification_sha256") != specification_hash
        or not isinstance(authorization.get("authorizer"), str)
        or not authorization["authorizer"].strip()
    ):
        raise RunnerGateError("max-token ratification is not authenticated")
    reference_time = _utc(REFERENCE_TIME_UTC, "reference time")
    specified_at = _utc(specification.get("specified_at_utc"), "specification time")
    authorized_at = _utc(authorization.get("authorized_at_utc"), "authorization time")
    if not reference_time < specified_at < authorized_at:
        raise RunnerGateError("max-token ratification chronology is invalid")
    reference = _reference(Path(summary_path))
    try:
        return MaxTokenFreezeEvidence(
            specification_hash,
            reference.summary_sha256,
            authorization_hash,
            reference,
            tuple(specification["candidate_max_tokens"]),
            specification["maximum_reference_censor_rate"],
            specification["minimum_answer_terminations"],
            specification["require_zero_censored_exact_successes"],
            specification["provisional_failure_inferred_from_max_tokens"],
            specification["selected_max_tokens"],
            tuple(specification["common_apply_to_methods"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError("max-token ratification is malformed") from error


__all__ = ("load_ratification",)
