"""Authenticated local launch envelope for Pilot-0 source artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from duraseed.data.leakage import (
    LeakageAuditReport,
    LeakageCode,
    audit_leakage,
)
from duraseed.data.sealing import inspect_seal
from duraseed.provenance import canonical_json_hash, sha256_bytes, validate_sha256_id
from duraseed.runners import RunnerGateError
from duraseed.runners.boundary_launch import authenticate_live_smoke
from duraseed.schemas import StrictModel


SOURCE_BUNDLE_SCHEMA = "duraseed-pilot0-source-bundle-v1"
AUTHORIZATION_SCHEMA = "duraseed-pilot0-launch-authorization-v1"
PILOT_AUTHORIZED_USD = 600.0


class Pilot0SeedSourceIDs(StrictModel):
    seed: int
    prompt_pool_artifact_id: str
    a_rl_train_manifest_id: str
    a_monitor_manifest_id: str
    a_cadence_manifest_id: str
    a_validation_manifest_id: str
    b_train_manifest_id: str
    b_validation_manifest_id: str

    @field_validator(
        "prompt_pool_artifact_id",
        "a_rl_train_manifest_id",
        "a_monitor_manifest_id",
        "a_cadence_manifest_id",
        "a_validation_manifest_id",
        "b_train_manifest_id",
        "b_validation_manifest_id",
    )
    @classmethod
    def ids_are_hashes(cls, value: str) -> str:
        return validate_sha256_id(value)


class Pilot0SourceBundle(StrictModel):
    schema_version: Literal["duraseed-pilot0-source-bundle-v1"]
    status: Literal["ready_for_authorization"]
    project_id: str
    git_commit: str
    resolved_config_hash: str
    completed_live_smoke_sha256: str
    completed_live_smoke_run_id: str
    completed_live_smoke_finished_at_utc: str
    post_calibration_billing_sha256: str
    post_calibration_raw_billing_sha256: str
    billing_cutoff_utc: str
    latest_calibration_finished_at_utc: str
    uncommitted_grant_balance_usd: float = Field(ge=0)
    protected_reserve_usd: float = Field(ge=0)
    m0_sampler_path: str
    m0_state_path: str
    m0_selection_sha256: str
    m0_ttl_sha256: str
    panel_artifact_sha256: str
    teacher_recipe_artifact_sha256: str
    acquisition_artifact_sha256: str
    stage_b_recipe_artifact_sha256: str
    visible_leakage_sha256: str
    sealed_b_test_envelope_sha256: str
    sealed_b_test_plaintext_sha256: str
    seed_sources: tuple[Pilot0SeedSourceIDs, ...]

    @field_validator(
        "project_id",
        "git_commit",
        "completed_live_smoke_run_id",
        "completed_live_smoke_finished_at_utc",
        "billing_cutoff_utc",
        "latest_calibration_finished_at_utc",
        "m0_sampler_path",
        "m0_state_path",
    )
    @classmethod
    def text_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Pilot-0 source identity must not be empty")
        return value

    @field_validator(
        "resolved_config_hash",
        "completed_live_smoke_sha256",
        "post_calibration_billing_sha256",
        "post_calibration_raw_billing_sha256",
        "m0_selection_sha256",
        "m0_ttl_sha256",
        "panel_artifact_sha256",
        "teacher_recipe_artifact_sha256",
        "acquisition_artifact_sha256",
        "stage_b_recipe_artifact_sha256",
        "visible_leakage_sha256",
        "sealed_b_test_envelope_sha256",
        "sealed_b_test_plaintext_sha256",
    )
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)


class Pilot0LaunchAuthorization(StrictModel):
    schema_version: Literal["duraseed-pilot0-launch-authorization-v1"]
    status: Literal["accepted"]
    source_bundle_sha256: str
    post_calibration_billing_sha256: str
    project_id: str
    authorizer: str
    authorized_at_utc: str
    authorized_usd: Literal[600.0]
    no_rerun_authorized: Literal[True]

    @field_validator("project_id", "authorizer", "authorized_at_utc")
    @classmethod
    def text_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Pilot-0 authorization identity must not be empty")
        return value

    @field_validator("source_bundle_sha256", "post_calibration_billing_sha256")
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)


@dataclass(frozen=True, slots=True)
class Pilot0SourceAuthentication:
    bundle: Pilot0SourceBundle
    authorization: Pilot0LaunchAuthorization
    bundle_sha256: str
    authorization_sha256: str
    sealed_envelope_sha256: str


def _utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise RunnerGateError(f"Pilot-0 {label} is not an ISO UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise RunnerGateError(f"Pilot-0 {label} is not UTC")
    return parsed


def _read_model(path: Path, model: type[StrictModel], label: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        value = model.model_validate_json(raw)
    except (OSError, ValueError) as error:
        raise RunnerGateError(f"Pilot-0 {label} is missing or invalid") from error
    return value, raw


def load_pilot0_source_authentication(
    bundle_path: str | Path,
    authorization_path: str | Path,
    sealed_b_test_path: str | Path,
    *,
    completed_live_smoke_path: str | Path,
    post_calibration_billing_path: str | Path,
    post_calibration_raw_billing_path: str | Path,
    m0_selection_path: str | Path,
    m0_ttl_path: str | Path,
) -> Pilot0SourceAuthentication:
    """Authenticate authorization and public seal metadata without opening b_test."""

    bundle, bundle_raw = _read_model(
        Path(bundle_path), Pilot0SourceBundle, "source bundle"
    )
    authorization, authorization_raw = _read_model(
        Path(authorization_path), Pilot0LaunchAuthorization, "launch authorization"
    )
    bundle_hash = sha256_bytes(bundle_raw)
    if (
        authorization.source_bundle_sha256 != bundle_hash
        or authorization.post_calibration_billing_sha256
        != bundle.post_calibration_billing_sha256
        or authorization.project_id != bundle.project_id
    ):
        raise RunnerGateError("Pilot-0 authorization does not bind the source bundle")
    smoke_run, smoke_hash, smoke_finished = authenticate_live_smoke(
        completed_live_smoke_path, project_id=bundle.project_id
    )
    sampler, state, step, selection_hash, ttl_hash = _load_m0_evidence(
        m0_selection_path, m0_ttl_path
    )
    billing, billing_raw = _read_object(
        Path(post_calibration_billing_path), "post-calibration billing"
    )
    try:
        raw_billing = Path(post_calibration_raw_billing_path).read_bytes()
    except OSError as error:
        raise RunnerGateError("Pilot-0 raw billing evidence is unreadable") from error
    cutoff = _utc(bundle.billing_cutoff_utc, "billing cutoff")
    latest = _utc(
        bundle.latest_calibration_finished_at_utc, "latest calibration finish"
    )
    if cutoff < latest:
        raise RunnerGateError("Pilot-0 billing cutoff predates calibration completion")
    if _utc(authorization.authorized_at_utc, "authorization time") < cutoff:
        raise RunnerGateError("Pilot-0 authorization predates its billing cutoff")
    remaining = bundle.uncommitted_grant_balance_usd
    reserve = bundle.protected_reserve_usd
    if reserve < 0.2 * remaining or remaining - PILOT_AUTHORIZED_USD < reserve:
        raise RunnerGateError(
            "Pilot-0 authorization would consume the protected reserve"
        )
    source_hashes = billing.get("source_artifact_sha256s")
    if (
        smoke_run != bundle.completed_live_smoke_run_id
        or smoke_hash != bundle.completed_live_smoke_sha256
        or smoke_finished
        != _utc(bundle.completed_live_smoke_finished_at_utc, "smoke finish")
        or (sampler, state, step) != (bundle.m0_sampler_path, bundle.m0_state_path, 2)
        or selection_hash != bundle.m0_selection_sha256
        or ttl_hash != bundle.m0_ttl_sha256
        or sha256_bytes(billing_raw) != bundle.post_calibration_billing_sha256
        or sha256_bytes(raw_billing) != bundle.post_calibration_raw_billing_sha256
        or billing.get("schema_version") != "duraseed-post-calibration-billing-v1"
        or billing.get("status") != "reconciled"
        or billing.get("project_id") != bundle.project_id
        or billing.get("source_run_id") != smoke_run
        or billing.get("source_acceptance_sha256") != smoke_hash
        or billing.get("raw_usage_sha256") != bundle.post_calibration_raw_billing_sha256
        or billing.get("raw_usage_cutoff_utc") != bundle.billing_cutoff_utc
        or billing.get("latest_calibration_finished_at_utc")
        != bundle.latest_calibration_finished_at_utc
        or billing.get("remaining_balance_usd") != remaining
        or billing.get("protected_reserve_usd") != reserve
        or billing.get("remaining_balance_verified") is not True
        or billing.get("protected_reserve_survives") is not True
        or billing.get("pilot0_authorization_usd") != PILOT_AUTHORIZED_USD
        or not isinstance(billing.get("raw_billing_entry_count"), int)
        or billing["raw_billing_entry_count"] < 1
        or not isinstance(source_hashes, list)
        or set(source_hashes)
        != {
            bundle.teacher_recipe_artifact_sha256,
            bundle.acquisition_artifact_sha256,
        }
    ):
        raise RunnerGateError("Pilot-0 smoke, M0, or billing source is unauthenticated")
    sealed_path = Path(sealed_b_test_path)
    try:
        sealed_raw = sealed_path.read_bytes()
        seal = inspect_seal(sealed_path)
    except (OSError, ValueError) as error:
        raise RunnerGateError("Pilot-0 b_test seal is missing or invalid") from error
    sealed_hash = sha256_bytes(sealed_raw)
    if (
        seal.declared_split != "b_test"
        or sealed_hash != bundle.sealed_b_test_envelope_sha256
        or f"sha256:{seal.plaintext_sha256}" != bundle.sealed_b_test_plaintext_sha256
    ):
        raise RunnerGateError("Pilot-0 b_test seal differs from the source bundle")
    return Pilot0SourceAuthentication(
        bundle,
        authorization,
        bundle_hash,
        sha256_bytes(authorization_raw),
        sealed_hash,
    )


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as error:
        raise RunnerGateError(f"Pilot-0 {label} is missing or invalid") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"Pilot-0 {label} is not an object")
    return value, raw


def _load_m0_evidence(
    selection_path: str | Path, ttl_path: str | Path
) -> tuple[str, str, int, str, str]:
    """Authenticate the selected step-2 M0 without importing calibration code."""

    selection, selection_raw = _read_object(Path(selection_path), "M0 selection")
    ttl, ttl_raw = _read_object(Path(ttl_path), "M0 TTL")
    sampler = selection.get("selected_sampler_checkpoint_path")
    state = selection.get("selected_state_checkpoint_path")
    step = selection.get("selected_training_step")
    expected = ((sampler, "sampler"), (state, "training"))
    if (
        selection.get("status") != "completed"
        or selection.get("scientific_m0_selected") is not True
        or selection.get("pilot_started") is not False
        or type(sampler) is not str
        or type(state) is not str
        or step != 2
        or sampler.split("/sampler_weights/", 1)[0] != state.split("/weights/", 1)[0]
        or any(
            not isinstance(ttl.get(path), dict)
            or ttl[path].get("checkpoint_type") != kind
            or ttl[path].get("expires_at") is not None
            or ttl[path].get("ttl_seconds") is not None
            for path, kind in expected
        )
    ):
        raise RunnerGateError("Pilot-0 selected M0 is not the non-expiring step-2 pair")
    return sampler, state, step, sha256_bytes(selection_raw), sha256_bytes(ttl_raw)


def seed_source_ids(source: Any) -> Pilot0SeedSourceIDs:
    pools = source.prompt_pools
    return Pilot0SeedSourceIDs(
        seed=source.seed,
        prompt_pool_artifact_id=pools.artifact.artifact_id,
        a_rl_train_manifest_id=pools.a_rl_train_manifest.manifest_id,
        a_monitor_manifest_id=pools.a_monitor_manifest.manifest_id,
        a_cadence_manifest_id=source.a_cadence.manifest_id,
        a_validation_manifest_id=source.a_validation.manifest_id,
        b_train_manifest_id=source.b_train.manifest_id,
        b_validation_manifest_id=source.b_validation.manifest_id,
    )


def _maps_semantic_leakage(train: Any, validation: Any) -> dict[str, Any]:
    """Require distinct MAPS tasks while recording unavoidable state reuse."""

    report = audit_leakage({"b_train": train, "b_validation": validation})
    numeric_only = {
        LeakageCode.DUPLICATE_OPERANDS_TARGET,
        LeakageCode.TEACHER_EVALUATION_NUMERIC_OVERLAP,
    }
    semantic_findings = tuple(
        finding
        for finding in report.findings
        if finding.code not in numeric_only
        and not (
            finding.code is LeakageCode.TASK_ACROSS_SPLITS
            and finding.key.startswith("numeric:")
        )
    )
    semantic = LeakageAuditReport(
        report.record_count,
        report.audited_splits,
        semantic_findings,
    ).assert_clean()
    payload = semantic.to_dict()
    payload["numeric_state_reuse_diagnostic"] = {
        code.value: sum(finding.code is code for finding in report.findings)
        for code in numeric_only
    }
    return payload


def visible_leakage_hash(seed_sources: tuple[Any, ...]) -> str:
    """Audit every visible Pilot split while treating paired replicas separately."""

    reports = {}
    for source in sorted(seed_sources, key=lambda row: row.seed):
        tces = audit_leakage(
            {
                "a_rl_train": source.prompt_pools.a_rl_train_manifest,
                "a_monitor": source.prompt_pools.a_monitor_manifest,
                "a_validation": source.a_validation,
            }
        ).assert_clean()
        maps = _maps_semantic_leakage(source.b_train, source.b_validation)
        reports[str(source.seed)] = {"tces": tces.to_dict(), "maps": maps}
    return canonical_json_hash(reports)


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "PILOT_AUTHORIZED_USD",
    "SOURCE_BUNDLE_SCHEMA",
    "Pilot0LaunchAuthorization",
    "Pilot0SeedSourceIDs",
    "Pilot0SourceAuthentication",
    "Pilot0SourceBundle",
    "load_pilot0_source_authentication",
    "seed_source_ids",
    "visible_leakage_hash",
]
