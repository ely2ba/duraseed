"""Stable preflight identity and restart billing binding for calibration."""

from __future__ import annotations

from typing import Any

from duraseed.provenance import canonical_json_value
from duraseed.runners import RunnerGateError
from duraseed.teacher_exposure_spec import (
    AMENDED_AGGREGATE_CAP_USD,
    AMENDED_STAGE_A_CAP_USD,
    AMENDED_STAGE_A_TOKEN_CEILINGS,
    DIRECT_M0_TEACHER_CAP_USD,
    LIFETIME_CALIBRATION_CAP_USD,
    ORIGINAL_TEACHER_CAP_USD,
)


def validate_repair_allocation(inputs: Any) -> None:
    child_cap = (
        inputs.teacher_ledger.authorized_usd + inputs.stage_a_ledger.authorized_usd
    )
    parent_spend = inputs.parent_teacher_evidence.lifetime_sunk_usd
    teacher_spend = inputs.parent_teacher_evidence.teacher_lifetime_sunk_usd
    teacher_limits = inputs.teacher_ledger.limits
    stage_a_limits = inputs.stage_a_ledger.limits
    if (
        inputs.teacher_ledger.authorized_usd != DIRECT_M0_TEACHER_CAP_USD
        or inputs.stage_a_ledger.authorized_usd != AMENDED_STAGE_A_CAP_USD
        or child_cap != AMENDED_AGGREGATE_CAP_USD
        or (teacher_limits.prefill, teacher_limits.sample, teacher_limits.train)
        != (0, 0, 0)
        or (stage_a_limits.prefill, stage_a_limits.sample, stage_a_limits.train)
        != AMENDED_STAGE_A_TOKEN_CEILINGS
        or parent_spend + child_cap > LIFETIME_CALIBRATION_CAP_USD
        or teacher_spend > ORIGINAL_TEACHER_CAP_USD
    ):
        raise ValueError("direct-M0 calibration allocations exceed a frozen cap")


def calibration_preflight(inputs: Any, schema_version: str) -> dict[str, Any]:
    child_cap = (
        inputs.teacher_ledger.authorized_usd + inputs.stage_a_ledger.authorized_usd
    )
    return canonical_json_value(
        {
            "schema_version": schema_version,
            "run_id": inputs.run_id,
            "project_id": inputs.project_id,
            "cost_caps_usd": {
                "teacher-dose": inputs.teacher_ledger.authorized_usd,
                "teacher-allocation": 0,
                "stage-a": inputs.stage_a_ledger.authorized_usd,
                "total": child_cap,
            },
            "lifetime_calibration_cap_usd": LIFETIME_CALIBRATION_CAP_USD,
            "lifetime_worst_case_usd": (
                inputs.parent_teacher_evidence.lifetime_sunk_usd + child_cap
            ),
            "stage_a_origin": "direct-m0",
            "parent_calibration": inputs.parent_teacher_evidence.lineage,
            "prior_repair": inputs.parent_teacher_evidence.prior_repair_lineage,
            "interrupted_m1": inputs.parent_teacher_evidence.m1_lineage,
            "prior_direct_stage_a": (
                inputs.parent_teacher_evidence.prior_stage_a_lineage
            ),
            "boundary_config_sha256": inputs.sources.boundary_config_sha256,
            "current_config_sha256": inputs.sources.current_config_sha256,
            "nonprotocol_config_sha256": inputs.sources.nonprotocol_config_sha256,
            "resolved_config_hash": inputs.config.resolved_config_hash(),
            "model_id": inputs.config.tinker.model_id,
            "renderer": inputs.config.tinker.renderer_name,
            "lora_rank": inputs.config.tinker.lora_rank,
            "m0_selection_sha256": inputs.sources.m0_selection_sha256,
            "m0_ttl_sha256": inputs.sources.m0_ttl_sha256,
            "boundary_bundle_sha256": inputs.sources.boundary_bundle_sha256,
            "panel_split_authorization_sha256": (
                inputs.panel_split_authorization_sha256
            ),
            "panel_split_equivalence_sha256": inputs.panel_split_equivalence_sha256,
            "precalibration_billing_sha256": inputs.precalibration_billing_sha256,
            "precalibration_raw_billing_sha256": (
                inputs.precalibration_raw_billing_sha256
            ),
            "live_smoke_sha256": inputs.smoke.artifact_sha256,
            "max_token_specification_sha256": inputs.max_tokens.specification_sha256,
            "max_token_specification_authorization_sha256": (
                inputs.max_tokens.specification_authorization_sha256
            ),
            "max_token_evidence_sha256": inputs.max_tokens.evidence_sha256,
            "source_manifest_ids": {
                "a_rl_train": inputs.prompt_pools.a_rl_train_manifest.manifest_id,
                "a_monitor": inputs.prompt_pools.a_monitor_manifest.manifest_id,
            },
        }
    )


def validate_restart_reconciliations(inputs: Any, preflight_sha256: str) -> None:
    maxima = {
        action: max(
            (
                row.cumulative_billed_usd
                for row in inputs.reconciled_restarts
                if row.action == action
            ),
            default=0.0,
        )
        for action in ("teacher-dose", "stage-a")
    }
    aggregate = max(
        (row.aggregate_billed_usd for row in inputs.reconciled_restarts), default=0.0
    )
    if (
        any(
            row.run_id != inputs.run_id
            or row.action not in {"teacher-dose", "stage-a"}
            or row.project_id != inputs.project_id
            or row.preflight_sha256 != preflight_sha256
            for row in inputs.reconciled_restarts
        )
        or sum(maxima.values()) > aggregate
        or maxima["teacher-dose"] > inputs.teacher_ledger.authorized_usd
        or maxima["stage-a"] > inputs.stage_a_ledger.authorized_usd
        or aggregate
        > inputs.teacher_ledger.authorized_usd + inputs.stage_a_ledger.authorized_usd
        or inputs.parent_teacher_evidence.lifetime_sunk_usd + aggregate
        > LIFETIME_CALIBRATION_CAP_USD
    ):
        raise RunnerGateError("restart reconciliation differs from this launch")


__all__ = [
    "calibration_preflight",
    "validate_repair_allocation",
    "validate_restart_reconciliations",
]
