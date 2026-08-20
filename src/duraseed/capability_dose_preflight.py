"""Immutable launch identity for the frozen capability-dose run."""

from __future__ import annotations

from typing import Any

from duraseed.provenance import canonical_json_value
from duraseed.training.capability_dose_evidence import (
    CADENCE_UPDATES,
    CONFIRMATION_SUCCESSES,
    DOSE_LEARNING_RATE,
    EPOCH_UPDATES,
    MAX_CONFIRMATIONS,
    MAX_UPDATES,
    MAXIMUM_LENGTH_STOP_RATE,
    MAXIMUM_LOOP_FRACTION,
    MAXIMUM_VALID_TAG_DROP,
    THETA_SUCCESSES,
)


def capability_dose_preflight(inputs: Any) -> dict[str, Any]:
    budget = inputs.dose_budget
    return canonical_json_value(
        {
            "schema_version": "duraseed-capability-dose-preflight-v1",
            "run_id": inputs.run_id,
            "project_id": inputs.project_id,
            "git_commit": inputs.git_commit,
            "charter_sha256": inputs.charter_sha256,
            "resolved_config_hash": inputs.config.resolved_config_hash(),
            "model_id": inputs.config.tinker.model_id,
            "renderer": inputs.config.tinker.renderer_name,
            "lora_rank": inputs.config.tinker.lora_rank,
            "m0_selection_sha256": inputs.sources.m0_selection_sha256,
            "m0_ttl_sha256": inputs.sources.m0_ttl_sha256,
            "boundary_bundle_sha256": inputs.sources.boundary_bundle_sha256,
            "panel_split_authorization_sha256": inputs.panel_split_authorization_sha256,
            "panel_split_equivalence_sha256": inputs.panel_split_equivalence_sha256,
            "source_manifest_ids": {
                "a_rl_train": inputs.prompt_pools.a_rl_train_manifest.manifest_id,
                "a_monitor": inputs.prompt_pools.a_monitor_manifest.manifest_id,
            },
            "dose": {
                "method": "B-S",
                "learning_rate": DOSE_LEARNING_RATE,
                "canonical_epoch_updates": EPOCH_UPDATES,
                "epochs": 6,
                "maximum_updates": MAX_UPDATES,
                "cadence_updates": CADENCE_UPDATES,
                "theta_successes": THETA_SUCCESSES,
                "confirmation_successes": CONFIRMATION_SUCCESSES,
                "maximum_confirmations": MAX_CONFIRMATIONS,
                "valid_tag_drop": MAXIMUM_VALID_TAG_DROP,
                "absolute_length_stop": MAXIMUM_LENGTH_STOP_RATE,
                "loop_fraction": MAXIMUM_LOOP_FRACTION,
                "loop_zero_denominator_passes": True,
            },
            "budget": {
                "token_ceiling": budget.tokens,
                "fixed_storage_usd": budget.fixed_storage_usd,
                "pinned_upper_usd": budget.upper_bound_usd,
                "authorized_usd": float(budget.cent_ceiling_usd),
            },
            "actual_lifetime_billing": inputs.actual_lifetime_billing.lineage,
            "lifetime_calibration_cap_usd": 300,
            "projected_lifetime_usd": (
                inputs.actual_lifetime_billing.actual_lifetime_spend_usd
                + float(budget.cent_ceiling_usd)
            ),
        }
    )


__all__ = ["capability_dose_preflight"]
