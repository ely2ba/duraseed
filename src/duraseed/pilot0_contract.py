"""Frozen inputs and gates for the two-seed B-S/B-G Pilot 0."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duraseed.config import PilotConfig
from duraseed.data.manifests import (
    DatasetManifest,
    MAPSTaskManifestRecord,
    TCESTaskManifestRecord,
)
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.data.stage_a_prompt_pools import StageAPromptPoolBundle
from duraseed.provenance import canonical_json_hash, validate_sha256_id
from duraseed.pilot0_sources import (
    PILOT_AUTHORIZED_USD,
    Pilot0SourceAuthentication,
    seed_source_ids,
    visible_leakage_hash,
)
from duraseed.runners import RunnerGateError
from duraseed.runtime import (
    LORA_RANK,
    MODEL_ID,
    RENDERER_NAME,
    RuntimeBundle,
    TokenLedger,
)
from duraseed.training.stage_a_calibration import StageADurationDecisionStatus
from duraseed.training.teacher_dose import TeacherDoseDecisionStatus


PILOT_SEEDS = (11, 29)
METHODS = ("B-S", "B-G")
STAGE_A_GRID = (0, 10, 25, 50)
STAGE_B_GRID = (0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480)
STAGE_B_PROFILE = "shortest2_cap2"
STAGE_B_LEARNING_RATE = 3e-4
STAGE_B_MAX_TOKENS = 128
STAGE_B_DECISION_SHA256 = (
    "sha256:36015e1e08a0a96b4f4b0d7dd8480a679294c1cfc2a026d3ab96073c5378acf8"
)
PILOT_COST_CAP_USD = 600.0
EPHEMERAL_SAMPLER_FIXED_USD = 0.05
COMMON_RL_CONFIGURATION = (
    ("advantage_normalization", "group_mean_no_std"),
    ("constant_reward_groups", "skip_no_resample"),
    ("entropy_collapse_response", "hard_stop_protocol_wide_fallback"),
    ("kl_penalty", False),
    ("loss", "importance_sampling"),
    ("objective", "group_relative_exact_binary"),
)
ALL_METHODS = ("G-U", "G-B", "R-G", "B-S", "B-O", "B-G")
COMMON_RL_METHODS = ("G-U", "G-B", "R-G", "B-G")


@dataclass(frozen=True, slots=True)
class PilotSeedSources:
    seed: int
    prompt_pools: StageAPromptPoolBundle
    teacher_train: DatasetManifest
    a_validation: DatasetManifest
    b_train: DatasetManifest
    b_validation: DatasetManifest


@dataclass(frozen=True, slots=True)
class Pilot0Inputs:
    config: PilotConfig
    runtime: RuntimeBundle
    ledger: TokenLedger
    output_root: Path
    run_id: str
    git_commit: str
    project_id: str
    m0_sampler_path: str
    m0_state_path: str
    panel: FamilyPanelArtifact
    acquisition: Any
    teacher_recipe: Any
    acquisition_artifact_sha256: str
    teacher_recipe_artifact_sha256: str
    panel_artifact_sha256: str
    stage_b_recipe_artifact_sha256: str
    seed_sources: tuple[PilotSeedSources, ...]
    source_authentication: Pilot0SourceAuthentication


def _manifest(
    value: DatasetManifest,
    *,
    family: str,
    split: str,
    count: int,
) -> None:
    if (
        not isinstance(value, DatasetManifest)
        or value.task_family != family
        or value.split != split
        or value.record_count != count
    ):
        raise RunnerGateError(f"Pilot 0 requires exact {family}/{split} population")


def _validate_seed_sources(inputs: Pilot0Inputs, source: PilotSeedSources) -> None:
    if source.seed not in PILOT_SEEDS:
        raise RunnerGateError("Pilot 0 source uses a non-pilot seed")
    assignment = next(
        (
            row
            for row in inputs.panel.seed_block_assignments
            if row.training_seed == source.seed
        ),
        None,
    )
    artifact = source.prompt_pools.artifact
    if (
        assignment is None
        or artifact.calibration_seed != source.seed
        or artifact.family_panel_artifact_id != canonical_json_hash(inputs.panel)
        or artifact.targeted_panel != assignment.targeted_panel
        or artifact.sentinel_panel != assignment.sentinel_panel
    ):
        raise RunnerGateError("Pilot 0 prompt allocation differs from crossed panels")
    _manifest(source.teacher_train, family="tces", split="a_seed_train", count=384)
    _manifest(source.a_validation, family="tces", split="a_validation", count=512)
    _manifest(source.b_train, family="maps", split="b_train", count=4096)
    _manifest(source.b_validation, family="maps", split="b_validation", count=512)
    panel_families = set(artifact.boundary_family_ids + artifact.sentinel_family_ids)
    for manifest in (source.teacher_train, source.a_validation):
        if any(not isinstance(row, TCESTaskManifestRecord) for row in manifest.records):
            raise RunnerGateError("Pilot 0 TCES manifest changed record type")
    teacher_families = Counter(
        row.intended_family
        for row in source.teacher_train.records
        if isinstance(row, TCESTaskManifestRecord)
    )
    if set(teacher_families) != panel_families or set(teacher_families.values()) != {
        16
    }:
        raise RunnerGateError("boundary teacher source differs from the frozen panels")
    validation_roles = [
        row.intended_family in set(artifact.boundary_family_ids)
        for row in source.a_validation.records
        if isinstance(row, TCESTaskManifestRecord)
    ]
    if (
        any(
            row.intended_family not in panel_families
            for row in source.a_validation.records
        )
        or validation_roles.count(True) != 256
        or validation_roles.count(False) != 256
    ):
        raise RunnerGateError(
            "a_validation is not the symmetric frozen panel population"
        )
    for manifest in (source.b_train, source.b_validation):
        if any(
            not isinstance(row, MAPSTaskManifestRecord)
            or row.max_program_length != 2
            or row.shortest_program_length != 2
            for row in manifest.records
        ):
            raise RunnerGateError("MAPS manifest is not shortest2_cap2")
        if manifest.metadata.get("profile") != STAGE_B_PROFILE:
            raise RunnerGateError("MAPS manifest omitted frozen profile lineage")


def validate_pilot0_inputs(inputs: Pilot0Inputs) -> Pilot0Inputs:
    """Fail closed before any Pilot-0 remote operation."""

    if inputs.ledger.authorized_usd != PILOT_AUTHORIZED_USD:
        raise RunnerGateError("Pilot 0 requires the exact $600 ledger")
    for value in (
        inputs.acquisition_artifact_sha256,
        inputs.teacher_recipe_artifact_sha256,
        inputs.panel_artifact_sha256,
    ):
        validate_sha256_id(value)
    if inputs.acquisition_artifact_sha256 != canonical_json_hash(inputs.acquisition):
        raise RunnerGateError("acquisition artifact hash mismatch")
    if inputs.teacher_recipe_artifact_sha256 != canonical_json_hash(
        inputs.teacher_recipe
    ):
        raise RunnerGateError("teacher recipe artifact hash mismatch")
    if inputs.panel_artifact_sha256 != canonical_json_hash(inputs.panel):
        raise RunnerGateError("panel artifact hash mismatch")
    if inputs.config.unresolved_values():
        raise RunnerGateError("Pilot 0 requires a launch-ready resolved config")
    if (
        inputs.config.stage_b.selected_profile != STAGE_B_PROFILE
        or inputs.config.stage_b.selected_max_updates != STAGE_B_GRID[-1]
        or inputs.config.stage_b.primary_task != "maps"
        or inputs.config.stage_b.adaptation != "continue_stage_a_lora"
        or inputs.config.stage_b.optimizer != "fresh"
        or inputs.config.stage_b.lora_parameterization != "same_as_stage_a"
        or tuple(inputs.config.stage_b.provisional_evaluation_updates)
        != (*STAGE_B_GRID, 640)
        or inputs.config.tinker.learning_rates.stage_b_sft.selected
        != STAGE_B_LEARNING_RATE
        or inputs.config.tinker.model_id != MODEL_ID
        or inputs.config.tinker.renderer_name != RENDERER_NAME
        or inputs.config.tinker.lora_rank != LORA_RANK
        or inputs.stage_b_recipe_artifact_sha256 != STAGE_B_DECISION_SHA256
    ):
        raise RunnerGateError("Stage-B recipe differs from the ratified freeze")
    duration = inputs.acquisition.duration.selected_max_updates
    learning_rates = inputs.acquisition.learning_rates
    selected_dose = inputs.teacher_recipe.decision.selected_dose
    configured_rl = inputs.config.stage_a.selected_rl_configuration
    if (
        inputs.acquisition.duration.status is not StageADurationDecisionStatus.FROZEN
        or duration != STAGE_A_GRID[-1]
        or inputs.acquisition.duration.automatic_extension_allowed is not False
        or selected_dose is None
        or inputs.teacher_recipe.decision.status
        is not TeacherDoseDecisionStatus.SELECTED
        or selected_dose not in inputs.config.teacher_dose.demonstrations_per_family
        or inputs.teacher_recipe.selected_learning_rate
        not in inputs.config.tinker.learning_rates.teacher_seed_sft.grid
        or inputs.teacher_recipe.evidence.verification_arm.learning_rate
        != inputs.teacher_recipe.selected_learning_rate
        or set(learning_rates)
        != {
            "static_sft",
            "on_policy_sft",
            "group_relative_rl",
        }
        or learning_rates["static_sft"]
        not in inputs.config.tinker.learning_rates.static_sft.grid
        or learning_rates["on_policy_sft"] != learning_rates["static_sft"]
        or learning_rates["group_relative_rl"]
        not in inputs.config.tinker.learning_rates.group_relative_rl.grid
        or inputs.acquisition.selected_max_tokens
        != inputs.acquisition.max_token_gate.selected_max_tokens
        or inputs.acquisition.max_tokens_apply_to != ALL_METHODS
        or inputs.acquisition.common_rl_apply_to != COMMON_RL_METHODS
        or inputs.acquisition.common_rl.configuration != COMMON_RL_CONFIGURATION
        or inputs.acquisition.common_rl.fallback_configuration is not None
        or not inputs.acquisition.common_rl.entropy_collapse_gate_passed
        or inputs.acquisition.selected_max_tokens < 256
        or inputs.config.teacher_dose.selected_demonstrations_per_family
        != selected_dose
        or inputs.config.stage_a.selected_max_updates != duration
        or inputs.config.stage_a.selected_max_tokens
        != inputs.acquisition.selected_max_tokens
        or inputs.config.stage_a.entropy_collapse_gate_passed is not True
        or configured_rl is None
        or tuple(sorted(configured_rl.items()))
        != inputs.acquisition.common_rl.configuration
        or inputs.config.tinker.learning_rates.teacher_seed_sft.selected
        != inputs.teacher_recipe.selected_learning_rate
        or inputs.config.tinker.learning_rates.static_sft.selected
        != learning_rates["static_sft"]
        or inputs.config.tinker.learning_rates.on_policy_sft.selected
        != learning_rates["on_policy_sft"]
        or inputs.config.tinker.learning_rates.group_relative_rl.selected
        != learning_rates["group_relative_rl"]
        or inputs.config.tasks.maps.min_shortest_length != 2
        or inputs.config.tasks.maps.max_shortest_length != 2
    ):
        raise RunnerGateError("acquisition calibration is not launch-ready")
    if tuple(sorted(source.seed for source in inputs.seed_sources)) != PILOT_SEEDS:
        raise RunnerGateError("Pilot 0 requires exactly paired seeds 11 and 29")
    for source in inputs.seed_sources:
        _validate_seed_sources(inputs, source)
    for name in ("teacher_train", "a_validation", "b_train", "b_validation"):
        if (
            len({getattr(source, name).manifest_id for source in inputs.seed_sources})
            != 1
        ):
            raise RunnerGateError(f"Pilot 0 changed the common {name} manifest")
    orientations = {
        source.prompt_pools.artifact.targeted_panel for source in inputs.seed_sources
    }
    if len(orientations) != 2:
        raise RunnerGateError("Pilot 0 seeds do not cross panel orientation")
    if (
        not inputs.run_id.strip()
        or Path(inputs.run_id).name != inputs.run_id
        or not inputs.git_commit.strip()
        or not inputs.project_id.strip()
    ):
        raise RunnerGateError("Pilot 0 run identity is incomplete")
    if not inputs.m0_sampler_path.strip() or not inputs.m0_state_path.strip():
        raise RunnerGateError("Pilot 0 M0 lineage is incomplete")
    auth = inputs.source_authentication
    bundle = auth.bundle
    expected_sources = tuple(
        seed_source_ids(source)
        for source in sorted(inputs.seed_sources, key=lambda row: row.seed)
    )
    if (
        bundle.project_id != inputs.project_id
        or bundle.git_commit != inputs.git_commit
        or bundle.resolved_config_hash != inputs.config.resolved_config_hash()
        or bundle.completed_live_smoke_sha256
        != inputs.acquisition.live_smoke.artifact_sha256
        or bundle.m0_sampler_path != inputs.m0_sampler_path
        or bundle.m0_state_path != inputs.m0_state_path
        or bundle.panel_artifact_sha256 != inputs.panel_artifact_sha256
        or bundle.teacher_recipe_artifact_sha256
        != inputs.teacher_recipe_artifact_sha256
        or bundle.acquisition_artifact_sha256 != inputs.acquisition_artifact_sha256
        or bundle.stage_b_recipe_artifact_sha256
        != inputs.stage_b_recipe_artifact_sha256
        or bundle.seed_sources != expected_sources
        or bundle.visible_leakage_sha256 != visible_leakage_hash(inputs.seed_sources)
        or auth.bundle_sha256 != auth.authorization.source_bundle_sha256
        or auth.authorization.post_calibration_billing_sha256
        != bundle.post_calibration_billing_sha256
        or auth.authorization.project_id != inputs.project_id
        or auth.sealed_envelope_sha256 != bundle.sealed_b_test_envelope_sha256
        or auth.authorization.authorized_usd != PILOT_COST_CAP_USD
        or auth.authorization.no_rerun_authorized is not True
    ):
        raise RunnerGateError("Pilot 0 source bundle is not authenticated")
    return inputs


__all__ = [
    "METHODS",
    "EPHEMERAL_SAMPLER_FIXED_USD",
    "PILOT_COST_CAP_USD",
    "PILOT_SEEDS",
    "STAGE_A_GRID",
    "STAGE_B_GRID",
    "STAGE_B_LEARNING_RATE",
    "STAGE_B_MAX_TOKENS",
    "STAGE_B_PROFILE",
    "STAGE_B_DECISION_SHA256",
    "Pilot0Inputs",
    "PilotSeedSources",
    "validate_pilot0_inputs",
]
