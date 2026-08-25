"""Frozen one-paired-seed Pilot-0 execution contract."""

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
from duraseed.runners import RunnerGateError
from duraseed.runtime import (
    LORA_RANK,
    MODEL_ID,
    RENDERER_NAME,
    RuntimeBundle,
    TokenLedger,
)


PILOT_SEEDS = (11, 29)
METHODS = ("B-S", "B-G")
STAGE_A_GRID = (0, 10, 25, 50)  # retained only for the legacy matched reducer
BS_STAGE_A_GRID = (0, *range(10, 291, 10), 294)
BG_STAGE_A_GRID = (0, 10, 20, 30, 40, 50)
STAGE_A_GRIDS = {"B-S": BS_STAGE_A_GRID, "B-G": BG_STAGE_A_GRID}
STAGE_B_GRID = (0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480)
STAGE_B_PROFILE = "shortest2_cap2"
STAGE_B_LEARNING_RATE = 3e-4
STAGE_B_MAX_TOKENS = 128
STAGE_B_DECISION_SHA256 = (
    "sha256:36015e1e08a0a96b4f4b0d7dd8480a679294c1cfc2a026d3ab96073c5378acf8"
)
STAGE_B_PUBLIC_DECISION_SHA256 = (
    "sha256:dfb9d0cdf21b47653b90a901490ec618e6b5c203c753b6ac9184d13fdf3e295e"
)
BS_LEARNING_RATE = 1e-4
BG_LEARNING_RATE = 1e-5
TCES_MAX_TOKENS = 4096
PILOT_PAIR_PLANNING_CAP_USD = 774.04
PILOT_TWO_PAIR_PLANNING_CAP_USD = 1548.08
EPHEMERAL_SAMPLER_FIXED_USD = 0.05
CADENCE_CHECKPOINT_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class PilotStageARecipe:
    selected_max_tokens: int = TCES_MAX_TOKENS

    @property
    def learning_rates(self) -> dict[str, float]:
        return {
            "static_sft": BS_LEARNING_RATE,
            "group_relative_rl": BG_LEARNING_RATE,
        }


@dataclass(frozen=True, slots=True)
class PilotSeedSources:
    seed: int
    prompt_pools: StageAPromptPoolBundle
    a_cadence: DatasetManifest
    a_validation: DatasetManifest
    b_train: DatasetManifest
    b_validation: DatasetManifest


@dataclass(frozen=True, slots=True)
class PilotPairSourceAuthentication:
    bundle_sha256: str
    lineage: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Pilot0Inputs:
    config: PilotConfig
    runtime: RuntimeBundle
    ledger: TokenLedger
    output_root: Path
    run_id: str
    git_commit: str
    project_id: str
    pair_index: int
    m0_sampler_path: str
    m0_state_path: str
    panel: FamilyPanelArtifact
    source: PilotSeedSources
    source_authentication: PilotPairSourceAuthentication
    billing: Any
    session_id: str
    dose_terminal_sha256: str
    stage_b_recipe_artifact_sha256: str
    acquisition: PilotStageARecipe = PilotStageARecipe()
    prior_pair_result_sha256: str | None = None

    @property
    def seed_sources(self) -> tuple[PilotSeedSources, ...]:
        return (self.source,)


def stage_a_grid(method: str) -> tuple[int, ...]:
    try:
        return STAGE_A_GRIDS[method]
    except KeyError as error:
        raise RunnerGateError(f"unknown Pilot-0 method: {method}") from error


def _manifest(value: DatasetManifest, *, family: str, split: str, count: int) -> None:
    if (
        not isinstance(value, DatasetManifest)
        or value.task_family != family
        or value.split != split
        or value.record_count != count
    ):
        raise RunnerGateError(f"Pilot 0 requires exact {family}/{split} population")


def _validate_source(inputs: Pilot0Inputs) -> None:
    source = inputs.source
    expected_seed = PILOT_SEEDS[inputs.pair_index - 1]
    if source.seed != expected_seed:
        raise RunnerGateError("Pilot pair index and seed differ")
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
        raise RunnerGateError("Pilot prompt allocation differs from crossed panels")
    _manifest(source.a_validation, family="tces", split="a_validation", count=512)
    _manifest(source.a_cadence, family="tces", split="a_monitor", count=192)
    _manifest(source.b_train, family="maps", split="b_train", count=4096)
    _manifest(source.b_validation, family="maps", split="b_validation", count=512)
    panel_families = set(artifact.boundary_family_ids + artifact.sentinel_family_ids)
    cadence_counts = Counter(
        row.intended_family
        for row in source.a_cadence.records
        if isinstance(row, TCESTaskManifestRecord)
    )
    if set(cadence_counts) != panel_families or set(cadence_counts.values()) != {8}:
        raise RunnerGateError("Pilot cadence is not 8 items per crossed family")
    validation_counts = Counter(
        row.intended_family
        for row in source.a_validation.records
        if isinstance(row, TCESTaskManifestRecord)
    )
    if (
        len(validation_counts) != 24
        or set(validation_counts) != panel_families
        or set(validation_counts.values()) != {21, 22}
        or sum(validation_counts[family] for family in artifact.boundary_family_ids)
        != 256
        or sum(validation_counts[family] for family in artifact.sentinel_family_ids)
        != 256
        or any(
            not isinstance(row, TCESTaskManifestRecord)
            for row in source.a_validation.records
        )
    ):
        raise RunnerGateError(
            "a_validation is not the balanced frozen panel population"
        )
    for manifest in (source.b_train, source.b_validation):
        if (
            any(
                not isinstance(row, MAPSTaskManifestRecord)
                or row.max_program_length != 2
                or row.shortest_program_length != 2
                for row in manifest.records
            )
            or manifest.metadata.get("profile") != STAGE_B_PROFILE
        ):
            raise RunnerGateError("MAPS manifest is not shortest2_cap2")


def validate_pilot0_inputs(inputs: Pilot0Inputs) -> Pilot0Inputs:
    """Fail closed before any Pilot remote operation."""

    if inputs.pair_index not in (1, 2):
        raise RunnerGateError("Pilot 0 supports exactly pair 1 or pair 2")
    for value in (
        inputs.source_authentication.bundle_sha256,
        inputs.dose_terminal_sha256,
        inputs.stage_b_recipe_artifact_sha256,
    ):
        validate_sha256_id(value)
    if inputs.pair_index == 1 and inputs.prior_pair_result_sha256 is not None:
        raise RunnerGateError("pair 1 cannot claim prior-pair evidence")
    if inputs.pair_index == 2:
        if inputs.prior_pair_result_sha256 is None:
            raise RunnerGateError("pair 2 requires durable pair-1 F1/F2/F3 evidence")
        validate_sha256_id(inputs.prior_pair_result_sha256)
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
        or inputs.config.tinker.group_size != 8
        or inputs.config.tinker.groups_per_batch != 16
        or inputs.config.tinker.max_sampled_tokens != TCES_MAX_TOKENS
        or inputs.stage_b_recipe_artifact_sha256 != STAGE_B_DECISION_SHA256
        or inputs.acquisition.selected_max_tokens != TCES_MAX_TOKENS
    ):
        raise RunnerGateError("Pilot recipe differs from the frozen charter")
    _validate_source(inputs)
    if (
        not inputs.run_id.strip()
        or Path(inputs.run_id).name != inputs.run_id
        or not inputs.git_commit.strip()
        or not inputs.project_id.strip()
        or not inputs.session_id.strip()
        or not inputs.m0_sampler_path.strip()
        or not inputs.m0_state_path.strip()
    ):
        raise RunnerGateError("Pilot run identity or M0 lineage is incomplete")
    return inputs


__all__ = [
    "BG_LEARNING_RATE",
    "BG_STAGE_A_GRID",
    "BS_LEARNING_RATE",
    "BS_STAGE_A_GRID",
    "CADENCE_CHECKPOINT_TTL_SECONDS",
    "EPHEMERAL_SAMPLER_FIXED_USD",
    "METHODS",
    "PILOT_PAIR_PLANNING_CAP_USD",
    "PILOT_SEEDS",
    "PILOT_TWO_PAIR_PLANNING_CAP_USD",
    "STAGE_B_DECISION_SHA256",
    "STAGE_A_GRID",
    "STAGE_B_GRID",
    "STAGE_B_LEARNING_RATE",
    "STAGE_B_MAX_TOKENS",
    "STAGE_B_PROFILE",
    "STAGE_B_PUBLIC_DECISION_SHA256",
    "TCES_MAX_TOKENS",
    "Pilot0Inputs",
    "PilotPairSourceAuthentication",
    "PilotSeedSources",
    "PilotStageARecipe",
    "stage_a_grid",
    "validate_pilot0_inputs",
]
