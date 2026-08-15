"""Private comparison contracts for the unverified three-cohort freeze."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from duraseed.config import PilotConfig
from duraseed.data.boundary import BoundaryFamilySummary
from duraseed.data.manifests import DatasetManifest
from duraseed.data.panel_capacity import FamilyCapacityAudit
from duraseed.data.panel_matching import (
    PANEL_MATCHING_COVARIATES,
    FamilyPanelCandidate,
    FamilyPanelMatch,
)
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.provenance import MAX_ROOT_SEED, canonical_json_bytes, sha256_bytes
from duraseed.tasks.tces import TCESGeneratorConfig


def _immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class BoundaryFreezeSettings:
    generator_kwargs: Mapping[str, Any]
    capacity_root_seed: int
    allocation_seed: int
    panel_size: int
    matching_covariates: tuple[str, ...]
    training_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        generator_kwargs = dict(self.generator_kwargs)
        TCESGeneratorConfig(**generator_kwargs)
        object.__setattr__(self, "generator_kwargs", _immutable(generator_kwargs))
        integers = (self.capacity_root_seed, self.allocation_seed, self.panel_size)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in integers
        ):
            raise ValueError("freeze seeds and panel size must be integers")
        if (
            not 0 <= self.capacity_root_seed <= MAX_ROOT_SEED
            or not 0 <= self.allocation_seed <= MAX_ROOT_SEED
            or self.panel_size < 1
        ):
            raise ValueError("freeze seeds and panel size must be valid")
        if self.matching_covariates != PANEL_MATCHING_COVARIATES:
            raise ValueError("freeze matching covariates differ from the carried order")
        seeds = self.training_seeds
        if (
            len(seeds) < 2
            or len(seeds) % 2
            or len(seeds) != len(set(seeds))
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= MAX_ROOT_SEED
                for seed in seeds
            )
        ):
            raise ValueError("freeze training seeds must support a balanced crossover")

    def projection(self) -> dict[str, Any]:
        return {
            "allocation_seed": self.allocation_seed,
            "capacity_root_seed": self.capacity_root_seed,
            "generator_kwargs": dict(self.generator_kwargs),
            "matching_covariates": list(self.matching_covariates),
            "panel_size": self.panel_size,
            "training_seeds": list(self.training_seeds),
        }

    @property
    def projection_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.projection()))


def freeze_settings_from_config(config: PilotConfig) -> BoundaryFreezeSettings:
    """Extract only the fields that can change the archived freeze reduction."""

    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    panels = config.model_extra.get("family_panels")
    statistics = config.statistics.model_dump(mode="python")
    if not isinstance(panels, Mapping):
        raise ValueError("family-panel config is missing")
    try:
        confirmatory = panels["size_profiles"]["confirmatory"]
        return BoundaryFreezeSettings(
            generator_kwargs=config.tasks.tces.generator_kwargs(),
            capacity_root_seed=config.seed,
            allocation_seed=panels["frozen_allocation_seed"],
            panel_size=confirmatory["targeted"],
            matching_covariates=tuple(panels["matching_covariates"]),
            training_seeds=(
                *config.statistics.calibration_seeds,
                *statistics["pilot_seeds"],
                *statistics["confirmatory_seeds"],
            ),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("freeze config projection is incomplete") from error


@dataclass(frozen=True, slots=True)
class BoundaryFreezeCohort:
    cohort_id: str
    broad_manifest: DatasetManifest
    confirmation_manifest: DatasetManifest
    finalist_summaries: tuple[BoundaryFamilySummary, ...]
    sampler_checkpoint_path: str
    locked_eligible_family_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BoundaryFreezeResult:
    combined_broad_manifest: DatasetManifest
    combined_confirmation_manifest: DatasetManifest
    finalist_summaries: tuple[BoundaryFamilySummary, ...]
    confirmation_family_table: tuple[dict[str, Any], ...]
    capacity_audits: tuple[FamilyCapacityAudit, ...]
    observation_eligible_family_ids: tuple[str, ...]
    eligible_family_ids: tuple[str, ...]
    teacher_trace_token_counts: dict[str, dict[str, int]]
    candidates: tuple[FamilyPanelCandidate, ...]
    ranked_family_ids: tuple[str, ...]
    candidate_payload: dict[str, Any] | None
    candidate_table_id: str | None
    match: FamilyPanelMatch | None
    matching_report_payload: dict[str, Any] | None
    matching_report_id: str | None
    intermediate_family_ids: tuple[str, ...]
    selected_capacity_audits: tuple[FamilyCapacityAudit, ...]
    panel_artifact: FamilyPanelArtifact | None
    confirmation_summary: dict[str, Any]

    @property
    def selection_performed(self) -> bool:
        return self.panel_artifact is not None


__all__ = [
    "BoundaryFreezeCohort",
    "BoundaryFreezeResult",
    "BoundaryFreezeSettings",
    "freeze_settings_from_config",
]
