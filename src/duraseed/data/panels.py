"""Typed post-M0 target/sentinel family-panel artifacts.

This module defines the resolved artifact contract only.  It deliberately does
not select families: real membership can be supplied only after the M0 boundary
scan and panel-matching report exist.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from duraseed.provenance import MAX_ROOT_SEED, validate_sha256_id
from duraseed.schemas import StrictModel


PANEL_ARTIFACT_SCHEMA_VERSION = "duraseed-family-panels-v1"


class PanelLabel(StrEnum):
    """The two matched family panels used by the crossed design."""

    A = "A"
    B = "B"


class _PanelModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SeedBlockPanelAssignment(_PanelModel):
    """Target/sentinel roles for one training-seed block."""

    training_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    targeted_panel: PanelLabel
    sentinel_panel: PanelLabel

    @model_validator(mode="after")
    def roles_are_distinct(self) -> Self:
        if self.targeted_panel is self.sentinel_panel:
            raise ValueError("targeted and sentinel panels must be distinct")
        return self


class FamilyPanelArtifact(_PanelModel):
    """Resolved matched panels and their crossed seed-block schedule."""

    schema_version: Literal["duraseed-family-panels-v1"] = PANEL_ARTIFACT_SCHEMA_VERSION
    m0_checkpoint_path: str
    candidate_family_table_manifest_id: str
    panel_matching_report_id: str
    allocation_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    panel_a_family_ids: tuple[str, ...] = Field(min_length=1)
    panel_b_family_ids: tuple[str, ...] = Field(min_length=1)
    seed_block_assignments: tuple[SeedBlockPanelAssignment, ...] = Field(min_length=2)

    @field_validator("m0_checkpoint_path")
    @classmethod
    def m0_path_is_nonempty(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("m0_checkpoint_path must be non-empty canonical text")
        return value

    @field_validator(
        "candidate_family_table_manifest_id",
        "panel_matching_report_id",
    )
    @classmethod
    def provenance_ids_are_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)

    @field_validator("panel_a_family_ids", "panel_b_family_ids")
    @classmethod
    def family_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not family_id or family_id != family_id.strip() for family_id in value):
            raise ValueError("family IDs must be non-empty canonical text")
        if value != tuple(sorted(set(value))):
            raise ValueError("family IDs must be unique and canonically sorted")
        return value

    @field_validator("seed_block_assignments")
    @classmethod
    def assignments_are_canonical(
        cls,
        value: tuple[SeedBlockPanelAssignment, ...],
    ) -> tuple[SeedBlockPanelAssignment, ...]:
        seeds = tuple(assignment.training_seed for assignment in value)
        if len(seeds) != len(set(seeds)):
            raise ValueError("seed-block assignments must use unique training seeds")
        if seeds != tuple(sorted(seeds)):
            raise ValueError("seed-block assignments must be sorted by training seed")
        return value

    @model_validator(mode="after")
    def panel_design_is_crossed_and_balanced(self) -> Self:
        if len(self.panel_a_family_ids) != len(self.panel_b_family_ids):
            raise ValueError("Panel A and Panel B must contain equal family counts")
        overlap = set(self.panel_a_family_ids).intersection(self.panel_b_family_ids)
        if overlap:
            raise ValueError("Panel A and Panel B family sets must be disjoint")

        targeted_counts = Counter(
            assignment.targeted_panel for assignment in self.seed_block_assignments
        )
        if targeted_counts[PanelLabel.A] != targeted_counts[PanelLabel.B]:
            raise ValueError(
                "seed-block assignments must balance targeted roles across A and B"
            )
        return self


__all__ = [
    "FamilyPanelArtifact",
    "PANEL_ARTIFACT_SCHEMA_VERSION",
    "PanelLabel",
    "SeedBlockPanelAssignment",
]
