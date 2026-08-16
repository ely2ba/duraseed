"""Load only equivalence-authorized local calibration source objects."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from duraseed.config import PilotConfig
from duraseed.data.manifests import read_manifest
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.data.sealing import ExecutionContext
from duraseed.data.stage_a_prompt_pools import (
    StageAPromptPoolBundle,
    read_stage_a_prompt_pool_bundle,
)
from duraseed.provenance import sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.training.teacher_allocation_sources import TeacherAllocationSources


# Accepted exact archived-v0/current-v1 production split authorization.
ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256: str | None = (
    "sha256:420421f8bf0d8fbac08791d72b908e627f6a4ed845834d91b818eac0ab064e12"
)


@dataclass(frozen=True, slots=True)
class LoadedCalibrationSources:
    teacher: TeacherAllocationSources
    prompts: StageAPromptPoolBundle
    authorization_sha256: str
    equivalence_sha256: str


def _object(path: Path, label: str) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid {label} artifact") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"{label} artifact is not an object")
    return value, raw


def load_calibration_source_objects(
    *,
    config: PilotConfig,
    boundary_directory: str | Path,
    source_directory: str | Path,
    panel_split_authorization_path: str | Path,
    panel_split_equivalence_path: str | Path,
) -> LoadedCalibrationSources:
    """Bind prebuilt a_seed manifests to an accepted exact-v0 equivalence."""

    if ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256 is None:
        raise RunnerGateError(
            "panel-split extraction RFC/equivalence is not accepted in this build"
        )
    authorization, authorization_raw = _object(
        Path(panel_split_authorization_path), "panel-split authorization"
    )
    equivalence, equivalence_raw = _object(
        Path(panel_split_equivalence_path), "panel-split equivalence"
    )
    authorization_hash = sha256_bytes(authorization_raw)
    equivalence_hash = sha256_bytes(equivalence_raw)
    boundary = Path(boundary_directory)
    source = Path(source_directory)
    prompts = read_stage_a_prompt_pool_bundle(source)
    target = read_manifest(
        source / "a_seed_train_manifest.json", context=ExecutionContext.TRAINING
    )
    gate = read_manifest(
        source / "a_seed_gate_manifest.json", context=ExecutionContext.SELECTION
    )
    if (
        authorization_hash != ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256
        or authorization.get("schema_version")
        != "duraseed-calibration-panel-split-authorization-v1"
        or authorization.get("status") != "accepted"
        or authorization.get("equivalence_sha256") != equivalence_hash
        or not isinstance(authorization.get("authorizer"), str)
        or not authorization["authorizer"].strip()
        or not isinstance(authorization.get("accepted_at_utc"), str)
        or equivalence.get("schema_version")
        != "duraseed-calibration-panel-split-equivalence-v1"
        or equivalence.get("status") != "passed"
        or equivalence.get("old_new_identical") is not True
        or equivalence.get("a_seed_train_manifest_id") != target.manifest_id
        or equivalence.get("a_seed_gate_manifest_id") != gate.manifest_id
        or equivalence.get("a_seed_train_sha256")
        != sha256_bytes((source / "a_seed_train_manifest.json").read_bytes())
        or equivalence.get("a_seed_gate_sha256")
        != sha256_bytes((source / "a_seed_gate_manifest.json").read_bytes())
    ):
        raise RunnerGateError("panel-split source is not accepted and equivalent")
    teacher = TeacherAllocationSources(
        config=config,
        panel=FamilyPanelArtifact.model_validate_json(
            (boundary / "target_sentinel_panels.json").read_bytes()
        ),
        broad_manifest=read_manifest(
            boundary / "a_candidate_manifest.json",
            context=ExecutionContext.SELECTION,
        ),
        confirmation_manifest=read_manifest(
            boundary / "a_candidate_confirmation_manifest.json",
            context=ExecutionContext.SELECTION,
        ),
        a_rl_train_manifest=prompts.a_rl_train_manifest,
        a_monitor_manifest=prompts.a_monitor_manifest,
        target_train_manifest=target,
        gate_manifest=gate,
        selected_dose=None,
        optimizer_updates=config.teacher_dose.calibration_updates,
    )
    return LoadedCalibrationSources(
        teacher, prompts, authorization_hash, equivalence_hash
    )


__all__ = ["LoadedCalibrationSources", "load_calibration_source_objects"]
