"""Authenticate every local source before acquisition calibration can spend."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from duraseed.config import PilotConfig
from duraseed.data.manifests import DatasetManifest
from duraseed.data.panels import FamilyPanelArtifact, PanelLabel
from duraseed.data.stage_a_prompt_pools import StageAPromptPoolBundle
from duraseed.max_token_ratification import load_ratification
from duraseed.provenance import canonical_json_hash, sha256_bytes
from duraseed.run_records import RunStatus, read_run_record
from duraseed.runners import RunnerGateError
from duraseed.runtime import LORA_RANK, MODEL_ID, RENDERER_NAME
from duraseed.training.acquisition_freeze import (
    FROZEN_PROTOCOL_MAX_TOKENS,
    LiveSmokeCalibrationEvidence,
    MaxTokenFreezeEvidence,
)
from duraseed.training.teacher_allocation_sources import (
    TeacherAllocationSources,
    validate_teacher_allocation_base_sources,
)


_BOUNDARY_FILES = (
    "a_candidate_confirmation_manifest.json",
    "a_candidate_manifest.json",
    "confirmation_summary.json",
    "panel_candidate_table.json",
    "panel_matching_report.json",
    "preflight.json",
    "run.json",
    "selected_panel_capacity_audits.jsonl",
    "split_capacity_audits.jsonl",
    "target_sentinel_panels.json",
    "three_cohort_equivalence.json",
)

# Accepted completed three-cohort reduction used by the amended production panels.
ACCEPTED_BOUNDARY_FREEZE_EQUIVALENCE_SHA256: str | None = (
    "sha256:e003fe85289f29915e25582b22ae582182b89185ea66f0d74fdcb4a202653f15"
)
# Accepted §10.1 decision and authorization, stored under provenance/.
ACCEPTED_MAX_TOKEN_SPECIFICATION_SHA256: str | None = (
    "sha256:801a43db3bfcd6b025757601ef49321409a328893d46b68290a3cbd32a452c00"
)
ACCEPTED_MAX_TOKEN_AUTHORIZATION_SHA256: str | None = (
    "sha256:a89bab743f2a70f116b5b4bb1c5767836984550f127ebc3801ed0d69c1905c93"
)
ACCEPTED_BOUNDARY_CONFIG_SHA256 = (
    "sha256:6d0caf9912e1cbafecb1103fe9e4999f62ab9fae4f9d2ee71f34b86f177748c1"
)
ACCEPTED_NONPROTOCOL_CONFIG_SHA256 = (
    "sha256:9ed41168a43609be90878fa624565a60cb202c8be99937fcf3aa78acccce45ca"
)


@dataclass(frozen=True, slots=True)
class CalibrationSourceEvidence:
    smoke: LiveSmokeCalibrationEvidence
    m0_selection_sha256: str
    m0_ttl_sha256: str
    boundary_bundle_sha256: str
    m0_sampler_path: str
    m0_state_path: str
    m0_training_step: int
    boundary_config_sha256: str
    current_config_sha256: str
    nonprotocol_config_sha256: str


def _nonprotocol_config_hash(config: PilotConfig) -> str:
    payload = config.model_dump(mode="json")
    payload.pop("protocol", None)
    return canonical_json_hash(payload)


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid {label} artifact") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"{label} artifact is not an object")
    return value, raw


def load_live_smoke_evidence(
    path: str | Path, *, project_id: str
) -> LiveSmokeCalibrationEvidence:
    """Hash and verify the completed real live-smoke acceptance artifact."""

    source = Path(path)
    value, raw = _object(source, "live-smoke acceptance")
    required = (
        value.get("phase_label") == "live-smoke-gate",
        value.get("status") == "passed",
        value.get("real_data") is True,
        value.get("online_offline_reward_parity") is True,
        value.get("stop_contract_verified") is True,
        value.get("full_state_resume") is True,
        value.get("weights_only_branch") is True,
    )
    updates = value.get("updates")
    maximum = value.get("max_tokens")
    if (
        not all(required)
        or not isinstance(updates, dict)
        or any(
            updates.get(name) is not True
            for name in ("tces_sft", "tces_group_relative_rl", "maps_sft")
        )
        or not isinstance(maximum, dict)
        or maximum.get("protocol_value") != FROZEN_PROTOCOL_MAX_TOKENS
        or maximum.get("runtime_diagnostic_passed") is not True
    ):
        raise RunnerGateError("live-smoke acceptance contract did not pass")
    try:
        run = read_run_record(source.parent)
    except (OSError, ValueError) as error:
        raise RunnerGateError("live-smoke RunRecord is invalid") from error
    if (
        run.status is not RunStatus.COMPLETED
        or run.run_kind != "engineering_smoke"
        or run.project_id != project_id
        or run.model_id != MODEL_ID
        or run.renderer != RENDERER_NAME
        or run.lora_rank != LORA_RANK
        or run.authorized_cost_usd != 25
        or run.reserved_cost_usd != 25
        or run.finished_at is None
        or run.cost_usd > 25
    ):
        raise RunnerGateError("live-smoke RunRecord identity differs")
    return LiveSmokeCalibrationEvidence(
        sha256_bytes(raw),
        int(maximum["protocol_value"]),
        maximum.get("sample_count"),
        maximum.get("truncated_count"),
        True,
    )


def load_m0_evidence(
    selection_path: str | Path, ttl_path: str | Path
) -> tuple[str, str, int, str, str]:
    """Authenticate the selected step-2 sampler/state pair and non-expiring TTLs."""

    selection, selection_raw = _object(Path(selection_path), "M0 selection")
    ttl, ttl_raw = _object(Path(ttl_path), "M0 TTL")
    sampler = selection.get("selected_sampler_checkpoint_path")
    state = selection.get("selected_state_checkpoint_path")
    step = selection.get("selected_training_step")
    if (
        selection.get("status") != "completed"
        or selection.get("scientific_m0_selected") is not True
        or selection.get("pilot_started") is not False
        or type(sampler) is not str
        or type(state) is not str
        or step != 2
        or sampler.split("/sampler_weights/", 1)[0] != state.split("/weights/", 1)[0]
    ):
        raise RunnerGateError("selected M0 lineage is not the frozen step-2 pair")
    expected = ((sampler, "sampler"), (state, "training"))
    if any(
        not isinstance(ttl.get(path), dict)
        or ttl[path].get("checkpoint_type") != kind
        or ttl[path].get("expires_at") is not None
        or ttl[path].get("ttl_seconds") is not None
        for path, kind in expected
    ):
        raise RunnerGateError("selected M0 pair is not verified non-expiring")
    return (
        sampler,
        state,
        step,
        sha256_bytes(selection_raw),
        sha256_bytes(ttl_raw),
    )


def load_max_token_evidence(
    specification_path: str | Path,
    authorization_path: str | Path,
    evidence_path: str | Path,
) -> MaxTokenFreezeEvidence:
    """Authenticate an accepted ratification of the single frozen M0 source."""

    if (
        ACCEPTED_MAX_TOKEN_SPECIFICATION_SHA256 is None
        or ACCEPTED_MAX_TOKEN_AUTHORIZATION_SHA256 is None
    ):
        raise RunnerGateError(
            "no prospective acquisition max-token rule is accepted in this build"
        )

    return load_ratification(
        specification_path,
        authorization_path,
        evidence_path,
        accepted_specification_sha256=ACCEPTED_MAX_TOKEN_SPECIFICATION_SHA256,
        accepted_authorization_sha256=ACCEPTED_MAX_TOKEN_AUTHORIZATION_SHA256,
    )


def _assignment(panel: FamilyPanelArtifact, seed: int):
    value = next(
        (row for row in panel.seed_block_assignments if row.training_seed == seed),
        None,
    )
    if value is None:
        raise RunnerGateError(f"panel schedule omits calibration seed {seed}")
    return value


def _families(panel: FamilyPanelArtifact, label: PanelLabel) -> tuple[str, ...]:
    return (
        panel.panel_a_family_ids if label is PanelLabel.A else panel.panel_b_family_ids
    )


def authenticate_calibration_sources(
    *,
    config: PilotConfig,
    project_id: str,
    smoke_acceptance_path: str | Path,
    m0_selection_path: str | Path,
    m0_ttl_path: str | Path,
    boundary_directory: str | Path,
    teacher_sources: TeacherAllocationSources,
    prompt_pools: StageAPromptPoolBundle,
) -> CalibrationSourceEvidence:
    """Verify smoke, M0, three-cohort boundary, config, and seed orientation."""

    validate_teacher_allocation_base_sources(teacher_sources)
    smoke = load_live_smoke_evidence(smoke_acceptance_path, project_id=project_id)
    sampler, state, step, selection_hash, ttl_hash = load_m0_evidence(
        m0_selection_path, m0_ttl_path
    )
    boundary = Path(boundary_directory)
    file_hashes = {}
    for name in _BOUNDARY_FILES:
        try:
            file_hashes[name] = sha256_bytes((boundary / name).read_bytes())
        except OSError as error:
            raise RunnerGateError(f"boundary source omits {name}") from error
    run = read_run_record(boundary)
    preflight, _ = _object(boundary / "preflight.json", "boundary preflight")
    equivalence, equivalence_raw = _object(
        boundary / "three_cohort_equivalence.json", "boundary freeze equivalence"
    )
    panel = FamilyPanelArtifact.model_validate_json(
        (boundary / "target_sentinel_panels.json").read_bytes()
    )
    broad = DatasetManifest.model_validate_json(
        (boundary / "a_candidate_manifest.json").read_bytes()
    )
    confirmation = DatasetManifest.model_validate_json(
        (boundary / "a_candidate_confirmation_manifest.json").read_bytes()
    )
    current_config_hash = config.resolved_config_hash()
    current_nonprotocol_hash = _nonprotocol_config_hash(config)
    source_nonprotocol_hash = _nonprotocol_config_hash(teacher_sources.config)
    if (
        ACCEPTED_BOUNDARY_FREEZE_EQUIVALENCE_SHA256 is None
        or sha256_bytes(equivalence_raw) != ACCEPTED_BOUNDARY_FREEZE_EQUIVALENCE_SHA256
        or equivalence.get("schema_version")
        != "duraseed-three-cohort-freeze-equivalence-v1"
        or equivalence.get("status") != "passed"
        or equivalence.get("old_new_identical") is not True
        or run.status is not RunStatus.COMPLETED
        or run.run_kind != "m0_calibration"
        or preflight.get("source_kind") != "three_cohort_boundary_freeze_v1"
        or run.model_id != MODEL_ID
        or run.renderer != RENDERER_NAME
        or run.lora_rank != LORA_RANK
        or run.resolved_config_hash != ACCEPTED_BOUNDARY_CONFIG_SHA256
        or current_nonprotocol_hash != ACCEPTED_NONPROTOCOL_CONFIG_SHA256
        or source_nonprotocol_hash != ACCEPTED_NONPROTOCOL_CONFIG_SHA256
        or run.project_id != project_id
        or run.parent_tinker_checkpoint_path != state
        or panel.m0_checkpoint_path != sampler
        or teacher_sources.panel != panel
        or teacher_sources.broad_manifest != broad
        or teacher_sources.confirmation_manifest != confirmation
        or prompt_pools.a_rl_train_manifest != teacher_sources.a_rl_train_manifest
        or prompt_pools.a_monitor_manifest != teacher_sources.a_monitor_manifest
        or prompt_pools.artifact.family_panel_artifact_id != canonical_json_hash(panel)
    ):
        raise RunnerGateError("calibration source lineage or runtime identity differs")
    seed17, seed37 = _assignment(panel, 17), _assignment(panel, 37)
    artifact = prompt_pools.artifact
    if (
        seed17.targeted_panel is not seed37.sentinel_panel
        or seed17.sentinel_panel is not seed37.targeted_panel
        or artifact.calibration_seed != 17
        or artifact.targeted_panel is not seed17.targeted_panel
        or artifact.sentinel_panel is not seed17.sentinel_panel
        or artifact.boundary_family_ids != _families(panel, seed17.targeted_panel)
        or artifact.sentinel_family_ids != _families(panel, seed17.sentinel_panel)
    ):
        raise RunnerGateError("Stage-A seed-17/37 panel orientation differs")
    return CalibrationSourceEvidence(
        smoke,
        selection_hash,
        ttl_hash,
        canonical_json_hash(file_hashes),
        sampler,
        state,
        step,
        run.resolved_config_hash,
        current_config_hash,
        current_nonprotocol_hash,
    )


__all__ = [
    "CalibrationSourceEvidence",
    "authenticate_calibration_sources",
    "load_live_smoke_evidence",
    "load_max_token_evidence",
    "load_m0_evidence",
]
