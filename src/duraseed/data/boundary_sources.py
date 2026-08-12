"""Pure source-contract authentication for staged boundary evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from duraseed.config import PilotConfig
from duraseed.data.boundary_confirmation import broad_cohort
from duraseed.data.boundary_protocol import (
    BOUNDARY_ENGINEERING_SEED,
    validate_broad_cohort_provenance,
)
from duraseed.data.manifests import DatasetManifest
from duraseed.run_records import GenerationRecord, RunRecord, RunStatus


class BoundarySourceError(ValueError):
    """Staged evidence does not share one authenticated M0 source."""


@dataclass(frozen=True, slots=True)
class BoundarySourceContract:
    cohort_id: str
    cohort_ordinal_start: int
    prior_run_ids: tuple[str, str]
    sampler_checkpoint_path: str
    training_step: int
    model_id: str
    renderer: str
    lora_rank: int
    state_checkpoint_path: str
    project_id: str
    protocol_version: str
    resolved_config_hash: str
    broad_manifest_id: str


def validate_completed_confirmation_contract(
    *,
    source_contract: BoundarySourceContract,
    confirmation_run: RunRecord,
    confirmation_plan: Mapping[str, Any],
    broad_manifest: DatasetManifest,
    confirmation_manifest: DatasetManifest,
    confirmation_generations: Sequence[GenerationRecord],
    config: PilotConfig,
) -> tuple[str, tuple[str, ...]]:
    """Authenticate one completed confirmation against its staged source."""

    if (
        confirmation_run.status is not RunStatus.COMPLETED
        or confirmation_run.run_kind != "m0_calibration"
        or confirmation_run.method is not None
    ):
        raise BoundarySourceError("confirmation is not completed M0 evidence")
    stable_identity = (
        confirmation_run.model_id,
        confirmation_run.renderer,
        confirmation_run.lora_rank,
        confirmation_run.parent_tinker_checkpoint_path,
        str(confirmation_run.project_id),
    )
    if stable_identity != (
        source_contract.model_id,
        source_contract.renderer,
        source_contract.lora_rank,
        source_contract.state_checkpoint_path,
        source_contract.project_id,
    ):
        raise BoundarySourceError("confirmation changed the selected-M0 identity")
    source = confirmation_plan.get("source")
    measurement = confirmation_plan.get("measurement")
    expected_run_id = confirmation_plan.get("run_id")
    if (
        not isinstance(source, Mapping)
        or not isinstance(measurement, Mapping)
        or not isinstance(expected_run_id, str)
        or not expected_run_id
        or confirmation_plan.get("project_id") != confirmation_run.project_id
        or confirmation_run.task_manifest_ids.get("boundary_confirmation_a_candidate")
        != confirmation_manifest.manifest_id
        or confirmation_manifest.metadata.get("source_broad_manifest_id")
        != broad_manifest.manifest_id
        or broad_manifest.manifest_id != source_contract.broad_manifest_id
        or source.get("broad_manifest_id") != broad_manifest.manifest_id
        or measurement.get("manifest_id") != confirmation_manifest.manifest_id
        or tuple(source.get("prior_run_ids", ())) != source_contract.prior_run_ids
        or source.get("sampler_checkpoint_path")
        != source_contract.sampler_checkpoint_path
        or source.get("training_step") != source_contract.training_step
    ):
        raise BoundarySourceError("confirmation manifest or source lineage differs")
    generations = tuple(confirmation_generations)
    run_ids = {row.run_id for row in generations}
    sample_ids = tuple(row.sample_id for row in generations)
    evaluation = config.evaluation
    if not isinstance(evaluation, Mapping):
        raise BoundarySourceError("config evaluation contract is malformed")
    expected_sampling = (
        evaluation.get("temperature"),
        evaluation.get("top_p"),
        config.tinker.max_sampled_tokens,
    )
    protocol_matches = (
        confirmation_run.protocol_version == source_contract.protocol_version
        and confirmation_run.resolved_config_hash
        == source_contract.resolved_config_hash
    )
    amendment_matches = (
        source.get("source_protocol_version") == source_contract.protocol_version
        and source.get("source_resolved_config_hash")
        == source_contract.resolved_config_hash
        and source.get("current_protocol_version") == confirmation_run.protocol_version
        and source.get("current_resolved_config_hash")
        == confirmation_run.resolved_config_hash
        and source.get("compatible_pre_pilot_amendment")
        == "docs/rfc-prepilot-estimands-and-selection.md"
    )
    if (
        run_ids != {expected_run_id}
        or not sample_ids
        or len(sample_ids) != len(set(sample_ids))
        or {(row.sampler_checkpoint_path, row.training_step) for row in generations}
        != {
            (
                source_contract.sampler_checkpoint_path,
                source_contract.training_step,
            )
        }
        or {row.origin_sampler_checkpoint_path for row in generations}
        != {source_contract.sampler_checkpoint_path}
        or {row.seed for row in generations} != {BOUNDARY_ENGINEERING_SEED}
        or {
            (
                row.sampling_temperature,
                row.sampling_top_p,
                row.sampling_max_tokens,
            )
            for row in generations
        }
        != {expected_sampling}
        or (
            measurement.get("temperature"),
            measurement.get("top_p"),
            measurement.get("max_tokens"),
        )
        != expected_sampling
        or not (protocol_matches or amendment_matches)
    ):
        raise BoundarySourceError("confirmation observation contract differs")
    return str(next(iter(run_ids))), sample_ids


def validate_confirmation_source_contract(
    *,
    config: PilotConfig,
    broad_run: RunRecord,
    refinement_run: RunRecord,
    broad_plan: Mapping[str, Any],
    refinement_plan: Mapping[str, Any],
    broad_manifest: DatasetManifest,
    refinement_manifest: DatasetManifest,
    broad_generations: Sequence[GenerationRecord],
    refinement_generations: Sequence[GenerationRecord],
    expected_parent_manifest_id: str | None,
) -> BoundarySourceContract:
    """Authenticate the exact local inputs accepted by confirmation."""

    runs = (broad_run, refinement_run)
    if any(
        run.status is not RunStatus.COMPLETED
        or run.run_kind != "m0_calibration"
        or run.method is not None
        for run in runs
    ):
        raise BoundarySourceError("boundary sources are not completed M0 evidence")
    identity = (
        "protocol_version",
        "resolved_config_hash",
        "model_id",
        "renderer",
        "lora_rank",
        "parent_tinker_checkpoint_path",
        "project_id",
    )
    if any(
        getattr(broad_run, field) != getattr(refinement_run, field)
        for field in identity
    ):
        raise BoundarySourceError("broad and refinement sources changed identity")
    if (
        broad_run.model_id != config.tinker.model_id
        or broad_run.renderer != config.tinker.renderer_name
        or broad_run.lora_rank != config.tinker.lora_rank
        or not broad_run.parent_tinker_checkpoint_path
        or not broad_run.project_id
    ):
        raise BoundarySourceError("boundary source differs from frozen config")
    source = refinement_plan.get("source")
    broad_source = broad_plan.get("source")
    broad_run_id = broad_plan.get("run_id")
    refinement_run_id = refinement_plan.get("run_id")
    if (
        broad_manifest != refinement_manifest
        or broad_run.task_manifest_ids.get("boundary_broad_a_candidate")
        != broad_manifest.manifest_id
        or refinement_run.task_manifest_ids.get("boundary_refinement_a_candidate")
        != broad_manifest.manifest_id
        or not isinstance(source, Mapping)
        or not isinstance(broad_source, Mapping)
        or not isinstance(broad_run_id, str)
        or not broad_run_id
        or not isinstance(refinement_run_id, str)
        or not refinement_run_id
        or broad_plan.get("project_id") != broad_run.project_id
        or refinement_plan.get("project_id") != refinement_run.project_id
        or source.get("broad_run_id") != broad_run_id
        or source.get("manifest_id") != broad_manifest.manifest_id
    ):
        raise BoundarySourceError("boundary manifest lineage differs")
    cohort_id, cohort_start = broad_cohort(broad_manifest)
    validate_broad_cohort_provenance(
        broad_manifest,
        broad_plan,
        expected_cohort=cast(Any, cohort_id),
        expected_parent_manifest_id=expected_parent_manifest_id,
    )
    broad_measurement = broad_plan.get("measurement")
    refinement_measurement = refinement_plan.get("measurement")
    if not isinstance(broad_measurement, Mapping) or not isinstance(
        refinement_measurement, Mapping
    ):
        raise BoundarySourceError("boundary plans omitted sampling contracts")
    evaluation = config.evaluation
    if not isinstance(evaluation, Mapping):
        raise BoundarySourceError("config evaluation contract is malformed")
    expected = (
        evaluation.get("temperature"),
        evaluation.get("top_p"),
        config.tinker.max_sampled_tokens,
    )
    declared = {
        (
            value.get("temperature"),
            value.get("top_p"),
            value.get("max_tokens"),
        )
        for value in (broad_measurement, refinement_measurement)
    }
    generations = tuple(broad_generations) + tuple(refinement_generations)
    observed = {
        (
            row.sampling_temperature,
            row.sampling_top_p,
            row.sampling_max_tokens,
        )
        for row in generations
    }
    sampler_paths = {row.sampler_checkpoint_path for row in generations}
    training_steps = {row.training_step for row in generations}
    origins = {row.origin_sampler_checkpoint_path for row in generations}
    seeds = {row.seed for row in generations}
    if (
        declared != {expected}
        or observed != {expected}
        or origins != sampler_paths
        or seeds != {BOUNDARY_ENGINEERING_SEED}
        or broad_measurement.get("group_size_for_i8") != config.tinker.group_size
    ):
        raise BoundarySourceError("boundary sampling contract differs")
    broad_ids = {row.run_id for row in broad_generations}
    refinement_ids = {row.run_id for row in refinement_generations}
    if (
        len(broad_ids) != 1
        or len(refinement_ids) != 1
        or None in broad_ids
        or None in refinement_ids
        or len(sampler_paths) != 1
        or len(training_steps) != 1
        or broad_ids != {broad_run_id}
        or refinement_ids != {refinement_run_id}
    ):
        raise BoundarySourceError("boundary observation identity is ambiguous")
    sampler = str(next(iter(sampler_paths)))
    step = int(next(iter(training_steps)))
    if (
        source.get("sampler_checkpoint_path") != sampler
        or source.get("training_step") != step
        or broad_source.get("sampler_checkpoint_path") != sampler
        or broad_source.get("training_step") != step
        or broad_source.get("state_checkpoint_path")
        != broad_run.parent_tinker_checkpoint_path
    ):
        raise BoundarySourceError("boundary plans differ from observed checkpoint")
    return BoundarySourceContract(
        cohort_id=cohort_id,
        cohort_ordinal_start=cohort_start,
        prior_run_ids=(broad_run_id, refinement_run_id),
        sampler_checkpoint_path=sampler,
        training_step=step,
        model_id=broad_run.model_id,
        renderer=broad_run.renderer,
        lora_rank=broad_run.lora_rank,
        state_checkpoint_path=broad_run.parent_tinker_checkpoint_path,
        project_id=str(broad_run.project_id),
        protocol_version=broad_run.protocol_version,
        resolved_config_hash=broad_run.resolved_config_hash,
        broad_manifest_id=broad_manifest.manifest_id,
    )


__all__ = [
    "BoundarySourceContract",
    "BoundarySourceError",
    "validate_completed_confirmation_contract",
    "validate_confirmation_source_contract",
]
