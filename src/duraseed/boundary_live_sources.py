"""Authenticate and reduce the fixed local handoff for the live boundary gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from duraseed.boundary_capacity import audit_family_split_capacities
from duraseed.config import PilotConfig
from duraseed.data.boundary import (
    BoundaryFamilySummary,
    assess_refinement_finalist_gate,
    summarize_m0_boundary,
)
from duraseed.data.boundary_confirmation import (
    BoundaryConfirmationError,
    build_confirmation_manifest,
    choose_refinement_family_ids,
)
from duraseed.data.boundary_protocol import (
    BOUNDARY_BROAD_EXTENSION_1_COHORT,
    BOUNDARY_ENGINEERING_SEED,
    audit_new_broad_cohort,
)
from duraseed.data.boundary_sources import BoundarySourceContract
from duraseed.data.manifests import DatasetManifest
from duraseed.data.panel_capacity import FamilyCapacityAudit
from duraseed.run_records import GenerationRecord, RewardRecord, RunRecord, RunStatus
from duraseed.runners import RunnerGateError, authenticate_extension1_source
from duraseed.tasks.tces import (
    GeneratedTCESInstance,
    TCESGeneratorConfig,
    enumerate_task,
    generate_teacher_trace,
)


@dataclass(frozen=True, slots=True)
class BoundaryLiveSource:
    contract: BoundarySourceContract
    initial_broad_manifest: DatasetManifest
    initial_confirmation_manifest: DatasetManifest
    extension1_broad_manifest: DatasetManifest
    extension1_broad_generations: tuple[GenerationRecord, ...]
    extension1_broad_rewards: tuple[RewardRecord, ...]
    extension1_refinement_generations: tuple[GenerationRecord, ...]
    extension1_refinement_rewards: tuple[RewardRecord, ...]
    extension1_family_successes: dict[str, int]
    extension1_refinement_summaries: tuple[BoundaryFamilySummary, ...]


def reconstruct_family_templates(
    broad: DatasetManifest, family_ids: tuple[str, ...]
) -> dict[str, GeneratedTCESInstance]:
    """Rebuild equivalent generators from authenticated broad members."""

    requested = tuple(sorted(set(family_ids)))
    if not requested or len(requested) != len(family_ids):
        raise ValueError("family IDs must be nonempty and unique")
    result: dict[str, GeneratedTCESInstance] = {}
    for family_id in requested:
        record = min(
            (row for row in broad.records if row.intended_family == family_id),
            key=lambda row: (row.item_index, row.task_id),
            default=None,
        )
        if record is None:
            raise BoundaryConfirmationError("a requested family is absent")
        task = record.to_task()
        enumeration = enumerate_task(task)
        latent = enumeration.family_representatives.get(family_id)
        if not enumeration.complete or latent is None:
            raise BoundaryConfirmationError("broad family reconstruction diverged")
        result[family_id] = GeneratedTCESInstance(
            task,
            record.content_hash,
            latent,
            family_id,
            enumeration,
            generate_teacher_trace(latent),
            record.generator_seed,
            record.item_index,
            record.accepted_attempt,
        )
    return result


def capacity_cleared_confirmation(
    generator: TCESGeneratorConfig,
    broad: DatasetManifest,
    refinement: tuple[BoundaryFamilySummary, ...],
) -> tuple[DatasetManifest, tuple[FamilyCapacityAudit, ...]]:
    """Apply the existing Stage-2 and split-capacity gates before paid Stage 3."""

    finalists = tuple(
        sorted(
            row.intended_family_id
            for row in refinement
            if assess_refinement_finalist_gate(row).eligible
        )
    )
    if not finalists:
        return build_confirmation_manifest(generator, broad, ()), ()
    templates = reconstruct_family_templates(broad, finalists)
    provisional = build_confirmation_manifest(
        generator, broad, finalists, templates=templates
    )
    forbidden = (*broad.records, *provisional.records)
    audits = audit_family_split_capacities(
        templates,
        finalists,
        generator,
        root_seed=BOUNDARY_ENGINEERING_SEED,
        forbidden_records=forbidden,
        protected_family_ids=finalists,
    )
    cleared = tuple(row.family_id for row in audits if row.passed)
    return (
        build_confirmation_manifest(
            generator,
            broad,
            cleared,
            templates={family_id: templates[family_id] for family_id in cleared},
        ),
        audits,
    )


def load_frozen_extension1_confirmation(
    path: str | Path, broad: DatasetManifest
) -> DatasetManifest:
    """Load the production manifest already compared exactly with archived v0."""

    candidate = Path(path)
    try:
        raw = candidate.read_bytes()
        manifest = DatasetManifest.model_validate_json(raw)
    except (OSError, ValueError) as error:
        raise RunnerGateError("invalid frozen Extension-1 confirmation") from error
    expected_id = (
        "sha256:23665b09e285af293d8bda200317056f35ae091014462f721226b89c46eef4a6"
    )
    if (
        hashlib.sha256(raw).hexdigest()
        != "7455d7d77d8702d309de7ca63fd4d7da820cf73236e7a02787105596679c9aca"
        or manifest.manifest_id != expected_id
        or manifest.record_count != 136
        or manifest.metadata.get("source_broad_manifest_id") != broad.manifest_id
        or len({row.intended_family for row in manifest.records}) != 34
    ):
        raise RunnerGateError("frozen Extension-1 confirmation identity changed")
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    if len(payload) != 2_510_974:
        raise RunnerGateError("frozen Extension-1 canonical payload changed")
    return manifest


def confirmed_family_summaries(
    broad: DatasetManifest,
    confirmation: DatasetManifest,
    generations: tuple[GenerationRecord, ...],
    rewards: tuple[RewardRecord, ...],
    *,
    group_size: int,
    expected_run_ids: tuple[str, ...],
) -> tuple[BoundaryFamilySummary, ...]:
    """Reduce each Stage-3 family over its four refined and four held-out items."""

    family_ids = {row.intended_family for row in confirmation.records}
    if not family_ids:
        return ()
    summaries = summarize_m0_boundary(
        broad,
        generations,
        rewards,
        group_size=group_size,
        expected_run_ids=expected_run_ids,
        additional_manifests=(confirmation,),
    )
    return tuple(row for row in summaries if row.intended_family_id in family_ids)


def _manifest(directory: Path, name: str) -> DatasetManifest:
    try:
        return DatasetManifest.model_validate_json((directory / name).read_bytes())
    except (OSError, ValueError) as error:
        raise RunnerGateError(f"invalid boundary source manifest: {name}") from error


def _rows(path: Path, model: type[GenerationRecord] | type[RewardRecord]):
    try:
        return tuple(
            model.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValueError) as error:
        raise RunnerGateError(f"invalid boundary source rows: {path.name}") from error


def load_boundary_live_source(
    config: PilotConfig, source_root: str | Path
) -> BoundaryLiveSource:
    """Load only the named, frozen initial and Extension-1 evidence."""

    root = Path(source_root)
    initial_broad_dir = root / "boundary-broad-20260809T214400Z"
    initial_confirm_dir = root / "boundary-confirm-20260810T144517Z"
    extension1_broad_dir = root / "boundary-broad-extension1-20260810T223000Z"
    extension1_refine_dir = root / "boundary-refine-extension1-20260811T104725Z"
    initial_broad = _manifest(initial_broad_dir, "a_candidate_manifest.json")
    initial_confirmation = _manifest(
        initial_confirm_dir, "a_candidate_confirmation_manifest.json"
    )
    if (
        initial_confirmation.metadata.get("source_broad_manifest_id")
        != initial_broad.manifest_id
    ):
        raise RunnerGateError(
            "initial confirmation is not linked to the initial broad scan"
        )
    try:
        initial_run = RunRecord.model_validate_json(
            (initial_confirm_dir / "run.json").read_bytes()
        )
    except (OSError, ValueError) as error:
        raise RunnerGateError("initial confirmation run record is invalid") from error
    if initial_run.status is not RunStatus.COMPLETED:
        raise RunnerGateError("initial confirmation is not completed evidence")
    contract = authenticate_extension1_source(
        config,
        extension1_broad_dir,
        extension1_refine_dir,
        expected_parent_manifest_id=initial_broad.manifest_id,
    )
    if contract.cohort_id != BOUNDARY_BROAD_EXTENSION_1_COHORT:
        raise RunnerGateError("authenticated source is not Extension-1")
    broad_manifest = _manifest(extension1_broad_dir, "a_candidate_manifest.json")
    broad_generations = _rows(
        extension1_broad_dir / "generations.jsonl", GenerationRecord
    )
    broad_rewards = _rows(extension1_broad_dir / "rewards.jsonl", RewardRecord)
    refine_generations = _rows(
        extension1_refine_dir / "generations.jsonl", GenerationRecord
    )
    refine_rewards = _rows(extension1_refine_dir / "rewards.jsonl", RewardRecord)
    audit_new_broad_cohort(broad_manifest, (initial_broad,), (initial_confirmation,))
    broad_summaries = summarize_m0_boundary(
        broad_manifest,
        broad_generations,
        broad_rewards,
        group_size=config.tinker.group_size,
        expected_run_ids=(contract.prior_run_ids[0],),
    )
    successes = {row.intended_family_id: row.total_successes for row in broad_summaries}
    candidates, audit = choose_refinement_family_ids(successes)
    refined_ids = set((*candidates, *audit))
    combined = summarize_m0_boundary(
        broad_manifest,
        (*broad_generations, *refine_generations),
        (*broad_rewards, *refine_rewards),
        group_size=config.tinker.group_size,
        expected_run_ids=contract.prior_run_ids,
    )
    refinement = tuple(row for row in combined if row.intended_family_id in refined_ids)
    if {row.intended_family_id for row in refinement} != refined_ids:
        raise RunnerGateError("Extension-1 refinement source grid is incomplete")
    return BoundaryLiveSource(
        contract,
        initial_broad,
        initial_confirmation,
        broad_manifest,
        broad_generations,
        broad_rewards,
        refine_generations,
        refine_rewards,
        successes,
        refinement,
    )


__all__ = [
    "BoundaryLiveSource",
    "capacity_cleared_confirmation",
    "confirmed_family_summaries",
    "load_boundary_live_source",
    "load_frozen_extension1_confirmation",
    "reconstruct_family_templates",
]
