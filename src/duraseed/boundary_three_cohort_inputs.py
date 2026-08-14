"""Read-only authentication and replay for the three boundary cohorts."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from duraseed.boundary_consolidated_journal import load_consolidated_journal
from duraseed.boundary_live_fresh_resume import (
    STALE_RUNTIME_INCIDENT,
    load_saved_extension2_manifest,
)
from duraseed.boundary_live_sources import (
    BoundaryLiveSource,
    confirmed_family_summaries,
    load_boundary_live_source,
    load_frozen_extension1_confirmation,
    reconstruct_family_templates,
)
from duraseed.config import PilotConfig
from duraseed.data.boundary import (
    assess_confirmation_observation_gate,
    assess_refinement_finalist_gate,
    summarize_m0_boundary,
)
from duraseed.data.boundary_confirmation import (
    build_confirmation_manifest,
    choose_refinement_family_ids,
)
from duraseed.data.boundary_freeze_contracts import BoundaryFreezeCohort
from duraseed.data.boundary_protocol import (
    BOUNDARY_BROAD_EXTENSION_1_COHORT,
    BOUNDARY_BROAD_EXTENSION_2_COHORT,
    BOUNDARY_BROAD_INITIAL_COHORT,
    BOUNDARY_ENGINEERING_SEED,
)
from duraseed.data.manifests import DatasetManifest
from duraseed.data.panel_capacity import FamilyCapacityAudit, FamilySplitCapacity
from duraseed.run_records import GenerationRecord, RewardRecord, RunRecord, RunStatus
from duraseed.runners import RunnerGateError
from duraseed.tasks.tces import TCESGeneratorConfig


_REPO = Path(__file__).resolve().parents[2]
_FROZEN_PREFIX = Path("frozen/v0/runs/tinker-calibration/boundary")
_PRESERVED_FILES = (
    "boundary-broad-20260809T214400Z/a_candidate_manifest.json",
    "boundary-broad-20260809T214400Z/generations.jsonl",
    "boundary-broad-20260809T214400Z/rewards.jsonl",
    "boundary-refine-20260810T005730Z/generations.jsonl",
    "boundary-refine-20260810T005730Z/rewards.jsonl",
    "boundary-confirm-20260810T144517Z/a_candidate_confirmation_manifest.json",
    "boundary-confirm-20260810T144517Z/generations.jsonl",
    "boundary-confirm-20260810T144517Z/rewards.jsonl",
    "boundary-confirm-20260810T144517Z/run.json",
    "boundary-broad-extension1-20260810T223000Z/a_candidate_manifest.json",
    "boundary-broad-extension1-20260810T223000Z/generations.jsonl",
    "boundary-broad-extension1-20260810T223000Z/rewards.jsonl",
    "boundary-broad-extension1-20260810T223000Z/run.json",
    "boundary-broad-extension1-20260810T223000Z/preflight.public.json",
    "boundary-refine-extension1-20260811T104725Z/a_candidate_manifest.json",
    "boundary-refine-extension1-20260811T104725Z/generations.jsonl",
    "boundary-refine-extension1-20260811T104725Z/rewards.jsonl",
    "boundary-refine-extension1-20260811T104725Z/run.json",
    "boundary-refine-extension1-20260811T104725Z/preflight.public.json",
)
_Model = TypeVar("_Model", bound=BaseModel)


def _provenance_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("v0-artifacts.sha256", "public-projections.sha256"):
        for line in (_REPO / "provenance" / name).read_text().splitlines():
            digest, path = line.split("  ", 1)
            result[path] = digest
    return result


def _authenticate_preserved_source(root: Path) -> None:
    expected = _provenance_hashes()
    for relative in _PRESERVED_FILES:
        key = (_FROZEN_PREFIX / relative).as_posix()
        path = root / relative
        if key not in expected or not path.is_file():
            raise RunnerGateError(f"missing authenticated boundary source: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[key]:
            raise RunnerGateError(f"boundary source SHA-256 changed: {relative}")


def _manifest(path: Path, label: str) -> DatasetManifest:
    try:
        return DatasetManifest.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise RunnerGateError(f"invalid {label} manifest") from error


def _rows(path: Path, model: type[_Model], label: str) -> tuple[_Model, ...]:
    try:
        return tuple(
            model.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValueError) as error:
        raise RunnerGateError(f"invalid {label} rows") from error


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"invalid {label}")
    return value


def _load_capacity_audits(path: Path) -> tuple[FamilyCapacityAudit, ...]:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            FamilyCapacityAudit(
                row["family_id"],
                tuple(FamilySplitCapacity(**item) for item in row["split_capacities"]),
                row["available_disjoint_instances"],
                row["passed"],
            )
            for row in values
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerGateError("invalid Extension-2 capacity evidence") from error


def _validate_completed_run(
    directory: Path,
    config: PilotConfig,
    source: BoundaryLiveSource,
    manifests: Mapping[str, DatasetManifest],
) -> RunRecord:
    try:
        run = RunRecord.model_validate_json((directory / "run.json").read_bytes())
    except (OSError, ValueError) as error:
        raise RunnerGateError("invalid completed boundary run") from error
    expected_ids = {
        "boundary_extension1_confirmation": manifests["extension1-confirm"].manifest_id,
        "boundary_extension2_broad": manifests["extension2-broad"].manifest_id,
        "boundary_extension2_confirmation": manifests["extension2-confirm"].manifest_id,
    }
    if (
        directory.name != STALE_RUNTIME_INCIDENT["run_id"]
        or (directory / "pending_group.json").exists()
        or run.status is not RunStatus.COMPLETED
        or run.finished_at is None
        or run.run_kind != "m0_calibration"
        or run.method is not None
        or run.seed != BOUNDARY_ENGINEERING_SEED
        or run.task_manifest_ids != expected_ids
        or run.resolved_config_hash != config.resolved_config_hash()
        or run.model_id != source.contract.model_id
        or run.renderer != source.contract.renderer
        or run.lora_rank != source.contract.lora_rank
        or run.project_id != source.contract.project_id
        or run.parent_tinker_checkpoint_path != source.contract.state_checkpoint_path
    ):
        raise RunnerGateError("completed boundary run identity changed")
    return run


def _initial_cohort(
    config: PilotConfig, source: BoundaryLiveSource, root: Path
) -> BoundaryFreezeCohort:
    broad_dir = root / "boundary-broad-20260809T214400Z"
    refine_dir = root / "boundary-refine-20260810T005730Z"
    confirm_dir = root / "boundary-confirm-20260810T144517Z"
    generations = tuple(
        row
        for directory in (broad_dir, refine_dir, confirm_dir)
        for row in _rows(
            directory / "generations.jsonl", GenerationRecord, directory.name
        )
    )
    rewards = tuple(
        row
        for directory in (broad_dir, refine_dir, confirm_dir)
        for row in _rows(directory / "rewards.jsonl", RewardRecord, directory.name)
    )
    summaries = summarize_m0_boundary(
        source.initial_broad_manifest,
        generations,
        rewards,
        group_size=config.tinker.group_size,
        expected_run_ids=(broad_dir.name, refine_dir.name, confirm_dir.name),
        additional_manifests=(source.initial_confirmation_manifest,),
    )
    confirmed = {
        row.intended_family for row in source.initial_confirmation_manifest.records
    }
    finalists = tuple(row for row in summaries if row.intended_family_id in confirmed)
    locked = tuple(
        row.intended_family_id
        for row in finalists
        if assess_confirmation_observation_gate(row).eligible
    )
    if len(finalists) != len(confirmed) or len(locked) != 15:
        raise RunnerGateError("initial cohort replay changed")
    return BoundaryFreezeCohort(
        BOUNDARY_BROAD_INITIAL_COHORT,
        source.initial_broad_manifest,
        source.initial_confirmation_manifest,
        finalists,
        source.contract.sampler_checkpoint_path,
        locked,
    )


def load_three_cohort_inputs(
    config: PilotConfig,
    source_root: str | Path,
    consolidated_run: str | Path,
) -> tuple[BoundaryFreezeCohort, BoundaryFreezeCohort, BoundaryFreezeCohort]:
    """Replay the completed raw evidence into the private reducer's typed inputs."""

    root = Path(source_root).resolve()
    directory = Path(consolidated_run).resolve()
    try:
        if not (directory / "extension2_confirmation_manifest.json").is_file():
            raise RunnerGateError(
                "three-cohort replay requires completed Extension-2 confirmation"
            )
        _authenticate_preserved_source(root)
        source = load_boundary_live_source(config, root)
        extension1 = load_frozen_extension1_confirmation(
            directory / "extension1_confirmation_manifest.json",
            source.extension1_broad_manifest,
        )
        extension2 = load_saved_extension2_manifest(directory)
        broad_path = directory / "extension2_broad_manifest.json"
        expected_broad_sha = STALE_RUNTIME_INCIDENT["snapshot_sha256"][broad_path.name]
        if (
            "sha256:" + hashlib.sha256(broad_path.read_bytes()).hexdigest()
            != expected_broad_sha
        ):
            raise RunnerGateError("saved Extension-2 manifest bytes changed")
        confirmation = _manifest(
            directory / "extension2_confirmation_manifest.json",
            "Extension-2 confirmation",
        )
        manifests = {
            "extension1-confirm": extension1,
            "extension2-broad": extension2,
            "extension2-refine": extension2,
            "extension2-confirm": confirmation,
        }
        _validate_completed_run(directory, config, source, manifests)
        generation, reward, tasks = load_consolidated_journal(
            directory, run_id=directory.name, manifests=manifests
        )
        expected_sampling = (
            config.evaluation["temperature"],
            config.evaluation["top_p"],
            config.tinker.max_sampled_tokens,
        )
        if any(
            row.seed != BOUNDARY_ENGINEERING_SEED
            or row.sampler_checkpoint_path != source.contract.sampler_checkpoint_path
            or row.origin_sampler_checkpoint_path
            != source.contract.sampler_checkpoint_path
            or (row.sampling_temperature, row.sampling_top_p, row.sampling_max_tokens)
            != expected_sampling
            for rows in generation.values()
            for row in rows
        ):
            raise RunnerGateError("consolidated boundary sampling identity changed")
        if (
            tasks["extension1-confirm"] != {row.task_id for row in extension1.records}
            or tasks["extension2-broad"] != {row.task_id for row in extension2.records}
            or tasks["extension2-confirm"]
            != {row.task_id for row in confirmation.records}
        ):
            raise RunnerGateError("consolidated boundary task coverage changed")
        broad = summarize_m0_boundary(
            extension2,
            generation["extension2-broad"],
            reward["extension2-broad"],
            group_size=config.tinker.group_size,
            expected_run_ids=(f"{directory.name}:extension2-broad",),
        )
        successes = {row.intended_family_id: row.total_successes for row in broad}
        positive, audit = choose_refinement_family_ids(successes)
        selected = frozenset((*positive, *audit))
        selected_tasks = {
            row.task_id for row in extension2.records if row.intended_family in selected
        }
        if tasks["extension2-refine"] != selected_tasks:
            raise RunnerGateError("Extension-2 refinement selection changed")
        combined = summarize_m0_boundary(
            extension2,
            (*generation["extension2-broad"], *generation["extension2-refine"]),
            (*reward["extension2-broad"], *reward["extension2-refine"]),
            group_size=config.tinker.group_size,
            expected_run_ids=(
                f"{directory.name}:extension2-broad",
                f"{directory.name}:extension2-refine",
            ),
        )
        refinement = tuple(
            row for row in combined if row.intended_family_id in selected
        )
        finalists = tuple(
            sorted(
                row.intended_family_id
                for row in refinement
                if assess_refinement_finalist_gate(row).eligible
            )
        )
        audits = _load_capacity_audits(directory / "extension2_capacity_audits.json")
        cleared = tuple(row.family_id for row in audits if row.passed)
        templates = reconstruct_family_templates(extension2, finalists)
        expected_confirmation = build_confirmation_manifest(
            TCESGeneratorConfig(**config.tasks.tces.generator_kwargs()),
            extension2,
            cleared,
            templates={family_id: templates[family_id] for family_id in cleared},
        )
        marker = _object(
            directory / "extension2_confirmation_preparation.json",
            "Extension-2 confirmation preparation",
        )
        if (
            tuple(row.family_id for row in audits) != finalists
            or confirmation != expected_confirmation
            or marker.get("run_id") != directory.name
            or marker.get("capacity_family_ids") != list(finalists)
            or marker.get("capacity_passed_family_ids") != list(cleared)
            or marker.get("confirmation_manifest_id") != confirmation.manifest_id
        ):
            raise RunnerGateError("Extension-2 confirmation preparation changed")
        extension1_summaries = confirmed_family_summaries(
            source.extension1_broad_manifest,
            extension1,
            (
                *source.extension1_broad_generations,
                *source.extension1_refinement_generations,
                *generation["extension1-confirm"],
            ),
            (
                *source.extension1_broad_rewards,
                *source.extension1_refinement_rewards,
                *reward["extension1-confirm"],
            ),
            group_size=config.tinker.group_size,
            expected_run_ids=(
                *source.contract.prior_run_ids,
                f"{directory.name}:extension1-confirm",
            ),
        )
        extension2_summaries = confirmed_family_summaries(
            extension2,
            confirmation,
            (
                *generation["extension2-broad"],
                *generation["extension2-refine"],
                *generation["extension2-confirm"],
            ),
            (
                *reward["extension2-broad"],
                *reward["extension2-refine"],
                *reward["extension2-confirm"],
            ),
            group_size=config.tinker.group_size,
            expected_run_ids=(
                f"{directory.name}:extension2-broad",
                f"{directory.name}:extension2-refine",
                f"{directory.name}:extension2-confirm",
            ),
        )
        return (
            _initial_cohort(config, source, root),
            BoundaryFreezeCohort(
                BOUNDARY_BROAD_EXTENSION_1_COHORT,
                source.extension1_broad_manifest,
                extension1,
                extension1_summaries,
                source.contract.sampler_checkpoint_path,
            ),
            BoundaryFreezeCohort(
                BOUNDARY_BROAD_EXTENSION_2_COHORT,
                extension2,
                confirmation,
                extension2_summaries,
                source.contract.sampler_checkpoint_path,
            ),
        )
    except RunnerGateError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RunnerGateError("three-cohort evidence replay failed") from error
