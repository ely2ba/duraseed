"""Authenticate the one completed-refinement boundary continuation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from duraseed.boundary_live_fresh_resume import (
    FRESH_RESUME_MARKER,
    STALE_RUNTIME_INCIDENT,
    load_saved_extension2_manifest,
)
from duraseed.boundary_live_retry import INCIDENT
from duraseed.boundary_live_sampling import ACTION_CAPS, summarize
from duraseed.boundary_live_sources import (
    BoundaryLiveSource,
    load_frozen_extension1_confirmation,
)
from duraseed.config import PilotConfig
from duraseed.data.boundary import (
    BoundaryFamilySummary,
    assess_refinement_finalist_gate,
)
from duraseed.data.boundary_confirmation import choose_refinement_family_ids
from duraseed.data.manifests import DatasetManifest
from duraseed.run_records import (
    GenerationRecord,
    RewardRecord,
    RunRecord,
    RunStatus,
)
from duraseed.runners import RunnerGateError


CONFIRMATION_RESUME_RUN_ID = "boundary-live-20260813T130013Z"
CONFIRMATION_RESUME_GROUPS = {
    "extension1-confirm": 136,
    "extension2-broad": 256,
    "extension2-refine": 216,
}
CONFIRMATION_RESUME_SAMPLES = 5_792
CONFIRMATION_RESUME_FINALISTS = 43


@dataclass(frozen=True, slots=True)
class BoundaryConfirmationResumeSnapshot:
    """Fully authenticated local inputs needed after the refinement stop."""

    directory: Path
    run: RunRecord
    preflight: dict[str, Any]
    extension1_confirmation: DatasetManifest
    extension2: DatasetManifest
    broad_generations: tuple[GenerationRecord, ...]
    broad_rewards: tuple[RewardRecord, ...]
    refinement_generations: tuple[GenerationRecord, ...]
    refinement_rewards: tuple[RewardRecord, ...]
    family_successes: dict[str, int]
    refinement_summaries: tuple[BoundaryFamilySummary, ...]


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"invalid {label}")
    return value


def _rows(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    try:
        values = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid {label}") from error
    if any(not isinstance(row, dict) for row in values):
        raise RunnerGateError(f"invalid {label}")
    return values


def _validate_run_and_preflight(
    directory: Path, config: PilotConfig, source: BoundaryLiveSource
) -> tuple[RunRecord, dict[str, Any]]:
    try:
        run = RunRecord.model_validate_json((directory / "run.json").read_bytes())
    except (OSError, ValueError) as error:
        raise RunnerGateError("invalid completed-refinement run record") from error
    preflight = _object(directory / "preflight.json", "boundary preflight")
    expected_actions = {name: str(cap) for name, cap in ACTION_CAPS.items()}
    if (
        directory.name != CONFIRMATION_RESUME_RUN_ID
        or run.status is not RunStatus.INTERRUPTED
        or run.finished_at is None
        or run.run_kind != "m0_calibration"
        or run.method is not None
        or run.seed != 5
        or run.project_id != source.contract.project_id
        or run.parent_tinker_checkpoint_path != source.contract.state_checkpoint_path
        or run.resolved_config_hash != config.resolved_config_hash()
        or run.authorized_cost_usd != 120.0
        or run.reserved_cost_usd != 120.0
        or preflight.get("gate_name") != "boundary-extension"
        or preflight.get("run_id") != CONFIRMATION_RESUME_RUN_ID
        or preflight.get("authorized_cost_usd") != "120"
        or preflight.get("actions") != expected_actions
        or preflight.get("source")
        != json.loads(json.dumps(asdict(source.contract), default=str))
    ):
        raise RunnerGateError("completed-refinement run identity changed")
    return run, preflight


def _validate_markers_and_billing(directory: Path, run: RunRecord) -> None:
    if (directory / "pending_group.json").exists():
        raise RunnerGateError("completed refinement has an ambiguous pending group")
    marker = _object(directory / FRESH_RESUME_MARKER, "fresh refinement marker")
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "completed"
        or marker.get("action") != "extension2-refine"
        or marker.get("incident") != STALE_RUNTIME_INCIDENT
        or marker.get("recovery_git_commit")
        != "3f9fb51fbbabc934eae031592741ceb41bd914bf"
    ):
        raise RunnerGateError("fresh refinement completion marker changed")
    errors = _rows(directory / "errors.jsonl", "boundary error journal")
    if errors != (INCIDENT["error"],):
        raise RunnerGateError("boundary error history changed after refinement")
    billing = _object(directory / "billing.json", "boundary billing journal")
    actions = billing.get("actions")
    if not isinstance(actions, dict) or set(actions) != {
        "extension1-confirm",
        "extension2-broad",
        "extension2-refine",
    }:
        raise RunnerGateError("completed-refinement billing actions changed")
    try:
        observed_cost = sum(float(row["observed_cost_usd"]) for row in actions.values())
        observed_prefill = sum(
            int(row["observed_tokens"]["prefill"]) for row in actions.values()
        )
        observed_sample = sum(
            int(row["observed_tokens"]["sample"]) for row in actions.values()
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError(
            "completed-refinement billing journal is malformed"
        ) from error
    if (
        any(
            Decimal(str(actions[name].get("authorized_cost_usd"))) != ACTION_CAPS[name]
            for name in actions
        )
        or not math.isclose(
            observed_cost,
            float(billing.get("observed_cost_usd")),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(observed_cost, run.cost_usd, rel_tol=0.0, abs_tol=1e-12)
        or observed_prefill != run.prompt_tokens
        or observed_sample != run.sampled_tokens
    ):
        raise RunnerGateError("completed-refinement billing totals changed")


def _validate_journal(
    directory: Path,
    *,
    extension1: DatasetManifest,
    extension2: DatasetManifest,
) -> tuple[
    tuple[GenerationRecord, ...],
    tuple[RewardRecord, ...],
    tuple[GenerationRecord, ...],
    tuple[RewardRecord, ...],
    dict[str, set[str]],
]:
    expected = {
        "extension1-confirm": (extension1.manifest_id, tuple(range(16))),
        "extension2-broad": (extension2.manifest_id, tuple(range(4))),
        "extension2-refine": (extension2.manifest_id, tuple(range(4, 16))),
    }
    counts: Counter[str] = Counter()
    task_ids = {name: set() for name in expected}
    sample_ids: set[str] = set()
    broad_g: list[GenerationRecord] = []
    broad_r: list[RewardRecord] = []
    refine_g: list[GenerationRecord] = []
    refine_r: list[RewardRecord] = []
    total_samples = 0
    try:
        handle = (directory / "observation_groups.jsonl").open("r", encoding="utf-8")
    except OSError as error:
        raise RunnerGateError("completed-refinement journal is unreadable") from error
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                action = value["action"]
                task_id = value["task_id"]
                generations = tuple(
                    GenerationRecord.model_validate_json(json.dumps(row))
                    for row in value["generations"]
                )
                rewards = tuple(
                    RewardRecord.model_validate_json(json.dumps(row))
                    for row in value["rewards"]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise RunnerGateError(
                    "completed-refinement journal is malformed"
                ) from error
            if action not in expected or not isinstance(task_id, str):
                raise RunnerGateError(
                    "completed-refinement journal contains a later action"
                )
            manifest_id, indices = expected[action]
            by_reward = {row.sample_id: row for row in rewards}
            ids = {row.sample_id for row in generations}
            if (
                task_id in task_ids[action]
                or not generations
                or len(generations) != len(rewards)
                or tuple(row.sample_index for row in generations) != indices
                or len(ids) != len(generations)
                or set(by_reward) != ids
                or sample_ids.intersection(ids)
                or any(
                    row.task_id != task_id
                    or row.task_manifest_id != manifest_id
                    or row.run_id != f"{CONFIRMATION_RESUME_RUN_ID}:{action}"
                    or by_reward[row.sample_id].task_id != task_id
                    or by_reward[row.sample_id].reward != row.reward
                    for row in generations
                )
            ):
                raise RunnerGateError("completed-refinement group coordinates changed")
            counts[action] += 1
            task_ids[action].add(task_id)
            sample_ids.update(ids)
            total_samples += len(generations)
            if action == "extension2-broad":
                broad_g.extend(generations)
                broad_r.extend(by_reward[row.sample_id] for row in generations)
            elif action == "extension2-refine":
                refine_g.extend(generations)
                refine_r.extend(by_reward[row.sample_id] for row in generations)
    if (
        dict(counts) != CONFIRMATION_RESUME_GROUPS
        or total_samples != CONFIRMATION_RESUME_SAMPLES
    ):
        raise RunnerGateError("completed-refinement journal grid changed")
    if task_ids["extension1-confirm"] != {row.task_id for row in extension1.records}:
        raise RunnerGateError("Extension-1 confirmation group set changed")
    if task_ids["extension2-broad"] != {row.task_id for row in extension2.records}:
        raise RunnerGateError("Extension-2 broad group set changed")
    return (
        tuple(broad_g),
        tuple(broad_r),
        tuple(refine_g),
        tuple(refine_r),
        task_ids,
    )


def validate_confirmation_resume(
    directory: str | Path,
    *,
    config: PilotConfig,
    source: BoundaryLiveSource,
) -> BoundaryConfirmationResumeSnapshot:
    """Validate the exact 608-group stop before creating a remote client."""

    root = Path(directory).resolve()
    run, preflight = _validate_run_and_preflight(root, config, source)
    _validate_markers_and_billing(root, run)
    extension1 = load_frozen_extension1_confirmation(
        root / "extension1_confirmation_manifest.json",
        source.extension1_broad_manifest,
    )
    manifest_path = root / "extension2_broad_manifest.json"
    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as error:
        raise RunnerGateError("saved Extension-2 broad manifest is invalid") from error
    expected_digest = STALE_RUNTIME_INCIDENT["snapshot_sha256"][
        "extension2_broad_manifest.json"
    ]
    if "sha256:" + hashlib.sha256(raw_manifest).hexdigest() != expected_digest:
        raise RunnerGateError("saved Extension-2 broad manifest changed")
    extension2 = load_saved_extension2_manifest(root)
    if run.task_manifest_ids != {
        "boundary_extension2_broad": extension2.manifest_id,
        "boundary_extension1_confirmation": extension1.manifest_id,
    }:
        raise RunnerGateError("completed-refinement manifest identities changed")
    broad_g, broad_r, refine_g, refine_r, task_ids = _validate_journal(
        root, extension1=extension1, extension2=extension2
    )
    broad_summaries = summarize(
        extension2,
        broad_g,
        broad_r,
        config,
        expected=(f"{CONFIRMATION_RESUME_RUN_ID}:extension2-broad",),
    )
    successes = {row.intended_family_id: row.total_successes for row in broad_summaries}
    candidates, audit = choose_refinement_family_ids(successes)
    selected = frozenset((*candidates, *audit))
    selected_tasks = {
        row.task_id for row in extension2.records if row.intended_family in selected
    }
    combined = summarize(
        extension2,
        (*broad_g, *refine_g),
        (*broad_r, *refine_r),
        config,
        expected=(
            f"{CONFIRMATION_RESUME_RUN_ID}:extension2-broad",
            f"{CONFIRMATION_RESUME_RUN_ID}:extension2-refine",
        ),
    )
    refinement = tuple(row for row in combined if row.intended_family_id in selected)
    finalists = tuple(
        row.intended_family_id
        for row in refinement
        if assess_refinement_finalist_gate(row).eligible
    )
    if (
        len(candidates) != STALE_RUNTIME_INCIDENT["refinement"]["positive_families"]
        or len(audit) != STALE_RUNTIME_INCIDENT["refinement"]["audit_families"]
        or task_ids["extension2-refine"] != selected_tasks
        or len(refinement) != len(selected)
        or len(finalists) != CONFIRMATION_RESUME_FINALISTS
    ):
        raise RunnerGateError("completed Extension-2 refinement selection changed")
    return BoundaryConfirmationResumeSnapshot(
        root,
        run,
        preflight,
        extension1,
        extension2,
        broad_g,
        broad_r,
        refine_g,
        refine_r,
        successes,
        refinement,
    )


__all__ = [
    "BoundaryConfirmationResumeSnapshot",
    "CONFIRMATION_RESUME_FINALISTS",
    "CONFIRMATION_RESUME_GROUPS",
    "CONFIRMATION_RESUME_RUN_ID",
    "CONFIRMATION_RESUME_SAMPLES",
    "validate_confirmation_resume",
]
