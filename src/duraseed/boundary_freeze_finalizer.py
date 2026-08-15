"""One-shot local three-cohort freeze with exact archived-v0 equivalence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from duraseed.boundary_teacher_tokens import archived_token_counter
from duraseed.boundary_three_cohort_inputs import load_three_cohort_inputs
from duraseed.config import load_pilot_config
from duraseed.data.boundary_freeze import _reduce_three_cohort_panels
from duraseed.data.boundary_freeze_contracts import (
    BoundaryFreezeResult,
    freeze_settings_from_config,
)
from duraseed.data.io import atomic_write_bytes
from duraseed.data.manifests import write_manifest
from duraseed.provenance import (
    canonical_json_bytes,
    canonical_json_value,
    sha256_bytes,
)
from duraseed.run_records import (
    RunRecord,
    RunStatus,
    read_run_record,
    write_jsonl,
    write_run_record,
)
from duraseed.runtime import MODEL_ID, RENDERER_NAME
from duraseed.runners import RunnerGateError


_REPO = Path(__file__).resolve().parents[2]
_V0_SOURCE_FILES = (
    "src/duraseed/tinker_boundary_panel_freeze.py",
    "src/duraseed/tinker_boundary_confirmation.py",
)
_V1_SOURCE_FILES = (
    "src/duraseed/data/boundary_freeze.py",
    "src/duraseed/data/boundary_freeze_contracts.py",
    "src/duraseed/boundary_teacher_tokens.py",
)


class BoundaryFreezeFinalizationError(RunnerGateError):
    """The production freeze or exact old/new comparison did not pass."""


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise BoundaryFreezeFinalizationError(
            "cannot identify reducer commit"
        ) from error


def _json(path: Path, value: Any) -> bytes:
    payload = canonical_json_bytes(value) + b"\n"
    atomic_write_bytes(path, payload)
    return payload


def _source_payload(cohorts, settings) -> dict[str, Any]:
    return {
        "cohorts": canonical_json_value(cohorts),
        "model_id": MODEL_ID,
        "renderer": RENDERER_NAME,
        "settings": settings.projection(),
    }


def _scientific_payload(result: BoundaryFreezeResult) -> dict[str, Any]:
    value = canonical_json_value(result)
    if not isinstance(value, dict):  # pragma: no cover - dataclass contract
        raise BoundaryFreezeFinalizationError("freeze result is not an object")
    return value


def _source_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: _sha256(root / name) for name in names}


def _consolidated_source_identity(
    directory: Path, completed: RunRecord, sampler_checkpoint_path: str
) -> dict[str, Any]:
    try:
        preflight = json.loads((directory / "preflight.json").read_text())
        source = preflight["source"]
    except (KeyError, OSError, json.JSONDecodeError) as error:
        raise BoundaryFreezeFinalizationError(
            "consolidated boundary preflight is invalid"
        ) from error
    if not isinstance(source, dict):
        raise BoundaryFreezeFinalizationError("consolidated boundary source is invalid")
    step = source.get("training_step")
    if (
        source.get("sampler_checkpoint_path") != sampler_checkpoint_path
        or source.get("state_checkpoint_path")
        != completed.parent_tinker_checkpoint_path
        or type(step) is not int
        or step != 2
    ):
        raise BoundaryFreezeFinalizationError(
            "consolidated M0 checkpoint identity changed"
        )
    return source


def _run_reducers(
    *,
    source_payload: dict[str, Any],
    cohorts,
    settings,
    staging: Path,
    v0_source_root: Path,
) -> tuple[BoundaryFreezeResult, bytes, bytes]:
    normalized = staging / "normalized_input.json"
    old_output = staging / "v0_result.json"
    _json(normalized, source_payload)
    python = v0_source_root / ".venv/bin/python"
    oracle = _REPO / "tools/three_cohort_v0_oracle.py"
    if not python.is_file() or not oracle.is_file():
        raise BoundaryFreezeFinalizationError("archived v0 oracle runtime is missing")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(v0_source_root / "src")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    old = subprocess.Popen(
        [
            str(python),
            str(oracle),
            "--source-root",
            str(v0_source_root),
            "--input",
            str(normalized),
            "--output",
            str(old_output),
        ],
        cwd=v0_source_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        current = _reduce_three_cohort_panels(
            cohorts,
            settings=settings,
            teacher_trace_token_counter=archived_token_counter(),
        )
    except BaseException:
        old.terminate()
        old.communicate()
        raise
    stdout, stderr = old.communicate()
    if old.returncode != 0:
        detail = (stderr or stdout).strip()[-4000:]
        raise BoundaryFreezeFinalizationError(
            f"archived v0 reducer failed ({old.returncode}): {detail}"
        )
    old_value = json.loads(old_output.read_text(encoding="utf-8"))
    old_bytes = canonical_json_bytes(old_value)
    current_bytes = canonical_json_bytes(_scientific_payload(current))
    if old_bytes != current_bytes:
        raise BoundaryFreezeFinalizationError(
            "production old-v0/current-v1 scientific outputs differ: "
            f"old={sha256_bytes(old_bytes)}, current={sha256_bytes(current_bytes)}"
        )
    normalized.unlink()
    old_output.unlink()
    return current, old_bytes, current_bytes


def _write_scientific_artifacts(directory: Path, result: BoundaryFreezeResult) -> None:
    write_manifest(
        directory / "a_candidate_manifest.json", result.combined_broad_manifest
    )
    write_manifest(
        directory / "a_candidate_confirmation_manifest.json",
        result.combined_confirmation_manifest,
    )
    write_jsonl(
        directory / "confirmation_family_table.jsonl", result.confirmation_family_table
    )
    write_jsonl(
        directory / "split_capacity_audits.jsonl",
        (asdict(row) for row in result.capacity_audits),
    )
    write_jsonl(
        directory / "panel_candidate_table.jsonl",
        (asdict(row) for row in result.candidates),
    )
    write_jsonl(
        directory / "selected_panel_capacity_audits.jsonl",
        (asdict(row) for row in result.selected_capacity_audits),
    )
    _json(
        directory / "teacher_trace_token_counts.json",
        result.teacher_trace_token_counts,
    )
    _json(directory / "confirmation_summary.json", result.confirmation_summary)
    if result.panel_artifact is not None:
        _json(directory / "panel_candidate_table.json", result.candidate_payload)
        _json(
            directory / "panel_matching_report.json",
            result.matching_report_payload,
        )
        _json(directory / "target_sentinel_panels.json", result.panel_artifact)


def finalize_three_cohort_freeze(
    *,
    run_id: str,
    source_root: str | Path,
    consolidated_run: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    v0_source_root: str | Path,
) -> Path:
    """Publish a local freeze only after independent production reducers agree."""

    if not run_id.strip() or any(value in run_id for value in "/\\"):
        raise BoundaryFreezeFinalizationError("run_id must be one filename token")
    output = Path(output_root).resolve() / run_id
    if output.exists():
        raise BoundaryFreezeFinalizationError(f"output already exists: {output}")
    root = output.parent
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=root))
    try:
        config = load_pilot_config(config_path)
        consolidated = Path(consolidated_run).resolve()
        completed = read_run_record(consolidated)
        if completed.status is not RunStatus.COMPLETED:
            raise BoundaryFreezeFinalizationError("boundary run is not completed")
        cohorts = load_three_cohort_inputs(config, source_root, consolidated)
        source_identity = _consolidated_source_identity(
            consolidated, completed, cohorts[0].sampler_checkpoint_path
        )
        settings = freeze_settings_from_config(config)
        source = _source_payload(cohorts, settings)
        result, old_bytes, current_bytes = _run_reducers(
            source_payload=source,
            cohorts=cohorts,
            settings=settings,
            staging=staging,
            v0_source_root=Path(v0_source_root).resolve(),
        )
        scientific_hash = sha256_bytes(current_bytes)
        token_hash = sha256_bytes(
            canonical_json_bytes(result.teacher_trace_token_counts)
        )
        equivalence = {
            "schema_version": "duraseed-three-cohort-freeze-equivalence-v1",
            "status": "passed",
            "old_new_identical": old_bytes == current_bytes,
            "token_count_map_identical": True,
            "source_run_id": consolidated.name,
            "source_input_sha256": sha256_bytes(canonical_json_bytes(source)),
            "freeze_settings_sha256": settings.projection_sha256,
            "scientific_outputs_sha256": scientific_hash,
            "teacher_trace_token_counts_sha256": token_hash,
            "compared_fields": sorted(_scientific_payload(result)),
            "archived_v0": {
                "git_commit": _git_commit(Path(v0_source_root).resolve()),
                "source_sha256": _source_hashes(
                    Path(v0_source_root).resolve(), _V0_SOURCE_FILES
                ),
                "scientific_outputs_sha256": sha256_bytes(old_bytes),
            },
            "current_v1": {
                "git_commit": _git_commit(_REPO),
                "source_sha256": _source_hashes(_REPO, _V1_SOURCE_FILES),
                "scientific_outputs_sha256": scientific_hash,
            },
        }
        _write_scientific_artifacts(staging, result)
        equivalence_raw = _json(staging / "three_cohort_equivalence.json", equivalence)
        initial = Path(source_root).resolve() / "boundary-confirm-20260810T144517Z"
        preflight = {
            "status": "completed_local_freeze",
            "phase": "m0_tces_boundary_three_cohort_panel_freeze",
            "source_kind": "three_cohort_boundary_freeze_v1",
            "remote_execution": False,
            "execution_authorized": False,
            "pilot_started": False,
            "run_id": run_id,
            "source": {
                "source_kind": "three_cohort_boundary_freeze_v1",
                "confirmation_run_directories": [
                    str(initial),
                    str(consolidated),
                    str(consolidated),
                ],
                "consolidated_run_directory": str(consolidated),
                "cohort_ids": [row.cohort_id for row in cohorts],
                "combined_broad_manifest_id": (
                    result.combined_broad_manifest.manifest_id
                ),
                "combined_confirmation_manifest_id": (
                    result.combined_confirmation_manifest.manifest_id
                ),
                "sampler_checkpoint_path": source_identity["sampler_checkpoint_path"],
                "state_checkpoint_path": source_identity["state_checkpoint_path"],
                "training_step": source_identity["training_step"],
                "equivalence_sha256": sha256_bytes(equivalence_raw),
            },
            "selection": {
                "minimum_candidate_count": 36,
                "panel_family_count": 24,
                "intermediate_family_ids": list(result.intermediate_family_ids),
                "allocation_seed": settings.allocation_seed,
                "both_extensions_required": True,
                "further_extension_authorized": False,
            },
            "cost": {
                "prompt_tokens": 0,
                "sampled_tokens": 0,
                "train_tokens": 0,
                "cost_usd": 0.0,
            },
            "unresolved_deviations": [],
        }
        _json(staging / "preflight.json", preflight)
        now = datetime.now(UTC)
        write_run_record(
            staging,
            RunRecord(
                protocol_version=config.protocol.version,
                git_commit=_git_commit(_REPO),
                resolved_config_hash=config.resolved_config_hash(),
                run_kind="m0_calibration",
                method=None,
                seed=result.combined_broad_manifest.root_seed,
                model_id=completed.model_id,
                renderer=completed.renderer,
                lora_rank=completed.lora_rank,
                task_manifest_ids={
                    "boundary_composite_a_candidate": (
                        result.combined_broad_manifest.manifest_id
                    ),
                    "boundary_confirmation_a_candidate": (
                        result.combined_confirmation_manifest.manifest_id
                    ),
                },
                parent_tinker_checkpoint_path=completed.parent_tinker_checkpoint_path,
                status=RunStatus.COMPLETED,
                started_at=now,
                updated_at=now,
                finished_at=now,
                project_id=completed.project_id,
                tinker_sdk_version=completed.tinker_sdk_version,
                tinker_cookbook_version=completed.tinker_cookbook_version,
                cost_usd=0.0,
                deviations=[
                    "local-only exact old-v0/current-v1 three-cohort freeze",
                    "no Tinker API call",
                    "Pilot 0 not started",
                ],
            ),
        )
        os.replace(staging, output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["BoundaryFreezeFinalizationError", "finalize_three_cohort_freeze"]
