"""One-shot local construction and v0/v1 equivalence of calibration sources."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from duraseed.calibration_build import write_calibration_sources
from duraseed.config import load_pilot_config
from duraseed.data.manifests import DatasetManifest, read_manifest
from duraseed.data.panel_split_production import (
    PanelSplitManifests,
    build_production_panel_splits,
    production_forbidden_records,
    write_equivalent_panel_splits,
)
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.data.sealing import ExecutionContext
from duraseed.data.stage_a_prompt_pools import (
    build_stage_a_prompt_pools,
    write_stage_a_prompt_pool_bundle,
)
from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.training.teacher_allocation_sources import (
    TeacherAllocationSources,
    validate_teacher_allocation_base_sources,
)


_REPO = Path(__file__).resolve().parents[2]
_V0_BUILDER = Path("src/duraseed/tinker_teacher_dose.py")
_V0_BUILDER_SHA256 = (
    "sha256:5e461b768f7b00deac02a101a272242bd3fff910e222f8c8c8a2426a6827944e"
)
_V1_BUILDER = Path("src/duraseed/data/panel_split_manifest.py")
_V1_BUILDER_SHA256 = (
    "sha256:f8dc0c60ee4fdae7adccb5862c8c10171f346a0e3e3bd5fed5929409446c5a9a"
)


class PanelSourceFinalizationError(RunnerGateError):
    """The local source construction or exact old/new comparison failed."""


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary_objects(
    directory: Path,
) -> tuple[FamilyPanelArtifact, DatasetManifest, DatasetManifest]:
    source = _require_boundary_equivalence(directory)
    try:
        panel_path = directory / "target_sentinel_panels.json"
        panel = FamilyPanelArtifact.model_validate_json(panel_path.read_bytes())
        broad = read_manifest(
            directory / "a_candidate_manifest.json", context=ExecutionContext.SELECTION
        )
        confirmation = read_manifest(
            directory / "a_candidate_confirmation_manifest.json",
            context=ExecutionContext.SELECTION,
        )
    except (OSError, ValueError) as error:
        raise PanelSourceFinalizationError(
            "frozen boundary source objects are invalid"
        ) from error
    if (
        source.get("combined_broad_manifest_id") != broad.manifest_id
        or source.get("combined_confirmation_manifest_id") != confirmation.manifest_id
    ):
        raise PanelSourceFinalizationError("three-cohort boundary manifest IDs differ")
    return panel, broad, confirmation


def _require_boundary_equivalence(directory: Path) -> dict[str, object]:
    try:
        raw = (directory / "three_cohort_equivalence.json").read_bytes()
        value = json.loads(raw)
        preflight = json.loads((directory / "preflight.json").read_bytes())
        source = preflight["source"]
        archived = value["archived_v0"]
        current = value["current_v1"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise PanelSourceFinalizationError(
            "three-cohort boundary equivalence is invalid"
        ) from error
    scientific_hash = value.get("scientific_outputs_sha256")
    consolidated = source.get("consolidated_run_directory")
    if (
        not isinstance(value, dict)
        or not isinstance(preflight, dict)
        or not isinstance(source, dict)
        or not isinstance(archived, dict)
        or not isinstance(current, dict)
        or raw != canonical_json_bytes(value) + b"\n"
        or value.get("schema_version") != "duraseed-three-cohort-freeze-equivalence-v1"
        or value.get("status") != "passed"
        or value.get("old_new_identical") is not True
        or value.get("token_count_map_identical") is not True
        or not isinstance(scientific_hash, str)
        or not scientific_hash
        or archived.get("scientific_outputs_sha256") != scientific_hash
        or current.get("scientific_outputs_sha256") != scientific_hash
        or source.get("equivalence_sha256") != sha256_bytes(raw)
        or not isinstance(consolidated, str)
        or not consolidated
        or value.get("source_run_id") != Path(consolidated).name
    ):
        raise PanelSourceFinalizationError(
            "three-cohort boundary equivalence did not pass"
        )
    return source


def _start_archived_builder(
    *,
    v0_source_root: Path,
    boundary_directory: Path,
    prompt_directory: Path,
    output_path: Path,
) -> subprocess.Popen[str]:
    python = v0_source_root / ".venv/bin/python"
    oracle = _REPO / "tools/panel_split_v0_oracle.py"
    if (
        not python.is_file()
        or not oracle.is_file()
        or _sha256(v0_source_root / _V0_BUILDER) != _V0_BUILDER_SHA256
    ):
        raise PanelSourceFinalizationError("accepted archived v0 builder is missing")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(v0_source_root / "src")
    environment["HF_HUB_OFFLINE"] = "1"
    environment.pop("TINKER_API_KEY", None)
    environment.pop("TINKER_PROJECT_ID", None)
    command = [
        str(python),
        str(oracle),
        "--source-root",
        str(v0_source_root),
        "--boundary",
        str(boundary_directory),
        "--prompts",
        str(prompt_directory),
        "--output",
        str(output_path),
    ]
    try:
        return subprocess.Popen(
            command,
            cwd=v0_source_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise PanelSourceFinalizationError(
            "cannot start archived v0 builder"
        ) from error


def _finish_archived_builder(
    process: subprocess.Popen[str], output_path: Path
) -> PanelSplitManifests:
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        detail = (stderr or stdout).strip()[-4000:]
        raise PanelSourceFinalizationError(
            f"archived v0 panel builder failed ({process.returncode}): {detail}"
        )
    try:
        raw = output_path.read_bytes()
        value = json.loads(raw)
        if raw != canonical_json_bytes(value) + b"\n":
            raise ValueError("noncanonical oracle output")
        return PanelSplitManifests(
            DatasetManifest.model_validate_json(
                json.dumps(value["a_seed_train"], separators=(",", ":"))
            ),
            DatasetManifest.model_validate_json(
                json.dumps(value["a_seed_gate"], separators=(",", ":"))
            ),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PanelSourceFinalizationError(
            "archived v0 panel output is invalid"
        ) from error


def finalize_panel_sources(
    *,
    run_id: str,
    boundary_directory: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    v0_source_root: str | Path,
    authorizer: str,
    accepted_at: datetime | None = None,
) -> Path:
    if not run_id.strip() or any(value in run_id for value in "/\\"):
        raise PanelSourceFinalizationError("run_id must be one filename token")
    authorizer = authorizer.strip()
    if not authorizer:
        raise PanelSourceFinalizationError("panel-source authorizer is required")
    output = Path(output_root).resolve() / run_id
    if output.exists():
        raise PanelSourceFinalizationError(f"output already exists: {output}")
    root = output.parent
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=root))
    try:
        if _sha256(_REPO / _V1_BUILDER) != _V1_BUILDER_SHA256:
            raise PanelSourceFinalizationError("accepted extracted v1 builder changed")
        config = load_pilot_config(config_path)
        boundary = Path(boundary_directory).resolve()
        panel, broad, confirmation = _boundary_objects(boundary)
        prompts = build_stage_a_prompt_pools(
            boundary, config=config, calibration_seed=17
        )
        write_stage_a_prompt_pool_bundle(staging, prompts)
        oracle_output = staging / ".v0_panel_splits.json"
        old_process = _start_archived_builder(
            v0_source_root=Path(v0_source_root).resolve(),
            boundary_directory=boundary,
            prompt_directory=staging,
            output_path=oracle_output,
        )
        try:
            new = build_production_panel_splits(
                config,
                artifact=panel,
                broad_manifest=broad,
                confirmation_manifest=confirmation,
                prior_manifests=(
                    prompts.a_rl_train_manifest,
                    prompts.a_monitor_manifest,
                ),
            )
        except BaseException:
            old_process.terminate()
            old_process.communicate()
            raise
        old = _finish_archived_builder(old_process, oracle_output)
        sources = TeacherAllocationSources(
            config=config,
            panel=panel,
            broad_manifest=broad,
            confirmation_manifest=confirmation,
            a_rl_train_manifest=prompts.a_rl_train_manifest,
            a_monitor_manifest=prompts.a_monitor_manifest,
            target_train_manifest=new.train,
            gate_manifest=new.gate,
            selected_dose=None,
            optimizer_updates=config.teacher_dose.calibration_updates,
        )
        validate_teacher_allocation_base_sources(sources)
        forbidden = production_forbidden_records(
            broad,
            confirmation,
            (prompts.a_rl_train_manifest, prompts.a_monitor_manifest),
        )
        write_equivalent_panel_splits(
            staging,
            old=old,
            new=new,
            artifact=panel,
            broad_manifest=broad,
            confirmation_manifest=confirmation,
            forbidden_records=forbidden,
        )
        write_calibration_sources(staging, sources, prompts)
        oracle_output.unlink()
        equivalence_raw = (staging / "panel_split_equivalence.json").read_bytes()
        timestamp = (accepted_at or datetime.now(UTC)).astimezone(UTC)
        authorization = {
            "schema_version": "duraseed-calibration-panel-split-authorization-v1",
            "status": "accepted",
            "equivalence_sha256": sha256_bytes(equivalence_raw),
            "authorizer": authorizer,
            "accepted_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        }
        atomic_write_bytes(
            staging / "panel_split_authorization.json",
            canonical_json_bytes(authorization) + b"\n",
        )
        os.replace(staging, output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
