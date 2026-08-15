"""One-shot local publication of the accepted boundary-panel amendment."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from duraseed.boundary_capacity import audit_family_split_capacities
from duraseed.config import load_pilot_config
from duraseed.data.boundary_confirmation import regenerate_family_templates
from duraseed.data.boundary_freeze_contracts import freeze_settings_from_config
from duraseed.data.boundary_panel_amendment import (
    algebraic_family_classes,
    reduce_panel_amendment,
)
from duraseed.data.boundary_panel_amendment_source import (
    AuthenticatedUnresolvedFreeze,
    BoundaryPanelAmendmentSourceError,
    PUBLISHED_UNRESOLVED_RUN_ID,
    SOURCE_KIND,
    authenticate_published_unresolved_freeze as _authenticate_source,
    candidate_from_mapping,
)
from duraseed.data.io import atomic_write_bytes
from duraseed.data.panel_capacity import (
    PANEL_SELECTED_TEST_SINGLE_MINIMUM,
    PANEL_SPLIT_MINIMUMS,
)
from duraseed.provenance import canonical_json_bytes, canonical_json_hash
from duraseed.run_records import RunRecord, RunStatus, write_jsonl
from duraseed.runners import RunnerGateError
from duraseed.tasks.tces import TCESGeneratorConfig


_REPO = Path(__file__).resolve().parents[2]
_PANEL_REQUIREMENTS = {
    **dict(PANEL_SPLIT_MINIMUMS),
    "a_test_single": PANEL_SELECTED_TEST_SINGLE_MINIMUM,
}


class BoundaryPanelAmendmentFinalizationError(RunnerGateError):
    """The fixed unresolved source or amended reduction failed closed."""


def authenticate_published_unresolved_freeze(
    directory: str | Path,
) -> AuthenticatedUnresolvedFreeze:
    try:
        return _authenticate_source(directory)
    except BoundaryPanelAmendmentSourceError as error:
        raise BoundaryPanelAmendmentFinalizationError(str(error)) from error


def _candidate(value: dict[str, Any]):
    try:
        return candidate_from_mapping(value)
    except BoundaryPanelAmendmentSourceError as error:
        raise BoundaryPanelAmendmentFinalizationError(str(error)) from error


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise BoundaryPanelAmendmentFinalizationError(
            "cannot identify amendment commit"
        ) from error


def _json(path: Path, value: Any) -> bytes:
    raw = canonical_json_bytes(value) + b"\n"
    atomic_write_bytes(path, raw)
    return raw


def _write_completed(
    staging: Path,
    source: AuthenticatedUnresolvedFreeze,
    result: Any,
    config: Any,
    run_id: str,
) -> None:
    representative_ids = tuple(
        row.family_id for row in result.representative_candidates
    )
    ordinary_by_id = {
        row["family_id"]: row for row in source.scientific["capacity_audits"]
    }
    ordinary = tuple(ordinary_by_id[family_id] for family_id in representative_ids)
    amendment = {
        **result.amendment_payload,
        "source_run_id": source.directory.name,
        "source_scientific_outputs_sha256": source.source_hashes[
            "scientific_outputs.json"
        ],
    }
    amendment_id = canonical_json_hash(amendment)
    summary = {
        **source.scientific["confirmation_summary"],
        **result.confirmation_summary,
        "panel_assignment_performed": True,
        "split_capacity_eligible_family_count": len(representative_ids),
        "panel_candidate_family_count": len(representative_ids),
        "panel_candidate_family_ids": list(representative_ids),
        "intermediate_family_ids": list(result.intermediate_family_ids),
        "panel_amendment_id": amendment_id,
        "next_step": "continue remaining pre-pilot calibration",
    }
    scientific = {
        **source.scientific,
        "capacity_audits": ordinary,
        "panel_capacity_audits": result.panel_capacity_audits,
        "eligible_family_ids": representative_ids,
        "teacher_trace_token_counts": {
            family_id: source.scientific["teacher_trace_token_counts"][family_id]
            for family_id in representative_ids
        },
        "candidates": result.representative_candidates,
        "ranked_family_ids": representative_ids,
        "candidate_payload": result.candidate_payload,
        "candidate_table_id": result.panel_artifact.candidate_family_table_manifest_id,
        "match": result.match,
        "matching_report_payload": result.matching_report_payload,
        "matching_report_id": result.panel_artifact.panel_matching_report_id,
        "intermediate_family_ids": result.intermediate_family_ids,
        "selected_capacity_audits": result.selected_capacity_audits,
        "panel_artifact": result.panel_artifact,
        "confirmation_summary": summary,
        "panel_amendment": amendment,
    }
    for name in (
        "a_candidate_manifest.json",
        "a_candidate_confirmation_manifest.json",
        "confirmation_family_table.jsonl",
        "three_cohort_equivalence.json",
    ):
        shutil.copyfile(source.directory / name, staging / name)
    write_jsonl(staging / "split_capacity_audits.jsonl", ordinary)
    write_jsonl(
        staging / "panel_capacity_audits.jsonl",
        (asdict(row) for row in result.panel_capacity_audits),
    )
    write_jsonl(
        staging / "selected_panel_capacity_audits.jsonl",
        (asdict(row) for row in result.selected_capacity_audits),
    )
    write_jsonl(
        staging / "panel_candidate_table.jsonl",
        (asdict(row) for row in result.representative_candidates),
    )
    _json(
        staging / "teacher_trace_token_counts.json",
        scientific["teacher_trace_token_counts"],
    )
    _json(staging / "panel_candidate_table.json", result.candidate_payload)
    _json(staging / "panel_matching_report.json", result.matching_report_payload)
    _json(staging / "target_sentinel_panels.json", result.panel_artifact)
    _json(staging / "panel_amendment.json", amendment)
    _json(staging / "confirmation_summary.json", summary)
    _json(staging / "scientific_outputs.json", scientific)
    old_source = source.preflight["source"]
    preflight = {
        "status": "completed_local_freeze",
        "phase": "m0_tces_boundary_three_cohort_panel_freeze",
        "source_kind": SOURCE_KIND,
        "remote_execution": False,
        "execution_authorized": False,
        "pilot_started": False,
        "run_id": run_id,
        "source": {
            **old_source,
            "source_kind": SOURCE_KIND,
            "unresolved_freeze_directory": str(source.directory),
            "unresolved_freeze_scientific_outputs_sha256": source.source_hashes[
                "scientific_outputs.json"
            ],
            "panel_amendment_id": amendment_id,
        },
        "selection": {
            "allocation_seed": result.panel_artifact.allocation_seed,
            "panel_family_count": 24,
            "panel_capacity_family_count": len(result.panel_eligible_family_ids),
            "intermediate_family_count": len(result.intermediate_family_ids),
            "intermediate_a_test_single_required": False,
            "intermediate_family_ids": list(result.intermediate_family_ids),
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
    record = source.run.model_dump(mode="python")
    record.update(
        git_commit=_git_commit(),
        resolved_config_hash=config.resolved_config_hash(),
        status=RunStatus.COMPLETED,
        started_at=now,
        updated_at=now,
        finished_at=now,
        prompt_tokens=0,
        sampled_tokens=0,
        train_tokens=0,
        cost_usd=0.0,
        deviations=[
            "accepted prospective de-duplicate-before-match panel amendment",
            "panel 22/22 and intermediate ordinary-only eligibility kept separate",
            "local-only; no Tinker API call; Pilot 0 not started",
        ],
    )
    _json(
        staging / "run.json", RunRecord.model_validate(record).model_dump(mode="json")
    )


def finalize_boundary_panel_amendment(
    *,
    run_id: str,
    source_freeze: str | Path,
    output_root: str | Path,
    config_path: str | Path,
) -> Path:
    """Run the accepted deterministic amendment and atomically publish its freeze."""

    if not run_id.strip() or any(character in run_id for character in "/\\"):
        raise BoundaryPanelAmendmentFinalizationError(
            "run_id must be one filename token"
        )
    output = Path(output_root).resolve() / run_id
    if output.exists():
        raise BoundaryPanelAmendmentFinalizationError(
            f"output already exists: {output}"
        )
    source = authenticate_published_unresolved_freeze(source_freeze)
    config = load_pilot_config(config_path)
    if source.run.resolved_config_hash != config.resolved_config_hash():
        raise BoundaryPanelAmendmentFinalizationError("freeze config changed")
    settings = freeze_settings_from_config(config)
    candidates = tuple(_candidate(row) for row in source.scientific["candidates"])
    classes = algebraic_family_classes(candidates, settings.allocation_seed)
    representative_ids = tuple(row.representative_family_id for row in classes)
    generator = TCESGeneratorConfig(**dict(settings.generator_kwargs))
    templates = regenerate_family_templates(generator, source.broad, representative_ids)
    audits = audit_family_split_capacities(
        templates,
        representative_ids,
        generator,
        root_seed=settings.capacity_root_seed,
        requirements=_PANEL_REQUIREMENTS,
        forbidden_records=(*source.broad.records, *source.confirmation.records),
        protected_family_ids=representative_ids,
        max_workers=6,
    )
    result = reduce_panel_amendment(
        candidates,
        audits,
        panel_size=settings.panel_size,
        intermediate_size=12,
        allocation_seed=settings.allocation_seed,
        training_seeds=settings.training_seeds,
        m0_checkpoint_path=source.preflight["source"]["sampler_checkpoint_path"],
    )
    if result.panel_artifact is None or len(result.intermediate_family_ids) != 12:
        raise BoundaryPanelAmendmentFinalizationError(
            "accepted amendment did not resolve 12/12 panels plus 12 intermediates"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output.parent))
    try:
        _write_completed(staging, source, result, config, run_id)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


__all__ = [
    "BoundaryPanelAmendmentFinalizationError",
    "PUBLISHED_UNRESOLVED_RUN_ID",
    "authenticate_published_unresolved_freeze",
    "finalize_boundary_panel_amendment",
]
