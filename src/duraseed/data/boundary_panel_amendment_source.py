"""Authentication for the published unresolved boundary-freeze source."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from duraseed.data.manifests import read_manifest
from duraseed.data.panel_matching import FamilyPanelCandidate
from duraseed.data.sealing import ExecutionContext
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import RunRecord, RunStatus, read_run_record


PUBLISHED_UNRESOLVED_RUN_ID = "boundary-panel-freeze-20260815T032009Z"
SOURCE_KIND = "three_cohort_boundary_freeze_v1"
SOURCE_RAW_SHA256 = {
    "preflight.json": "sha256:45122f2800055cbdeccdf6415a9b386ac2aeac1bd3bff38b5a02207f3d7315f4",
    "run.json": "sha256:dcb88ca30ef3a52a8e9e8bad3287727d6e27ddfa71fdb0c9951ca33dbd3ce2ab",
    "scientific_outputs.json": "sha256:98ac79747b2d81e3b4a89c22e39ef8bf536ae919eda02a31579ff50a43dd9704",
    "three_cohort_equivalence.json": "sha256:e003fe85289f29915e25582b22ae582182b89185ea66f0d74fdcb4a202653f15",
}


class BoundaryPanelAmendmentSourceError(ValueError):
    """The fixed unresolved freeze failed authentication."""


@dataclass(frozen=True, slots=True)
class AuthenticatedUnresolvedFreeze:
    directory: Path
    scientific: dict[str, Any]
    preflight: dict[str, Any]
    equivalence: dict[str, Any]
    run: RunRecord
    broad: Any
    confirmation: Any
    source_hashes: dict[str, str]


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BoundaryPanelAmendmentSourceError(
            f"published {label} is invalid JSON"
        ) from error
    if not isinstance(value, dict):
        raise BoundaryPanelAmendmentSourceError(f"published {label} is not an object")
    return value


def validate_source_metadata(
    *,
    scientific: dict[str, Any],
    preflight: dict[str, Any],
    equivalence: dict[str, Any],
    run: RunRecord,
    source_hashes: dict[str, str],
) -> None:
    summary = scientific.get("confirmation_summary")
    source = preflight.get("source")
    candidates = scientific.get("candidates")
    scientific_hash = sha256_bytes(canonical_json_bytes(scientific))
    token_hash = sha256_bytes(
        canonical_json_bytes(scientific.get("teacher_trace_token_counts"))
    )
    if (
        source_hashes != SOURCE_RAW_SHA256
        or run.status is not RunStatus.COMPLETED
        or run.run_kind != "m0_calibration"
        or run.cost_usd != 0.0
        or preflight.get("status") != "completed_local_freeze"
        or preflight.get("source_kind") != SOURCE_KIND
        or preflight.get("remote_execution") is not False
        or not isinstance(source, dict)
        or source.get("source_kind") != SOURCE_KIND
        or source.get("equivalence_sha256")
        != source_hashes["three_cohort_equivalence.json"]
        or equivalence.get("schema_version")
        != "duraseed-three-cohort-freeze-equivalence-v1"
        or equivalence.get("status") != "passed"
        or equivalence.get("old_new_identical") is not True
        or equivalence.get("token_count_map_identical") is not True
        or equivalence.get("scientific_outputs_sha256") != scientific_hash
        or equivalence.get("teacher_trace_token_counts_sha256") != token_hash
        or equivalence.get("compared_fields") != sorted(scientific)
        or not isinstance(summary, dict)
        or summary.get("status") != "confirmation_complete_panel_selection_unresolved"
        or scientific.get("panel_artifact") is not None
        or scientific.get("candidate_payload") is not None
        or scientific.get("matching_report_payload") is not None
        or not isinstance(candidates, list)
        or len(candidates) != 49
    ):
        raise BoundaryPanelAmendmentSourceError(
            "published unresolved freeze authentication failed"
        )


def authenticate_published_unresolved_freeze(
    directory: str | Path,
) -> AuthenticatedUnresolvedFreeze:
    """Authenticate the exact accepted input without recomputing it."""

    source = Path(directory).resolve()
    if source.name != PUBLISHED_UNRESOLVED_RUN_ID:
        raise BoundaryPanelAmendmentSourceError(
            "source is not the published unresolved freeze"
        )
    raw: dict[str, bytes] = {}
    try:
        for name in SOURCE_RAW_SHA256:
            raw[name] = (source / name).read_bytes()
        scientific = _object(raw["scientific_outputs.json"], "scientific output")
        preflight = _object(raw["preflight.json"], "preflight")
        equivalence = _object(raw["three_cohort_equivalence.json"], "equivalence")
        run = read_run_record(source)
        broad = read_manifest(
            source / "a_candidate_manifest.json", context=ExecutionContext.SELECTION
        )
        confirmation = read_manifest(
            source / "a_candidate_confirmation_manifest.json",
            context=ExecutionContext.SELECTION,
        )
        family_table = [
            json.loads(line)
            for line in (source / "confirmation_family_table.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise BoundaryPanelAmendmentSourceError(
            "published unresolved freeze is incomplete or invalid"
        ) from error
    hashes = {name: sha256_bytes(value) for name, value in raw.items()}
    validate_source_metadata(
        scientific=scientific,
        preflight=preflight,
        equivalence=equivalence,
        run=run,
        source_hashes=hashes,
    )
    source_meta = preflight["source"]
    if (
        source_meta.get("combined_broad_manifest_id") != broad.manifest_id
        or source_meta.get("combined_confirmation_manifest_id")
        != confirmation.manifest_id
        or run.task_manifest_ids.get("boundary_composite_a_candidate")
        != broad.manifest_id
        or run.task_manifest_ids.get("boundary_confirmation_a_candidate")
        != confirmation.manifest_id
        or scientific.get("combined_broad_manifest")
        != json.loads((source / "a_candidate_manifest.json").read_bytes())
        or scientific.get("combined_confirmation_manifest")
        != json.loads((source / "a_candidate_confirmation_manifest.json").read_bytes())
        or scientific.get("confirmation_family_table") != family_table
    ):
        raise BoundaryPanelAmendmentSourceError(
            "published raw artifacts differ from the scientific output"
        )
    return AuthenticatedUnresolvedFreeze(
        source, scientific, preflight, equivalence, run, broad, confirmation, hashes
    )


def candidate_from_mapping(value: dict[str, Any]) -> FamilyPanelCandidate:
    row = dict(value)
    row["operator_multiset"] = tuple(row.get("operator_multiset", ()))
    row["fractional_intermediate_profile"] = tuple(
        row.get("fractional_intermediate_profile", ())
    )
    try:
        return FamilyPanelCandidate(**row)
    except (TypeError, ValueError) as error:
        raise BoundaryPanelAmendmentSourceError(
            "published panel candidate is malformed"
        ) from error


__all__ = [
    "AuthenticatedUnresolvedFreeze",
    "BoundaryPanelAmendmentSourceError",
    "PUBLISHED_UNRESOLVED_RUN_ID",
    "SOURCE_KIND",
    "authenticate_published_unresolved_freeze",
    "candidate_from_mapping",
    "validate_source_metadata",
]
