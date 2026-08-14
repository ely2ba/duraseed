"""Open the authenticated run directory for the fixed boundary live gate."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from duraseed.boundary_live_artifacts import BoundaryLiveArtifacts
from duraseed.boundary_live_fresh_resume import (
    BoundaryFreshResumeArtifacts,
    FRESH_RESUME_MARKER,
    is_fresh_resume_trace,
)
from duraseed.boundary_live_retry import BoundaryRetryArtifacts, RETRY_MARKER
from duraseed.data.boundary_protocol import BOUNDARY_ENGINEERING_SEED
from duraseed.data.manifests import DatasetManifest
from duraseed.runtime import PRICE_SNAPSHOT
from duraseed.run_records import RunRecord, RunStatus


def open_boundary_artifacts(
    output_root: str | Path,
    run_id: str,
    *,
    git_commit: str,
    runtime: Any,
    config: Any,
    source_contract: Any,
    extension2: DatasetManifest,
    action_caps: dict[str, Decimal],
    refine_retry_trace: str | Path | None = None,
) -> BoundaryLiveArtifacts:
    if not run_id.strip() or any(character in run_id for character in "/\\"):
        raise ValueError("run_id must be a nonempty filename token")
    if sum(action_caps.values(), start=Decimal("0")) != Decimal("120"):
        raise ValueError("boundary action caps must total exactly $120")
    preflight = {
        "gate_name": "boundary-extension",
        "run_id": run_id,
        "actions": {name: str(cap) for name, cap in action_caps.items()},
        "authorized_cost_usd": "120",
        "source": asdict(source_contract),
        "extension2_manifest_id": extension2.manifest_id,
        "composite_status": "blocked_pending_three_cohort_equivalence",
    }
    now = datetime.now(UTC)
    run = RunRecord(
        protocol_version=str(config.protocol["version"]),
        git_commit=git_commit,
        resolved_config_hash=config.resolved_config_hash(),
        run_kind="m0_calibration",
        method=None,
        seed=BOUNDARY_ENGINEERING_SEED,
        model_id=source_contract.model_id,
        renderer=source_contract.renderer,
        lora_rank=source_contract.lora_rank,
        task_manifest_ids={"boundary_extension2_broad": extension2.manifest_id},
        parent_tinker_checkpoint_path=source_contract.state_checkpoint_path,
        status=RunStatus.RUNNING,
        started_at=now,
        updated_at=now,
        project_id=source_contract.project_id,
        authorized_cost_usd=120.0,
        reserved_cost_usd=120.0,
        price_snapshot_id=PRICE_SNAPSHOT.snapshot_id,
        tinker_sdk_version=runtime.sdk.sdk_version,
        tinker_cookbook_version=runtime.sdk.cookbook_version,
        deviations=["boundary calibration only; Pilot 0 not started"],
    )
    directory = Path(output_root) / run_id
    if (
        is_fresh_resume_trace(refine_retry_trace)
        or (directory / FRESH_RESUME_MARKER).exists()
    ):
        artifact_type = BoundaryFreshResumeArtifacts
    elif refine_retry_trace is not None or (directory / RETRY_MARKER).exists():
        artifact_type = BoundaryRetryArtifacts
    else:
        artifact_type = BoundaryLiveArtifacts
    artifact_kwargs = (
        {
            "trace_path": (
                Path(refine_retry_trace).resolve()
                if refine_retry_trace is not None
                else None
            )
        }
        if artifact_type in (BoundaryRetryArtifacts, BoundaryFreshResumeArtifacts)
        else {}
    )
    artifacts = artifact_type(
        directory, preflight=preflight, new_run=run, **artifact_kwargs
    )
    artifacts.write_manifest("extension2_broad_manifest.json", extension2)
    return artifacts


__all__ = ["open_boundary_artifacts"]
