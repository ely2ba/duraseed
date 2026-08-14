"""Confirmation-only continuation of the completed Extension-2 refinement."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from duraseed.boundary_confirmation_preparation import PreparedBoundaryConfirmation
from duraseed.boundary_confirmation_resume import CONFIRMATION_RESUME_GROUPS
from duraseed.boundary_live_artifacts import BoundaryLiveArtifacts
from duraseed.boundary_live_retry import INCIDENT
from duraseed.boundary_live_sampling import (
    ACTION_CAPS,
    action_limits,
    collect_groups,
)
from duraseed.boundary_live_sources import (
    BoundaryLiveSource,
    confirmed_family_summaries,
)
from duraseed.config import PilotConfig
from duraseed.data.boundary_confirmation import (
    BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
    BOUNDARY_REFINEMENT_TOTAL_SAMPLES_PER_ITEM,
)
from duraseed.data.boundary_protocol import (
    BOUNDARY_BROAD_EXTENSION_2_COHORT,
    BOUNDARY_BROAD_SAMPLES_PER_ITEM,
)
from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_value
from duraseed.runners import RunnerGateError
from duraseed.runners.boundary_extension import (
    BoundaryBlockInputs,
    BoundaryBlockResult,
    reduce_block,
)
from duraseed.runtime import RuntimeBundle, TokenBudget, TokenLedger
from duraseed.run_records import RunStatus
from duraseed.tasks.tces import TCESGeneratorConfig


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    atomic_write_bytes(path, payload + b"\n")


def _result_payload(result: BoundaryBlockResult) -> dict[str, Any]:
    value = canonical_json_value(asdict(result))
    if not isinstance(value, dict):  # pragma: no cover - dataclass contract
        raise TypeError("boundary result did not serialize to an object")
    return value


def _prior_ledgers(
    artifacts: BoundaryLiveArtifacts,
    prepared: PreparedBoundaryConfirmation,
    config: PilotConfig,
) -> dict[str, TokenLedger]:
    snapshot = prepared.snapshot
    extra = BOUNDARY_REFINEMENT_TOTAL_SAMPLES_PER_ITEM - BOUNDARY_BROAD_SAMPLES_PER_ITEM
    failed = TokenBudget(**INCIDENT["pending"]["reserved_tokens"])
    return {
        "extension1-confirm": artifacts.restore_ledger(
            "extension1-confirm",
            action_limits(
                "extension1-confirm",
                snapshot.extension1_confirmation,
                BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
                config.tinker.max_sampled_tokens,
            ),
            ACTION_CAPS["extension1-confirm"],
        ),
        "extension2-broad": artifacts.restore_ledger(
            "extension2-broad",
            action_limits(
                "extension2-broad",
                snapshot.extension2,
                BOUNDARY_BROAD_SAMPLES_PER_ITEM,
                config.tinker.max_sampled_tokens,
            ),
            ACTION_CAPS["extension2-broad"],
        ),
        "extension2-refine": artifacts.restore_ledger(
            "extension2-refine",
            action_limits(
                "extension2-refine",
                snapshot.extension2,
                extra,
                config.tinker.max_sampled_tokens,
                task_count=CONFIRMATION_RESUME_GROUPS["extension2-refine"],
            ).plus(failed),
            ACTION_CAPS["extension2-refine"],
        ),
    }


async def execute_boundary_confirmation_resume(
    runtime: RuntimeBundle,
    sampler: Any,
    *,
    source: BoundaryLiveSource,
    config: PilotConfig,
    prepared: PreparedBoundaryConfirmation,
    git_commit: str,
) -> BoundaryBlockResult:
    """Sample only Extension-2 confirmation, then reduce with carried audits."""

    snapshot = prepared.snapshot
    new_run = snapshot.run.model_copy(update={"git_commit": git_commit})
    artifacts = BoundaryLiveArtifacts(
        snapshot.directory,
        preflight=snapshot.preflight,
        new_run=new_run,
        _allow_git_change=True,
    )
    artifacts.write_manifest("extension2_confirmation_manifest.json", prepared.manifest)
    artifacts.add_manifest_identity(
        "boundary_extension2_confirmation", prepared.manifest.manifest_id
    )
    ledgers = _prior_ledgers(artifacts, prepared, config)
    action = "extension2-confirm"
    ledgers[action] = artifacts.restore_ledger(
        action,
        action_limits(
            action,
            prepared.manifest,
            BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
            config.tinker.max_sampled_tokens,
        ),
        ACTION_CAPS[action],
    )
    try:
        confirm_g, confirm_r = await collect_groups(
            artifacts,
            runtime,
            sampler,
            prepared.manifest,
            action=action,
            run_id=snapshot.directory.name,
            source=source,
            samples=BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
            sample_start=0,
            config=config,
            ledger=ledgers[action],
            allow_empty=True,
        )
        summaries = confirmed_family_summaries(
            snapshot.extension2,
            prepared.manifest,
            (
                *snapshot.broad_generations,
                *snapshot.refinement_generations,
                *confirm_g,
            ),
            (*snapshot.broad_rewards, *snapshot.refinement_rewards, *confirm_r),
            group_size=config.tinker.group_size,
            expected_run_ids=(
                f"{snapshot.directory.name}:extension2-broad",
                f"{snapshot.directory.name}:extension2-refine",
                f"{snapshot.directory.name}:extension2-confirm",
            ),
        )
        generator = TCESGeneratorConfig(**config.tasks.tces.generator_kwargs())
        result = reduce_block(
            generator,
            BoundaryBlockInputs(
                BOUNDARY_BROAD_EXTENSION_2_COHORT,
                snapshot.extension2,
                snapshot.family_successes,
                snapshot.refinement_summaries,
                summaries,
                (source.initial_broad_manifest, source.extension1_broad_manifest),
                (
                    source.initial_confirmation_manifest,
                    snapshot.extension1_confirmation,
                ),
                prepared.capacity_audits,
            ),
        )
        if (
            result.confirmation_manifest != prepared.manifest
            or result.capacity_audits != prepared.capacity_audits
        ):
            raise RunnerGateError("Extension-2 confirmation reduction changed")
        _write_json(
            snapshot.directory / "extension2_result.json",
            _result_payload(result),
        )
        artifacts.finish(RunStatus.COMPLETED, ledgers)
        return result
    except BaseException as error:
        artifacts.record_error(action, error)
        artifacts.finish(
            RunStatus.INTERRUPTED
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            else RunStatus.FAILED,
            ledgers,
        )
        raise


__all__ = [
    "execute_boundary_confirmation_resume",
]
