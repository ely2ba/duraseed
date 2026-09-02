"""Bounded concurrent Pilot evaluation groups with durable per-item journals."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.data.manifests import DatasetManifest, TaskManifestRecord
from duraseed.pilot0_contract import Pilot0Inputs
from duraseed.provenance import canonical_json_bytes
from duraseed.run_records import GenerationRecord, RewardRecord
from duraseed.runners import RunnerGateError
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import (
    SamplingCoordinates,
    SamplingTask,
    TokenBudget,
    TokenLedger,
    sample_seeded,
)


EVALUATION_CONCURRENCY = 4
GROUP_DIRECTORY = ".concurrent-groups"
GroupRows = tuple[tuple[GenerationRecord, RewardRecord], ...]


def _evidence_path(output: Path, index: int) -> Path:
    return output / GROUP_DIRECTORY / f"{index:04d}" / "evidence.json"


def _read_rows(path: Path, task_id: str) -> GroupRows:
    try:
        value = json.loads(path.read_bytes())
        generations = tuple(
            GenerationRecord.model_validate_json(canonical_json_bytes(row))
            for row in value["generations"]
        )
        rewards = tuple(
            RewardRecord.model_validate_json(canonical_json_bytes(row))
            for row in value["rewards"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerGateError(
            "concurrent Pilot-0 group evidence is unreadable"
        ) from error
    if (
        value.get("task_id") != task_id
        or len(generations) != len(rewards)
        or any(
            generation.task_id != task_id
            or reward.task_id != task_id
            or generation.sample_id != reward.sample_id
            or generation.reward != reward.reward
            for generation, reward in zip(generations, rewards, strict=True)
        )
    ):
        raise RunnerGateError("concurrent Pilot-0 group evidence changed")
    return tuple(zip(generations, rewards, strict=True))


def _write_rows(path: Path, task_id: str, rows: tuple[Any, ...]) -> GroupRows:
    atomic_write_bytes(
        path,
        canonical_json_bytes(
            {
                "task_id": task_id,
                "generations": [row.generation.model_dump(mode="json") for row in rows],
                "rewards": [row.reward.model_dump(mode="json") for row in rows],
            }
        ),
    )
    return _read_rows(path, task_id)


def _reservation(
    inputs: Pilot0Inputs,
    contract: dict[str, Any],
    *,
    samples_per_item: int,
    max_tokens: int,
) -> TokenBudget:
    prompt = inputs.runtime.renderer.build_generation_prompt(
        [{"role": "user", "content": contract["prompt_text"]}], role="assistant"
    )
    return TokenBudget(
        int(prompt.length) * samples_per_item,
        max_tokens * samples_per_item,
        0,
    )


async def _sample_group(
    inputs: Pilot0Inputs,
    *,
    index: int,
    record: TaskManifestRecord,
    contract: dict[str, Any],
    manifest: DatasetManifest,
    sampler: object,
    coordinates: SamplingCoordinates,
    samples_per_item: int,
    sample_index_start: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    output: Path,
    ledger: TokenLedger,
) -> tuple[int, GroupRows]:
    evidence_path = _evidence_path(output, index)
    group_output = evidence_path.parent
    journal = RemoteJournal(group_output)
    reserved = _reservation(
        inputs,
        contract,
        samples_per_item=samples_per_item,
        max_tokens=max_tokens,
    )
    journal.begin(
        "pilot0-validation-group",
        {"label": coordinates.label, "task_id": record.task_id},
        {
            "prefill_tokens": reserved.prefill,
            "sample_tokens": reserved.sample,
            "train_tokens": 0,
        },
    )
    rows = await sample_seeded(
        inputs.runtime,
        sampler,
        SamplingTask(
            manifest.manifest_id,
            record.task_id,
            record.task_family,
            manifest.split,
            contract["prompt_text"],
            record.to_task(),
            record.item_index,
            contract["assigned_family_id"],
            contract["panel_role"],
        ),
        coordinates,
        group_size=samples_per_item,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        ledger=ledger,
        sample_index_start=sample_index_start,
    )
    persisted = _write_rows(evidence_path, record.task_id, rows)
    journal.complete({"operation": "pilot0-validation-group", "row_count": len(rows)})
    return index, persisted


async def sample_manifest_groups(
    inputs: Pilot0Inputs,
    *,
    manifest: DatasetManifest,
    sampler: object,
    coordinates: SamplingCoordinates,
    contracts: dict[str, dict[str, Any]],
    pending: list[tuple[int, TaskManifestRecord]],
    samples_per_item: int,
    sample_index_start: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    output: Path,
) -> dict[int, GroupRows]:
    """Sample pending items four at a time while reserving each batch once."""

    sampled: dict[int, GroupRows] = {}
    for offset in range(0, len(pending), EVALUATION_CONCURRENCY):
        batch = pending[offset : offset + EVALUATION_CONCURRENCY]
        fresh = []
        for index, record in batch:
            path = _evidence_path(output, index)
            if path.exists():
                RemoteJournal(path.parent, reconciled_resume=True)
                sampled[index] = _read_rows(path, record.task_id)
            else:
                fresh.append((index, record))
        if not fresh:
            continue
        reservations = {
            index: _reservation(
                inputs,
                contracts[record.task_id],
                samples_per_item=samples_per_item,
                max_tokens=max_tokens,
            )
            for index, record in fresh
        }
        total = TokenBudget(0, 0, 0)
        for reserved in reservations.values():
            total = total.plus(reserved)
        inputs.ledger.reserve_call(total)
        children = {
            index: TokenLedger(reserved, inputs.ledger.authorized_usd)
            for index, reserved in reservations.items()
        }
        try:
            results = await asyncio.gather(
                *(
                    _sample_group(
                        inputs,
                        index=index,
                        record=record,
                        contract=contracts[record.task_id],
                        manifest=manifest,
                        sampler=sampler,
                        coordinates=coordinates,
                        samples_per_item=samples_per_item,
                        sample_index_start=sample_index_start,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        output=output,
                        ledger=children[index],
                    )
                    for index, record in fresh
                )
            )
        except BaseException:
            inputs.ledger.abort_call()
            raise
        actual = TokenBudget(0, 0, 0)
        for child in children.values():
            actual = actual.plus(child.observed)
        inputs.ledger.settle_call(actual)
        sampled.update(results)
    return sampled


def cleanup_concurrent_groups(output: Path) -> None:
    shutil.rmtree(output / GROUP_DIRECTORY, ignore_errors=True)


__all__ = ["cleanup_concurrent_groups", "sample_manifest_groups"]
