"""Validation-only sampling and exact joins for Pilot 0."""

from __future__ import annotations

import json
from pathlib import Path

from duraseed.data.manifests import (
    DatasetManifest,
    MAPSTaskManifestRecord,
    TCESTaskManifestRecord,
)
from duraseed.pilot0_contract import Pilot0Inputs, PilotSeedSources
from duraseed.pilot0_recovery import reconciled_evaluation
from duraseed.pilot0_evidence import (
    EvaluationStage,
    finish_evaluation,
    load_evaluation_prefix,
    read_evaluation,
)
from duraseed.provenance import canonical_json_hash
from duraseed.run_records import GenerationRecord, MethodCode, RewardRecord, write_jsonl
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_concurrent_sampling import (
    cleanup_concurrent_groups,
    sample_manifest_groups,
)
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import SamplingCoordinates
from duraseed.tasks.maps import render_prompt as render_maps_prompt
from duraseed.tasks.tces import render_prompt as render_tces_prompt


_ALLOWED_SPLITS = {"a_monitor", "a_validation", "b_validation"}


def _panel_role(source: PilotSeedSources, record: object) -> str:
    if isinstance(record, MAPSTaskManifestRecord):
        return "stage-b"
    assert isinstance(record, TCESTaskManifestRecord)
    if record.intended_family in source.prompt_pools.artifact.boundary_family_ids:
        return "targeted"
    if record.intended_family in source.prompt_pools.artifact.sentinel_family_ids:
        return "sentinel"
    raise RunnerGateError("Pilot-0 evaluation row is outside the crossed panels")


def _prompt(record: TCESTaskManifestRecord | MAPSTaskManifestRecord) -> str:
    if isinstance(record, TCESTaskManifestRecord):
        return render_tces_prompt(record.to_task())
    return render_maps_prompt(record.to_task())


def _task_contracts(
    source: PilotSeedSources, manifest: DatasetManifest
) -> dict[str, dict]:
    return {
        record.task_id: {
            "task_family": record.task_family,
            "item_index": record.item_index,
            "prompt_text": _prompt(record),
            "assigned_family_id": getattr(record, "intended_family", None),
            "panel_role": _panel_role(source, record),
        }
        for record in manifest.records
    }


async def evaluate_manifest(
    inputs: Pilot0Inputs,
    source: PilotSeedSources,
    *,
    manifest: DatasetManifest,
    sampler: object,
    sampler_path: str,
    origin_sampler_path: str,
    method: MethodCode | None,
    checkpoint_stage: EvaluationStage,
    training_step: int,
    label: str,
    samples_per_item: int,
    sample_index_start: int = 0,
    max_tokens: int,
    seed_namespace: str,
    output: Path,
) -> dict:
    """Sample one authenticated validation population and persist exact joins."""

    if manifest.split not in _ALLOWED_SPLITS or "test" in manifest.split:
        raise RunnerGateError("Pilot 0 cannot read a sealed-test split")
    output.mkdir(parents=True, exist_ok=True)
    temperature = float(inputs.config.evaluation["temperature"])
    top_p = float(inputs.config.evaluation["top_p"])
    contracts = _task_contracts(source, manifest)
    coordinates = {
        "run_id": inputs.run_id,
        "label": label,
        "origin_sampler_path": origin_sampler_path,
        "method": method,
        "checkpoint_stage": checkpoint_stage,
        "training_step": training_step,
        "seed": source.seed,
        "seed_namespace": seed_namespace,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "task_contract_sha256": canonical_json_hash(contracts),
    }
    reconciled_resume = (
        hasattr(inputs, "output_root")
        and output.is_relative_to(Path(inputs.output_root) / inputs.run_id)
        and reconciled_evaluation(inputs, output)
    )
    completed = read_evaluation(output, reconciled_resume=reconciled_resume)
    journal = (
        None
        if completed is not None
        else RemoteJournal(output, reconciled_resume=reconciled_resume)
    )
    if completed is not None:
        try:
            call_state = json.loads((output / "remote-call-state.json").read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise RunnerGateError(
                "completed Pilot-0 evaluation omitted its call state"
            ) from error
        if call_state.get("pending") is not None:
            raise RunnerGateError("completed Pilot-0 evaluation has a pending call")
    counts, sample_ids, reward_sample_ids, completed_tasks = load_evaluation_prefix(
        output,
        samples_per_item,
        sample_index_start=sample_index_start,
        run_id=inputs.run_id,
        label=label,
        manifest=manifest,
        sampler_path=sampler_path,
        origin_sampler_path=origin_sampler_path,
        method=method,
        checkpoint_stage=checkpoint_stage,
        training_step=training_step,
        seed=source.seed,
        seed_namespace=seed_namespace,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        task_contracts=contracts,
    )
    if not completed_tasks.issubset({record.task_id for record in manifest.records}):
        raise RunnerGateError("Pilot-0 partial evaluation contains an unknown task")
    expected_item_counts = [
        {
            "task_id": task_id,
            "panel_role": values["panel_role"],
            "successes": values["successes"],
            "trials": values["trials"],
        }
        for task_id, values in sorted(counts.items())
    ]
    if completed is not None:
        if (
            completed.get("manifest_id") != manifest.manifest_id
            or completed.get("sampler_path") != sampler_path
            or completed.get("coordinates") != coordinates
            or completed.get("samples_per_item") != samples_per_item
            or completed.get("row_count") != len(sample_ids)
            or completed.get("item_counts") != expected_item_counts
        ):
            raise RunnerGateError("completed Pilot-0 evaluation changed evidence")
        return completed
    assert journal is not None
    sample_coordinates = SamplingCoordinates(
        inputs.run_id,
        label,
        "evaluation",
        checkpoint_stage,
        training_step,
        sampler_path,
        origin_sampler_path,
        source.seed,
        seed_namespace,
        method,
    )
    pending = [
        (index, record)
        for index, record in enumerate(manifest.records)
        if record.task_id not in completed_tasks
    ]
    sampled = await sample_manifest_groups(
        inputs,
        manifest=manifest,
        sampler=sampler,
        coordinates=sample_coordinates,
        contracts=contracts,
        pending=pending,
        samples_per_item=samples_per_item,
        sample_index_start=sample_index_start,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        output=output,
    )

    existing_generations = (
        [
            GenerationRecord.model_validate_json(row)
            for row in (output / "generations.jsonl").read_text().splitlines()
        ]
        if (output / "generations.jsonl").exists()
        else []
    )
    existing_rewards = (
        [
            RewardRecord.model_validate_json(row)
            for row in (output / "rewards.jsonl").read_text().splitlines()
        ]
        if (output / "rewards.jsonl").exists()
        else []
    )
    reward_by_sample = {row.sample_id: row for row in existing_rewards}
    all_rows: dict[str, list[tuple[GenerationRecord, RewardRecord]]] = {}
    for generation in existing_generations:
        all_rows.setdefault(generation.task_id, []).append(
            (generation, reward_by_sample[generation.sample_id])
        )
    for index, rows in sampled.items():
        record = manifest.records[index]
        role = _panel_role(source, record)
        values = counts[record.task_id]
        values["panel_role"] = role
        all_rows[record.task_id] = list(rows)
        for generation, reward in rows:
            sample_id = generation.sample_id
            if sample_id in sample_ids or reward.sample_id in reward_sample_ids:
                raise RunnerGateError(
                    "Pilot-0 evaluation produced duplicate sample IDs"
                )
            if (
                reward.sample_id != sample_id
                or reward.task_id != generation.task_id
                or reward.reward != generation.reward
            ):
                raise RunnerGateError("Pilot-0 generation/reward row mismatch")
            sample_ids.add(sample_id)
            reward_sample_ids.add(reward.sample_id)
            values["successes"] += int(reward.reward)
            values["trials"] += 1
    ordered = [
        pair
        for record in manifest.records
        for pair in sorted(
            all_rows.get(record.task_id, ()), key=lambda row: row[0].sample_index
        )
    ]
    write_jsonl(output / "generations.jsonl", (row[0] for row in ordered))
    write_jsonl(output / "rewards.jsonl", (row[1] for row in ordered))
    cleanup_concurrent_groups(output)
    expected = manifest.record_count * samples_per_item
    if len(sample_ids) != expected or any(
        row["trials"] != samples_per_item for row in counts.values()
    ):
        raise RunnerGateError("Pilot-0 evaluation row count is incomplete")
    return finish_evaluation(
        output,
        manifest=manifest,
        label=label,
        sampler_path=sampler_path,
        samples_per_item=samples_per_item,
        counts=counts,
        sample_ids=sample_ids,
        reward_ids=reward_sample_ids,
        coordinates=coordinates,
    )


__all__ = ["evaluate_manifest"]
