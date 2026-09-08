"""Read-only authentication of the two original B-G acquisition archives."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterator

from duraseed.pilot0_data import ordered_stage_a_pools, scheduled_stage_a_records
from duraseed.pilot0_source_build import _read_pilot_seed_source
from duraseed.provenance import canonical_json_hash, sha256_bytes
from duraseed.runners.pilot0_remote import read_segment
from duraseed.runners.pilot0_updates import _group_seeds
from duraseed.runtime.sampling import _completion


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def read_source(run_root: Path, seed: int):
    preflight = json.loads((run_root / "preflight.json").read_bytes())
    if preflight["seed"] != seed or preflight["run_id"] != run_root.name:
        raise ValueError("Pilot source run coordinates differ")
    source = _read_pilot_seed_source(run_root / "pilot-inputs", seed)
    declared = preflight["lineage"]["manifest_ids"]
    actual = {
        "a_rl_train": source.prompt_pools.a_rl_train_manifest.manifest_id,
        "a_monitor": source.prompt_pools.a_monitor_manifest.manifest_id,
        "a_cadence": source.a_cadence.manifest_id,
        "a_validation": source.a_validation.manifest_id,
        "b_train": source.b_train.manifest_id,
        "b_validation": source.b_validation.manifest_id,
    }
    if actual != declared:
        raise ValueError("Pilot source manifest identities changed")
    return source, preflight


def authenticated_rollouts(
    run_root: Path, seed: int, tokenizer: Any, source: Any, preflight: dict
) -> Iterator[dict[str, Any]]:
    """The sampler for rollout update u was saved before optimizer update u."""

    pools = ordered_stage_a_pools(source)
    order = source.prompt_pools.artifact.bg_group_order
    parent_state = preflight["lineage"]["m0_state_path"]
    parent_sampler = preflight["lineage"]["m0_sampler_path"]
    for start in range(0, 50, 10):
        directory = run_root / f"seed-{seed}" / "B-G" / f"steps-{start}-{start + 10}"
        segment = read_segment(
            directory,
            {"method": "B-G", "start": start, "stop": start + 10, "seed": seed},
        )
        if segment is None:
            raise ValueError(f"missing completed source segment: {directory}")
        if (
            segment["parent_state_path"] != parent_state
            or segment["parent_sampler_path"] != parent_sampler
            or segment["origin_state_path"] != preflight["lineage"]["m0_state_path"]
            or segment["run_id"] != run_root.name
            or segment["source_manifest_ids"]["a_rl_train_manifest_id"]
            != source.prompt_pools.a_rl_train_manifest.manifest_id
        ):
            raise ValueError("B-G source checkpoint lineage changed")
        metrics = list(jsonl(directory / "metrics.jsonl"))
        if [row["training_step"] for row in metrics] != list(
            range(start + 1, start + 11)
        ) or any(row["method"] != "B-G" for row in metrics):
            raise ValueError("source segment lacks all ten committed B-G updates")
        calls = list(jsonl(directory / "remote-calls.jsonl"))
        update_calls = [
            row for row in calls if row.get("operation") == "pilot0-stage-a-rl-update"
        ]
        if [row["step"] for row in update_calls] != list(
            range(start + 1, start + 11)
        ) or any(row["status"] != "completed" for row in calls):
            raise ValueError("source call journal has incomplete/ambiguous updates")
        for step in range(start + 1, start + 11):
            coordinate = {"seed": seed, "method": "B-G", "step": step}
            path = f"ephemeral:{run_root.name}:{canonical_json_hash(coordinate).removeprefix('sha256:')}"
            sampler_call = [
                row
                for row in calls
                if row.get("operation") == "pilot0-ephemeral-sampler"
                and row.get("path") == path
            ]
            update = next(row for row in update_calls if row["step"] == step)
            if (
                len(sampler_call) != 1
                or sampler_call[0]["sequence"] >= update["sequence"]
            ):
                raise ValueError(
                    "rollout sampler was not recorded before its optimizer update"
                )
            between = [
                row
                for row in calls
                if sampler_call[0]["sequence"] < row["sequence"] < update["sequence"]
            ]
            if len(between) != 16 or any(
                row.get("operation") != "pilot0-stage-a-rl-group"
                or row.get("row_count") != 8
                for row in between
            ):
                raise ValueError(
                    "source update does not contain sixteen committed eight-draw groups"
                )
        source_files = {
            name: sha256_bytes((directory / name).read_bytes())
            for name in (
                "segment.json",
                "metrics.jsonl",
                "remote-calls.jsonl",
                "generations.jsonl",
                "rewards.jsonl",
            )
        }
        rewards = {row["sample_id"]: row for row in jsonl(directory / "rewards.jsonl")}
        counts, sample_ids = Counter(), set()
        for row in jsonl(directory / "generations.jsonl"):
            step, sample_id = row["training_step"], row["sample_id"]
            if sample_id in sample_ids or not start < step <= start + 10:
                raise ValueError("duplicate sample or wrong update in source segment")
            sample_ids.add(sample_id)
            group_text = sample_id.split(f":seed-{seed}-B-G-step-{step}-group-", 1)
            if len(group_text) != 2:
                raise ValueError("source sample lacks its authenticated group index")
            group = int(group_text[1].split(":", 1)[0])
            if not 0 <= group < 16 or not 0 <= row["sample_index"] < 8:
                raise ValueError(
                    "source group/sample is outside the frozen eight-draw groups"
                )
            task = scheduled_stage_a_records(pools, order, step)[group]
            expected_id = f"{run_root.name}:seed-{seed}-B-G-step-{step}-group-{group}:tces:{task.task_id}:{task.item_index}:sample-{row['sample_index']}:cap-4096"
            ephemeral = f"ephemeral:{run_root.name}:{canonical_json_hash({'seed': seed, 'method': 'B-G', 'step': step}).removeprefix('sha256:')}"
            if (
                sample_id != expected_id
                or row["task_id"] != task.task_id
                or row["run_id"] != run_root.name
                or row["sampler_checkpoint_path"] != ephemeral
                or row["origin_sampler_checkpoint_path"]
                != segment["origin_sampler_path"]
                or row["sampling_seed"]
                != _group_seeds(seed, step, group, task.task_id)[row["sample_index"]]
                or row["sampling_temperature"] != 1.0
                or row["sampling_top_p"] != 0.95
                or row["sampling_max_tokens"] != 4096
                or row["stop_reason"] not in {"stop", "length"}
            ):
                raise ValueError(f"source sampling contract differs: {sample_id}")
            tokens = tuple(row["completion_token_ids"])
            if (
                len(tokens) != row["sampled_tokens"]
                or _completion(tokenizer, tokens) != row["completion_text"]
            ):
                raise ValueError(
                    f"historical stop-normalized text/tokenizer mismatch: {sample_id}"
                )
            counts[(step, group)] += 1
            yield {
                "generation": row,
                "reward": rewards[sample_id],
                "source": {
                    "run_id": run_root.name,
                    "segment": directory.relative_to(run_root).as_posix(),
                    "files_sha256": source_files,
                    "parent_state_path": segment["parent_state_path"],
                    "parent_completed_update": start,
                    "update_provenance": "ephemeral sampler before optimizer update; intermediate sampler is not a retained checkpoint",
                },
            }
        if (
            sample_ids != set(rewards)
            or len(counts) != 160
            or set(counts.values()) != {8}
        ):
            raise ValueError(
                "source generations/rewards are not the complete aligned rollout groups"
            )
        parent_state, parent_sampler = segment["state_path"], segment["sampler_path"]
