"""Read-only validation of the consolidated boundary observation journal."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from duraseed.data.manifests import DatasetManifest
from duraseed.run_records import GenerationRecord, RewardRecord
from duraseed.runners import RunnerGateError


ACTION_INDICES = {
    "extension1-confirm": tuple(range(16)),
    "extension2-broad": tuple(range(4)),
    "extension2-refine": tuple(range(4, 16)),
    "extension2-confirm": tuple(range(16)),
}


def _rows(path: Path, model, label: str):
    try:
        return tuple(
            model.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValueError) as error:
        raise RunnerGateError(f"invalid {label} rows") from error


def load_consolidated_journal(
    directory: Path,
    *,
    run_id: str,
    manifests: Mapping[str, DatasetManifest],
) -> tuple[
    dict[str, tuple[GenerationRecord, ...]],
    dict[str, tuple[RewardRecord, ...]],
    dict[str, frozenset[str]],
]:
    """Validate journal groups and require identical raw row projections."""

    generations = {action: [] for action in manifests}
    rewards = {action: [] for action in manifests}
    task_ids = {action: set() for action in manifests}
    all_generations: list[GenerationRecord] = []
    all_rewards: list[RewardRecord] = []
    sample_ids: set[str] = set()
    reward_ids: set[str] = set()
    try:
        lines = (directory / "observation_groups.jsonl").read_text().splitlines()
    except OSError as error:
        raise RunnerGateError("consolidated boundary journal is unreadable") from error
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            action, task_id = value["action"], value["task_id"]
            group_g = tuple(
                GenerationRecord.model_validate(row) for row in value["generations"]
            )
            group_r = tuple(
                RewardRecord.model_validate(row) for row in value["rewards"]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RunnerGateError(
                "consolidated boundary journal is malformed"
            ) from error
        if action not in manifests or not isinstance(task_id, str):
            raise RunnerGateError("consolidated boundary journal has an unknown action")
        manifest = manifests[action]
        records = {row.task_id: row for row in manifest.records}
        by_sample = {row.sample_id: row for row in group_r}
        ids = {row.sample_id for row in group_g}
        if (
            task_id not in records
            or task_id in task_ids[action]
            or tuple(row.sample_index for row in group_g) != ACTION_INDICES[action]
            or len(group_g) != len(group_r)
            or len(ids) != len(group_g)
            or set(by_sample) != ids
            or sample_ids.intersection(ids)
            or reward_ids.intersection(row.reward_id for row in group_r)
            or any(
                row.task_id != task_id
                or row.task_manifest_id != manifest.manifest_id
                or row.run_id != f"{run_id}:{action}"
                or by_sample[row.sample_id].task_id != task_id
                or by_sample[row.sample_id].reward != row.reward
                for row in group_g
            )
        ):
            raise RunnerGateError("consolidated boundary group coordinates changed")
        task_ids[action].add(task_id)
        sample_ids.update(ids)
        reward_ids.update(row.reward_id for row in group_r)
        generations[action].extend(group_g)
        rewards[action].extend(group_r)
        all_generations.extend(group_g)
        all_rewards.extend(group_r)
    projected_g = _rows(
        directory / "generations.jsonl", GenerationRecord, "generation projection"
    )
    projected_r = _rows(directory / "rewards.jsonl", RewardRecord, "reward projection")
    if tuple(all_generations) != projected_g or tuple(all_rewards) != projected_r:
        raise RunnerGateError("consolidated projections differ from the group journal")
    return (
        {key: tuple(value) for key, value in generations.items()},
        {key: tuple(value) for key, value in rewards.items()},
        {key: frozenset(value) for key, value in task_ids.items()},
    )


__all__ = ["ACTION_INDICES", "load_consolidated_journal"]
