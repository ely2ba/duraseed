"""Frozen local inputs and original Pilot populations for supervised replay."""

from __future__ import annotations

import json
from pathlib import Path

from duraseed.data.manifests import build_manifest
from duraseed.provenance import sha256_bytes
from duraseed.replay_data_archive import read_source
from duraseed.replay_data_tokens import local_runtime, measure_source
from duraseed.replay_matching import ARMS, STAGE_A_GRID, STAGE_B_GRID
from duraseed.training.sft import VerifiedSourceRecord


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def verified_file(repo: Path, entry: dict) -> Path:
    path = (repo / entry["path"]).resolve()
    if not path.is_relative_to(repo.resolve()):
        raise ValueError("replay inputs must be inside the repository")
    if sha256_bytes(path.read_bytes()) != entry["sha256"]:
        raise ValueError(f"frozen input changed: {entry['path']}")
    return path


def load_config(repo: Path, path: Path) -> dict:
    config = read_json(path)
    if (
        config["namespace"] != "replay-v1"
        or config["arms"] != list(ARMS)
        or config["seeds"] != [11, 29]
        or config["stage_a_grid"] != list(STAGE_A_GRID)
        or config["stage_b_grid"] != list(STAGE_B_GRID)
        or config["stage_a_lr"] != 1e-4
        or config["stage_b_lr"] != 3e-4
        or config["batch_size"] != 32
        or config["evaluation"] != {"temperature": 1.0, "top_p": 0.95}
        or config["checkpoint_ttl_seconds"] != 30 * 86400
    ):
        raise ValueError("replay configuration contradicts the fixed experiment")
    verified_file(repo, config["protocol"])
    verified_file(repo, config["preflight"])
    for block in config["blocks"].values():
        verified_file(repo, block["corpus"])
        verified_file(repo, block["audit"])
    return config


def block_inputs(repo: Path, config: dict, seed: int):
    block = config["blocks"][str(seed)]
    source, old_preflight = read_source(repo / block["source_run"], seed)
    audit = read_json(verified_file(repo, block["audit"]))
    if (
        audit["status"] != "READY"
        or old_preflight["lineage"]["m0_state_path"] != block["m0_state_path"]
        or old_preflight["lineage"]["manifest_ids"] != block["manifest_ids"]
    ):
        raise ValueError("replay source is unavailable or differs from its freeze")
    rows = [
        json.loads(line)
        for line in verified_file(repo, block["corpus"]).read_text().splitlines()
    ]
    if len(rows) < 32 or len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("DATA_BLOCKED: replay needs at least 32 unique prompts")
    records = {
        arm: tuple(
            VerifiedSourceRecord.model_validate_json(json.dumps(row[arm]))
            for row in rows
        )
        for arm in ARMS
    }
    return source, records


def candidate_manifest(source):
    """Exactly the existing 256 targeted a_validation items, no new task build."""
    original = source.a_validation
    families = set(source.prompt_pools.artifact.boundary_family_ids)
    records = [row for row in original.records if row.intended_family in families]
    if len(records) != 256:
        raise ValueError("candidate assessment requires 256 original targeted items")
    return build_manifest(
        name="replay-v1-targeted-candidates",
        split=original.split,
        generator_version=original.generator_version,
        root_seed=original.root_seed,
        records=records,
        parent_manifest_id=original.manifest_id,
        metadata={"scope": "nominee assessment only, not independent holdout"},
    )


def verify_tokenizer(runtime, repo: Path, config: dict, sources) -> None:
    """Compare every full datum against the frozen local renderer before training."""
    local = local_runtime(repo / config["tokenizer_path"])
    for source in sources:
        expected = measure_source(local, source)
        actual = measure_source(runtime, source)
        messages = [
            {"role": "user", "content": source.prompt_text},
            {"role": "assistant", "content": source.verified_completion_text},
        ]
        token_arrays = [
            bundle.renderer.build_supervised_example(
                messages, train_on_what=bundle.sdk.train_on_what.LAST_ASSISTANT_MESSAGE
            )[0].to_ints()
            for bundle in (local, runtime)
        ]
        if expected != actual or list(token_arrays[0]) != list(token_arrays[1]):
            raise ValueError("provider tokenizer differs from frozen replay datums")
        if actual["full_rendered_tokens"] > config["max_length"]:
            raise ValueError("frozen replay max_length cannot contain a full datum")
