#!/usr/bin/env python3
"""One authorized evaluation-only replication of the saved replay R-P u10."""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path

from duraseed.pilot0_evidence import read_evaluation
from duraseed.provenance import derive_namespaced_seed
from duraseed.replay_data_archive import read_source
from duraseed.replay_data_tokens import local_runtime
from duraseed.replay_evaluation import evaluate
from duraseed.replay_inputs import read_json
from duraseed.replay_remote import ReplayRemote, write_json
from duraseed.replay_rolling import sample_manifest_groups
from duraseed.runners import pilot0_sampling
from duraseed.runners.pilot0_sampling import _task_contracts
from duraseed.runtime import RuntimeBundle, TokenBudget
from duraseed.runtime.billing import PRICE_SNAPSHOT, UsageQuantities

SOURCE = Path("runs/replay-v1/replay-v1-continuation-20260907T075721Z")
POINT = Path("seed-11/R-P/stage_b/u10")


def prepare(repo, root):
    config = read_json(repo / SOURCE / "config.json")
    source, _ = read_source(repo / config["blocks"]["11"]["source_run"], 11)
    manifest = source.prompt_pools.a_monitor_manifest
    original = repo / SOURCE / POINT / "a_monitor"
    previous = read_evaluation(original)
    if previous is None or previous["row_count"] != 1536:
        raise ValueError("original complete 384-item, four-draw evaluation is required")
    identity = read_json(original / "replay-identity.json")
    checkpoint = read_json(repo / SOURCE / POINT / "checkpoint.json")
    assert identity["sampler_path"] == checkpoint["sampler_path"]
    assert identity["manifest_id"] == manifest.manifest_id
    checkpoint["origin_sampler_path"] = identity["origin_sampler_path"]
    local = local_runtime(Path(config["tokenizer_path"]))
    contracts = _task_contracts(source, manifest)
    rows = [
        json.loads(line)
        for line in (original / "generations.jsonl").read_text().splitlines()
    ]
    prompt_tokens = {}
    for task, contract in contracts.items():
        prompt = local.renderer.build_generation_prompt(
            [{"role": "user", "content": contract["prompt_text"]}], role="assistant"
        )
        prompt_tokens[task] = int(prompt.length)
    for row in rows:
        task = row["task_id"]
        expected_seed = derive_namespaced_seed(
            11,
            identity["seed_namespace"],
            "tces",
            task,
            contracts[task]["item_index"],
            row["sample_index"],
        )
        if (
            row["prompt_text"] != contracts[task]["prompt_text"]
            or row["prompt_tokens"] != prompt_tokens[task]
            or row["sampling_seed"] != expected_seed
        ):
            raise ValueError("historical prompt, token count, or sampling seed differs")
    tokens = TokenBudget(sum(prompt_tokens.values()) * 4, 1536 * 4096, 0)
    cost = PRICE_SNAPSHOT.cost(
        UsageQuantities(tokens.prefill, sample_tokens=tokens.sample)
    )
    if root.exists():
        raise FileExistsError(
            "choose a fresh recheck directory; never retry an ambiguous request"
        )
    write_json(
        root / "preflight.json", {"token_budget": asdict(tokens), "main_usd": str(cost)}
    )
    config = deepcopy(config)
    config["preflight"] = {"path": str((root / "preflight.json").relative_to(repo))}
    write_json(root / "config.json", config)
    write_json(
        root / "check.json",
        {
            "date": datetime.now(UTC).isoformat(),
            "purpose": "post-hoc evaluation reproducibility check; original observations remain unchanged",
            "source_evaluation": str(original.relative_to(repo)),
            "checkpoint": checkpoint,
            "items": 384,
            "draws_per_item": 4,
            "sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096},
            "sampling_seeds": "same recorded per-item/per-draw realization as the original u10 evaluation",
            "training_updates": 0,
            "new_remote_checkpoints": 0,
            "token_prices": {"prefill_per_million": 0.66, "sample_per_million": 1.995},
            "price_checked": "2026-09-09; official Tinker models.json",
            "scope": "replicates the explicitly named saved sampler in a new client session; not retrospective provider weight attestation",
        },
    )
    print(
        f"Prepared {manifest.record_count} items / 1536 completions; original prompts and seeds agree.",
        flush=True,
    )
    return config, source, manifest, checkpoint, local


async def execute(repo, root, prepared):
    config, source, manifest, checkpoint, local = prepared
    remote = ReplayRemote(repo, config, root, config["project_id"])
    rest = remote.runtime.service.create_rest_client()
    from tinker import ParsedCheckpointTinkerPath

    parsed = ParsedCheckpointTinkerPath.from_tinker_path(checkpoint["sampler_path"])
    listing = await rest.list_checkpoints_async(parsed.training_run_id)
    saved = next(
        row
        for row in listing.checkpoints
        if row.tinker_path == checkpoint["sampler_path"]
    )
    write_json(root / "checkpoint-metadata.json", saved.model_dump(mode="json"))
    remote.runtime = RuntimeBundle(
        local.sdk, remote.runtime.service, None, local.renderer, local.tokenizer
    )
    remote.eval_inputs.runtime = remote.runtime
    pilot0_sampling.sample_manifest_groups = sample_manifest_groups
    print(
        f"Saved sampler available; expiry {saved.expires_at}. Beginning evaluation, no training.",
        flush=True,
    )
    result = await evaluate(
        remote,
        source,
        checkpoint,
        manifest,
        stage="stage_b",
        update=10,
        arm="R-P",
        purpose="a_monitor",
        draws=4,
        cap=4096,
        output=root / "a_monitor",
    )
    write_json(
        root / "result.json",
        {
            "status": "COMPLETED",
            "completed_at": datetime.now(UTC).isoformat(),
            "items": result["item_count"],
            "completions": result["row_count"],
            "training_updates": 0,
            "billing": remote.snapshot(),
        },
    )
    print(json.dumps(read_json(root / "result.json"), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        parser.error("run-id must be a simple directory name")
    repo = args.repo.resolve()
    root = repo / "runs/replay-v1" / args.run_id
    prepared = prepare(repo, root)
    if args.execute:
        asyncio.run(execute(repo, root, prepared))


if __name__ == "__main__":
    main()
