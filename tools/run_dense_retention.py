#!/usr/bin/env python3
"""Authorized two-arm 20-update replication with a dense arithmetic-only grid."""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from duraseed.pilot0_data import stage_b_sources
from duraseed.pilot0_evidence import read_evaluation
from duraseed.replay_data_archive import read_source
from duraseed.replay_data_tokens import local_runtime
from duraseed.replay_evaluation import evaluate
from duraseed.replay_inputs import read_json
from duraseed.replay_remote import ReplayRemote, write_json
from duraseed.replay_rolling import sample_manifest_groups
from duraseed.runners import pilot0_concurrent_sampling as groups
from duraseed.runners import pilot0_sampling
from duraseed.runners.replay import train_segment
from duraseed.runtime import TokenBudget, sft_datum
from duraseed.runtime.billing import PRICE_SNAPSHOT, UsageQuantities

SOURCE = Path("runs/replay-v1/replay-v1-continuation-20260907T075721Z")
SELECTED = {"R-S": 220, "R-P": 20}
GRID = tuple(range(1, 21))
SAMPLING_WORKERS = 8


def inputs(repo):
    config = read_json(repo / SOURCE / "config.json")
    if (
        config["stage_b_lr"] != 3e-4
        or config["batch_size"] != 32
        or config["evaluation"] != {"temperature": 1.0, "top_p": 0.95}
    ):
        raise ValueError("original Stage-B recipe differs")
    source, _ = read_source(repo / config["blocks"]["11"]["source_run"], 11)
    return config, source, stage_b_sources(source)


def prepare(repo, root):
    if root.exists():
        raise FileExistsError("run already prepared; use --execute, never overwrite")
    config, source, records = inputs(repo)
    local = local_runtime(Path(config["tokenizer_path"]))
    manifest = source.prompt_pools.a_monitor_manifest
    if manifest.record_count != 384 or len(records) != 4096:
        raise ValueError("original monitor or MAPS training population differs")
    datums = [sft_datum(local, r, max_length=config["max_length"]) for r in records]
    train = sum(int(d.model_input.length) for d in datums[: 20 * 32]) * 2
    prefill = (
        40
        * 4
        * sum(
            int(
                local.renderer.build_generation_prompt(
                    [{"role": "user", "content": pilot0_sampling._prompt(r)}],
                    role="assistant",
                ).length
            )
            for r in manifest.records
        )
    )
    tokens = TokenBudget(prefill, 40 * 384 * 4 * 4096, train)
    storage = 40 * config["storage_per_pair_usd"]
    token_cost = PRICE_SNAPSHOT.cost(
        UsageQuantities(prefill, sample_tokens=tokens.sample, train_tokens=train)
    )
    origins, baselines = {}, {}
    for arm, update in SELECTED.items():
        relative = Path(f"seed-11/{arm}/stage_a/u{update}/checkpoint.json")
        checkpoint = read_json(repo / SOURCE / relative)
        baseline = repo / SOURCE / f"seed-11/{arm}/pre-b/a_monitor"
        result = read_evaluation(baseline)
        if (
            checkpoint["update"] != update
            or checkpoint["replay_arm"] != arm
            or result is None
            or result["row_count"] != 1536
            or result["sampler_path"] != checkpoint["sampler_path"]
            or result["manifest_id"] != manifest.manifest_id
        ):
            raise ValueError("selected checkpoint or reused baseline differs")
        # The reconstructed first 20 updates must consume the original tokens.
        old_tokens = sum(
            read_json(p)["train_tokens"]
            for p in (repo / SOURCE / f"seed-11/{arm}/stage_b").glob("u*/update-*.json")
            if int(p.stem.split("-")[1]) <= 20
        )
        if old_tokens != train // 2:
            raise ValueError("reconstructed Stage-B datum tokens differ")
        origins[arm], baselines[arm] = checkpoint, str(baseline.relative_to(repo))
    config = deepcopy(config)
    config["preflight"] = {"path": str((root / "preflight.json").relative_to(repo))}
    write_json(root / "config.json", config)
    write_json(
        root / "preflight.json",
        {
            "token_budget": asdict(tokens),
            "main_usd": str(token_cost + storage),
            "storage_allowance_usd": storage,
            "scope": "exact finite 40-update/40-evaluation workload; no extra approval step",
        },
    )
    write_json(
        root / "design.json",
        {
            "authorized_at_utc": datetime.now(UTC).isoformat(),
            "owner_direction": "do it",
            "source_run": str(SOURCE),
            "seed": 11,
            "selected": SELECTED,
            "stage_b_grid": [0, *GRID],
            "baseline_reuse": True,
            "baselines": baselines,
            "purpose": "post-hoc dense-grid replication of the first 20 Stage-B updates; original results unchanged",
            "training": "same MAPS shortest2_cap2 corpus/order, batch32, LR3e-4, completion-only cross-entropy; fresh optimizer per arm and full-state continuity thereafter",
            "optimizer": {
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-12,
                "weight_decay": 0,
                "grad_clip_norm": 0,
            },
            "evaluation": {
                "items": 384,
                "targeted": 192,
                "sentinel": 192,
                "draws_per_item": 4,
                "cap": 4096,
                "temperature": 1,
                "top_p": 0.95,
            },
            "randomness": "original replay-v1.stage_b.a_monitor.<update> seed derivation, shared across arms; same namespaces at historical anchors",
            "checkpointing": "sampler/state pair each update; full optimizer restore between one-update segments",
            "sampling_workers": SAMPLING_WORKERS,
            "summaries": "raw Pass@1 curves, dense trapezoid/20, first downward crossing of half own baseline with linear interpolation, length/validity; paired item-bootstrap uncertainty",
            "not_in_scope": "new matching, acquisition, seeds, MAPS evaluations, altered LR/prompts, or continuation beyond20",
        },
    )
    for arm, checkpoint in origins.items():
        write_json(
            root / f"seed-11/{arm}/stage_a/u{SELECTED[arm]}/checkpoint.json", checkpoint
        )
    print(f"Prepared two arms, 40 updates, 61,440 new completions: {root}", flush=True)


async def run_grid(remote, source, records):
    manifest = source.prompt_pools.a_monitor_manifest
    for arm, update in SELECTED.items():
        previous = read_json(
            remote.root / f"seed-11/{arm}/stage_a/u{update}/checkpoint.json"
        )
        origin = previous["sampler_path"]
        for stop in GRID:
            previous = await train_segment(
                remote, source, arm, "stage_b", records, previous, stop - 1, stop
            )
            previous["origin_sampler_path"] = origin
            result = await evaluate(
                remote,
                source,
                previous,
                manifest,
                stage="stage_b",
                update=stop,
                arm=arm,
                purpose="a_monitor",
                draws=4,
                cap=4096,
                output=remote.root / f"seed-11/{arm}/stage_b/u{stop}/a_monitor",
            )
            print(
                f"{datetime.now(UTC).isoformat()} {arm} u{stop}/20: {result['row_count']} completions committed",
                flush=True,
            )


async def execute(repo, root):
    if (root / "remote").exists():
        raise RuntimeError(
            "already dispatched; inspect existing requests, do not automatically relaunch"
        )
    config, source, records = inputs(repo)
    prepared = read_json(root / "config.json")
    if {k: v for k, v in prepared.items() if k != "preflight"} != {
        k: v for k, v in config.items() if k != "preflight"
    }:
        raise ValueError("prepared training configuration changed")
    remote = ReplayRemote(repo, prepared, root, prepared["project_id"])
    groups.EVALUATION_CONCURRENCY = SAMPLING_WORKERS
    pilot0_sampling.sample_manifest_groups = sample_manifest_groups
    await run_grid(remote, source, records)
    write_json(
        root / "result.json",
        {
            "status": "COMPLETED",
            "completed_at": datetime.now(UTC).isoformat(),
            "training_updates": 40,
            "new_evaluation_items": 40 * 384,
            "new_completions": 40 * 1536,
            "reused_baseline_completions": 3072,
            "billing": remote.snapshot(),
        },
    )
    print("Dense-grid collection complete; offline analysis can now run.", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.run_id or "/" in args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a simple directory name")
    repo = args.repo.resolve()
    root = repo / "runs/replay-v1" / args.run_id
    if args.execute:
        try:
            asyncio.run(execute(repo, root))
        except Exception as error:
            write_json(
                root / "error.json",
                {
                    "at": datetime.now(UTC).isoformat(),
                    "type": type(error).__name__,
                    "message": str(error),
                    "action": "inspect durable requests; never automatically retry",
                },
            )
            raise
    else:
        prepare(repo, root)


if __name__ == "__main__":
    main()
