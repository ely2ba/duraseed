#!/usr/bin/env python3
"""One authorized replay-order replication, separate from all original runs."""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from duraseed.replay_evaluation import evaluate, raw_counts
from duraseed.replay_inputs import candidate_manifest, read_json, verified_file
from duraseed.replay_remote import ReplayRemote, write_json
from duraseed.replay_replication_inputs import SEED, SOURCE_BLOCK, load_inputs, prepare
from duraseed.replay_replication_matching import match, nominate
from duraseed.replay_rolling import sample_manifest_groups
from duraseed.runners import pilot0_concurrent_sampling as groups
from duraseed.runners import pilot0_sampling
from duraseed.runners import replay
from duraseed.runners.replay import stage_a, stage_b
from duraseed.runtime import sft_datum

ARMS = ("R-S", "R-P")
PROTOCOL = Path("docs/experiments/replay-order-replication.md")


class CachedDatumRemote(ReplayRemote):
    """Reuse pure conversion results; retain every original remote operation."""

    def __init__(self, *args, **kwargs):
        self.datums = {}
        self.renderer_signature = None
        super().__init__(*args, **kwargs)

    async def restore(self, path, *, full_state, coordinate, sources=()):
        await super().restore(
            path, full_state=full_state, coordinate=coordinate, sources=sources
        )
        tokenizer = self.runtime.tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        signature = (
            (
                type(self.runtime.renderer),
                backend.to_str(),
                tokenizer.special_tokens_map,
            )
            if backend is not None
            else None
        )
        if not full_state or signature is None or signature != self.renderer_signature:
            self.datums.clear()
        self.renderer_signature = signature

    def datum(self, runtime, source, *, max_length):
        key = (source.prompt_text, source.verified_completion_text, max_length)
        if key not in self.datums:
            self.datums[key] = sft_datum(runtime, source, max_length=max_length)
        return self.datums[key]


async def run_pair(remote, source, records):
    cadence = {}
    for arm in ARMS:
        cadence[arm] = await stage_a(remote, source, arm, records[arm])
        print(f"{datetime.now(UTC).isoformat()} {arm} acquisition complete", flush=True)
    nomination = nominate(SEED, cadence)
    write_json(remote.root / "nominations.json", nomination)
    manifest, assessments = candidate_manifest(source), {}
    for arm in ARMS:
        assessments[arm] = []
        for row in nomination["nominees"][arm]:
            step = row["update"]
            point = remote.root / f"seed-{SEED}/{arm}/stage_a/u{step}"
            checkpoint = read_json(point / "checkpoint.json")
            result = await evaluate(
                remote,
                source,
                checkpoint,
                manifest,
                stage="stage_a",
                update=step,
                arm=arm,
                purpose="candidate",
                draws=16,
                cap=4096,
                output=point / "candidate",
            )
            assessments[arm].append({"update": step, **raw_counts(result, "targeted")})
    selection = match(nomination, assessments)
    matching = {str(SEED): selection}
    write_json(remote.root / "matching.json", matching)
    if selection["stage_b_allowed"]:
        for arm in ARMS:
            await stage_b(remote, source, arm, selection["selected"][arm])
            print(f"{datetime.now(UTC).isoformat()} {arm} Stage B complete", flush=True)
    return {
        "status": "COMPLETED" if selection["stage_b_allowed"] else "NO_MATCH",
        "source_block": SOURCE_BLOCK,
        "acquisition_order_seed": SEED,
        "matching": matching,
        "completed_at": datetime.now(UTC).isoformat(),
        "billing": remote.snapshot(),
    }


async def execute(repo, root):
    if (root / "remote").exists():
        raise RuntimeError(
            "already dispatched; inspect requests, never blindly relaunch"
        )
    config, source, records = load_inputs(repo)
    prepared = read_json(root / "config.json")
    # The offline preparation binds the exact ordered input and recipe.
    references = {"protocol", "preflight", "source_config"}
    if {k: v for k, v in prepared.items() if k not in references} != {
        k: v for k, v in config.items() if k not in references
    }:
        raise ValueError("prepared recipe or source mapping changed")
    for key in references:
        verified_file(repo, prepared[key])
    expected_order = [row.task_id for row in records["R-S"]]
    if read_json(root / "corpus_order.json")["task_ids"] != expected_order:
        raise ValueError("prepared acquisition order changed")
    remote = CachedDatumRemote(repo, prepared, root, prepared["project_id"])
    groups.EVALUATION_CONCURRENCY = 8
    pilot0_sampling.sample_manifest_groups = sample_manifest_groups
    replay.sft_datum = remote.datum
    result = await run_pair(remote, source, records)
    write_json(root / "result.json", result)
    print(f"Replication collection terminal: {result['status']}", flush=True)


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
    if not args.execute:
        prepare(repo, root, repo / PROTOCOL)
        print(f"Prepared source block {SOURCE_BLOCK}, order seed {SEED}: {root}")
        return
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


if __name__ == "__main__":
    main()
