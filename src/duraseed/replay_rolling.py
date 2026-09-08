"""Four rolling evaluation workers; resume the existing run at a clean boundary."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess

from duraseed.provenance import sha256_bytes
from duraseed.replay_continuation import execute, revise_matching
from duraseed.replay_inputs import block_inputs, load_config, read_json
from duraseed.replay_remote import write_json
from duraseed.runners import pilot0_concurrent_sampling as groups
from duraseed.runners import pilot0_sampling
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import TokenBudget, TokenLedger


async def sample_manifest_groups(
    inputs,
    *,
    pending,
    output,
    contracts,
    samples_per_item,
    max_tokens,
    **arguments,
):
    """Keep four item groups occupied without changing any sampling request."""
    sampled, fresh = {}, []
    for index, record in pending:
        path = groups._evidence_path(output, index)
        if path.exists():
            RemoteJournal(path.parent, reconciled_resume=True)
            sampled[index] = groups._read_rows(path, record.task_id)
        else:
            fresh.append((index, record))
    if not fresh:
        return sampled
    reservations = {
        index: groups._reservation(
            inputs,
            contracts[record.task_id],
            samples_per_item=samples_per_item,
            max_tokens=max_tokens,
        )
        for index, record in fresh
    }
    total = TokenBudget(0, 0, 0)
    for reservation in reservations.values():
        total = total.plus(reservation)
    # The existing parent ledger permits one reservation at a time. Reserve the
    # entire remaining panel once; each worker still has its own bounded ledger.
    inputs.ledger.reserve_call(total)
    work, children, failure = iter(fresh), [], None

    async def worker():
        nonlocal failure
        while failure is None:
            item = next(work, None)
            if item is None:
                return
            index, record = item
            ledger = TokenLedger(reservations[index], inputs.ledger.authorized_usd)
            children.append(ledger)
            try:
                _, rows = await groups._sample_group(
                    inputs,
                    index=index,
                    record=record,
                    contract=contracts[record.task_id],
                    samples_per_item=samples_per_item,
                    max_tokens=max_tokens,
                    output=output,
                    ledger=ledger,
                    **arguments,
                )
                sampled[index] = rows
            except BaseException as error:
                # Do not cancel or retry the other submitted requests. Drain
                # them, persist their evidence, and stop dispatching new items.
                if failure is None:
                    failure = error

    try:
        await asyncio.gather(
            *(worker() for _ in range(min(groups.EVALUATION_CONCURRENCY, len(fresh))))
        )
        if failure is not None:
            raise failure
    except BaseException:
        inputs.ledger.abort_call()
        raise
    actual = TokenBudget(0, 0, 0)
    for ledger in children:
        actual = actual.plus(ledger.observed)
    inputs.ledger.settle_call(actual)
    return sampled


def clean_boundary(root: Path, previous_pid: int) -> dict:
    """Reject a live predecessor, ambiguous calls, or uncheckpointed updates."""
    if previous_pid <= 0:
        raise ValueError("the exact previous runner PID is required")
    try:
        os.kill(previous_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("previous runner is still alive; do not overlap owners")
    states = list(root.rglob("remote-call-state.json"))
    if not states or any(read_json(path)["pending"] is not None for path in states):
        raise ValueError("pending remote call; no scheduling handoff or retry")
    progress = read_json(root / "progress.json")
    if progress["pending"] or progress["phase"] != "evaluation":
        raise ValueError("handoff requires a completed evaluation boundary")
    directory = (
        root
        / f"seed-{progress['seed']}"
        / progress["replay_arm"]
        / "stage_b"
        / f"u{progress['update']}"
    )
    checkpoint = read_json(directory / "checkpoint.json")
    if checkpoint["update"] != progress["update"]:
        raise ValueError("boundary checkpoint does not match progress")
    updates = list(root.glob("seed-*/*/stage_b/u*/update-*.json"))
    if any(not (path.parent / "checkpoint.json").exists() for path in updates):
        raise ValueError("incomplete training segment; never replay it")
    billing = read_json(root / "billing.json")
    journal = read_json(root / "remote/remote-call-state.json")
    floor = journal["reserved_floor"]
    if (
        any(
            billing["committed"][key] != floor[f"{key}_tokens"]
            for key in ("prefill", "sample", "train")
        )
        or billing["committed_fixed_usd"] != floor["fixed_usd"]
    ):
        raise ValueError("billing and completed-call floor disagree")
    return {
        "previous_pid": previous_pid,
        "completed_calls": journal["completed_count"],
        "committed_updates": len(updates),
        "boundary": {
            key: progress[key] for key in ("seed", "replay_arm", "update", "purpose")
        },
        "checkpoint": str((directory / "checkpoint.json").relative_to(root)),
        "prior_billing": billing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--previous-pid", type=int, required=True)
    parser.add_argument("--approval", default="")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.approval != "ok do it":
        parser.error("the owner's scheduling-change approval is required")
    repo, root = args.repo.resolve(), args.run_root.resolve()
    if not root.is_relative_to(repo / "runs/replay-v1"):
        parser.error("resume the existing replay run inside the repository")
    boundary = clean_boundary(root, args.previous_pid)
    config = load_config(repo, root / "config.json")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            config["implementation_commit"],
            "--",
            *config["implementation_files"],
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    matching = read_json(root / "matching.json")
    continuation = read_json(root / "continuation.json")
    original = read_json(repo / continuation["source_run"] / "matching.json")
    if matching != revise_matching(original) or config["continuation"] != continuation:
        raise ValueError("matching or continuation settings changed")
    record = {
        "authorized_at_utc": datetime.now(UTC).isoformat(),
        "owner_direction": args.approval,
        "change": "Replace four-item batch barriers with four rolling workers only.",
        "handoff": "Read-only empty next-panel directory stopped the predecessor before its next remote call; no request cancelled or repeated.",
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        **boundary,
    }
    if args.check:
        print(
            {
                key: record[key]
                for key in ("boundary", "committed_updates", "completed_calls")
            }
        )
        print("Clean local handoff checks passed; no remote calls made.")
        return
    record_path = root / "scheduling-handoff.json"
    if record_path.exists():
        raise ValueError("handoff already launched; inspect it rather than retry")
    sources = {
        int(seed): block_inputs(repo, config, int(seed))[0]
        for seed, row in matching.items()
        if row["selected"] is not None
    }
    write_json(record_path, record)
    pilot0_sampling.sample_manifest_groups = sample_manifest_groups
    asyncio.run(execute(repo, root, config, matching, sources))


if __name__ == "__main__":
    main()
