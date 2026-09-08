"""Explicit human approval boundary; offline checking never creates a service."""

from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal
from pathlib import Path
import subprocess

from duraseed.provenance import sha256_bytes
from duraseed.replay_budget import PROTECTED_USD, money
from duraseed.replay_inputs import block_inputs, load_config, read_json
from duraseed.replay_remote import ReplayRemote, write_json
from duraseed.runners.replay import run_package
from duraseed.runtime import sft_datum


def approve(repo: Path, config_path: Path, config: dict, phrase: str):
    preflight = read_json(repo / config["preflight"]["path"])
    ceiling = money(preflight["approval_ceiling_usd"])
    expected = (
        f"AUTHORIZE DURASEED REPLAY {config['protocol']['sha256']} "
        f"{sha256_bytes(config_path.read_bytes())} CEILING ${ceiling:.2f}"
    )
    direction = config.get("owner_launch_direction") or {}
    owner_directed = (
        phrase == direction.get("text")
        and direction.get("ceiling_usd") == str(ceiling)
        and direction.get("billing_assumptions_disclosed") is True
    )
    if phrase != expected and not owner_directed:
        raise ValueError(f"explicit launch approval required: {expected}")
    unverified = {
        "CHECKPOINT_SIZE_BOUND_UNVERIFIED",
        "BACKUP_LIABILITY_UNVERIFIED",
        "LIVE_BALANCE_AND_OTHER_COMMITMENTS_UNVERIFIED",
    }
    blockers = set(preflight["blockers"])
    liability = preflight.get("all_in_maximum_liability_usd")
    if owner_directed and blockers <= unverified:
        # Owner directed launch after disclosure. Keep unknowns in the record;
        # this authorizes the operational allowance, not a verified invoice bound.
        liability = preflight["package_usd"] if liability is None else liability
        blockers = set()
    if blockers or liability is None:
        raise ValueError("preflight still has unresolved service/billing liability")
    if (
        ceiling > Decimal("2400")
        or money(liability) > ceiling
        or money(preflight["current_balance_usd"])
        - money(preflight["other_committed_usd"] or "0")
        - ceiling
        < PROTECTED_USD
    ):
        raise ValueError("budget or protected-credit reserve would be violated")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != config["implementation_commit"]:
        raise ValueError("implementation commit differs from the approved config")
    for path in config["implementation_files"]:
        if subprocess.check_output(["git", "diff", "HEAD", "--", path], cwd=repo):
            raise ValueError(f"approved implementation has uncommitted edits: {path}")
    return preflight


async def engineering_smoke(repo, config, root, project_id):
    """Only the newly required full-length path, never a scientific fifth arm."""
    complete = root / "result.json"
    if complete.exists():
        if read_json(complete).get("status") != "ENGINEERING_ONLY_PASSED":
            raise ValueError("engineering evidence is not a passing completed smoke")
        return
    records = []
    for seed in (11, 29):
        _, arms = block_inputs(repo, config, seed)
        records.extend((seed, row) for values in arms.values() for row in values)
    # Use the frozen tokenized maximum, not a newly sampled or hand-written input.
    from duraseed.replay_data_tokens import local_runtime, measure_source

    local = local_runtime(repo / config["tokenizer_path"])
    seed, record = max(
        records, key=lambda row: measure_source(local, row[1])["full_rendered_tokens"]
    )
    remote = ReplayRemote(repo, config, root, project_id, smoke=True)
    coordinate = {"stage": "engineering_only", "seed": seed}
    await remote.restore(
        config["blocks"][str(seed)]["m0_state_path"],
        full_state=False,
        coordinate=coordinate,
        sources=(record,),
    )
    datum = sft_datum(remote.runtime, record, max_length=config["max_length"])
    await remote.update(
        [datum], 1e-4, root / "update.json", {**coordinate, "update": 1}
    )
    checkpoint = await remote.save(
        f"{root.name}-longest", root / "checkpoint.json", coordinate
    )
    await remote.restore(
        checkpoint["state_path"], full_state=True, coordinate=coordinate
    )
    await remote.restore(
        checkpoint["state_path"], full_state=False, coordinate=coordinate
    )
    write_json(
        complete,
        {
            "status": "ENGINEERING_ONLY_PASSED",
            "scientific_evidence": False,
            "max_length": config["max_length"],
            "billing": remote.snapshot(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="local input checks only")
    parser.add_argument("--approval", default="")
    parser.add_argument("--project-id")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    repo, config_path = args.repo.resolve(), args.config.resolve()
    config = load_config(repo, config_path)
    for seed in (11, 29):
        block_inputs(repo, config, seed)
    if args.check:
        print(
            "Local frozen inputs valid; no service was created, no training authorized."
        )
        return
    preflight = approve(repo, config_path, config, args.approval)
    if (
        args.project_id != config["project_id"]
        or not args.run_id
        or "/" in args.run_id
        or args.run_id in {".", ".."}
    ):
        raise ValueError(
            "the explicit frozen project ID and a simple run ID are required"
        )
    root = repo / "runs" / "replay-v1" / args.run_id
    approval_record = {
        "phrase": args.approval,
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "preflight": config["preflight"],
        "ceiling_usd": str(preflight["approval_ceiling_usd"]),
        "owner_launch_direction": config.get("owner_launch_direction"),
        "unverified_billing_items": preflight["blockers"],
    }
    if (root / "authorization.json").exists() and read_json(
        root / "authorization.json"
    ) != approval_record:
        raise ValueError("run authorization differs; no reused run directory")
    write_json(root / "authorization.json", approval_record)

    async def execute():
        await engineering_smoke(
            repo, config, root / "engineering-only", args.project_id
        )
        remote = ReplayRemote(repo, config, root, args.project_id)
        await run_package(remote)
        from duraseed.replay_results import write_report

        write_report(root)

    asyncio.run(execute())


if __name__ == "__main__":
    main()
