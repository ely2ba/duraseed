"""Approved between-arm-only matching revision; reuse retained Stage-A evidence."""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction
from pathlib import Path
import subprocess

from duraseed.pilot0_evidence import read_evaluation
from duraseed.provenance import sha256_bytes
from duraseed.replay_budget import money, token_cost
from duraseed.replay_evaluation import raw_counts
from duraseed.replay_inputs import block_inputs, load_config, read_json
from duraseed.replay_matching import ARMS, STAGE_B_GRID, TOLERANCE, match
from duraseed.replay_remote import ReplayRemote, write_json
from duraseed.replay_results import write_report
from duraseed.runners.replay import stage_b
from duraseed.runtime import TokenBudget

OWNER_DIRECTION = "Ok continue working since we agreed on the matching now"
COMPONENTS = {
    "stage_b_train",
    "pre_b_and_final_validation",
    "stage_b_monitor_including_pre_b",
    "stage_b_maps_including_pre_b",
}


def revise_matching(original: dict) -> dict:
    """Remove only target-band eligibility; preserve nominees and every tie-break."""
    revised = deepcopy(original)
    for selection in revised.values():
        for row in selection["combinations"]:
            row["eligible"] = (
                Fraction(
                    abs(row["R-S_successes"] - row["R-P_successes"]),
                    row["trials_per_arm"],
                )
                <= TOLERANCE
            )
        eligible = [row for row in selection["combinations"] if row["eligible"]]
        selection["selected"] = (
            min(eligible, key=lambda row: tuple(map(Fraction, row["selection_key"])))
            if eligible
            else None
        )
        selection["status"] = "MATCHED" if eligible else "NO_MATCH"
        selection["stage_b_allowed"] = bool(eligible)
        selection["historical_target_required"] = False
    return revised


def completed_source(root: Path) -> dict:
    terminal, original = (
        read_json(root / "result.json"),
        read_json(root / "matching.json"),
    )
    if terminal["status"] != "NO_MATCH" or terminal["matching"] != original:
        raise ValueError("continuation requires the original terminal NO_MATCH record")
    nominees = read_json(root / "nominations.json")
    for seed in (11, 29):
        assessments = {}
        for arm in ARMS:
            base = root / f"seed-{seed}" / arm
            if (base / "stage_b").exists() or (base / "pre-b").exists():
                raise ValueError("original run already contains Stage-B work")
            assessments[arm] = []
            for row in nominees[str(seed)]["nominees"][arm]:
                step = row["update"]
                result = read_evaluation(base / "stage_a" / f"u{step}" / "candidate")
                if result is None:
                    raise ValueError("a frozen candidate assessment is missing")
                assessments[arm].append(
                    {"update": step, **raw_counts(result, "targeted")}
                )
        if match(nominees[str(seed)], assessments) != original[str(seed)]:
            raise ValueError("original matching does not reproduce from saved counts")
    return original


def continuation_preflight(source: Path, original: dict, matching: dict, config: dict):
    selected = {int(seed) for seed, row in matching.items() if row["selected"]}
    components = [
        row
        for row in original["components"]
        if row["seed"] in selected and row["component"] in COMPONENTS
    ]
    if len(components) != 4 * len(selected) or not selected:
        raise ValueError(
            "continuation needs one complete Stage-B budget per matched block"
        )
    tokens = TokenBudget(
        *(sum(row[key] for row in components) for key in ("prefill", "sample", "train"))
    )
    pairs = len(selected) * len(ARMS) * (len(STAGE_B_GRID) - 1)
    storage = pairs * money(config["storage_per_pair_usd"])
    main = token_cost(tokens) + storage
    observed = reserved = Decimal(0)
    for directory in (source, source / "engineering-only"):
        state = read_json(directory / "remote" / "remote-call-state.json")
        billing = read_json(directory / "billing.json")
        floor = state["reserved_floor"]
        if (
            state["pending"] is not None
            or any(
                billing["committed"][key] != floor[f"{key}_tokens"]
                for key in ("prefill", "sample", "train")
            )
            or money(billing["committed_fixed_usd"]) != money(floor["fixed_usd"])
        ):
            raise ValueError("source billing is inconsistent or a request is pending")
        observed += token_cost(TokenBudget(**billing["observed"]))
        observed += money(billing["observed_fixed_usd"])
        reserved += token_cost(TokenBudget(**billing["committed"]))
        reserved += money(billing["committed_fixed_usd"])
    ceiling = money(original["approval_ceiling_usd"])
    recovery = money(original["recovery_reservation_usd"])
    if reserved + main + recovery > ceiling or ceiling > Decimal("2400"):
        raise ValueError("continuation exceeds the existing approved package ceiling")
    return {
        "components": components,
        "token_budget": {
            key: getattr(tokens, key) for key in ("prefill", "sample", "train")
        },
        "main_usd": str(main),
        "continuation_ceiling_usd": str(
            main.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        ),
        "checkpoint_pairs": pairs,
        "storage_allowance_usd": str(storage),
        "source_observed_tokens_plus_storage_allowances_usd": str(observed),
        "source_full_cap_reservations_usd": str(reserved),
        "combined_observed_plus_continuation_ceiling_usd": str(observed + main),
        "recovery_reservation_usd": str(recovery),
        "conservative_package_bound_usd": str(reserved + main + recovery),
        "approval_ceiling_usd": str(ceiling),
        "prices_per_million_usd": original["prices_per_million_usd"],
        "unverified_billing_items": original["blockers"],
        "billing_note": "Local token observations plus storage allowances, not settled provider invoice; external commitments and backup retention remain unverified.",
    }


def keep(path: Path, value: dict):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"continuation record changed: {path.name}")
    write_json(path, value)


def prepare(repo: Path, source: Path, config_path: Path, root: Path):
    if source == root or not source.is_relative_to(repo / "runs/replay-v1"):
        raise ValueError("the original run must be preserved in a separate directory")
    config = load_config(repo, config_path)
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
    original = completed_source(source)
    matching = revise_matching(original)
    preflight = continuation_preflight(
        source,
        read_json(repo / config["preflight"]["path"]),
        matching,
        config,
    )
    record = {
        "source_run": str(source.relative_to(repo)),
        "source_config": str(config_path.relative_to(repo)),
        "source_matching_sha256": sha256_bytes((source / "matching.json").read_bytes()),
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "date": "2026-09-07",
        "owner_direction": OWNER_DIRECTION,
        "revision": "Remove historical-score eligibility only; preserve between-arm tolerance, nominees, tie-breaks and all Stage-B settings.",
    }
    keep(root / "continuation.json", record)
    keep(root / "matching.json", matching)
    keep(root / "preflight.json", preflight)
    keep(root / "nominations.json", read_json(source / "nominations.json"))
    sources = {}
    for seed, selection in matching.items():
        if selection["selected"] is None:
            continue
        sources[int(seed)], _ = block_inputs(repo, config, int(seed))
        for arm in ARMS:
            step = selection["selected"][arm]
            relative = (
                Path(f"seed-{seed}") / arm / "stage_a" / f"u{step}" / "checkpoint.json"
            )
            checkpoint = read_json(source / relative)
            if any(
                checkpoint[key] != value
                for key, value in {
                    "seed": int(seed),
                    "replay_arm": arm,
                    "stage": "stage_a",
                    "update": step,
                    "ttl_seconds": config["checkpoint_ttl_seconds"],
                }.items()
            ):
                raise ValueError("selected checkpoint identity differs")
            keep(root / relative, checkpoint)
    config = deepcopy(config)
    config["preflight"] = {
        "path": str((root / "preflight.json").relative_to(repo)),
        "sha256": sha256_bytes((root / "preflight.json").read_bytes()),
    }
    config["continuation"] = record
    keep(root / "config.json", config)
    return config, matching, sources, preflight


async def execute(repo, root, config, matching, sources):
    if (root / "result.json").exists():
        write_report(root)
        return
    remote = ReplayRemote(repo, config, root, config["project_id"])
    first_seed = min(sources)
    first_update = matching[str(first_seed)]["selected"][ARMS[0]]
    checkpoint = read_json(
        root
        / f"seed-{first_seed}"
        / ARMS[0]
        / "stage_a"
        / f"u{first_update}"
        / "checkpoint.json"
    )
    # Bind the original provider renderer before pre-B sampling. No update occurs;
    # stage_b still performs its own fresh-optimizer restore before update 1.
    await remote.restore(
        checkpoint["state_path"],
        full_state=False,
        coordinate={
            "seed": first_seed,
            "replay_arm": ARMS[0],
            "stage": "stage_b_setup",
            "update": 0,
        },
    )
    for seed, source in sorted(sources.items()):
        for arm in ARMS:
            await stage_b(remote, source, arm, matching[str(seed)]["selected"][arm])
    write_json(
        root / "result.json",
        {
            "status": "COMPLETED",
            "matching": matching,
            "billing": remote.snapshot(),
            "continuation": config["continuation"],
        },
    )
    write_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--check", action="store_true", help="prepare locally; no remote calls"
    )
    parser.add_argument("--approval", default="")
    args = parser.parse_args()
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        parser.error("a simple, separate continuation run ID is required")
    if not args.check and args.approval != OWNER_DIRECTION:
        parser.error("explicit owner continuation direction is required")
    repo = args.repo.resolve()
    root = repo / "runs/replay-v1" / args.run_id
    config, matching, sources, preflight = prepare(
        repo,
        args.source_run.resolve(),
        args.config.resolve(),
        root,
    )
    print(f"Continuation ceiling: ${preflight['continuation_ceiling_usd']}", flush=True)
    print({seed: row["selected"] for seed, row in matching.items()}, flush=True)
    if args.check:
        print("Prepared locally; no service, new sampling or training.")
        return
    keep(root / "authorization.json", {"phrase": args.approval, "preflight": preflight})
    asyncio.run(execute(repo, root, config, matching, sources))


if __name__ == "__main__":
    main()
