"""Bind the completed local replay preparation; never creates a remote client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from duraseed.provenance import sha256_bytes
from duraseed.replay_inputs import block_inputs, load_config, read_json
from duraseed.replay_matching import ARMS, STAGE_A_GRID, STAGE_B_GRID
from duraseed.replay_remote import write_json


RUNS = {
    11: "runs/pilot0/pilot0-pair1-seed11-20260825T125100Z",
    29: "runs/pilot0/pilot0-pair2-seed29-20260830T204211Z",
}


def freeze(repo: Path, tokenizer: Path) -> tuple[Path, dict]:
    prepared = repo / "runs/replay-v1/preparation"

    def bound(path):
        return {
            "path": str(path.relative_to(repo)),
            "sha256": sha256_bytes(path.read_bytes()),
        }

    preflight = read_json(prepared / "preflight.json")
    blocks, project_ids = {}, set()
    for seed, run in RUNS.items():
        old = read_json(repo / run / "preflight.json")
        project_ids.add(old["lineage"]["project_id"])
        blocks[str(seed)] = {
            "source_run": run,
            "corpus": bound(prepared / f"block-{seed}/corpus.jsonl"),
            "audit": bound(prepared / f"block-{seed}/audit.json"),
            **{
                key: old["lineage"][key]
                for key in ("m0_state_path", "m0_sampler_path", "manifest_ids")
            },
        }
    if len(project_ids) != 1:
        raise ValueError("original source blocks do not share one provider project")
    tracked = subprocess.check_output(
        ["git", "ls-files", "src/duraseed"], cwd=repo, text=True
    ).splitlines()
    implementation = sorted(
        set(
            [path for path in tracked if path != "src/duraseed/cli.py"]
            + [
                str(path.relative_to(repo))
                for path in (repo / "src/duraseed").glob("replay*.py")
            ]
            + [
                "src/duraseed/runners/replay.py",
                "tools/freeze_replay.py",
                "tools/prepare_replay_budget.py",
            ]
        )
    )
    config = {
        "namespace": "replay-v1",
        "arms": list(ARMS),
        "seeds": [11, 29],
        "blocks": blocks,
        "project_id": project_ids.pop(),
        "protocol": bound(repo / "docs/replay-v1-protocol.md"),
        "preflight": bound(prepared / "preflight.json"),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "implementation_files": implementation,
        "stage_a_grid": list(STAGE_A_GRID),
        "stage_b_grid": list(STAGE_B_GRID),
        "stage_a_lr": 1e-4,
        "stage_b_lr": 3e-4,
        "batch_size": 32,
        "max_length": max(
            read_json(prepared / f"block-{seed}/audit.json")["max_length"]
            for seed in RUNS
        ),
        "tokenizer_path": str(tokenizer.resolve()),
        "evaluation": {"temperature": 1.0, "top_p": 0.95},
        "checkpoint_ttl_seconds": preflight["storage"]["ttl_seconds"],
        "storage_per_pair_usd": float(preflight["storage"]["cost_usd"])
        / preflight["storage"]["pairs"],
        "targeted_matching": {
            "11": "31/96",
            "29": "17/96",
            "tolerance": "3/100",
            "nominees_per_arm": 3,
        },
        "primary": "per-block R-P-minus-R-S targeted raw Pass@1 trapezoidal AUC0-20 /20",
        "key_f2": "per-block absolute raw MAPS trapezoidal AUC0-480 /480",
        "protected_reserve_usd": "1343.74",
        "owner_launch_direction": preflight.get("owner_launch_direction"),
        "launch_status": "OWNER_DIRECTED"
        if preflight.get("owner_launch_direction")
        else "NOT_AUTHORIZED",
    }
    destination = prepared / "config.json"
    if list((repo / "runs/replay-v1").glob("*/authorization.json")):
        raise ValueError("replay already authorized; do not replace its frozen inputs")
    write_json(destination, config)
    load_config(repo, destination)
    for seed in RUNS:
        block_inputs(repo, config, seed)
    return destination, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    destination, config = freeze(args.repo.resolve(), args.tokenizer)
    print(
        json.dumps(
            {
                "config": str(destination),
                "config_sha256": sha256_bytes(destination.read_bytes()),
                "protocol_sha256": config["protocol"]["sha256"],
                "status": config["launch_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
