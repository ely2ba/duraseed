"""Declared health, token, and cost summaries for collected Pilot-0 evidence."""

from __future__ import annotations

from collections import defaultdict
from math import fsum, isfinite
import json
from pathlib import Path
from typing import Any

from duraseed.pilot0_budget import Pilot0Budget
from duraseed.provenance import sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runtime import TokenLedger


def _bg_metrics(root: Path) -> dict[int, list[dict[str, Any]]]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("seed-*/B-G/steps-*/metrics.jsonl")):
        try:
            seed = int(path.parents[2].name.removeprefix("seed-"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise RunnerGateError(
                "Pilot-0 B-G health evidence is unreadable"
            ) from error
        by_seed[seed].extend(
            row
            for row in rows
            if row.get("phase") == "stage_a" and row.get("method") == "B-G"
        )
    return by_seed


def reward_group_health(root: Path, seeds: tuple[int, ...]) -> tuple[dict, ...]:
    """Reduce every B-G step, retaining the frozen final-ten collapse diagnostic."""

    by_seed = _bg_metrics(root)
    summaries = []
    for seed in seeds:
        rows = sorted(
            by_seed.get(seed, ()), key=lambda row: row.get("training_step", -1)
        )
        if tuple(row.get("training_step") for row in rows) != tuple(range(1, 51)):
            raise RunnerGateError(
                "Pilot-0 B-G health report requires steps 1 through 50"
            )
        try:
            metrics = [row["metrics"] for row in rows]
            mixed = [float(row["mixed_group_rate"]) for row in metrics]
            zero = [float(row["all_zero_group_count"]) for row in metrics]
            one = [float(row["all_one_group_count"]) for row in metrics]
            mixed_count = [float(row["mixed_group_count"]) for row in metrics]
            surprise = [float(row["mean_sampled_token_surprisal"]) for row in metrics]
        except (KeyError, TypeError, ValueError) as error:
            raise RunnerGateError(
                "Pilot-0 B-G health metrics are incomplete"
            ) from error
        finite = all(
            isfinite(float(value))
            for row in metrics
            for value in row.values()
            if isinstance(value, (int, float))
        )
        groups = fsum(zero) + fsum(one) + fsum(mixed_count)
        if (
            not finite
            or groups <= 0
            or any(not 0 <= value <= 1 for value in mixed)
            or any(value < 0 for value in (*zero, *one, *mixed_count, *surprise))
        ):
            raise RunnerGateError("Pilot-0 B-G health metrics are incoherent")
        summaries.append(
            {
                "seed": seed,
                "update_count": 50,
                "mean_mixed_group_rate": fsum(mixed) / 50,
                "final_ten_mixed_group_rate": fsum(mixed[-10:]) / 10,
                "constant_reward_group_rate": (fsum(zero) + fsum(one)) / groups,
                "mean_sampled_token_surprisal": fsum(surprise) / 50,
                "loss_health_passed": finite,
            }
        )
    return tuple(summaries)


def usage_summary(ledger: TokenLedger, budget: Pilot0Budget) -> dict:
    return {
        "authorized_usd": ledger.authorized_usd,
        "pessimistic_preflight_upper_bound_usd": budget.upper_bound_usd,
        "committed_tokens": {
            "prefill": ledger.committed.prefill,
            "sample": ledger.committed.sample,
            "train": ledger.committed.train,
        },
        "observed_tokens": {
            "prefill": ledger.observed.prefill,
            "sample": ledger.observed.sample,
            "train": ledger.observed.train,
        },
        "committed_fixed_usd": ledger.committed_fixed_usd,
        "observed_fixed_usd": ledger.observed_fixed_usd,
        "committed_cost_usd": ledger.committed_cost_usd,
        "observed_cost_usd": ledger.observed_cost_usd,
        "remaining_authorized_usd": ledger.authorized_usd - ledger.committed_cost_usd,
        "rerun_policy": budget.rerun_policy,
    }


def evidence_index(root: Path) -> dict:
    names = {
        "generations.jsonl",
        "metrics.jsonl",
        "remote-call-state.json",
        "remote-calls.jsonl",
        "rewards.jsonl",
        "segment.json",
    }
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.name in names or path.name == "result.json")
        and path.parent != root
    )
    return {
        "schema_version": "duraseed-pilot0-evidence-index-v1",
        "file_count": len(paths),
        "files": {
            path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
            for path in paths
        },
    }


__all__ = ["evidence_index", "reward_group_health", "usage_summary"]
