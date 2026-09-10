#!/usr/bin/env python3
"""Post-hoc paired item-trajectory uncertainty; no sampling or paper edits."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from duraseed.replay_analysis import GRID, bootstrap_seed, half_life
from duraseed.replay_remote import write_json


def crossing_batch(curves, grid=GRID):
    """Vectorized version of the existing first-downward-crossing definition."""
    values = np.asarray(curves, dtype=float)
    threshold = values[:, 0] / 2
    out = np.full(len(values), np.inf)
    out[values[:, 0] <= 0] = np.nan
    for i in range(1, len(grid)):
        take = (
            np.isposinf(out)
            & (values[:, i] <= threshold)
            & (threshold < values[:, i - 1])
        )
        fraction = (values[take, i - 1] - threshold[take]) / (
            values[take, i - 1] - values[take, i]
        )
        out[take] = grid[i - 1] + fraction * (grid[i] - grid[i - 1])
    return out


def interval(samples, estimate):
    finite = np.isfinite(samples)
    return {
        "estimate": estimate,
        "ci95": np.quantile(samples, [0.025, 0.975], method="linear").tolist()
        if finite.all()
        else None,
        "finite_replicates": int(finite.sum()),
        "undefined_replicates": int(np.isnan(samples).sum()),
        "censored_replicates": int(np.isinf(samples).sum()),
        "nonfinite_handling": "No replicates discarded; interval withheld if any replicate is undefined or censored.",
    }


def paired_half_lives(raw, names, label, replicates=50_000):
    raw = np.asarray(raw, dtype=float)
    assert raw.shape[0] == 2 and raw.shape[2] == len(GRID)
    seed = bootstrap_seed(label, namespace="duraseed-publication-checks-20260909|")
    rng = np.random.default_rng(seed)
    samples = np.empty((2, replicates))
    for start in range(0, replicates, 250):
        stop = min(replicates, start + 250)
        indices = rng.integers(raw.shape[1], size=(stop - start, raw.shape[1]))
        curves = raw[:, indices, :].mean(axis=2)
        for arm in range(2):
            samples[arm, start:stop] = crossing_batch(curves[arm])
    points = [half_life(values.mean(axis=0)) for values in raw]
    with np.errstate(invalid="ignore"):
        difference = samples[1] - samples[0]
    return {
        "label": label,
        "paired_items": raw.shape[1],
        "replicates": replicates,
        "rng_seed": seed,
        "arms": {
            name: {**interval(samples[i], points[i]["update"]), "crossing": points[i]}
            for i, name in enumerate(names)
        },
        "difference": {
            "direction": f"{names[1]} minus {names[0]}",
            **interval(difference, points[1]["update"] - points[0]["update"]),
        },
        "source_mean_trajectories": {
            name: raw[i].mean(axis=0).tolist() for i, name in enumerate(names)
        },
    }


def compute(repo, output):
    replay = json.loads(
        (repo / "artifacts/replay-v1/followup/readout.json").read_text()
    )
    pilot = [
        json.loads(line)
        for line in (repo / "artifacts/replay-v1/pilot-audit/per-item-counts.jsonl")
        .read_text()
        .splitlines()
    ]
    results = []
    for role in ("targeted", "sentinel"):
        arms = replay["blocks"]["11"]["arms"]
        rows = [arms[name]["curves"][role] for name in ("R-S", "R-P")]
        assert rows[0]["task_ids"] == rows[1]["task_ids"]
        raw = np.array([np.array(row["successes"]) / row["trials"] for row in rows])
        results.append(paired_half_lives(raw, ("R-S", "R-P"), f"replay-seed11-{role}"))
        for pair in (1, 2):
            counts = [
                row
                for row in pilot
                if row["pair"] == pair
                and row["panel"] == "a-monitor"
                and row["panel_role"] == role
            ]
            ids = sorted({row["task_id"] for row in counts})
            by_key = {
                (row["method"], row["task_id"], row["update"]): row for row in counts
            }
            assert len(by_key) == 2 * len(ids) * len(GRID)
            assert set(Counter(row["trials"] for row in counts)) == {4}
            raw = np.array(
                [
                    [
                        [by_key[name, task, step]["successes"] / 4 for step in GRID]
                        for task in ids
                    ]
                    for name in ("B-S", "B-G")
                ]
            )
            results.append(
                paired_half_lives(raw, ("B-S", "B-G"), f"pilot-pair{pair}-{role}")
            )
    value = {
        "date": "2026-09-09",
        "scientific_role": "post-hoc descriptive uncertainty; no gate or selection use",
        "grid": list(GRID),
        "definition": "first downward crossing of half of each resampled arm's own update-0 raw Pass@1, linearly interpolated",
        "bootstrap": "50,000 NumPy PCG64 percentile replicates; identical sampled item indices for both arms and all checkpoints; each item's four realized draws kept together",
        "scope": "Treats evaluation items as independent exchangeable clusters. Conditional on selected checkpoints, training seeds, families, realized draws, and grid interpolation. Does not measure training-seed or selection uncertainty, or uncertainty from unobserved times between checkpoints. No family-cluster adjustment or multiplicity correction.",
        "results": results,
    }
    write_json(output / "half-life-uncertainty.json", value)
    for row in results:
        print(row["label"], row["difference"], flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compute(args.repo.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
