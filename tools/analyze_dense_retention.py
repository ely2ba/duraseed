#!/usr/bin/env python3
"""Offline analysis of the separately authorized 0–20 dense replay replication."""

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np

from duraseed.pilot0_evidence import read_evaluation
from duraseed.replay_analysis import auc, bootstrap_seed, failure_summary, half_life
from duraseed.replay_analysis import paired_interval
from duraseed.replay_remote import write_json
from replay_half_life_uncertainty import crossing_batch, interval

ARMS, ROLES = ("R-S", "R-P"), ("targeted", "sentinel")
GRID, ANCHORS = list(range(21)), [0, 1, 2, 5, 10, 20]
NAMESPACE = "dense-retention-20260909|"


@cache
def evaluation(directory):
    result = read_evaluation(directory)
    if result is None or result["row_count"] != 1536:
        raise ValueError(
            f"Complete 384-item × four-draw evaluation required: {directory}"
        )
    rewards = {
        r["sample_id"]: r["exact_verification"]
        for r in map(json.loads, (directory / "rewards.jsonl").read_text().splitlines())
    }
    rows = list(
        map(json.loads, (directory / "generations.jsonl").read_text().splitlines())
    )
    stats = {}
    for role in ROLES:
        chosen = [r for r in rows if r["panel_role"] == role]
        stats[role] = failure_summary(
            [
                {
                    "verification": rewards[r["sample_id"]],
                    "sampled_tokens": r["sampled_tokens"],
                    "sampling_max_tokens": r["sampling_max_tokens"],
                }
                for r in chosen
            ]
        )
        stats[role]["successes"] = sum(int(r["reward"]) for r in chosen)
        if len(chosen) != 768:
            raise ValueError("Each monitor role requires 192 items × four draws")
    return result["item_counts"], stats


def uncertainty(raw, role, replicates=50_000):
    """Resample whole item trajectories identically across arms and checkpoints."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 3 or raw.shape[0] != 2 or raw.shape[2] != len(GRID):
        raise ValueError("Expected two aligned item-by-21-update matrices")
    seed = bootstrap_seed(f"{role}-half-life", namespace=NAMESPACE)
    rng = np.random.default_rng(seed)
    boots = np.empty((2, replicates))
    for start in range(0, replicates, 250):
        end = min(replicates, start + 250)
        indices = rng.integers(raw.shape[1], size=(end - start, raw.shape[1]))
        means = raw[:, indices, :].mean(axis=2)
        for arm in range(2):
            boots[arm, start:end] = crossing_batch(means[arm], GRID)
    points = [half_life(x.mean(axis=0), GRID) for x in raw]
    point_difference = (
        points[1]["update"] - points[0]["update"]
        if all(p["status"] == "crossed" for p in points)
        else None
    )
    with np.errstate(invalid="ignore"):
        difference = boots[1] - boots[0]
    item_auc = auc(raw, GRID)
    result = {
        "role": role,
        "paired_items": raw.shape[1],
        "replicates": replicates,
        "half_life_seed": seed,
        "arms": {},
    }
    for i, name in enumerate(ARMS):
        result["arms"][name] = {
            "half_life": {
                **interval(boots[i], points[i]["update"]),
                "crossing": points[i],
            },
            "auc_0_20": paired_interval(
                item_auc[i],
                f"{role}-{name}-auc",
                namespace=NAMESPACE,
                replicates=replicates,
            ),
        }
    result["R-P_minus_R-S"] = {
        "half_life": interval(difference, point_difference),
        "auc_0_20": paired_interval(
            item_auc[1] - item_auc[0],
            f"{role}-auc-difference",
            namespace=NAMESPACE,
            replicates=replicates,
        ),
    }
    return result


def collect(run, source):
    curves, portable, original = [], [], []
    for arm in ARMS:
        for step in GRID:
            directory = (
                source / f"seed-11/{arm}/pre-b/a_monitor"
                if step == 0
                else run / f"seed-11/{arm}/stage_b/u{step}/a_monitor"
            )
            items, stats = evaluation(directory)
            base = {"arm": arm, "update": step, "baseline_reused": step == 0}
            for role in ROLES:
                curves.append({**base, "role": role, **stats[role]})
            portable += [{"arm": arm, "update": step, **r} for r in items]
        for step in ANCHORS:
            directory = (
                source
                / f"seed-11/{arm}"
                / ("pre-b/a_monitor" if step == 0 else f"stage_b/u{step}/a_monitor")
            )
            _, stats = evaluation(directory)
            original += [
                {"arm": arm, "update": step, "role": role, **stats[role]}
                for role in ROLES
            ]
    return curves, portable, original


def matrices(portable, role):
    rows = [r for r in portable if r["panel_role"] == role]
    ids = sorted({r["task_id"] for r in rows})
    by_key = {(r["arm"], r["task_id"], r["update"]): r for r in rows}
    if len(ids) != 192 or len(by_key) != 2 * 192 * 21 or len(rows) != len(by_key):
        raise ValueError(
            "Items must be shared exactly across both arms and the full grid"
        )
    if any(r["trials"] != 4 for r in rows):
        raise ValueError("Expected four stored draws per item")
    return np.array(
        [
            [
                [by_key[arm, task, step]["successes"] / 4 for step in GRID]
                for task in ids
            ]
            for arm in ARMS
        ]
    )


def plot(curves, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained", sharey=True)
    for ax, role in zip(axes, ROLES, strict=True):
        for arm, color in zip(ARMS, ("#087F6D", "#7652A3"), strict=True):
            y = [
                r["unconditional_success"] * 100
                for r in curves
                if r["arm"] == arm and r["role"] == role
            ]
            ax.plot(GRID, y, "o-", color=color, markersize=3, label=arm)
        ax.set(title=role.title(), xlabel="Stage-B update", xlim=(0, 20))
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Arithmetic raw Pass@1 (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output / "dense-retention.png", dpi=180)
    fig.savefig(output / "dense-retention.svg", metadata={"Date": None})
    plt.close(fig)


def report(value, output):
    def estimate(r):
        point = "unavailable" if r["estimate"] is None else f"{r['estimate']:.6f}"
        return point + (
            f" [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}]"
            if r["ci95"] is not None
            else " [interval unavailable]"
        )

    lines = [
        f"""# Dense arithmetic-retention replication

{value["scope"]}

{value["baseline"]}

{value["definitions"]}

{value["bootstrap"]}

![Dense replication retention curves](dense-retention.png)

## Dense-grid summaries

All intervals are paired item-bootstrap 95% intervals, conditional on this training realization.

| Role | Arm / contrast | 0–20 raw retention AUC | First-crossing half-life (updates) |
|---|---|---:|---:|"""
    ]
    for summary in value["summaries"]:
        for arm in (*ARMS, "R-P_minus_R-S"):
            r = summary["arms"][arm] if arm in ARMS else summary[arm]
            lines.append(
                f"| {summary['role']} | {arm} | {estimate(r['auc_0_20'])} | {estimate(r['half_life'])} |"
            )
    lines += [
        """
## Full trajectory

Valid output requires valid tag, syntax, and lexing together. Cap reached means recorded tokens ≥ recorded cap; failure indicators can overlap.

| Role | Arm | Update | Correct / 768 | Raw Pass@1 | Mean tokens | Median tokens | Valid / 768 | Cap reached / 768 |
|---|---|---:|---:|---:|---:|---:|---:|---:|"""
    ]
    for r in value["trajectories"]:
        count = r["overlapping_indicators"]
        lines.append(
            f"| {r['role']} | {r['arm']} | {r['update']} | {r['successes']} | {r['unconditional_success']:.6f} | {r['tokens_mean']:.2f} | {r['tokens_median']} | {count['output_valid']} | {count['cap_reached']} |"
        )
    lines += [
        """
## Original coarse anchors alongside the new replication

Update 0 is the same reused observation, not a second baseline measurement.

| Role | Arm | Update | Original correct / 768 | Dense correct / 768 |
|---|---|---:|---:|---:|"""
    ]
    lookup = {(r["arm"], r["role"], r["update"]): r for r in value["trajectories"]}
    for old in value["original_coarse_anchors"]:
        new = lookup[old["arm"], old["role"], old["update"]]
        lines.append(
            f"| {old['role']} | {old['arm']} | {old['update']} | {old['successes']} | {new['successes']} |"
        )
    lines += [
        "",
        "Full failure-code counts and bootstrap details: `dense-retention.json`. Portable item success counts: `per-item-counts.jsonl`. Original frozen results are not replaced.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def analyze(run, output):
    design = json.loads((run / "design.json").read_text())
    source = Path(design["source_run"])
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[1] / source
    if design["baseline_reuse"] is not True:
        raise ValueError(
            "This analysis is fixed to reuse of the original pre-B monitor"
        )
    curves, portable, original = collect(run, source)
    summaries = [uncertainty(matrices(portable, role), role) for role in ROLES]
    value = {
        "run_id": run.name,
        "source_run_id": source.name,
        "grid": GRID,
        "scope": "Separate dense-grid replication of the two replay arms from seed 11. No original frozen observation, primary result, or uncertainty estimate is replaced. Arithmetic monitoring only; no new MAPS outcome is computed.",
        "baseline": "Update 0 reuses each arm's original pre-B a_monitor observation (192 targeted and 192 sentinel items, four draws each). Updates 1–20 are new training/evaluation observations from the saved selected origins.",
        "definitions": "Raw Pass@1 includes every sampled completion. Retention AUC is trapezoidally integrated over updates 0–20 and divided by 20, without baseline subtraction. Half-life is the first downward crossing of half of each arm's own update-0 raw Pass@1, linearly interpolated between adjacent updates.",
        "bootstrap": "50,000 NumPy PCG64 paired whole-item trajectory percentile replicates per contrast, namespace dense-retention-20260909|. Both arms and all checkpoints use identical resampled item indices; each item's four realized draws stay together. Items are treated as independent exchangeable clusters, without family clustering. Intervals do not quantify training-seed variation, selection uncertainty, or unsampled within-update behavior. No half-life replicate is discarded; intervals are withheld if any are undefined or censored.",
        "summaries": summaries,
        "trajectories": curves,
        "original_coarse_anchors": original,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "dense-retention.json", value)
    (output / "per-item-counts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in portable)
    )
    plot(curves, output)
    report(value, output)
    print(output / "README.md")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
