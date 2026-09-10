#!/usr/bin/env python3
"""Offline readout for source-block-11 replay with acquisition-order seed 47."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from duraseed.replay_analysis import (
    BOOTSTRAP_REPLICATES,
    GRID,
    bootstrap_seed,
    half_life,
)
from duraseed.replay_inputs import read_json
from duraseed.replay_matching import ARMS
from duraseed.replay_remote import write_json
from duraseed.replay_results import BOOTSTRAP_NAMESPACE, _arm, _contrasts, markdown
from replay_half_life_uncertainty import crossing_batch, interval

SEED, SOURCE_BLOCK = 47, 11


def half_life_intervals(arms, role, replicates=BOOTSTRAP_REPLICATES):
    """Same first-crossing estimand and paired whole-item resampling as before."""
    rows = [arms[arm]["curves"][role] for arm in ARMS]
    if any(
        rows[0][k] != rows[1][k] for k in ("task_ids", "trials", "grid", "manifest_id")
    ):
        raise ValueError("Half-life contrast requires paired item trajectories")
    raw = np.array([np.asarray(r["successes"]) / r["trials"] for r in rows])
    seed = bootstrap_seed(
        f"replay-orderseed{SEED}-{role}",
        namespace="duraseed-publication-checks-20260909|",
    )
    rng, samples = np.random.default_rng(seed), np.empty((2, replicates))
    for start in range(0, replicates, 250):
        end = min(start + 250, replicates)
        indices = rng.integers(raw.shape[1], size=(end - start, raw.shape[1]))
        means = raw[:, indices, :].mean(axis=2)
        for i in range(2):
            samples[i, start:end] = crossing_batch(means[i])
    points = [half_life(r.mean(axis=0)) for r in raw]
    estimate = (
        points[1]["update"] - points[0]["update"]
        if all(p["status"] == "crossed" for p in points)
        else None
    )
    with np.errstate(invalid="ignore"):
        difference = samples[1] - samples[0]
    return {
        "paired_items": raw.shape[1],
        "replicates": replicates,
        "rng_seed": seed,
        "arms": {
            arm: {**interval(samples[i], points[i]["update"]), "crossing": points[i]}
            for i, arm in enumerate(ARMS)
        },
        "R-P_minus_R-S": interval(difference, estimate),
    }


def reduce(run):
    terminal, matching = (
        read_json(run / "result.json"),
        read_json(run / "matching.json"),
    )
    if (
        terminal.get("status") not in ("COMPLETED", "NO_MATCH")
        or terminal.get("matching") != matching
    ):
        raise ValueError("Replication is not terminal with durable matching")
    if set(matching) != {str(SEED)}:
        raise ValueError("Expected only acquisition-order seed 47")
    if (
        terminal.get("source_block") != SOURCE_BLOCK
        or terminal.get("acquisition_order_seed") != SEED
    ):
        raise ValueError("Replication source block or acquisition-order seed differs")
    states = list(run.rglob("remote-call-state.json"))
    if run / "remote/remote-call-state.json" not in states:
        raise ValueError("Missing durable root remote-call journal")
    if any(read_json(path).get("pending") is not None for path in states):
        raise ValueError("Pending remote call: do not analyze incomplete evidence")
    billing = read_json(run / "billing.json")
    if terminal.get("billing") != billing:
        raise ValueError("Terminal billing differs from durable run ledger")
    selection = matching[str(SEED)]
    block = {
        "status": terminal["status"],
        "matching": selection,
        "arms": {},
        "contrasts": {},
    }
    if selection["selected"] is None:
        if (
            selection["status"] != "NO_MATCH"
            or selection["stage_b_allowed"]
            or terminal["status"] != "NO_MATCH"
        ):
            raise ValueError("Inconsistent unavailable matching")
        for arm in ARMS:
            base = run / f"seed-{SEED}" / arm
            if (base / "pre-b").exists() or (base / "stage_b").exists():
                raise ValueError("Unmatched replication contains Stage-B evidence")
    else:
        if (
            selection["status"] != "MATCHED"
            or not selection["stage_b_allowed"]
            or terminal["status"] != "COMPLETED"
        ):
            raise ValueError("Inconsistent available matching")
        arms = {arm: _arm(run, SEED, arm, selection["selected"][arm]) for arm in ARMS}
        contrasts = _contrasts(SEED, arms)
        estimates = {name: row["estimate"] for name, row in contrasts.items()}
        block.update(
            arms=arms,
            contrasts=contrasts,
            gain_auc_decomposition={
                "absolute_auc_difference": estimates["stage-b/raw-absolute-AUC-0-40"],
                "baseline_difference": estimates["stage-b/raw-baseline"],
                "gain_auc_difference": estimates["stage-b/raw-gain-AUC-0-40"],
                "identity": "gain AUC difference = absolute AUC difference - baseline difference",
            },
            half_life_intervals={
                role: half_life_intervals(arms, role)
                for role in ("targeted", "sentinel")
            },
        )
    return {
        "namespace": "replay-order-replication",
        "run_id": run.name,
        "source_block": SOURCE_BLOCK,
        "acquisition_order_seed": SEED,
        "grid": list(GRID),
        "blocks": {str(SEED): block},
        "scope": "Additional acquisition-order realization on the same 579-example source-block-11 corpus; not a new source block. Original Pilot, replay, and dense-grid observations are unchanged. No pooled comparisons.",
        "primary": "Targeted raw-retention AUC over the registered sparse grid [0,1,2,5,10,20], trapezoidally integrated and divided by 20.",
        "bootstrap": {
            "namespace": BOOTSTRAP_NAMESPACE,
            "replicates": BOOTSTRAP_REPLICATES,
            "contrast_direction": "R-P-minus-R-S",
            "scope": "Paired whole-item trajectories; within-item draws retained together. Items are exchangeable independent clusters, conditional on selected checkpoints, training realization, families, observed draws and grid. No training-seed or checkpoint-selection uncertainty, family clustering, or multiplicity correction. Half-life intervals withheld if any replicate is undefined or censored; no replicates discarded.",
        },
        "raw_definition": "Equal-item observed successes/trials; unconditional exact-verifier success.",
        "posterior_definition": "Equal-item Jeffreys mean (successes+0.5)/(trials+1), separate from raw estimands.",
        "integrity": {"journals_without_pending": len(states)},
        "billing": billing,
    }


def portable_counts(arms):
    for arm, data in arms.items():
        for panel, curves in (
            ("a_monitor", data["curves"]),
            ("a_validation", data["validation"]),
        ):
            for role, curve in curves.items():
                actual_panel = "b_validation" if role == "stage-b" else panel
                for i, task in enumerate(curve["task_ids"]):
                    for j, step in enumerate(curve["grid"]):
                        yield {
                            "source_block": SOURCE_BLOCK,
                            "acquisition_order_seed": SEED,
                            "arm": arm,
                            "panel": actual_panel,
                            "panel_role": role,
                            "task_id": task,
                            "update": step,
                            "manifest_id": curve["manifest_id"],
                            "successes": curve["successes"][i][j],
                            "trials": curve["trials"][i][j],
                        }


def plot(arms, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for arm, color in zip(ARMS, ("#087F6D", "#7652A3"), strict=True):
        curves = arms[arm]["curves"]
        for role, style in (("targeted", "-"), ("sentinel", ":")):
            axes[0].plot(
                GRID[:6],
                np.array(curves[role]["raw"][:6]) * 100,
                style,
                marker="o",
                color=color,
                markersize=3,
                label=f"{arm} {role}",
            )
        for ax, metric in zip(axes[1:], ("raw", "own_baseline_gain"), strict=True):
            ax.plot(
                GRID,
                np.array(curves["stage-b"][metric]) * 100,
                "o-",
                color=color,
                markersize=3,
                label=arm,
            )
    for ax, title in zip(
        axes, ("Arithmetic retention", "New-task success", "New-task gain"), strict=True
    ):
        ax.set(title=title, xlabel="Stage-B update")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set(ylabel="Raw Pass@1 (%)", xlim=(0, 20), xticks=[0, 5, 10, 15, 20])
    axes[1].set_ylabel("Raw Pass@1 (%)")
    axes[2].set_ylabel("Gain from own baseline (points)")
    for suffix in ("svg", "png"):
        fig.savefig(output / f"trajectories.{suffix}", dpi=180)
    plt.close(fig)


def analyze(run, output):
    value = reduce(run)
    output.mkdir(parents=True, exist_ok=True)
    block = value["blocks"][str(SEED)]
    arms = block["arms"]
    for arm, data in arms.items():
        source = run / data["F3_profile"]
        destination = output / f"{arm}-profile.json"
        shutil.copyfile(source, destination)
        data["F3_profile"] = destination.name
    text = markdown(value).replace(
        "# Replay-v1 descriptive readout", "# Replay acquisition-order replication"
    )
    text = text.replace("Source block 47", "Source block 11 / order seed 47")
    for data in arms.values():
        name = data["F3_profile"]
        text = text.replace(f"](../{name})", f"]({name})")
    text += f"\n## Scope and definitions\n\n{value['scope']}\n\n{value['primary']}\n\n{value['bootstrap']['scope']}\n"
    if arms:
        plot(arms, output)
        text += "\n![Arithmetic retention and absolute/baseline-relative new-task learning](trajectories.svg)\n\n## Descriptive first-crossing half-lives\n\n| Role | Arm / contrast | Updates | Paired item 95% interval |\n|---|---|---:|---|\n"
        for role, uncertainty in block["half_life_intervals"].items():
            for name, row in (
                *uncertainty["arms"].items(),
                ("R-P minus R-S", uncertainty["R-P_minus_R-S"]),
            ):
                point = (
                    "unavailable"
                    if row["estimate"] is None
                    else f"{row['estimate']:.8g}"
                )
                ci = (
                    "unavailable"
                    if row["ci95"] is None
                    else f"[{row['ci95'][0]:.8g}, {row['ci95'][1]:.8g}]"
                )
                text += f"| {role} | {name} | {point} | {ci} |\n"
    (output / "readout.md").write_text(text)
    write_json(output / "readout.json", value)
    (output / "per-item-counts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in portable_counts(arms))
    )
    print(output / "readout.md")
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
