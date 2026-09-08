"""Render frozen REPLAY estimands from portable counts; no sampling or fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from duraseed.replay_analysis import auc
from duraseed.replay_analysis_report import fmt, table

COLORS = {"R-S": "#C72940", "R-P": "#245DE6"}
ARMS = ("R-S", "R-P")
GROUPS = (
    "correct",
    "executable_wrong_target",
    "other_well_formed_task_failure",
    "output_contract_or_parse",
)


def save(fig, output, name):
    svg = output / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def style(ax, title, xlabel="Stage-B update", ylabel="Raw Pass@1 (%)"):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def learning(block, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for col, stop in enumerate((40, 480)):
        for row, field in enumerate(("raw", "own_baseline_gain")):
            ax = axes[row, col]
            for arm in ARMS:
                c = block["arms"][arm]["curves"]["stage-b"]
                n = c["grid"].index(stop) + 1
                ax.plot(
                    c["grid"][:n],
                    np.array(c[field][:n]) * 100,
                    "o-",
                    color=COLORS[arm],
                    markersize=3,
                    label=arm,
                )
            name = (
                "Absolute new-task performance"
                if row == 0
                else "Gain from own baseline"
            )
            style(
                ax,
                f"{name} · 0–{stop}",
                ylabel="Raw Pass@1 (%)" if row == 0 else "Raw gain (percentage points)",
            )
            ax.set_xlim(0, stop)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Same later training; different starting points and learning paths",
        x=0.02,
        ha="left",
    )
    save(fig, output, "learning")


def retention(block, output):
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), layout="constrained")
    for col, role in enumerate(("targeted", "sentinel")):
        for row, field in enumerate(("raw", "relative_retention")):
            for arm in ARMS:
                c = block["arms"][arm]["curves"][role]
                y = np.array([np.nan if v is None else v for v in c[field][:6]])
                axes[row, col].plot(
                    c["grid"][:6],
                    100 * y,
                    "o-",
                    color=COLORS[arm],
                    markersize=4,
                    label=arm,
                )
            style(
                axes[row, col],
                f"{role.title()} · {'raw success' if row == 0 else 'relative to own start'}",
                ylabel="Raw Pass@1 (%)" if row == 0 else "Share of own baseline (%)",
            )
            axes[row, col].set_xlim(0, 20)
    axes[0, 0].legend(frameon=False)
    for col, arm in enumerate(ARMS):
        ax = axes[2, col]
        c = block["arms"][arm]["curves"]
        x, y = np.array(c["stage-b"]["raw"]) * 100, np.array(c["targeted"]["raw"]) * 100
        ax.plot(x, y, "o-", color=COLORS[arm], markersize=4, linewidth=1.4)
        for i, update in enumerate(c["targeted"]["grid"]):
            offset = (3, 7 if i % 2 == 0 else -13)
            ax.annotate(
                str(update),
                (x[i], y[i]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color=COLORS[arm],
            )
            if i:
                ax.annotate(
                    "",
                    xy=(x[i], y[i]),
                    xytext=(x[i - 1], y[i - 1]),
                    arrowprops={"arrowstyle": "->", "color": COLORS[arm], "lw": 0.8},
                )
        style(
            ax,
            f"{arm} · chronological path",
            "New-task raw Pass@1 (%)",
            "Targeted raw Pass@1 (%)",
        )
        ax.set_xlim(-2, 50)
        ax.set_ylim(-4, 42)
    fig.suptitle("Retention is not a monotone decay curve", x=0.02, ha="left")
    save(fig, output, "retention")


def failures(block, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    colors = ("#245DE6", "#A4B2C7", "#CFD5DF", "#C72940")
    for col, arm in enumerate(ARMS):
        for row, (panel, role) in enumerate(
            (("a_monitor", "targeted"), ("b_validation", "stage-b"))
        ):
            points = block["arms"][arm]["failures"][panel]
            bottom = np.zeros(len(points))
            for group, color in zip(GROUPS, colors, strict=True):
                y = (
                    np.array(
                        [
                            p[role]["disjoint_groups"].get(group, 0)
                            / p[role]["completions"]
                            for p in points
                        ]
                    )
                    * 100
                )
                axes[row, col].bar(
                    range(len(points)),
                    y,
                    bottom=bottom,
                    color=color,
                    width=0.78,
                    label=group.replace("_", " "),
                )
                bottom += y
            axes[row, col].set_xticks(range(len(points)), [p["update"] for p in points])
            axes[row, col].set_ylim(0, 100)
            task = "targeted arithmetic" if row == 0 else "new task"
            style(
                axes[row, col], f"{arm} · {task}", ylabel="Share of all completions (%)"
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9
    )
    fig.suptitle("Verifier outcomes · mutually exclusive categories", x=0.02, ha="left")
    save(fig, output, "failures")


def context(data, pilot, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    metrics = (
        ("targeted", "absolute-AUC-0-20", "Targeted retention AUC · 0–20"),
        ("maps", "absolute-AUC-0-480", "Absolute new-task AUC · 0–480"),
    )
    labels = [
        "Pilot 0 · pair 1\nB-G − B-S",
        "Pilot 0 · pair 2\nB-G − B-S",
        "Trace replay · block 11\nR-P − R-S",
    ]
    for ax, (role, metric, title) in zip(axes, metrics, strict=True):
        rows = [
            next(x for x in p["contrasts"] if f"-{role}-{metric}-" in x["contrast"])
            for p in pilot["pairs"]
        ]
        key = (
            "targeted/raw-absolute-AUC-0-20"
            if role == "targeted"
            else "stage-b/raw-absolute-AUC-0-480"
        )
        rows.append(data["blocks"]["11"]["contrasts"][key])
        for i, row in enumerate(rows):
            mid = row["estimate"] * 100
            lo, hi = np.array(row["ci95"]) * 100
            ax.errorbar(
                mid,
                i,
                xerr=[[mid - lo], [hi - mid]],
                fmt="o",
                capsize=4,
                color="#245DE6" if i == 2 else "#697586",
            )
        ax.axvline(0, color="#697586", linewidth=0.8)
        ax.set_yticks(range(3), labels)
        ax.invert_yaxis()
        style(ax, title, "Difference (percentage points)", "")
    fig.suptitle(
        "Pilot procedures are contextual references, not extra replay controls",
        x=0.02,
        ha="left",
        fontsize=12,
    )
    save(fig, output, "context")


def supplement(block, output):
    lines = [
        "# Supplementary frozen summaries",
        "",
        "All rates are raw. R-P minus R-S throughout. No fitted decay, smoothing, new windows or resampling settings.",
        "",
        "## Retention and starting-score sensitivity",
        "",
    ]
    rows = []
    for arm in ARMS:
        for role in ("targeted", "sentinel"):
            c = block["arms"][arm]["curves"][role]
            raw_auc = float(auc(c["raw"][:6], c["grid"][:6]))
            rows.append(
                [
                    arm,
                    role,
                    fmt(c["raw"][0]),
                    fmt(raw_auc),
                    fmt(raw_auc / c["raw"][0]) if c["raw"][0] else "undefined",
                    fmt(c["half_life"]["update"]),
                    str(c["half_life"].get("bracket", c["half_life"]["status"])),
                ]
            )
    lines += [
        table(
            "Arm|Role|Baseline|Raw AUC 0–20|AUC / own baseline|First half-life|Bracket/status".split(
                "|"
            ),
            rows,
        ),
        "",
        "Normalization is a descriptive starting-score sensitivity; it does not create equal initial functions or eliminate selection conditioning.",
        "",
        "## First attainment at fixed absolute new-task thresholds",
        "",
    ]
    rows = []
    for threshold in ("0.06", "0.07", "0.1", "0.2"):
        for role in ("targeted", "sentinel"):
            records = {
                a: block["arms"][a]["first_attainment"][threshold][role] for a in ARMS
            }
            eligible = all(r["status"] == "crossed" for r in records.values())
            difference = (
                records["R-P"]["retention"] - records["R-S"]["retention"]
                if eligible
                else None
            )
            for arm, record in records.items():
                rows.append(
                    [
                        threshold,
                        role,
                        arm,
                        record["status"],
                        fmt(record["update"]),
                        str(record["bracket"]),
                        fmt(record["retention"]),
                        fmt(difference) if arm == "R-P" else "—",
                    ]
                )
    lines += [
        table(
            "Threshold|Role|Arm|Status|Interpolated update|Observed bracket|TCES success|Eligible R-P − R-S".split(
                "|"
            ),
            rows,
        ),
        "",
        "Crossings follow training order, including reversals; both coordinates interpolate along the same observed interval. Baseline-exceeded and not-reached rows are not treated as matched post-training progress.",
        "",
        "## Authoritative failure and length decomposition",
        "",
        "Codes are mutually exclusive; cap/tag/syntax flags overlap and must not be summed. Conditional accuracy is not a repaired or counterfactual score.",
        "",
    ]
    rows, codes = [], []
    for arm in ARMS:
        for panel, points in block["arms"][arm]["failures"].items():
            for point in points:
                for role, value in point.items():
                    if role == "update":
                        continue
                    flags, valid = (
                        value["overlapping_indicators"],
                        value["conditional_output_valid"],
                    )
                    rows.append(
                        [
                            arm,
                            role,
                            point["update"],
                            value["completions"],
                            fmt(value["unconditional_success"]),
                            f"{valid['successes']}/{valid['denominator']}",
                            fmt(valid["accuracy"]),
                            flags["cap_reached"],
                            flags["invalid_tag"],
                            flags["invalid_syntax"],
                            fmt(value["tokens_mean"]),
                            fmt(value["tokens_median"]),
                        ]
                    )
                    codes.append(
                        [
                            arm,
                            role,
                            point["update"],
                            json.dumps(
                                value["disjoint_authoritative_codes"], sort_keys=True
                            ),
                        ]
                    )
    lines += [
        table(
            "Arm|Role|Update|N|Raw success|Valid successes/N|Conditional|Cap|Bad tag|Bad syntax|Mean tokens|Median tokens".split(
                "|"
            ),
            rows,
        ),
        "",
        table(["Arm", "Role", "Update", "Disjoint authoritative codes"], codes),
        "",
    ]
    (output / "supplement.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("artifacts/replay-v1/pilot-audit/summary.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/replay-v1/followup")
    )
    args = parser.parse_args()
    data, pilot = (
        json.loads(args.readout.read_text()),
        json.loads(args.pilot.read_text()),
    )
    block = data["blocks"]["11"]
    if block["status"] != "COMPLETED" or data["blocks"]["29"]["status"] != "NO_MATCH":
        raise ValueError(
            "expected the completed block-11 / unavailable block-29 package"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "svg.hashsalt": "duraseed-replay-v1-figures",
        }
    )
    learning(block, args.output)
    retention(block, args.output)
    failures(block, args.output)
    context(data, pilot, args.output)
    supplement(block, args.output)
    print("Rendered four figure pairs and supplement.md from existing counts only.")


if __name__ == "__main__":
    main()
