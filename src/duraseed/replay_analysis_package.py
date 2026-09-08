"""Static scientific figures and descriptive notebook for the replay audit."""

from __future__ import annotations

import json

from duraseed.replay_analysis_report import markdown


def figures(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"B-S": "#225AC7", "B-G": "#C92937"}
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    ):
        learning, ax = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        retention, ra = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
        failures, fa = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for p, pair in enumerate(data["pairs"]):
            grid = pair["grid"]
            for method in ("B-S", "B-G"):
                maps = pair["curves"]["maps"][method]["raw"]
                target = pair["curves"]["targeted"][method]["raw"]
                for col, ys in enumerate((maps, [q - maps[0] for q in maps])):
                    ax[p, col].plot(
                        grid, ys, "o-", color=colors[method], label=method, markersize=3
                    )
                    ax[p, col].set(
                        title=f"Pair {pair['pair']} · "
                        + ("absolute MAPS" if col == 0 else "MAPS minus own baseline"),
                        xlabel="Stage-B update",
                        ylabel="Raw Pass@1",
                    )
                for col, role in enumerate(("targeted", "sentinel")):
                    ra[p, col].plot(
                        grid[:6],
                        pair["curves"][role][method]["raw"][:6],
                        "o-",
                        color=colors[method],
                        label=method,
                        markersize=3,
                    )
                    ra[p, col].set(
                        title=f"Pair {pair['pair']} · {role}",
                        xlabel="Stage-B update",
                        ylabel="Unconditional raw Pass@1",
                    )
                ra[p, 2].plot(
                    maps, target, "o-", color=colors[method], label=method, markersize=3
                )
                for i, step in enumerate(grid):
                    ra[p, 2].annotate(
                        str(step),
                        (maps[i], target[i]),
                        xytext=(3, 5 if method == "B-S" else -9),
                        textcoords="offset points",
                        fontsize=6,
                        color=colors[method],
                    )
                    if i:
                        ra[p, 2].annotate(
                            "",
                            xy=(maps[i], target[i]),
                            xytext=(maps[i - 1], target[i - 1]),
                            arrowprops={
                                "arrowstyle": "->",
                                "color": colors[method],
                                "lw": 0.8,
                            },
                        )
                ra[p, 2].set(
                    title=f"Pair {pair['pair']} · chronological path",
                    xlabel="MAPS raw Pass@1",
                    ylabel="Targeted TCES raw Pass@1",
                )
            for axis in (*ax[p], *ra[p]):
                axis.legend(frameon=False, fontsize=8)
            for col, role in enumerate(("targeted", "sentinel", "stage-b")):
                rows = [r for r in pair["failures"] if r["role"] == role]
                rows.sort(key=lambda r: (r["update"], r["method"]))
                bottom = [0.0] * len(rows)
                for group, color in (
                    ("correct", "#365E8A"),
                    ("executable_wrong_target", "#A3ABB5"),
                    ("other_well_formed_task_failure", "#6B7684"),
                    ("output_contract_or_parse", "#D8DEE6"),
                ):
                    heights = [
                        r["disjoint_groups"].get(group, 0) / r["completions"]
                        for r in rows
                    ]
                    fa[p, col].bar(
                        range(len(rows)),
                        heights,
                        bottom=bottom,
                        color=color,
                        label=group,
                        width=0.85,
                    )
                    bottom = [a + b for a, b in zip(bottom, heights)]
                fa[p, col].set_xticks(
                    range(len(rows)),
                    [f"{r['update']}\n{r['method']}" for r in rows],
                    fontsize=6,
                )
                fa[p, col].set(
                    title=f"Pair {pair['pair']} · {role}",
                    ylabel="Fraction of all completions",
                    ylim=(0, 1),
                )
        handles, labels = fa[0, 0].get_legend_handles_labels()
        failures.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        for name, fig in (
            ("learning", learning),
            ("retention", retention),
            ("failures", failures),
        ):
            fig.savefig(output / f"{name}.svg")
            fig.savefig(output / f"{name}.png", dpi=160)
            plt.close(fig)


def write_package(data, counts, output, *, repo):
    private = repo / "runs/replay-v1" / output.name
    private.mkdir(parents=True, exist_ok=True)
    sources = {str(p["pair"]): p.pop("sources") for p in data["pairs"]}
    with (private / "source-index.json").open("x") as stream:
        json.dump(sources, stream, indent=2)
    output.mkdir(parents=True, exist_ok=False)
    document = markdown(data)
    (output / "summary.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    (output / "README.md").write_text(document)
    (output / "figure-data.json").write_text(
        json.dumps(
            {
                str(p["pair"]): {
                    "grid": p["grid"],
                    "curves": p["curves"],
                    "failures": p["failures"],
                }
                for p in data["pairs"]
            },
            indent=2,
        )
        + "\n"
    )
    with (output / "per-item-counts.jsonl").open("x") as stream:
        for row in counts:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    figures(data, output)
    code = "import json\nfrom pathlib import Path\np = Path('summary.json')\nif not p.exists(): p = Path('artifacts/replay-v1/pilot-audit/summary.json')\ndata = json.loads(p.read_text())\nfor pair in data['pairs']:\n    print('Pair', pair['pair'], 'gain = absolute − baseline:', pair['decomposition'])\n"
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Pilot-0 existing-data audit\n",
                    "Run the cell locally to inspect the saved analysis; no remote calls. Full computed report follows.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": code.splitlines(True),
                "outputs": [],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": document.splitlines(True),
            },
        ],
    }
    (output / "pilot-audit.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
