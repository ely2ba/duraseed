"""Endpoint-clone profile tables and figures, separated from numeric reduction."""

import numpy as np

from duraseed.endpoint_continuation import DENSE_GRID


def profile_markdown(value):
    rows = [
        "## Clone selection and confirmation",
        "",
        f"Status: {value['matching']['status']}",
        "",
        "| Assessment | Role | Metric | Student | Teacher | Passed |",
        "|---|---|---|---:|---:|---|",
    ]
    assessments = [
        (f"Selection u{row['update']}", row)
        for row in value["matching"].get("assessments", [])
    ]
    if "confirmation" in value["matching"]:
        assessments.append(("Confirmation", value["matching"]["confirmation"]))
    for name, assessment in assessments:
        rows += [
            f"| {name} | {g['role']} | {g['metric']} | {g['student']:.6f} | {g['teacher']:.6f} | {g['passed']} |"
            for g in assessment["gates"]
        ]
    rows += [
        "",
        "| Confirmation role | Student/teacher item distance | Teacher repeat distance | Excess | Passed |",
        "|---|---:|---:|---:|---|",
    ]
    rows += [
        f"| {r['role']} | {r['student_teacher_distance']:.6f} | {r['teacher_repeat_distance']:.6f} | {r['excess_item_distance']:.6f} | {r['passed']} |"
        for r in value["matching"].get("confirmation", {}).get("item_agreement", [])
    ]
    rows += [
        "",
        "Complete selection and confirmation profiles, including item-level counts, are in `readout.json`.",
        "",
    ]
    if "adapter_geometry" in value:
        rows += [
            "## Archived adapter measurements",
            "",
            "BA norms, largest singular values, and stable ranks use the block-diagonal union of module spectra.",
            "",
            "| Checkpoint | Selected | ‖A‖F | ‖B‖F | ‖BA‖F | σ₁ | Stable rank |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for record in value["adapter_geometry"]:
            a = record["aggregate"]
            rows += [
                f"| {record['arm']}@{record['update']} | {record['selected']} | {a['factor_a_frobenius_norm']:.6f} | {a['factor_b_frobenius_norm']:.6f} | {a['ba_frobenius_norm']:.6f} | {a['sigma1']:.6f} | {a['stable_rank']:.6f} |"
            ]
        rows += [""]
    return rows


def plot(value, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for label, run in value["runs"].items():
        color = "#087F6D" if label.startswith("T") else "#7652A3"
        style = "-" if label.endswith("1") else "--"
        for ax, role in zip(axes[:2], ("targeted", "sentinel"), strict=True):
            ax.plot(
                DENSE_GRID,
                np.asarray(run["curves"][role]["raw"][:21]) * 100,
                style,
                color=color,
                label=label,
                linewidth=1.7,
            )
        maps = run["curves"]["maps"]
        axes[2].plot(
            maps["grid"],
            np.asarray(maps["raw"]) * 100,
            style,
            color=color,
            label=label,
            linewidth=1.7,
        )
    for ax, title in zip(
        axes,
        ("Targeted arithmetic", "Sentinel arithmetic", "MAPS learning"),
        strict=True,
    ):
        ax.set(title=title, xlabel="Stage-B update", ylabel="Raw Pass@1 (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False)
    figure.savefig(output / "trajectories.svg")
    plt.close(figure)
