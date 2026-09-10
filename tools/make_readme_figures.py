"""Draw README SVGs from stored replay results, without fitting or resampling."""

from __future__ import annotations

import json
import re
from base64 import b64encode
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/replay-v1/followup"
OUTPUT = ROOT / "docs/assets"
INK, MUTED, PAPER, RULE = "#24292F", "#57606A", "#FFFFFF", "#D8DEE4"
COLORS = {"R-S": "#087F6D", "R-P": "#7652A3"}
ARMS = ("R-S", "R-P")


def load(name):
    return json.loads((SOURCE / name).read_text())


def font_style():
    rules = ["/* Source Sans 3, Adobe; SIL OFL 1.1: fonts/LICENSE.md. */"]
    for weight, name in ((400, "Regular"), (600, "Semibold")):
        data = b64encode(
            (OUTPUT / f"fonts/SourceSans3-{name}.woff2").read_bytes()
        ).decode()
        rules.append(
            '@font-face{font-family:"Source Sans 3";font-style:normal;'
            f'font-weight:{weight};src:url(data:font/woff2;base64,{data}) format("woff2")}}'
        )
    rules.append(
        'text{font-family:"Source Sans 3",sans-serif;'
        "font-variant-numeric:tabular-nums lining-nums}"
    )
    return "\n".join(rules)


def start(height, title, description):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="{height}" '
        f'viewBox="0 0 1050 {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        f"<style>{font_style()}</style>",
        f'<rect width="1050" height="{height}" fill="{PAPER}"/>',
    ]


def text(svg, x, y, value, size=15, color=INK, weight=400, anchor="start"):
    svg.append(
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def segment(svg, x1, y1, x2, y2, color=RULE, width=1):
    svg.append(
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
        f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>'
    )


def rectangle(svg, x, y, width, height, color, radius=0):
    svg.append(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{color}"/>'
    )


def save(svg, name):
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / f"{name}.svg"
    path.write_text("\n".join([*svg, "</svg>", ""]))
    print(path.relative_to(ROOT))


def coverage():
    data = load("retrospective-count-breakdowns.json")["coverage"]["targeted"]
    profiles = {arm: load(f"profiles/seed-11-{arm}.json") for arm in ARMS}
    shared = [
        {
            item["task_id"]
            for item in profiles[arm]["items"]
            if item["panel_role"] == "targeted"
        }
        for arm in ARMS
    ]
    assert shared[0] == shared[1] and len(shared[0]) == 256
    svg = start(
        608,
        "Arithmetic problem coverage",
        "Selected replay origins in source block 11, on the same 256 targeted arithmetic items. "
        "R-S has raw Pass@1 35.72% and at least one success on 170 items. R-P has raw Pass@1 "
        "34.59% and at least one success on 247 items. Two histograms show the number of problems "
        "at each observed success count, from zero to sixteen, on identical axes.",
    )
    text(svg, 60, 56, "Similar accuracy, different problem coverage", 28, weight=600)
    text(
        svg,
        60,
        87,
        "256 shared arithmetic problems · 16 attempts each · replay block 11",
        17,
        MUTED,
    )
    for arm, x, title in (("R-S", 75, "Solver traces"), ("R-P", 580, "Policy traces")):
        info = data[arm]
        panel = profiles[arm]["panels"]["targeted"]
        histogram = info["success_count_histogram_0_to_16"]
        items = [
            item for item in profiles[arm]["items"] if item["panel_role"] == "targeted"
        ]
        counts = Counter(item["successes"] for item in items)
        assert histogram == [counts[k] for k in range(17)]
        assert all(item["trials"] == 16 for item in items)
        assert (
            panel["exact_success_rate"] == info["recorded_panel"]["exact_success_rate"]
        )
        assert sum(histogram) == 256
        assert sum(histogram[1:]) == info["any_success"]
        text(svg, x, 137, f"{arm} / {title}", 19, COLORS[arm], 600)
        text(svg, x, 184, f"{info['any_success']} / 256", 38, weight=600)
        text(svg, x, 211, "problems solved at least once", 16, MUTED)
        text(
            svg,
            x + 397,
            178,
            f"{100 * panel['exact_success_rate']:.2f}%",
            25,
            weight=600,
            anchor="end",
        )
        text(svg, x + 397, 204, "correct attempts", 15, MUTED, anchor="end")
        top, bottom, width, ymax = 273, 478, 400, 90
        text(svg, x, 250, "Number of problems", 15, MUTED)
        for tick in (0, 30, 60, 90):
            y = bottom - tick / ymax * (bottom - top)
            segment(svg, x, y, x + width, y)
            text(svg, x - 12, y + 5, tick, 14, MUTED, anchor="end")
        for k, count in enumerate(histogram):
            cx = x + (k + 0.5) * width / 17
            height = count / ymax * (bottom - top)
            rectangle(svg, cx - 8, bottom - height, 16, height, COLORS[arm])
            segment(svg, cx, bottom, cx, bottom + 5, RULE)
            if k in (0, 4, 8, 12, 16):
                text(svg, cx, bottom + 25, k, 15, MUTED, anchor="middle")
            if k in (0, 16):
                text(
                    svg, cx, bottom - height - 9, count, 16, weight=600, anchor="middle"
                )
        text(svg, x + width / 34, 525, "None", 13, MUTED, anchor="middle")
        text(svg, x + width - width / 34, 525, "All 16", 13, MUTED, anchor="middle")
        text(
            svg,
            x + width / 2,
            554,
            "Correct attempts per problem (out of 16)",
            16,
            anchor="middle",
        )
    text(
        svg,
        60,
        589,
        "Each bar counts problems with exactly that many correct attempts. Both panels use the same scale.",
        15,
        MUTED,
    )
    svg.append(
        "<!-- Sources: artifacts/replay-v1/followup/retrospective-count-breakdowns.json; "
        "artifacts/replay-v1/followup/profiles/seed-11-R-{S,P}.json. -->"
    )
    save(svg, "results-coverage")


def axes(svg, left, right, top, bottom, xmax, ymax, xticks, yticks):
    for tick in yticks:
        y = bottom - tick / ymax * (bottom - top)
        segment(svg, left, y, right, y)
        text(svg, left - 12, y + 5, tick, 13, MUTED, anchor="end")
    segment(svg, left, bottom, right, bottom, "#969BA4")
    for tick in xticks:
        x = left + tick / xmax * (right - left)
        segment(svg, x, bottom, x, bottom + 5, "#969BA4")
        text(svg, x, bottom + 24, tick, 13, MUTED, anchor="middle")


def trajectory(svg, curve, arm, stop, left, right, top, bottom, ymax):
    points = [
        (
            left + update / stop * (right - left),
            bottom - 100 * value / ymax * (bottom - top),
        )
        for update, value in zip(curve["grid"], curve["raw"], strict=True)
        if update <= stop
    ]
    dash = ' stroke-dasharray="7 5"' if arm == "R-S" else ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    svg.append(
        f'<polyline points="{coords}" fill="none" stroke="{COLORS[arm]}" '
        f'stroke-width="2.8" stroke-linejoin="round"{dash}/>'
    )
    for x, y in points:
        if arm == "R-S":
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{COLORS[arm]}"/>'
            )
        else:
            rectangle(svg, x - 3.5, y - 3.5, 7, 7, COLORS[arm])
    return points


def trajectories():
    arms = load("readout.json")["blocks"]["11"]["arms"]
    svg = start(
        568,
        "Retention and subsequent learning",
        "Raw Pass@1 during common MAPS training in source block 11. The left panel shows targeted "
        "arithmetic through update 20, including the R-P rebound at update 10. The right panel shows "
        "absolute MAPS performance through update 480. Emerald dashed circles identify R-S solver traces; "
        "violet solid squares identify R-P policy traces. Both axes preserve the actual update spacing.",
    )
    text(svg, 60, 56, "What happens during the next task?", 28, weight=600)
    text(svg, 60, 87, "Same program-synthesis training · replay block 11", 17, MUTED)
    for arm, x, label in (("R-S", 75, "Solver traces"), ("R-P", 320, "Policy traces")):
        sample = {"grid": [0, 1], "raw": [0, 0]}
        trajectory(svg, sample, arm, 1, x, x + 30, 0, 128, 1)
        text(svg, x + 42, 133, f"{arm} / {label}", 16, COLORS[arm], 600)
    text(svg, 75, 173, "Arithmetic retention", 20, weight=600)
    text(svg, 75, 196, "Targeted arithmetic · raw Pass@1 (%)", 14, MUTED)
    text(svg, 600, 173, "New-task learning", 20, weight=600)
    text(svg, 600, 196, "Absolute MAPS · raw Pass@1 (%)", 14, MUTED)
    axes(svg, 75, 405, 222, 445, 20, 40, (0, 5, 10, 20), (0, 10, 20, 30, 40))
    axes(svg, 600, 880, 222, 445, 480, 50, (0, 160, 320, 480), (0, 10, 20, 30, 40, 50))
    for arm in ARMS:
        curve = arms[arm]["curves"]["targeted"]
        points = trajectory(svg, curve, arm, 20, 75, 405, 222, 445, 40)
        end = 100 * curve["raw"][curve["grid"].index(20)]
        text(svg, 421, points[-1][1] + 5, f"{arm}  {end:.2f}%", 14, COLORS[arm], 600)
        if arm == "R-P":
            x, y = points[curve["grid"].index(10)]
            text(svg, 267, 237, "Update 10", 15, weight=600)
            text(
                svg,
                267,
                258,
                f"{100 * curve['raw'][curve['grid'].index(10)]:.2f}%",
                15,
                MUTED,
            )
            segment(svg, x + 4, y - 6, 260, 251, COLORS[arm])
        curve = arms[arm]["curves"]["stage-b"]
        points = trajectory(svg, curve, arm, 480, 600, 880, 222, 445, 50)
        label_y = 226 if arm == "R-S" else 273
        segment(svg, *points[-1], 909, label_y - 5, COLORS[arm])
        text(
            svg,
            917,
            label_y,
            f"{arm}  {100 * curve['raw'][-1]:.2f}%",
            14,
            COLORS[arm],
            600,
        )
    text(svg, 240, 497, "Stage-B update", 15, MUTED, anchor="middle")
    text(svg, 740, 497, "Stage-B update", 15, MUTED, anchor="middle")
    text(svg, 75, 524, "192 monitor items · 4 draws per item", 14, MUTED)
    text(svg, 600, 524, "512 items · 16 draws per item", 14, MUTED)
    text(
        svg,
        60,
        552,
        "Points show recorded checkpoints; straight segments connect the observations.",
        14,
        MUTED,
    )
    svg.append(
        "<!-- Source: artifacts/replay-v1/followup/readout.json, blocks.11.arms.*.curves. -->"
    )
    save(svg, "results-trajectories")


def embed_pipeline_fonts():
    path = OUTPUT / "duraseed-pipeline.svg"
    svg = path.read_text()
    svg = re.sub(
        r"\s*<!-- Embedded fonts -->.*?<!-- End embedded fonts -->", "", svg, flags=re.S
    )
    svg = svg.replace(
        "</style>",
        f"<!-- Embedded fonts -->\n{font_style()}\n"
        "<!-- End embedded fonts -->\n</style>",
        1,
    )
    path.write_text(svg)


if __name__ == "__main__":
    coverage()
    trajectories()
    embed_pipeline_fonts()
