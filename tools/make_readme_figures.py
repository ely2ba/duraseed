"""Draw README SVGs from stored replay results, without fitting or resampling."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/replay-v1/followup"
OUTPUT = ROOT / "docs/assets"
INK, MUTED, PAPER, RULE = "#242735", "#676B78", "#FAF9F5", "#DADCD9"
COLORS = {"R-S": "#BC2344", "R-P": "#2545D9"}
ARMS = ("R-S", "R-P")


def load(name):
    return json.loads((SOURCE / name).read_text())


def start(height, title, description):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="{height}" '
        f'viewBox="0 0 1050 {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}'
        '.editorial{font-family:Georgia,"Times New Roman",serif}</style>',
        f'<rect width="1050" height="{height}" fill="{PAPER}"/>',
    ]


def text(svg, x, y, value, size=15, color=INK, weight=400, anchor="start", serif=False):
    klass = ' class="editorial"' if serif else ""
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
               f'font-weight="{weight}" text-anchor="{anchor}"{klass}>{escape(str(value))}</text>')


def segment(svg, x1, y1, x2, y2, color=RULE, width=1):
    svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
               f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>')


def rectangle(svg, x, y, width, height, color, radius=0):
    svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width}" height="{height}" '
               f'rx="{radius}" fill="{color}"/>')


def save(svg, name):
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / f"{name}.svg"
    path.write_text("\n".join([*svg, "</svg>", ""]))
    print(path.relative_to(ROOT))


def tint(arm, count):
    if count == 0:
        return "#DFE0DB"
    amount = 0.16 + 0.84 * count / 16
    rgb = [round(int(PAPER[i:i + 2], 16) * (1 - amount)
                 + int(COLORS[arm][i:i + 2], 16) * amount) for i in (1, 3, 5)]
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def coverage():
    data = load("retrospective-count-breakdowns.json")["coverage"]["targeted"]
    profiles = {arm: load(f"profiles/seed-11-{arm}.json") for arm in ARMS}
    shared = [{item["task_id"] for item in profiles[arm]["items"]
               if item["panel_role"] == "targeted"} for arm in ARMS]
    assert shared[0] == shared[1] and len(shared[0]) == 256
    svg = start(622, "Similar accuracy, different coverage",
                "Selected replay origins in source block 11, on the same 256 targeted arithmetic items. "
                "R-S has raw Pass@1 35.72% and at least one success on 170 items. R-P has raw Pass@1 "
                "34.59% and at least one success on 247 items. Each square represents 16 recorded draws "
                "on one item. Squares are sorted independently by success count, not paired by position.")
    text(svg, 60, 35, "DURASEED  /  OBSERVED RESULTS", 12, MUTED, 600)
    text(svg, 60, 86, "Similar accuracy, different coverage.", 35, serif=True)
    text(svg, 60, 117, "Same 256 targeted items · 16 draws per item · source block 11", 16, MUTED)
    segment(svg, 60, 138, 990, 138, INK)
    for arm, x, title in (("R-S", 60, "Solver traces"), ("R-P", 560, "Policy traces")):
        info = data[arm]
        panel = profiles[arm]["panels"]["targeted"]
        histogram = info["success_count_histogram_0_to_16"]
        items = [item for item in profiles[arm]["items"] if item["panel_role"] == "targeted"]
        counts = Counter(item["successes"] for item in items)
        assert histogram == [counts[k] for k in range(17)]
        assert all(item["trials"] == 16 for item in items)
        assert panel["exact_success_rate"] == info["recorded_panel"]["exact_success_rate"]
        text(svg, x, 177, f"{arm}  /  {title}", 18, COLORS[arm], 600)
        text(svg, x, 209, f'{100 * panel["exact_success_rate"]:.2f}%', 27, COLORS[arm], 600)
        text(svg, x + 112, 209, "raw Pass@1", 15, MUTED)
        values = [k for k, n in enumerate(histogram) for _ in range(n)]
        assert len(values) == 256
        for index, value in enumerate(values):
            rectangle(svg, x + (index % 16) * 16, 231 + (index // 16) * 16,
                      13.5, 13.5, tint(arm, value), 1.5)
        text(svg, x + 290, 314, info["any_success"], 53, COLORS[arm], serif=True)
        text(svg, x + 291, 341, "of 256 items", 16, INK)
        text(svg, x + 291, 365, "with at least", 15, MUTED)
        text(svg, x + 291, 386, "one success", 15, MUTED)
        text(svg, x, 516, "Observed successes in 16 draws", 14, MUTED)
        for k in range(17):
            rectangle(svg, x + k * 15, 530, 13.5, 12, tint(arm, k), 1)
        for k in (0, 4, 8, 12, 16):
            text(svg, x + k * 15 + 6.75, 562, k, 13, MUTED, anchor="middle")
    text(svg, 60, 595, "Each square is one item, sorted by success count within its arm; positions are not matched across grids.", 14, MUTED)
    svg.append('<!-- Sources: artifacts/replay-v1/followup/retrospective-count-breakdowns.json; '
               'artifacts/replay-v1/followup/profiles/seed-11-R-{S,P}.json. -->')
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
    points = [(left + update / stop * (right - left),
               bottom - 100 * value / ymax * (bottom - top))
              for update, value in zip(curve["grid"], curve["raw"], strict=True) if update <= stop]
    dash = ' stroke-dasharray="7 5"' if arm == "R-S" else ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[arm]}" '
               f'stroke-width="2.8" stroke-linejoin="round"{dash}/>')
    for x, y in points:
        if arm == "R-S":
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{COLORS[arm]}"/>')
        else:
            rectangle(svg, x - 3.5, y - 3.5, 7, 7, COLORS[arm])
    return points


def trajectories():
    arms = load("readout.json")["blocks"]["11"]["arms"]
    svg = start(568, "Two replay learning trajectories",
                "Raw Pass@1 during common MAPS training in source block 11. The left panel shows targeted "
                "arithmetic through update 20, including the R-P rebound at update 10. The right panel shows "
                "absolute MAPS performance through update 480. Red dashed circles identify R-S solver traces; "
                "blue solid squares identify R-P policy traces. Both axes preserve the actual update spacing.")
    text(svg, 60, 35, "DURASEED  /  OBSERVED RESULTS", 12, MUTED, 600)
    text(svg, 60, 86, "One continuation, different learning paths.", 35, serif=True)
    text(svg, 60, 117, "Source block 11 · common MAPS training from two selected origins", 16, MUTED)
    segment(svg, 60, 138, 990, 138, INK)
    text(svg, 75, 173, "Arithmetic retention", 20, weight=600)
    text(svg, 75, 196, "Targeted TCES · raw Pass@1 (%)", 14, MUTED)
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
            text(svg, 267, 237, "u10 rebound", 14, COLORS[arm], 600)
            text(svg, 267, 258, f'{100 * curve["raw"][curve["grid"].index(10)]:.2f}%', 14, COLORS[arm])
            segment(svg, x + 4, y - 6, 260, 251, COLORS[arm])
        curve = arms[arm]["curves"]["stage-b"]
        points = trajectory(svg, curve, arm, 480, 600, 880, 222, 445, 50)
        label_y = 226 if arm == "R-S" else 273
        segment(svg, *points[-1], 909, label_y - 5, COLORS[arm])
        text(svg, 917, label_y, f'{arm}  {100 * curve["raw"][-1]:.2f}%', 14, COLORS[arm], 600)
    text(svg, 240, 497, "Stage-B update", 15, MUTED, anchor="middle")
    text(svg, 740, 497, "Stage-B update", 15, MUTED, anchor="middle")
    text(svg, 75, 524, "192 monitor items · 4 draws per item", 14, MUTED)
    text(svg, 600, 524, "512 items · 16 draws per item", 14, MUTED)
    text(svg, 60, 552, "Points are recorded checkpoints. Straight segments connect observations; no smoothing or fitted curves.", 14, MUTED)
    svg.append('<!-- Source: artifacts/replay-v1/followup/readout.json, blocks.11.arms.*.curves. -->')
    save(svg, "results-trajectories")


if __name__ == "__main__":
    coverage()
    trajectories()
