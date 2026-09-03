"""Generate two descriptive README SVGs from the existing pair-1 count readout."""

import json
from pathlib import Path
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
DATA = json.loads((Path(__file__).parent / "readout.json").read_text())
OUTPUT = REPO / "docs/assets"
COLORS = {"B-S": "#164BEB", "B-G": "#D92432"}
GRID = DATA["stage_b_updates"]
WIDTH, HEIGHT = 960, 552
TOP, BOTTOM = 184, 422
LEFTS, PLOT_WIDTH = (80, 554), 342


def text(x, y, value, size=15, color="#424A52", anchor="start", weight=400):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')


def line(x1, y1, x2, y2, color="#DDE2E6", width=1, dash=""):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}" '
            + (f'stroke-dasharray="{dash}" ' if dash else "") + '/>')


def begin(title, description):
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
        'role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(description)}</desc>',
        '<style>text{font-family:Arial,Helvetica,sans-serif} '
        'line,polyline{stroke-linecap:round;stroke-linejoin:round}</style>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#FFFFFF"/>',
        text(40, 38, title, 25, "#242D35", weight=600),
        text(40, 66, "Pilot pair 1 · seed 11 · selected Stage-A checkpoints: B-S 140 / B-G 30", 15),
    ]


def legend(svg):
    for method, x, label in (("B-S", 40, "B-S · supervised fine-tuning"),
                              ("B-G", 398, "B-G · reinforcement learning")):
        svg.append(line(x, 99, x + 34, 99, COLORS[method], 2.5,
                        "6 5" if method == "B-G" else ""))
        svg.append(f'<circle cx="{x + 17}" cy="99" r="3.8" fill="{COLORS[method]}"/>')
        svg.append(text(x + 46, 104, label, 15))


def axes(svg, left, title, unit, xmax, ymax, xticks, yticks):
    svg.extend([text(left, 143, title, 18, "#242D35", weight=600),
                text(left, 167, unit, 13)])
    for value in yticks:
        y = BOTTOM - value / ymax * (BOTTOM - TOP)
        svg.append(line(left, y, left + PLOT_WIDTH, y))
        svg.append(text(left - 12, y + 5, value, 14, anchor="end"))
    for value in xticks:
        x = left + value / xmax * PLOT_WIDTH
        svg.extend([line(x, BOTTOM, x, BOTTOM + 5, "#8E969D"),
                    text(x, BOTTOM + 25, value, 14, anchor="middle")])
    svg.extend([line(left, TOP, left, BOTTOM, "#8E969D"),
                line(left, BOTTOM, left + PLOT_WIDTH, BOTTOM, "#8E969D"),
                text(left + PLOT_WIDTH / 2, BOTTOM + 52, "Stage-B updates", 15, anchor="middle")])


def curve(svg, left, method, updates, values, xmax, ymax):
    assert len(updates) == len(values)
    coords = [(left + step / xmax * PLOT_WIDTH,
               BOTTOM - value / ymax * (BOTTOM - TOP)) for step, value in zip(updates, values)]
    points = " ".join(f"{x:.4f},{y:.4f}" for x, y in coords)
    dash = ' stroke-dasharray="6 5"' if method == "B-G" else ""
    svg.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[method]}" '
               f'stroke-width="2.5"{dash}/>')
    for (x, y), step, value in zip(coords, updates, values):
        svg.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="3.8" '
                   f'fill="{COLORS[method]}" stroke="#FFFFFF" stroke-width="0.8">'
                   f'<title>{method}, update {step}: {value:.8f}</title></circle>')
    return coords[-1]


def finish(svg, filename):
    svg.append('</svg>')
    content = "\n".join(svg) + "\n"
    ET.fromstring(content)
    (OUTPUT / filename).write_text(content)


def count_rates(method, key, role=None):
    rows = DATA["derived"][method][key]
    if role:
        rows = rows[role]
    assert len(rows) == len(GRID)
    expected = 768 if key == "F1_monitor_counts" else 8192
    assert all(row["trials"] == expected for row in rows)
    return [100 * row["successes"] / row["trials"] for row in rows]


def retention():
    svg = begin("Arithmetic retention · first 20 updates", "Observed Pass at 1 on targeted and "
                "held-out sentinel TCES panels over Stage-B updates 0, 1, 2, 5, 10, and 20. "
                "Linear axes. One seed, point estimates, no uncertainty bands. Both arms "
                "recorded zero of 4096 successes per panel on final high-draw validation at update 480, "
                "which is outside this plot.")
    legend(svg)
    early = [index for index, step in enumerate(GRID) if step <= 20]
    assert [GRID[index] for index in early] == [0, 1, 2, 5, 10, 20]
    for left, role, label in zip(LEFTS, ("targeted", "sentinel"),
                                 ("Targeted skill", "Held-out families · sentinel")):
        axes(svg, left, label, "Observed Pass@1 (%)", 20, 40, (0, 5, 10, 15, 20), (0, 10, 20, 30, 40))
        for method in COLORS:
            values = count_rates(method, "F1_monitor_counts", role)
            curve(svg, left, method, [GRID[i] for i in early], [values[i] for i in early], 20, 40)
            endpoint = DATA["derived"][method]["F1_validation_counts"][role]["post_b"]
            assert endpoint["successes"] == 0 and endpoint["trials"] == 4096
    svg.extend([text(40, 508, "Points: 192 items × 4 draws per panel; one seed, no uncertainty bands. Linear x-axis.", 13),
                text(40, 532, "Only updates 0–20 shown. Final validation at 480: 0/4,096 successes per panel per arm.", 13)])
    finish(svg, "pilot-pair1-retention.svg")


def learning():
    svg = begin("Learning the next task · MAPS", "Observed MAPS Pass at 1 and percentage-point gain "
                "from each arm's own pre-Stage-B baseline, at all 11 recorded updates through 480. "
                "Both axes linear. One seed, point estimates, no uncertainty bands. "
                "Each point uses 512 items and 16 draws per item.")
    legend(svg)
    for index, left in enumerate(LEFTS):
        relative = index == 1
        axes(svg, left, "Gain from own pre-B baseline" if relative else "Absolute task performance",
             "Pass@1 gain (percentage points)" if relative else "Observed Pass@1 (%)",
             480, 45, (0, 80, 160, 320, 480), (0, 10, 20, 30, 40, 45))
        for method in COLORS:
            values = count_rates(method, "F2_counts")
            if relative:
                values = [value - values[0] for value in values]
            x, y = curve(svg, left, method, GRID, values, 480, 45)
            upper = (method == "B-S") if relative else (method == "B-G")
            suffix = " pp" if relative else "%"
            svg.append(text(x - 7, y - 13 if upper else y + 22,
                            f"{values[-1]:.2f}{suffix}", 14, COLORS[method], "end", 600))
    svg.extend([text(40, 508, "Points: 512 items × 16 draws; one seed, no uncertainty bands. Both panels use linear axes.", 13),
                text(40, 532, "Gain subtracts each arm’s observed update-0 Pass@1; it is not a posterior score or an AUC.", 13)])
    finish(svg, "pilot-pair1-learning.svg")


if __name__ == "__main__":
    assert DATA["seed"] == 11
    assert DATA["matching"]["B-S"]["step"] == 140 and DATA["matching"]["B-G"]["step"] == 30
    retention()
    learning()
    print("Saved retention and learning SVGs from exact observed counts; XML and denominators checked.")
