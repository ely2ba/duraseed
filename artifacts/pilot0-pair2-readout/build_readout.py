"""Local-only presentation of the completed pair-2 validation artifacts."""

from __future__ import annotations

from html import escape
import json
from math import isclose, log1p
from pathlib import Path

from duraseed.pilot0_analysis import summarize_selected_method
from duraseed.pilot0_contract import STAGE_B_GRID
from duraseed.provenance import canonical_json_value


REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ROOT = REPO / "runs/pilot0/pilot0-pair2-seed29-20260830T204211Z"
GRID = list(STAGE_B_GRID)
METHODS = ("B-S", "B-G")
COLORS = {"B-S": "#285b86", "B-G": "#b45a31"}


def read(path):
    return json.loads(path.read_text())


def eval_path(method, step, panel):
    base = ROOT / "seed-29" / method
    if panel == "a-monitor" and step == 0:
        return base / "selected-pre-b/a-monitor/result.json"
    segment = "step-0" if step == 0 else f"steps-{GRID[GRID.index(step)-1]}-{step}"
    name = "a-retention" if panel == "a-monitor" else panel
    return base / "stage-b" / segment / name / "result.json"


def counts(value, role):
    rows = [row for row in value["item_counts"] if row["panel_role"] == role]
    return {
        "items": len(rows),
        "successes": sum(row["successes"] for row in rows),
        "trials": sum(row["trials"] for row in rows),
        "draws_per_item": sorted({row["trials"] for row in rows}),
    }


def curve(values):
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def fmt_counts(rows):
    return "[" + ", ".join(f"{r['successes']}/{r['trials']}" for r in rows) + "]"


def make_figure(path, panels, title):
    """Simple vector scientific plot; points use log(1 + Stage-B update)."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1400" viewBox="0 0 1400 1400">',
        '<rect width="1400" height="1400" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#242424} .tick{font-size:13px} .small{font-size:15px}</style>',
        f'<text x="55" y="36" font-size="25">{escape(title)}</text>',
        '<text x="55" y="61" class="small">Pair 2 · seed 29 · x-axis: log(1 + update); AUC calculations use the linear update grid</text>',
    ]
    for i, (heading, series, ymax) in enumerate(panels):
        x0, y0 = 80 + (i % 2) * 690, 130 + (i // 2) * 580
        w, h = 575, 430
        parts.append(f'<text x="{x0}" y="{y0-28}" font-size="19">{escape(heading)}</text>')
        for j in range(6):
            value = ymax * j / 5
            y = y0 + h * (1 - value / ymax)
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0+w}" y2="{y}" stroke="#e2e5e8"/>')
            parts.append(f'<text x="{x0-12}" y="{y+5}" text-anchor="end" class="tick">{value:.2f}</text>')
        for step in GRID:
            x = x0 + w * log1p(step) / log1p(GRID[-1])
            parts.append(f'<text x="{x}" y="{y0+h+23}" text-anchor="middle" class="tick">{step}</text>')
        parts.append(f'<text x="{x0+w/2}" y="{y0+h+48}" text-anchor="middle" class="small">Stage-B update</text>')
        for method in METHODS:
            points = [(x0 + w * log1p(s) / log1p(GRID[-1]), y0 + h * (1-v/ymax)) for s, v in zip(GRID, series[method], strict=True)]
            poly = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
            dash = ' stroke-dasharray="8 5"' if method == "B-G" else ""
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{COLORS[method]}" stroke-width="2.5"{dash}/>')
            parts.extend(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="{COLORS[method]}"/>' for x, y in points)
    for i, method in enumerate(METHODS):
        x = 1030 + i * 155
        parts.append(f'<line x1="{x}" y1="42" x2="{x+36}" y2="42" stroke="{COLORS[method]}" stroke-width="3"/>')
        parts.append(f'<text x="{x+45}" y="47" class="small">{method}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n")


result = read(ROOT / "result.json")
assert result["status"] == "evidence_collected"
cells = {cell["method"]: cell for cell in result["F1_F2_cells"]}
matching = read(ROOT / "seed-29/matching.json")
extracted = {}
for method in METHODS:
    base = ROOT / "seed-29" / method
    monitor = [read(eval_path(method, step, "a-monitor")) for step in GRID]
    maps = [read(eval_path(method, step, "b-validation")) for step in GRID]
    before = read(base / "selected-pre-b/a-validation/result.json")
    after = read(base / "stage-b/steps-320-480/a-validation/result.json")
    reproduced = canonical_json_value(summarize_selected_method(
        seed=29, method=method, stage_a_selected=before,
        stage_b_maps=tuple(maps), stage_b_retention=tuple(monitor),
        stage_b_final_retention=after,
    ))
    assert reproduced == cells[method], method
    posterior = cells[method]["F2_stage_b_learning"]["maps_scores"]
    raw = [row["1"] for row in cells[method]["F2_stage_b_learning"]["maps_pass_at_k_curve"]]
    auc = cells[method]["F2_stage_b_learning"]["maps_auc"]
    relative = [v - posterior[0] for v in posterior]
    integrated = sum((b-a)*(x+y)/2 for a,b,x,y in zip(GRID, GRID[1:], posterior, posterior[1:])) / 480
    assert isclose(integrated, auc["absolute_auc"], abs_tol=1e-14)
    extracted[method] = {
        "F1_monitor_counts": {role: [counts(row, role) for row in monitor] for role in ("targeted", "sentinel")},
        "F1_validation_counts": {role: {"pre_b": counts(before, role), "post_b": counts(after, role)} for role in ("targeted", "sentinel")},
        "F2_counts": [counts(row, "stage-b") for row in maps],
        "F2_posterior_baseline_relative": relative,
        "F2_headroom_normalized_relative": [v/(1-posterior[0]+1e-12) for v in relative],
        "F2_raw_pass1_baseline_relative": [v-raw[0] for v in raw],
        "source_paths": {
            "a_monitor": [str(eval_path(method, step, "a-monitor")) for step in GRID],
            "b_validation": [str(eval_path(method, step, "b-validation")) for step in GRID],
            "a_validation_pre": str(base / "selected-pre-b/a-validation/result.json"),
            "a_validation_post": str(base / "stage-b/steps-320-480/a-validation/result.json"),
        },
    }

payload = {"run_id": ROOT.name, "seed": 29, "stage_b_updates": GRID,
           "matching": matching, "F1_F2_cells": cells, "derived": extracted,
           "paired_primary_contrast": result["paired_primary_contrast"],
           "F3_source": str(ROOT / "seed-29/pre-b-profiles.json")}
(OUT / "readout.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

f1panels = []
for role in ("targeted", "sentinel"):
    f1panels.append((f"F1 · {role} · observed Pass@1", {m: [r["1"] for r in cells[m]["monitor_retention"][f"{role}_pass_at_k_curve"]] for m in METHODS}, .4))
for role in ("targeted", "sentinel"):
    f1panels.append((f"F1 · {role} · posterior mean", {m: cells[m]["monitor_retention"][f"{role}_monitor_retention_curve"] for m in METHODS}, .4))
make_figure(OUT / "F1-retention.svg", f1panels, "F1 · TCES retention on a_monitor")
make_figure(OUT / "F2-learning.svg", [
    ("F2 · observed Pass@1", {m: [r["1"] for r in cells[m]["F2_stage_b_learning"]["maps_pass_at_k_curve"]] for m in METHODS}, .5),
    ("F2 · observed Pass@1 minus own baseline", {m: extracted[m]["F2_raw_pass1_baseline_relative"] for m in METHODS}, .5),
    ("F2 · posterior mean", {m: cells[m]["F2_stage_b_learning"]["maps_scores"] for m in METHODS}, .5),
    ("F2 · posterior mean minus own baseline", {m: extracted[m]["F2_posterior_baseline_relative"] for m in METHODS}, .5),
], "F2 · MAPS learning on b_validation")

lines = ["# Pair-2 scientific readout", "", f"Run: `{ROOT.name}`. Seed: 29. Local artifacts only.", "",
    "Matching selected B-S Stage-A update 40 and B-G update 20; each had 17/96 targeted exact successes on the cadence matching panel. Both Stage-B schedules ran 480 updates.", "",
    f"Sources: [terminal result]({ROOT / 'result.json'}), [matching record]({ROOT / 'seed-29/matching.json'}), [exact readout data](readout.json), [both F3 profiles](F3.md).", "",
    "All sequences below follow this Stage-B update grid:", "", f"`{GRID}`", "",
    "Rates are proportions, not percentages. Printed sequences use six decimal places; readout.json retains the stored precision, exact success/trial counts, and every source path. Pass@k uses the stored unbiased item-averaged estimator. Posterior means use the item-level Jeffreys estimator `(successes + 0.5)/(draws + 1)`, averaged equally over items.", "",
    "The zero-success posterior floor is 0.1 with four draws and 0.029411764705882353 with 16 draws; it is not observed Pass@1. a_monitor has 192 targeted and 192 sentinel items × 4 draws; a_validation has 256 targeted and 256 sentinel items × 16 draws; b_validation has 512 MAPS items × 16 draws.", "",
    "## F1 · retention trajectories", "", "![F1 retention](F1-retention.svg)", ""]
for method in METHODS:
    lines += [f"### {method}", ""]
    mon, f1 = cells[method]["monitor_retention"], cells[method]["F1_retention"]
    for role in ("targeted", "sentinel"):
        lines += [f"{role.capitalize()} a_monitor:", "",
            f"- Exact successes/trials: `{fmt_counts(extracted[method]['F1_monitor_counts'][role])}`.",
            f"- Pass@1: `{curve([r['1'] for r in mon[f'{role}_pass_at_k_curve']])}`.",
            f"- Pass@4: `{curve([r['4'] for r in mon[f'{role}_pass_at_k_curve']])}`.",
            f"- Posterior mean: `{curve(mon[f'{role}_monitor_retention_curve'])}`.",
            f"- Posterior absolute AUC: `{mon[f'{role}_monitor_retention_absolute_auc']}`.", ""]
        val = extracted[method]["F1_validation_counts"][role]
        lines += [f"{role.capitalize()} a_validation, selected checkpoint → Stage-B update 480:", "",
            f"- Exact successes/trials: {val['pre_b']['successes']}/{val['pre_b']['trials']} → {val['post_b']['successes']}/{val['post_b']['trials']}.",
            f"- Posterior mean: `{f1[f'pre_b_{role}_score']}` → `{f1[f'fixed_budget_{role}_retention']}`; change `{f1[f'fixed_budget_{role}_change']}`."]
        for k in (1,4,16):
            lines.append(f"- Pass@{k}: `{f1['pre_b_pass_at_k'][role][str(k)]}` → `{f1['post_b_pass_at_k'][role][str(k)]}`.")
        lines.append("")
lines += ["## F2 · learning curves", "", "![F2 learning](F2-learning.svg)", "",
    "Baseline-relative means subtract the same arm's update-0 score. The headroom-normalized series divides that difference by `1 - baseline + 1e-12`. AUC uses trapezoidal integration on the linear update grid, divided by 480; raw-gain AUC is absolute AUC minus the update-0 posterior mean. The figures use a transformed x-axis only for display.", ""]
for method in METHODS:
    f2, ex = cells[method]["F2_stage_b_learning"], extracted[method]
    lines += [f"### {method}", "", f"- Exact successes/trials: `{fmt_counts(ex['F2_counts'])}`."]
    for k in (1,4,16):
        lines.append(f"- Absolute Pass@{k}: `{curve([r[str(k)] for r in f2['maps_pass_at_k_curve']])}`.")
    lines += [f"- Pass@1 minus own baseline: `{curve(ex['F2_raw_pass1_baseline_relative'])}`.",
        f"- Absolute posterior mean: `{curve(f2['maps_scores'])}`.",
        f"- Posterior mean minus own baseline: `{curve(ex['F2_posterior_baseline_relative'])}`.",
        f"- Headroom-normalized posterior relative series: `{curve(ex['F2_headroom_normalized_relative'])}`.", ""]
    for key, value in f2["maps_auc"].items():
        lines.append(f"- `{key}`: `{value}`.")
    lines.append("")
contrast = result["paired_primary_contrast"]
lines += ["## Stored paired contrast · B-S minus B-G", "",
    f"- F2 raw-gain Stage-B AUC difference: `{contrast['raw_gain_stage_b_auc_difference']}`.",
    f"- F1 update-480 targeted posterior retention difference: `{contrast['fixed_budget_targeted_retention_difference']}`.", "",
    "## F3 · pre-Stage-B profiles", "", "[Both profiles, including panel metrics and full-source links](F3.md). The MAPS update-0 values above are the baseline Stage-B task performance before Stage-B training.", "",
    "## Local checks", "", "Both F1/F2 cells reproduced exactly from the saved evaluation item counts using the existing reducer. Both stored F2 AUCs reproduced from the 11-point linear grid. No new sampling, training, or remote calls were used for this readout.", ""]
(OUT / "README.md").write_text("\n".join(lines))
print("PASS: two F1/F2 cells reproduced; 11-point AUCs checked; local JSON, Markdown and SVG readout written.")
