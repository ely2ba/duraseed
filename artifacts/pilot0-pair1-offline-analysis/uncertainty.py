"""Descriptive paired item bootstrap and early-window retention; local only."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ROOT = REPO / "runs/pilot0/pilot0-pair1-seed11-20260825T125100Z"
GRID = np.array([0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480])
METHODS = ("B-S", "B-G")
REPS = 50_000
BOOTSTRAP_SEED = 314159


def read(path):
    return json.loads(path.read_text())


def evaluation(method, step, panel):
    base = ROOT / "seed-11" / method
    if panel in ("a-monitor", "a-validation") and step == 0:
        return base / "selected-pre-b" / panel / "result.json"
    segment = "step-0" if step == 0 else f"steps-{GRID[list(GRID).index(step)-1]}-{step}"
    name = "a-retention" if panel == "a-monitor" else panel
    return base / "stage-b" / segment / name / "result.json"


def paired_arrays(panel, role, steps):
    """Two methods × item IDs × time, with identical item/draw populations."""
    parsed, paths = [], []
    for method in METHODS:
        method_rows = []
        for step in steps:
            path = evaluation(method, step, panel)
            paths.append(str(path))
            rows = {r["task_id"]: r for r in read(path)["item_counts"] if r["panel_role"] == role}
            method_rows.append(rows)
        parsed.append(method_rows)
    ids = sorted(parsed[0][0])
    assert all(set(rows) == set(ids) for arm in parsed for rows in arm)
    c = np.array([[[rows[item]["successes"] for rows in arm] for item in ids] for arm in parsed], dtype=float)
    n = np.array([[[rows[item]["trials"] for rows in arm] for item in ids] for arm in parsed], dtype=float)
    assert np.all(n == n[0, 0, 0]) and np.all(c >= 0) and np.all(c <= n)
    return ids, c / n, (c + .5) / (n + 1), paths


def auc(values, grid):
    return np.sum(np.diff(grid) * (values[..., :-1] + values[..., 1:]) / 2, axis=-1) / (grid[-1] - grid[0])


def paired_bootstrap(vectors, names, seed):
    """Resample entire paired item vectors, not individual draws or checkpoints."""
    values = np.column_stack(vectors)
    n, k = values.shape
    rng = np.random.default_rng(seed)
    boots = np.empty((REPS, k))
    for begin in range(0, REPS, 250):
        end = min(begin + 250, REPS)
        indices = rng.integers(n, size=(end - begin, n))
        boots[begin:end] = values[indices].mean(axis=1)
    intervals = np.quantile(boots, [.025, .975], axis=0, method="linear")
    return [
        {"contrast": name, "direction": "B-S minus B-G", "items": n,
         "estimate": float(values[:, j].mean()), "ci95_lower": float(intervals[0, j]),
         "ci95_upper": float(intervals[1, j]), "bootstrap_seed": seed,
         "bootstrap_replicates": REPS}
        for j, name in enumerate(names)
    ]


def half_life(values):
    threshold = values[0] / 2
    if values[0] <= 0:
        return {"update": None, "status": "undefined_zero_baseline"}
    for j in range(1, len(GRID)):
        if values[j] <= threshold < values[j-1]:
            fraction = (values[j-1] - threshold) / (values[j-1] - values[j])
            return {"update": float(GRID[j-1] + fraction * (GRID[j]-GRID[j-1])),
                    "status": "first_downward_crossing_linear_interpolation",
                    "bracket": [int(GRID[j-1]), int(GRID[j])], "threshold": float(threshold)}
    return {"update": None, "status": "not_crossed_on_observed_grid_0_to_480", "threshold": float(threshold)}


def main():
    ids, raw, post, f2paths = paired_arrays("b-validation", "stage-b", GRID)
    f2diff = post[0] - post[1]
    rawdiff = raw[0] - raw[1]
    f2vectors = [rawdiff[:, -1], f2diff[:, -1], auc(f2diff, GRID)-f2diff[:, 0],
                 auc(rawdiff, GRID)-rawdiff[:, 0]]
    f2names = ["F2 update-480 Pass@1", "F2 update-480 posterior mean",
               "F2 raw-gain AUC (stored posterior-score estimand)",
               "F2 Pass@1 gain AUC (supplementary raw-rate estimand)"]
    cis = paired_bootstrap(f2vectors, f2names, BOOTSTRAP_SEED)
    result = read(ROOT / "result.json")
    stored = {r["method"]: r for r in result["F1_F2_cells"]}
    assert np.isclose(cis[2]["estimate"], result["paired_primary_contrast"]["raw_gain_stage_b_auc_difference"], atol=1e-14)
    assert np.allclose(f2vectors[1], f2vectors[0] * 16/17, atol=1e-14)
    assert np.allclose(f2vectors[2], f2vectors[3] * 16/17, atol=1e-14)
    for a, method in enumerate(METHODS):
        assert np.allclose(post[a].mean(axis=0), stored[method]["F2_stage_b_learning"]["maps_scores"], atol=1e-14)

    early, f1paths, family_counts = [], [], {}
    manifest = read(ROOT / "pilot-inputs/a_monitor_manifest.json")
    family_by_item = {r["task_id"]: r["intended_family"] for r in manifest["records"]}
    for role in ("targeted", "sentinel"):
        tids, raw, post, paths = paired_arrays("a-monitor", role, GRID)
        f1paths.extend(paths)
        family_counts[role] = len({family_by_item[i] for i in tids})
        if role == "targeted":
            vectors, names = [], []
            for step in (1, 2, 5, 10):
                j = list(GRID).index(step)
                for label, scores in (("Pass@1", raw), ("posterior mean", post)):
                    vectors.append(scores[0, :, j] - scores[1, :, j])
                    names.append(f"F1 targeted monitor update-{step} {label}")
            cis.extend(paired_bootstrap(vectors, names, BOOTSTRAP_SEED+1))
        for a, method in enumerate(METHODS):
            for label, scores in (("Pass@1", raw), ("posterior mean", post)):
                means = scores[a].mean(axis=0)
                first20 = GRID <= 20
                area = float(auc(means[first20], GRID[first20]))
                half = half_life(means)
                if label == "posterior mean" and means[0]/2 < .1:
                    half["posterior_floor_note"] = "With four draws the 0.1 Jeffreys floor exceeds half the update-0 score; posterior half-life cannot cross this threshold."
                early.append({"method": method, "role": role, "score": label,
                    "baseline": float(means[0]),
                    "relative_retention": {str(s): float(means[list(GRID).index(s)] / means[0]) for s in (1,2,5,10)},
                    "half_life": half, "normalized_absolute_auc_0_20": area,
                    "normalized_relative_auc_0_20": area / float(means[0]),
                    "unnormalized_absolute_area_0_20": area*20})
    pids, praw, ppost, prepaths = paired_arrays("a-validation", "targeted", [0])
    cis.extend(paired_bootstrap([praw[0,:,0]-praw[1,:,0], ppost[0,:,0]-ppost[1,:,0]],
        ["Pre-B 16-draw targeted Pass@1 (matching imperfection)",
         "Pre-B 16-draw targeted posterior mean (matching imperfection)"], BOOTSTRAP_SEED+2))
    data = {"run_id": ROOT.name, "seed": 11, "contrast_direction": "B-S minus B-G",
        "bootstrap": {"method": "paired item-cluster percentile", "replicates": REPS,
            "base_seed": BOOTSTRAP_SEED, "confidence": .95,
            "multiple_comparison_adjustment": "none; pointwise descriptive intervals",
            "resampling_unit": "item ID shared across both methods and all time points; all draws stay in each item"},
        "contrasts": cis, "early_window": early,
        "item_counts": {"F2": len(ids), "F1_targeted_monitor": len(tids), "pre_B_targeted": len(pids)},
        "monitor_assigned_families": family_counts,
        "source_paths": {"F2": f2paths, "F1_monitor": f1paths, "pre_B": prepaths}}
    (OUT / "uncertainty.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    lines = ["# Paired item-level uncertainty and early-window F1", "",
        "Descriptive, post-hoc, seed 11 only. No result enters a gate or changes the running experiment.", "",
        "## Paired uncertainty", "",
        "All contrasts are B-S minus B-G. Intervals are 95% paired item-cluster percentile bootstrap CIs with 50,000 replicates (NumPy PCG64, base seed 314159; seeds 314160 and 314161 for the monitor and pre-B panels). Sampling the same item index in both arms preserves pairing; for AUC, that index retains the entire trajectory and all within-item draws. No individual completions are treated as independent items.", "",
        "The sampling unit is an item, conditional on these fitted models, their selected checkpoints, the fixed family panel, and the observed per-item draw records. Items are treated as exchangeable independent clusters within each analyzed population. This estimates variability over empirical item clusters with realized draws retained, not fresh-completion uncertainty for the exact fixed panel. This is not a family-cluster or training-seed CI; correlation across items in a family could make intervals too narrow. The monitor has 12 assigned families per role. Intervals do not include training-seed variance, checkpoint-selection uncertainty, uncertainty from choosing these post-hoc windows, or a separate within-item resampling model. They are pointwise, without multiplicity adjustment. A posterior-score bootstrap CI is not a Bayesian credible interval.", "",
        "| Contrast | Paired items | Estimate | 95% CI lower | 95% CI upper |",
        "|---|---:|---:|---:|---:|"]
    for row in cis:
        lines.append(f"| {row['contrast']} | {row['items']} | {row['estimate']:.8f} | {row['ci95_lower']:.8f} | {row['ci95_upper']:.8f} |")
    lines += ["", "Units are proportions (multiply by 100 for percentage points). F2 raw-gain AUC here is the existing posterior-score estimand: trapezoidal absolute AUC over updates 0–480, normalized by 480, minus that arm's own update-0 posterior score. The supplementary Pass@1-gain AUC applies the same operation to raw rates. With 16 draws, posterior differences and gain differences equal 16/17 times their raw-rate counterparts.", "",
        "Pre-B uses the separate 256-targeted-item × 16-draw A-validation panel, not the one-draw cadence matching panel (31/96 for each selected checkpoint). Its CI is conditional on B-S@140 and B-G@30 having been selected; it does not certify an equivalence margin or adjust for selection.", "",
        "## Early-window retention", "",
        "Relative retention is the ratio of the panel's mean score at t to its own update-0 mean; it is not a mean of itemwise ratios and is not clipped at 1. Both raw Pass@1 and the stored Jeffreys posterior score are shown. Half-life is the first downward crossing of half the initial score, linearly interpolated between the adjacent observed checkpoints on the update axis; it is not an exponential-fit parameter or an observed intermediate checkpoint. All 11 available points (0–480) are searched.", "",
        "AUC(0–20) uses only the observed grid [0,1,2,5,10,20], trapezoidal integration, divided by 20. Relative AUC additionally divides by that arm/role's initial score; the unnormalized absolute area is in the JSON. These are point estimates, without bootstrap intervals.", "",
        "| Arm / role / score | Relative u1 | u2 | u5 | u10 | Half-life (updates) | Absolute AUC 0–20 | Relative AUC 0–20 |",
        "|---|---:|---:|---:|---:|---|---:|---:|"]
    for row in early:
        rel = row["relative_retention"]
        half = row["half_life"]["update"]
        bracket = row["half_life"].get("bracket")
        halftext = f"{half:.6f} [{bracket[0]}, {bracket[1]}]" if half is not None else "Not crossed (0–480)"
        lines.append(f"| {row['method']} / {row['role']} / {row['score']} | " + " | ".join(f"{rel[str(s)]:.6f}" for s in (1,2,5,10)) + f" | {halftext} | {row['normalized_absolute_auc_0_20']:.8f} | {row['normalized_relative_auc_0_20']:.8f} |")
    lines += ["", "For B-S sentinel posterior retention, half the initial score lies below the four-draw Jeffreys floor of 0.1, so the requested posterior half-life is unattainable under this score definition; the raw-rate half-life is reported separately. Relative retention does not subtract M0 ability and does not measure survival restricted to newly acquired items.", "",
        f"Sources: [pair-1 result]({ROOT / 'result.json'}); [matching]({ROOT / 'seed-11/matching.json'}); [all exact estimates and evaluation paths](uncertainty.json).", "",
        "Checks: identical paired item-ID and draw populations at every included checkpoint; existing stored F2 curves/AUC contrast reproduced; raw-to-posterior 16/17 identity checked. No remote calls or new samples.", ""]
    (OUT / "uncertainty.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
