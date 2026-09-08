"""Retrospective descriptive counts from the completed replay package only."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from math import comb
from pathlib import Path

ARMS = ("R-S", "R-P")
ROLES = ("targeted", "sentinel")


def read(path):
    return json.loads(path.read_text())


def panel_counts(row, role):
    items = [x for x in row["item_counts"] if x["panel_role"] == role]
    return {
        "successes": sum(x["successes"] for x in items),
        "trials": sum(x["trials"] for x in items),
        "items": len(items),
    }


def acquisition(selection):
    result = {}
    for seed in (11, 29):
        rows = [r for r in selection["cadence"] if r["seed"] == seed]
        grid = sorted({r["update"] for r in rows})
        indexed = {(r["arm"], r["update"]): r for r in rows}
        result[str(seed)] = {}
        for role in ROLES:
            series = [
                {"update": u, **{a: panel_counts(indexed[a, u], role) for a in ARMS}}
                for u in grid
            ]
            differences = [
                Fraction(r["R-P"]["successes"], r["R-P"]["trials"])
                - Fraction(r["R-S"]["successes"], r["R-S"]["trials"])
                for r in series
            ]
            result[str(seed)][role] = {
                "series": series,
                "R_P_above": sum(d > 0 for d in differences),
                "ties": [u for u, d in zip(grid, differences) if d == 0],
                "R_P_below": [u for u, d in zip(grid, differences) if d < 0],
            }
    candidates = [r for r in selection["candidates"] if r["seed"] == 29]
    pairs = []
    for rs in (r for r in candidates if r["arm"] == "R-S"):
        for rp in (r for r in candidates if r["arm"] == "R-P"):
            counts = {a: panel_counts(r, "targeted") for a, r in zip(ARMS, (rs, rp))}
            gap = abs(
                Fraction(counts["R-P"]["successes"], counts["R-P"]["trials"])
                - Fraction(counts["R-S"]["successes"], counts["R-S"]["trials"])
            )
            pairs.append(
                {"updates": [rs["update"], rp["update"]], **counts, "gap": float(gap)}
            )
    result["block_29_closest_assessed_pairs"] = [
        p for p in pairs if p["gap"] == min(x["gap"] for x in pairs)
    ]
    return result


def coverage(profiles):
    result = {}
    for role in ROLES:
        result[role] = {}
        for arm in ARMS:
            profile = profiles[arm]
            items = [r for r in profile["items"] if r["panel_role"] == role]
            histogram = Counter(r["successes"] for r in items)
            assert all(r["trials"] == 16 for r in items)
            pass_k = {
                str(k): sum(
                    1 - comb(16 - r["successes"], k) / comb(16, k) for r in items
                )
                / len(items)
                for k in (1, 4, 16)
            }
            stored = profile["panels"][role]
            assert all(
                abs(v - stored["mean_pass_at_k"][k]) < 1e-12 for k, v in pass_k.items()
            )
            result[role][arm] = {
                "items": len(items),
                "success_count_histogram_0_to_16": [histogram[k] for k in range(17)],
                "any_success": len(items) - histogram[0],
                "zero_success": histogram[0],
                "all_16_success": histogram[16],
                "recomputed_pass_at_k": pass_k,
                "recorded_panel": stored,
                "families": [r for r in profile["families"] if r["panel_role"] == role],
            }
        families = {
            a: {r["assigned_family_id"]: r for r in result[role][a]["families"]}
            for a in ARMS
        }
        assert families["R-S"].keys() == families["R-P"].keys()
        result[role]["R_P_above_family_count"] = sum(
            Fraction(families["R-P"][f]["successes"], families["R-P"][f]["trials"])
            > Fraction(families["R-S"][f]["successes"], families["R-S"][f]["trials"])
            for f in families["R-S"]
        )
    return result


def rebound(block):
    result = {}
    for role in ROLES:
        curve = block["arms"]["R-P"]["curves"][role]
        before, after = (curve["grid"].index(u) for u in (5, 10))
        pairs = [(r[before], r[after]) for r in curve["successes"]]
        result[role] = {
            "items": len(pairs),
            "any_success_u5_u10": [sum(p[j] > 0 for p in pairs) for j in (0, 1)],
            "correct_draws_u5_u10": [sum(p[j] for p in pairs) for j in (0, 1)],
            "more": sum(b > a for a, b in pairs),
            "fewer": sum(b < a for a, b in pairs),
            "unchanged": sum(b == a for a, b in pairs),
        }
    return result


def quadrature(block):
    grid = [0, 1, 2, 5, 10, 20]
    weights = [Fraction(0) for _ in grid]
    for i in range(len(grid) - 1):
        w = Fraction(grid[i + 1] - grid[i], 40)
        weights[i] += w
        weights[i + 1] += w
    curves = {a: block["arms"][a]["curves"]["targeted"] for a in ARMS}
    differences = []
    for u in grid:
        rates = {}
        for a, c in curves.items():
            i = c["grid"].index(u)
            rates[a] = Fraction(
                sum(r[i] for r in c["successes"]), sum(r[i] for r in c["trials"])
            )
        differences.append(rates["R-P"] - rates["R-S"])
    contributions = [w * d for w, d in zip(weights, differences)]
    total = sum(contributions)
    return {
        "grid": grid,
        "normalized_trapezoidal_weights": [float(w) for w in weights],
        "R_P_minus_R_S_point_contributions": [float(v) for v in contributions],
        "primary_absolute_auc_difference": float(total),
        "baseline_difference": float(differences[0]),
        "baseline_subtracted_auc_difference_descriptive_only": float(
            total - differences[0]
        ),
        "update_10_contribution": float(contributions[4]),
        "other_points_contribution": float(total - contributions[4]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/replay-v1/followup")
    )
    args = parser.parse_args()
    profiles = {a: read(args.root / f"profiles/seed-11-{a}.json") for a in ARMS}
    block = read(args.root / "readout.json")["blocks"]["11"]
    result = {
        "analysis_status": "retrospective descriptive breakdowns; no confirmation test or endpoint change",
        "acquisition": acquisition(read(args.root / "selection.json")),
        "coverage": coverage(profiles),
        "R_P_update_5_to_10": rebound(block),
        "targeted_auc_quadrature": quadrature(block),
    }
    target = args.root / "retrospective-count-breakdowns.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {target}; counts only, no new sampling or bootstrap.")


if __name__ == "__main__":
    main()
