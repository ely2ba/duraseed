#!/usr/bin/env python3
"""Local-only endpoint-clone readout: repeated futures, fixed grids, no verdict."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from endpoint_clone_report import plot, profile_markdown

from duraseed.data.io import atomic_write_bytes
from duraseed.endpoint_continuation import DENSE_GRID, PANELS, RUN_STOPS, grids
from duraseed.endpoint_matching import trajectory_analysis
from duraseed.pilot0_evidence import read_evaluation
from duraseed.replay_analysis import auc, half_life, paired_interval
from duraseed.replay_inputs import read_json
from duraseed.replay_remote import write_json

NAMESPACE = "duraseed-endpoint-clone|continuation|"


def evaluation(root, label, step, panel, plan):
    point = root / "stage_b" / f"u{step}"
    checkpoint = read_json(point / "checkpoint.json")
    if any(
        checkpoint.get(k) != v
        for k, v in {
            "replay_arm": label,
            "seed": plan["seed"],
            "stage": "stage_b",
            "update": step,
        }.items()
    ):
        raise ValueError("continuation checkpoint identity differs")
    directory = point / panel
    identity = read_json(directory / "replay-identity.json")
    items, draws, cap = PANELS[panel]
    namespace = f"endpoint_clone.{label}.{panel}.{step}"
    expected = {
        "seed": plan["seed"],
        "replay_arm": label,
        "stage": "stage_b",
        "update": step,
        "purpose": panel,
        "draws": draws,
        "cap": cap,
        "sampler_path": checkpoint["sampler_path"],
        "origin_sampler_path": plan["origin"]["sampler_path"],
        "seed_namespace": namespace,
    }
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("evaluation identity or independent RNG namespace differs")
    result = read_evaluation(directory)
    if result is None:
        raise ValueError(f"required evaluation is incomplete: {label}/{step}/{panel}")
    if any(
        result.get(k) != v
        for k, v in {
            "manifest_id": identity["manifest_id"],
            "item_count": items,
            "row_count": items * draws,
            "samples_per_item": draws,
            "sampler_path": checkpoint["sampler_path"],
        }.items()
    ):
        raise ValueError("evaluation count or sampler differs from the fixed schedule")
    if any(
        result["coordinates"].get(k) != v
        for k, v in {
            "run_id": read_json(root / "config.json")["run_id"],
            "seed_namespace": namespace,
            "max_tokens": cap,
            "temperature": 1.0,
            "top_p": 0.95,
            "training_step": step,
            "origin_sampler_path": plan["origin"]["sampler_path"],
        }.items()
    ):
        raise ValueError("evaluation coordinates differ from continuation identity")
    return {**result, "update": step, "source": f"stage_b/u{step}/{panel}"}


def curve(evaluations, role, grid):
    rows = [
        {r["task_id"]: r for r in point["item_counts"] if r["panel_role"] == role}
        for point in evaluations
    ]
    ids = sorted(rows[0])
    if not ids or any(set(point) != set(ids) for point in rows):
        raise ValueError("trajectory requires the same paired items at every update")
    if [point["update"] for point in evaluations] != list(grid):
        raise ValueError("trajectory differs from the registered grid")
    if len({point["manifest_id"] for point in evaluations}) != 1:
        raise ValueError("trajectory changed its manifest")
    successes = np.array([[point[task]["successes"] for point in rows] for task in ids])
    trials = np.array([[point[task]["trials"] for point in rows] for task in ids])
    if np.any(trials <= 0) or np.any(trials != trials[0, 0]):
        raise ValueError("trajectory changed its per-item draw budget")
    raw = (successes / trials).mean(axis=0)
    result = {
        "grid": list(grid),
        "manifest_id": evaluations[0]["manifest_id"],
        "task_ids": ids,
        "successes": successes.tolist(),
        "trials": trials.tolist(),
        "raw": raw.tolist(),
        "own_baseline_gain": (raw - raw[0]).tolist(),
    }
    if role != "stage-b":
        result["half_life"] = half_life(raw, grid)
        result["raw_auc_0_20"] = float(auc(raw[:21], DENSE_GRID))
    elif grid[-1] == 480:
        result["summary"] = {"baseline": float(raw[0]), "endpoint480": float(raw[-1])}
        for end in (40, 480):
            n = list(grid).index(end) + 1
            absolute = float(auc(raw[:n], grid[:n]))
            result["summary"].update(
                {
                    f"absolute_auc_0_{end}": absolute,
                    f"gain_auc_0_{end}": absolute - float(raw[0]),
                }
            )
    return result


def contrast_intervals(runs):
    """Whole-item linear functionals; no bootstrap verdict for nonlinear distance."""

    def paired(role, labels, end=None):
        curves = [runs[label]["curves"][role] for label in labels]
        if any(
            c["task_ids"] != curves[0]["task_ids"]
            or c["manifest_id"] != curves[0]["manifest_id"]
            for c in curves
        ):
            raise ValueError("contrasts require shared evaluation items and manifest")
        arrays, time = [], None
        for c in curves:
            n = c["grid"].index(end) + 1 if end is not None else len(c["grid"])
            selected_time = c["grid"][:n]
            if time is not None and selected_time != time:
                raise ValueError("contrast grids differ")
            time = selected_time
            arrays.append(
                np.asarray(c["successes"])[:, :n] / np.asarray(c["trials"])[:, :n]
            )
        return arrays, time

    contrasts = {}
    for role in ("targeted", "sentinel"):
        arrays, grid = paired(role, ("T1", "T2", "S1", "S2"), 20)
        delta = (arrays[2] + arrays[3] - arrays[0] - arrays[1]) / 2
        key = f"{role}/mean-student-minus-mean-teacher/signed-AUC-0-20"
        contrasts[key] = paired_interval(auc(delta, grid), key, namespace=NAMESPACE)
    (teacher, student), grid = paired("maps", ("T1", "S1"))
    delta = student - teacher
    functionals = {"baseline": delta[:, 0], "endpoint480": delta[:, -1]}
    for end in (40, 480):
        n = grid.index(end) + 1
        absolute = auc(delta[:, :n], grid[:n])
        functionals[f"absolute-AUC-0-{end}"] = absolute
        functionals[f"gain-AUC-0-{end}"] = absolute - delta[:, 0]
    for name, values in functionals.items():
        key = f"maps/S1-minus-T1/{name}"
        contrasts[key] = paired_interval(values, key, namespace=NAMESPACE)
    return contrasts


def reduce(package_root):
    package_root, runs, dense = Path(package_root), {}, {}
    terminal_package = read_json(package_root / "result.json")
    matching = read_json(package_root / "matching.json")
    profiles = {
        p.parent.name: read_json(p)
        for p in sorted((package_root / "acquisition/profiles").glob("*/profile.json"))
    }
    if terminal_package["status"] == "CLONE_UNAVAILABLE":
        if terminal_package.get("stage_b_started") or matching.get("stage_b_allowed"):
            raise ValueError("unavailable clone has inconsistent continuation status")
        if (package_root / "continuations").exists():
            raise ValueError("unavailable clone contains continuation artifacts")
        return {
            "run_id": package_root.name,
            "status": "CLONE_UNAVAILABLE",
            "matching": matching,
            "profiles": profiles,
            "billing": terminal_package["acquisition_billing"],
            "runs": {},
        }
    if terminal_package["status"] != "COMPLETED" or not matching["stage_b_allowed"]:
        raise ValueError("analysis requires completed evidence or unavailable cloning")
    for label, stop in RUN_STOPS.items():
        root = package_root / "continuations" / label
        plan, terminal = (
            read_json(root / "continuation.json"),
            read_json(root / "result.json"),
        )
        if terminal.get("status") != "COMPLETED" or terminal.get("updates") != stop:
            raise ValueError("all four predetermined continuations must be complete")
        if (
            plan["label"] != label
            or plan["stop"] != stop
            or not plan["fresh_optimizer"]
        ):
            raise ValueError("continuation plan differs from the approved matrix")
        states = list(root.rglob("remote-call-state.json"))
        if not states or any(read_json(p).get("pending") is not None for p in states):
            raise ValueError("missing journal or pending remote request")
        billing = read_json(root / "billing.json")
        if terminal["billing"] != billing:
            raise ValueError("terminal billing differs from durable ledger")
        panels = {
            panel: [evaluation(root, label, u, panel, plan) for u in updates]
            for panel, updates in grids(stop).items()
        }
        dense[label] = panels["a_monitor"][:21]
        runs[label] = {
            "origin": plan["origin"],
            "stop": stop,
            "billing": billing,
            "curves": {
                role: curve(panels[panel], role_name, grids(stop)[panel])
                for role, panel, role_name in (
                    ("targeted", "a_monitor", "targeted"),
                    ("sentinel", "a_monitor", "sentinel"),
                    ("maps", "b_validation", "stage-b"),
                )
            },
            "final_validation": panels["a_validation"],
        }
    for first, second in (("T1", "T2"), ("S1", "S2")):
        if runs[first]["origin"] != runs[second]["origin"]:
            raise ValueError("repeated continuations changed their selected origin")
    result = {
        "run_id": package_root.name,
        "status": "COMPLETED",
        "runs": runs,
        "profiles": profiles,
        "trajectory_comparison": trajectory_analysis(dense),
        "signed_contrasts": contrast_intervals(runs),
        "definitions": {
            "primary": "Targeted 0–20 trapezoidal mean absolute raw-Pass@1 distance; four cross-checkpoint distances minus mean of two repeat distances. Negative estimates retained. No equivalence threshold or binary verdict.",
            "bootstrap": "Paired whole-item trajectories, independent item clusters conditional on these checkpoints, continuation realizations, families, and observed draws. 50,000 percentile resamples for signed linear contrasts only; not training-seed uncertainty.",
            "half_life": "First downward crossing of half of own update-0 raw Pass@1, linearly interpolated. No crossing is right-censored; zero baseline undefined.",
        },
    }
    for filename in ("matching.json", "selection.json"):
        if (package_root / filename).exists():
            result[filename.removesuffix(".json")] = read_json(package_root / filename)
    return result


def markdown(value):
    if value["status"] == "CLONE_UNAVAILABLE":
        return "\n".join(
            [
                "# Endpoint-clone results",
                "",
                "No Stage-B continuation was started.",
                "",
                *profile_markdown(value),
            ]
        )
    rows = [
        "# Endpoint-clone continuation results",
        "",
        value["definitions"]["primary"],
        "",
    ]
    rows += profile_markdown(value)
    for role, comparison in value["trajectory_comparison"]["panels"].items():
        rows += [
            f"## {role.title()} trajectories",
            "",
            "| Distance | Raw Pass@1 |",
            "|---|---:|",
            *[f"| {k} | {v:.6f} |" for k, v in comparison["distances"].items()],
            f"| Mean cross-checkpoint distance | {comparison['mean_cross_distance']:.6f} |",
            f"| Mean repeat distance | {comparison['mean_repeat_distance']:.6f} |",
            f"| Excess separation | {comparison['excess_separation']:.6f} |",
            "",
            "| Update | T1 | T2 | S1 | S2 |",
            "|---:|---:|---:|---:|---:|",
        ]
        rows += [
            f"| {u} | "
            + " | ".join(f"{comparison['curves'][label][u]:.6f}" for label in RUN_STOPS)
            + " |"
            for u in DENSE_GRID
        ]
        rows += ["", "| Run | Signed AUC 0–20 | First half-life |", "|---|---:|---|"]
        for label, run in value["runs"].items():
            c = run["curves"][role]
            crossing = c["half_life"]
            half = (
                f"{crossing['update']:.6f}"
                if crossing["status"] == "crossed"
                else crossing["status"]
            )
            rows += [f"| {label} | {c['raw_auc_0_20']:.6f} | {half} |"]
        rows += [""]
    rows += [
        "## MAPS absolute performance and baseline-relative gain",
        "",
        "| Run | Update | Raw Pass@1 | Gain |",
        "|---|---:|---:|---:|",
    ]
    for label, run in value["runs"].items():
        c = run["curves"]["maps"]
        rows += [
            f"| {label} | {u} | {p:.6f} | {g:.6f} |"
            for u, p, g in zip(c["grid"], c["raw"], c["own_baseline_gain"], strict=True)
        ]
    rows += [
        "",
        "| Run | Gain AUC 0–40 | Gain AUC 0–480 | Endpoint |",
        "|---|---:|---:|---:|",
    ]
    for label in ("T1", "S1"):
        s = value["runs"][label]["curves"]["maps"]["summary"]
        rows += [
            f"| {label} | {s['gain_auc_0_40']:.6f} | {s['gain_auc_0_480']:.6f} | {s['endpoint480']:.6f} |"
        ]
    rows += [
        "",
        "## Paired item uncertainty",
        "",
        value["definitions"]["bootstrap"],
        "",
        "| Contrast | Estimate | 95% interval |",
        "|---|---:|---|",
    ]
    rows += [
        f"| {k} | {v['estimate']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |"
        for k, v in value["signed_contrasts"].items()
    ]
    rows += [
        "",
        value["definitions"]["half_life"],
        "",
        "Complete item counts, late retention points, endpoint profiles, selection records, and run-local billing are in `readout.json`.",
        "",
    ]
    return "\n".join(rows)


def analyze(root, output, geometry=None):
    value = reduce(root)
    if geometry is not None:
        manifest = read_json(Path(geometry) / "archive-manifest.json")
        value["adapter_geometry"] = [
            {key: row[key] for key in ("arm", "update", "selected", "aggregate")}
            for row in manifest["checkpoints"]
        ]
    output = Path(output)
    write_json(output / "readout.json", value)
    atomic_write_bytes(output / "readout.md", markdown(value).encode())
    if value["runs"]:
        plot(value, output)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, help="Local adapter archive directory")
    args = parser.parse_args()
    analyze(args.run, args.output, args.geometry)
