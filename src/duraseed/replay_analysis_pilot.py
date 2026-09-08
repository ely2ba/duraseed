"""Existing Pilot evidence reproduction followed by the fixed replay audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from duraseed.pilot0_analysis import summarize_selected_method
from duraseed.provenance import canonical_json_value
from duraseed.replay_analysis import (
    BOOTSTRAP_NAMESPACE,
    BOOTSTRAP_REPLICATES,
    GRID,
    THRESHOLDS,
    attainment,
    auc,
    failure_summary,
    half_life,
    paired_interval,
)

METHODS = ("B-S", "B-G")


def read(path):
    return json.loads(path.read_text())


def old_module(repo, pair):
    path = repo / f"artifacts/pilot0-pair{pair}-offline-analysis/uncertainty.py"
    spec = importlib.util.spec_from_file_location(
        f"pilot{pair}_historical_uncertainty", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reproduce(repo, pair):
    """Rerun original estimands/settings, redirecting every write away from Pilot."""
    import numpy as np

    module = old_module(repo, pair)
    bank = read(repo / f"artifacts/pilot0-pair{pair}-readout/readout.json")
    module.ROOT = repo / "runs/pilot0" / bank["run_id"]
    old = read(module.OUT / "uncertainty.json")
    with TemporaryDirectory(prefix="duraseed-pilot-reproduction-") as temporary:
        module.OUT = Path(temporary)
        with redirect_stdout(io.StringIO()):
            module.main()
        rebuilt = read(module.OUT / "uncertainty.json")
    errors = [
        abs(a[k] - b[k])
        for a, b in zip(old["contrasts"], rebuilt["contrasts"], strict=True)
        for k in ("estimate", "ci95_lower", "ci95_upper")
    ]
    if max(errors) > 1e-12:
        raise ValueError(
            f"pair {pair}: original confidence interval discrepancy {max(errors)}"
        )
    terminal = read(module.ROOT / "result.json")
    cells = {cell["method"]: cell for cell in terminal["F1_F2_cells"]}
    for method in METHODS:
        monitor = tuple(
            read(module.evaluation(method, step, "a-monitor")) for step in GRID
        )
        maps = tuple(
            read(module.evaluation(method, step, "b-validation")) for step in GRID
        )
        before = read(module.evaluation(method, 0, "a-validation"))
        after = read(module.evaluation(method, 480, "a-validation"))
        rebuilt_cell = canonical_json_value(
            summarize_selected_method(
                seed=bank["seed"],
                method=method,
                stage_a_selected=before,
                stage_b_maps=maps,
                stage_b_retention=monitor,
                stage_b_final_retention=after,
            )
        )
        if rebuilt_cell != cells[method] or rebuilt_cell != bank["F1_F2_cells"][method]:
            raise ValueError(f"pair {pair}/{method}: stored F1/F2 reducer discrepancy")
        for i, value in enumerate(maps):
            counts = bank["derived"][method]["F2_counts"][i]
            if (
                sum(r["successes"] for r in value["item_counts"]) != counts["successes"]
                or sum(r["trials"] for r in value["item_counts"]) != counts["trials"]
            ):
                raise ValueError("raw MAPS headline counts differ")
        raw = np.array(
            [
                sum(r["successes"] for r in e["item_counts"]) / e["row_count"]
                for e in maps
            ]
        )
        if not np.allclose(
            raw,
            [
                r["1"]
                for r in cells[method]["F2_stage_b_learning"]["maps_pass_at_k_curve"]
            ],
            rtol=0,
            atol=1e-14,
        ):
            raise ValueError("stored raw MAPS trajectory differs")
    return module, {
        "pair": pair,
        "seed": bank["seed"],
        "run_id": bank["run_id"],
        "old_intervals_reproduced": len(errors) // 3,
        "max_absolute_interval_discrepancy": max(errors),
        "stored_F1_F2_cells_exact": True,
        "raw_MAPS_counts_match": True,
        "original_bootstrap": old["bootstrap"],
        "original_intervals": rebuilt["contrasts"],
    }


def raw_evaluation(module, method, step, panel, repo):
    result_path = module.evaluation(method, step, panel)
    result = read(result_path)
    directory = result_path.parent
    expected = {r["task_id"]: r for r in result["item_counts"]}
    rewards = {}
    with (directory / "rewards.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_id"] in rewards:
                raise ValueError("duplicate authoritative reward")
            rewards[row["sample_id"]] = row
    role_records, items, seen = defaultdict(list), defaultdict(list), set()
    with (directory / "generations.jsonl").open() as stream:
        for line in stream:
            gen = json.loads(line)
            sample, task = gen["sample_id"], gen["task_id"]
            reward = rewards[sample]
            check = reward["exact_verification"]
            if sample in seen or task not in expected or reward["task_id"] != task:
                raise ValueError("generation/reward/manifest join differs")
            seen.add(sample)
            if (
                gen["reward"] != reward["reward"]
                or reward["reward"] != check["reward"]
                or gen["task_manifest_id"] != result["manifest_id"]
                or gen["method"] != method
            ):
                raise ValueError("stored generation provenance or verification differs")
            compact = {
                "verification": check,
                "sampled_tokens": gen["sampled_tokens"],
                "sampling_max_tokens": gen["sampling_max_tokens"],
            }
            role = expected[task]["panel_role"]
            role_records[role].append(compact)
            items[task].append((gen["sample_index"], int(check["reward"] == 1)))
    if seen != set(rewards) or set(items) != set(expected):
        raise ValueError("raw evidence population is incomplete")
    compact_items = []
    for task, draws in sorted(items.items()):
        draws.sort()
        row = expected[task]
        if [d[0] for d in draws] != list(range(row["trials"])) or sum(
            d[1] for d in draws
        ) != row["successes"]:
            raise ValueError("per-item raw draws do not reproduce frozen counts")
        compact_items.append({**row, "draw_rewards": [d[1] for d in draws]})
    return (
        {role: failure_summary(rows) for role, rows in role_records.items()},
        compact_items,
        str(result_path.relative_to(repo)),
    )


def extend(module, pair, repo):
    import numpy as np

    curves, intervals, counts, failures, sources = {}, [], [], [], []
    for panel, roles in (
        ("a-monitor", ("targeted", "sentinel")),
        ("b-validation", ("stage-b",)),
    ):
        for role in roles:
            ids, raw, posterior, _ = module.paired_arrays(panel, role, np.array(GRID))
            name = "maps" if role == "stage-b" else role
            curves[name] = {
                method: {
                    "raw": raw[a].mean(axis=0).tolist(),
                    "posterior": posterior[a].mean(axis=0).tolist(),
                    "items": len(ids),
                }
                for a, method in enumerate(METHODS)
            }
            windows = (40, 480) if name == "maps" else (20,)
            delta = raw[1] - raw[0]
            for end in windows:
                n = GRID.index(end) + 1
                vector = auc(delta[:, :n], GRID[:n])
                key = f"pair{pair}-seed{11 if pair == 1 else 29}-{name}-absolute-AUC-0-{end}-B-G-minus-B-S"
                intervals.append(paired_interval(vector, key))
                if name == "maps":
                    intervals.append(
                        paired_interval(
                            vector - delta[:, 0],
                            key.replace("absolute-AUC", "gain-AUC"),
                        )
                    )
            if name == "maps":
                for i, label in ((0, "baseline"), (-1, "endpoint480")):
                    intervals.append(
                        paired_interval(
                            delta[:, i], f"pair{pair}-maps-{label}-B-G-minus-B-S"
                        )
                    )
    for method in METHODS:
        for panel in ("a-monitor", "b-validation"):
            for step in GRID:
                failure, records, source = raw_evaluation(
                    module, method, step, panel, repo
                )
                sources.append(source)
                counts.extend(
                    {
                        "pair": pair,
                        "method": method,
                        "panel": panel,
                        "update": step,
                        **r,
                    }
                    for r in records
                )
                if step <= 40:
                    failures.extend(
                        {
                            "pair": pair,
                            "method": method,
                            "panel": panel,
                            "role": role,
                            "update": step,
                            **value,
                        }
                        for role, value in failure.items()
                    )
    summaries, crossing_rows = {}, []
    for method in METHODS:
        maps = curves["maps"][method]["raw"]
        summaries[method] = {
            "maps_baseline": maps[0],
            "maps_endpoint480": maps[-1],
            "maps_absolute_auc": {},
            "maps_gain_auc": {},
            "retention": {},
        }
        for end in (40, 480):
            n = GRID.index(end) + 1
            area = float(auc(maps[:n], GRID[:n]))
            summaries[method]["maps_absolute_auc"][str(end)] = area
            summaries[method]["maps_gain_auc"][str(end)] = area - maps[0]
        for role in ("targeted", "sentinel"):
            score = curves[role][method]["raw"]
            area = float(auc(score[:6], GRID[:6]))
            summaries[method]["retention"][role] = {
                "baseline": score[0],
                "half_life": half_life(score),
                "absolute_auc_0_20": area,
                "relative_auc_0_20": area / score[0] if score[0] else None,
                "relative_curve": [v / score[0] for v in score] if score[0] else None,
            }
    for threshold in THRESHOLDS:
        crosses = {
            m: attainment(
                curves["maps"][m]["raw"], curves["targeted"][m]["raw"], threshold
            )
            for m in METHODS
        }
        eligible = all(c["status"] == "crossed" for c in crosses.values())
        crossing_rows.append(
            {
                "threshold": threshold,
                "arms": crosses,
                "eligible_two_arm_post_training_comparison": eligible,
                "retention_difference_B_G_minus_B_S": crosses["B-G"]["retention"]
                - crosses["B-S"]["retention"]
                if eligible
                else None,
            }
        )
    decomposition = []
    for end in (40, 480):
        s, g = summaries["B-S"], summaries["B-G"]
        absolute = g["maps_absolute_auc"][str(end)] - s["maps_absolute_auc"][str(end)]
        baseline = g["maps_baseline"] - s["maps_baseline"]
        gain = g["maps_gain_auc"][str(end)] - s["maps_gain_auc"][str(end)]
        decomposition.append(
            {
                "window": [0, end],
                "absolute_difference": absolute,
                "baseline_difference": baseline,
                "gain_difference": gain,
                "identity_residual": gain - (absolute - baseline),
            }
        )
    return {
        "pair": pair,
        "grid": list(GRID),
        "curves": curves,
        "summaries": summaries,
        "contrasts": intervals,
        "decomposition": decomposition,
        "attainment": crossing_rows,
        "failures": failures,
        "sources": sources,
    }, counts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/replay-v1/pilot-audit")
    )
    parser.add_argument("--reproduce-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    modules, reproduction = [], []
    for pair in (1, 2):
        module, result = reproduce(repo, pair)
        modules.append(module)
        reproduction.append(result)
        print(
            f"Pair {pair}: original F1/F2 cells/counts and {result['old_intervals_reproduced']} intervals reproduced; max error {result['max_absolute_interval_discrepancy']}",
            flush=True,
        )
    if args.reproduce_only:
        return
    output = args.output if args.output.is_absolute() else repo / args.output
    if output.exists():
        raise FileExistsError(
            "choose a new audit destination; historical output is immutable"
        )
    results, counts = [], []
    for pair, module in enumerate(modules, 1):
        result, records = extend(module, pair, repo)
        results.append(result)
        counts.extend(records)
        print(f"Pair {pair}: raw joins and fixed analyses complete", flush=True)
    from duraseed.replay_analysis_package import write_package

    data = {
        "kind": "existing-pilot-post-hoc-audit",
        "grid": list(GRID),
        "thresholds": list(THRESHOLDS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_namespace": BOOTSTRAP_NAMESPACE,
        "bootstrap_seed_derivation": "first eight SHA256 digest bytes, unsigned big endian",
        "reproduction": reproduction,
        "pairs": results,
        "unavailable": [],
        "interval_scope": "pointwise evaluation-item uncertainty conditional on fitted checkpoints, selected origins and realized draws; not training-seed uncertainty; no pooling",
        "optional_family_sensitivity": "not performed",
        "crossing_intervals": "point estimates only; no undefined/censored replicate filtering",
    }
    write_package(data, counts, output, repo=repo)
    print(f"Audit saved to {output}", flush=True)


if __name__ == "__main__":
    main()
