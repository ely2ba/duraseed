"""Offline descriptive reduction of new replay-v1 evidence, never a decision path."""

from __future__ import annotations

import json
from pathlib import Path

from duraseed.pilot0_evidence import read_evaluation
from duraseed.replay_analysis import (
    BOOTSTRAP_REPLICATES,
    GRID,
    THRESHOLDS,
    attainment,
    auc,
    failure_summary,
    half_life,
    paired_interval,
)
from duraseed.replay_inputs import read_json
from duraseed.replay_matching import ARMS
from duraseed.replay_remote import write_json

BOOTSTRAP_NAMESPACE = "duraseed-replay-v1|followup|"
PANELS = {
    "a_monitor": (384, 4, 4096),
    "b_validation": (512, 16, 128),
    "a_validation": (512, 16, 4096),
}


def _evaluation(root, seed, arm, step, panel, selected):
    """Authenticate legacy evidence plus its explicit new replay arm identity."""
    base = root / f"seed-{seed}" / arm
    point = base / "pre-b" if step == 0 else base / "stage_b" / f"u{step}"
    directory = point / panel
    initial = read_json(base / "stage_a" / f"u{selected}" / "checkpoint.json")
    checkpoint = initial if step == 0 else read_json(point / "checkpoint.json")
    expected = {
        "seed": seed,
        "replay_arm": arm,
        "update": step or selected,
        "stage": "stage_a" if step == 0 else "stage_b",
    }
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError("replay checkpoint identity differs")
    items, draws, cap = PANELS[panel]
    identity = read_json(directory / "replay-identity.json")
    expected = {
        "seed": seed,
        "replay_arm": arm,
        "stage": "stage_b",
        "update": step,
        "purpose": panel,
        "draws": draws,
        "cap": cap,
        "sampler_path": checkpoint["sampler_path"],
        "origin_sampler_path": initial["sampler_path"],
        "seed_namespace": f"replay-v1.stage_b.{panel}.{step}",
    }
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("replay evaluation identity differs")
    result = read_evaluation(directory)
    if result is None:
        raise ValueError(f"missing completed replay evaluation: {directory}")
    label = f"replay-v1-seed{seed}-{arm}-stage_b-{step}-{panel}"
    coords = result["coordinates"]
    if (
        result["manifest_id"] != identity["manifest_id"]
        or result["sampler_path"] != checkpoint["sampler_path"]
        or result["item_count"] != items
        or result["samples_per_item"] != draws
        or result["label"] != label
        or any(
            coords.get(k) != v
            for k, v in {
                "run_id": root.name,
                "label": label,
                "method": None,
                "seed": seed,
                "checkpoint_stage": "stage_b",
                "training_step": step,
                "origin_sampler_path": initial["sampler_path"],
                "seed_namespace": expected["seed_namespace"],
                "max_tokens": cap,
                "temperature": 1.0,
                "top_p": 0.95,
            }.items()
        )
    ):
        raise ValueError("replay evaluation coordinates or frozen sample counts differ")
    rewards = {row["sample_id"]: row for row in _rows(directory / "rewards.jsonl")}
    failures = {}
    for gen in _rows(directory / "generations.jsonl"):
        if any(
            gen.get(k) != v
            for k, v in {
                "run_id": root.name,
                "method": None,
                "seed": seed,
                "checkpoint_stage": "stage_b",
                "training_step": step,
                "sampler_checkpoint_path": checkpoint["sampler_path"],
                "origin_sampler_checkpoint_path": initial["sampler_path"],
                "task_manifest_id": result["manifest_id"],
                "purpose": "evaluation",
                "sampling_max_tokens": cap,
                "sampling_temperature": 1.0,
                "sampling_top_p": 0.95,
            }.items()
        ):
            raise ValueError("replay generation identity differs")
        check = rewards[gen["sample_id"]]["exact_verification"]
        if check["reward"] != gen["reward"]:
            raise ValueError("authoritative verification differs from stored reward")
        if step <= 40:
            failures.setdefault(gen["panel_role"], []).append(
                {
                    "verification": check,
                    "sampled_tokens": gen["sampled_tokens"],
                    "sampling_max_tokens": cap,
                }
            )
    return {
        "result": result,
        "source": str(directory.relative_to(root)),
        "failures": {role: failure_summary(rows) for role, rows in failures.items()},
    }


def _rows(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def _curve(evaluations, role, grid=GRID):
    import numpy as np

    rows = [
        {
            row["task_id"]: row
            for row in e["result"]["item_counts"]
            if row["panel_role"] == role
        }
        for e in evaluations
    ]
    ids = sorted(rows[0])
    if not ids or any(set(point) != set(ids) for point in rows):
        raise ValueError("replay trajectory changed its item population")
    if len({e["result"]["manifest_id"] for e in evaluations}) != 1:
        raise ValueError("replay trajectory changed its task manifest")
    successes = np.array([[point[task]["successes"] for point in rows] for task in ids])
    trials = np.array([[point[task]["trials"] for point in rows] for task in ids])
    if np.any(trials <= 0) or np.any(trials != trials[0, 0]):
        raise ValueError("replay trajectory changed its per-item draw budget")
    raw = successes / trials
    mean = raw.mean(axis=0)
    return {
        "grid": list(grid),
        "manifest_id": evaluations[0]["result"]["manifest_id"],
        "task_ids": ids,
        "items": len(ids),
        "successes": successes.tolist(),
        "trials": trials.tolist(),
        "raw": mean.tolist(),
        "posterior": ((successes + 0.5) / (trials + 1)).mean(axis=0).tolist(),
        "own_baseline_gain": (mean - mean[0]).tolist(),
        "relative_retention": (mean / mean[0]).tolist()
        if mean[0] > 0
        else [None] * len(grid),
        "relative_retention_status": "defined"
        if mean[0] > 0
        else "undefined_zero_baseline",
        "half_life": half_life(mean, grid),
        "sources": [e["source"] for e in evaluations],
    }


def _arm(root, seed, arm, selected):
    panels = {
        name: [_evaluation(root, seed, arm, step, name, selected) for step in GRID]
        for name in ("a_monitor", "b_validation")
    }
    curves = {
        role: _curve(panels[panel], role)
        for panel, role in (
            ("a_monitor", "targeted"),
            ("a_monitor", "sentinel"),
            ("b_validation", "stage-b"),
        )
    }
    validation = [
        _evaluation(root, seed, arm, step, "a_validation", selected)
        for step in (0, 480)
    ]
    profiles = root / f"seed-{seed}" / arm / "pre-b" / "profile.json"
    profile = read_json(profiles)
    if (
        profile.get("replay_arm") != arm
        or profile.get("seed") != seed
        or profile.get("stage_a_update") != selected
    ):
        raise ValueError("F3 profile identity differs")
    downstream = curves["stage-b"]["raw"]
    return {
        "selected_update": selected,
        "curves": curves,
        "validation": {
            role: _curve(validation, role, (0, 480))
            for role in ("targeted", "sentinel")
        },
        "F3_profile": str(profiles.relative_to(root)),
        "failures": {
            panel: [
                {"update": step, **e["failures"]}
                for step, e in zip(GRID, values, strict=True)
                if step <= 40
            ]
            for panel, values in panels.items()
        },
        "first_attainment": {
            str(threshold): {
                role: attainment(downstream, curves[role]["raw"], threshold)
                for role in ("targeted", "sentinel")
            }
            for threshold in THRESHOLDS
        },
        "summary": {
            "targeted_raw_retention_auc_0_20": float(
                auc(curves["targeted"]["raw"][:6], GRID[:6])
            ),
            "maps_raw_absolute_auc_0_480": float(auc(downstream)),
            "maps_raw_endpoint": downstream[-1],
            "maps_raw_baseline": downstream[0],
            "maps_raw_absolute_auc_0_40": float(auc(downstream[:7], GRID[:7])),
            "maps_raw_gain_auc_0_40": float(
                auc(downstream[:7], GRID[:7]) - downstream[0]
            ),
        },
    }


def _contrasts(seed, arms):
    import numpy as np

    results = {}
    for role, windows in (("targeted", (20,)), ("stage-b", (40, 480))):
        left, right = (arms[arm]["curves"][role] for arm in ARMS)
        keys = ("task_ids", "trials", "manifest_id")
        if any(left[key] != right[key] for key in keys):
            raise ValueError(
                "replay contrast requires paired items and equal draw budgets"
            )
        if left["grid"] != right["grid"]:
            raise ValueError("replay contrast changed its evaluation grid")
        delta = np.asarray(right["successes"]) / np.asarray(
            right["trials"]
        ) - np.asarray(left["successes"]) / np.asarray(left["trials"])
        functionals = {}
        for end in windows:
            n = GRID.index(end) + 1
            vector = auc(delta[:, :n], GRID[:n])
            functionals[f"raw-absolute-AUC-0-{end}"] = vector
            if role == "stage-b" and end == 40:
                functionals["raw-gain-AUC-0-40"] = vector - delta[:, 0]
        if role == "stage-b":
            functionals.update(
                {"raw-baseline": delta[:, 0], "raw-endpoint480": delta[:, -1]}
            )
        for metric, vector in functionals.items():
            name = f"seed-{seed}|{role}|{metric}|R-P-minus-R-S"
            results[f"{role}/{metric}"] = paired_interval(
                vector, name, namespace=BOOTSTRAP_NAMESPACE
            )
    return results


def reduce_results(root: Path) -> dict:
    """Read only a completed follow-up package; never inspect original Pilot outputs."""
    root = Path(root)
    terminal, matching = (
        read_json(root / "result.json"),
        read_json(root / "matching.json"),
    )
    if (
        terminal.get("status") not in ("COMPLETED", "NO_MATCH")
        or terminal.get("matching") != matching
    ):
        raise ValueError("replay package is not terminal with durable matching")
    if set(matching) != {"11", "29"}:
        raise ValueError("replay package must retain both source blocks")
    blocks = {}
    for seed in (11, 29):
        selection = matching[str(seed)]
        if selection["selected"] is None:
            if selection["status"] != "NO_MATCH" or selection["stage_b_allowed"]:
                raise ValueError("inconsistent unavailable matching")
            for arm in ARMS:
                base = root / f"seed-{seed}" / arm
                if (base / "pre-b").exists() or (base / "stage_b").exists():
                    raise ValueError("unmatched replay block contains Stage-B evidence")
            blocks[str(seed)] = {
                "status": "NO_MATCH",
                "matching": selection,
                "arms": {},
                "contrasts": {},
            }
            continue
        if selection["status"] != "MATCHED" or not selection["stage_b_allowed"]:
            raise ValueError("inconsistent available matching")
        arms = {arm: _arm(root, seed, arm, selection["selected"][arm]) for arm in ARMS}
        contrasts = _contrasts(seed, arms)
        estimates = {name: row["estimate"] for name, row in contrasts.items()}
        blocks[str(seed)] = {
            "status": "COMPLETED",
            "matching": selection,
            "arms": arms,
            "contrasts": contrasts,
            "gain_auc_decomposition": {
                "absolute_auc_difference": estimates["stage-b/raw-absolute-AUC-0-40"],
                "baseline_difference": estimates["stage-b/raw-baseline"],
                "gain_auc_difference": estimates["stage-b/raw-gain-AUC-0-40"],
                "identity": "gain AUC difference = absolute AUC difference - baseline difference",
            },
        }
    return {
        "namespace": "replay-v1",
        "run_id": root.name,
        "grid": list(GRID),
        "blocks": blocks,
        "bootstrap": {
            "namespace": BOOTSTRAP_NAMESPACE,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_rule": "first 8 SHA256 bytes of namespace + contrast, unsigned big endian; NumPy PCG64",
            "contrast_direction": "R-P-minus-R-S",
            "scope": "paired item trajectories, retaining all within-item draws together; descriptive pointwise intervals conditional on these fixed source blocks, not training-seed intervals",
        },
        "raw_definition": "equal-item observed successes/trials; unconditional exact-verifier success",
        "posterior_definition": "equal-item Jeffreys mean (successes+0.5)/(trials+1), separate from raw estimands",
    }


def markdown(result):
    lines = [
        "# Replay-v1 descriptive readout",
        "",
        f"Run: `{result['run_id']}`.",
        "",
        "Contrasts are R-P minus R-S. Intervals are 50,000-resample paired-item trajectory 95% percentile intervals, conditional on each fixed source block. Raw rates and Jeffreys posterior means are reported separately.",
    ]
    for seed, block in result["blocks"].items():
        lines += ["", f"## Source block {seed}: {block['status']}", ""]
        if block["status"] == "NO_MATCH":
            lines += ["Matching unavailable; no Stage-B evidence collected."]
            continue
        lines += ["| Quantity | R-S | R-P |", "|---|---:|---:|"]
        for metric in ("selected_update", *block["arms"]["R-S"]["summary"]):
            values = [
                data[metric] if metric == "selected_update" else data["summary"][metric]
                for data in block["arms"].values()
            ]
            lines.append(f"| {metric} | {values[0]:.8g} | {values[1]:.8g} |")
        lines += ["", "| Contrast | Difference | 95% interval |", "|---|---:|---| "]
        for metric, value in block["contrasts"].items():
            lines.append(
                f"| {metric} | {value['estimate']:.8g} | [{value['ci95'][0]:.8g}, {value['ci95'][1]:.8g}] |"
            )
        d = block["gain_auc_decomposition"]
        lines += [
            "",
            f"Early gain-AUC difference {d['gain_auc_difference']:.8g} = absolute-AUC difference {d['absolute_auc_difference']:.8g} − baseline difference {d['baseline_difference']:.8g}.",
        ]
        for role in ("targeted", "sentinel", "stage-b"):
            lines += [
                "",
                f"### {role} trajectories",
                "",
                "| Update | R-S raw | R-P raw | R-S posterior | R-P posterior |",
                "|---:|---:|---:|---:|---:|",
            ]
            rs, rp = (block["arms"][arm]["curves"][role] for arm in ARMS)
            for i, step in enumerate(GRID):
                lines.append(
                    f"| {step} | {rs['raw'][i]:.8g} | {rp['raw'][i]:.8g} | {rs['posterior'][i]:.8g} | {rp['posterior'][i]:.8g} |"
                )
        for arm, data in block["arms"].items():
            lines += ["", f"{arm} F3: [{data['F3_profile']}](../{data['F3_profile']})."]
        lines += [
            "",
            "The JSON companion contains per-item counts, own-baseline curves, half-lives, fixed first-attainment rows, early authoritative failure summaries, and final validation counts.",
        ]
    return "\n".join(lines) + "\n"


def write_report(root: Path, output: Path | None = None) -> dict:
    result = reduce_results(root)
    output = Path(output) if output is not None else Path(root) / "analysis"
    write_json(output / "readout.json", result)
    (output / "readout.md").write_text(markdown(result))
    return result
