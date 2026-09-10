#!/usr/bin/env python3
"""Assemble completed computational checks for a separate manuscript writer."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from safetensors import safe_open
import torch

from duraseed.pilot0_evidence import read_evaluation
from duraseed.replay_analysis import GRID, auc, failure_summary, paired_interval
from duraseed.replay_inputs import read_json
from duraseed.replay_remote import write_json
from recheck_replay_u10 import POINT, SOURCE


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest_evaluation(directory):
    result = read_evaluation(directory)
    if result is None or result["row_count"] != 1536:
        raise ValueError("full recheck must finish before the writing handoff")
    rewards = {
        r["sample_id"]: r["exact_verification"]
        for r in rows(directory / "rewards.jsonl")
    }
    generations = rows(directory / "generations.jsonl")
    by_role = {}
    for role in ("targeted", "sentinel"):
        chosen = [r for r in generations if r["panel_role"] == role]
        summary = failure_summary(
            [
                {
                    "verification": rewards[r["sample_id"]],
                    "sampled_tokens": r["sampled_tokens"],
                    "sampling_max_tokens": r["sampling_max_tokens"],
                }
                for r in chosen
            ]
        )
        summary["successes"] = sum(int(r["reward"]) for r in chosen)
        summary["stop_reasons"] = dict(Counter(r["stop_reason"] for r in chosen))
        by_role[role] = summary
    return result, generations, by_role


def tensor_difference(archive):
    before = next(
        (archive / "seed-11-R-P-stage_a-u20").rglob("adapter_model.safetensors")
    )
    after = next(
        (archive / "seed-11-R-P-stage_b-u10").rglob("adapter_model.safetensors")
    )
    changed, squares, maxima = Counter(), Counter(), Counter()
    torch.set_num_threads(2)
    with (
        safe_open(str(before), framework="pt") as a,
        safe_open(str(after), framework="pt") as b,
    ):
        assert set(a.keys()) == set(b.keys())
        for key in a.keys():
            factor = (
                "A"
                if key.endswith(".lora_A.weight")
                else "B"
                if key.endswith(".lora_B.weight")
                else None
            )
            if factor is None:
                continue
            delta = a.get_tensor(key).double() - b.get_tensor(key).double()
            changed[factor] += int(torch.count_nonzero(delta))
            squares[factor] += float(torch.sum(delta * delta))
            maxima[factor] = max(maxima[factor], float(delta.abs().max()))
    return {
        "comparison": "stored R-P Stage-B u10 sampler versus stored R-P pre-B u20 sampler",
        "changed_elements": dict(changed),
        "delta_frobenius": {k: math.sqrt(v) for k, v in squares.items()},
        "maximum_absolute_element_difference": dict(maxima),
        "scope": "identifies differences between downloaded stored adapter tensors; does not attest historical serving",
    }


def summarize(repo, recheck, output, archive):
    old, old_gen, old_stats = digest_evaluation(repo / SOURCE / POINT / "a_monitor")
    new, new_gen, new_stats = digest_evaluation(recheck / "a_monitor")
    old_by_key = {(r["task_id"], r["sample_index"]): r for r in old_gen}
    equality = Counter()
    for row in new_gen:
        original = old_by_key[row["task_id"], row["sample_index"]]
        for key in (
            "sampling_seed",
            "prompt_text",
            "prompt_tokens",
            "sampling_temperature",
            "sampling_top_p",
            "sampling_max_tokens",
            "sampler_checkpoint_path",
        ):
            assert original[key] == row[key], key
        equality["token_identical_completions"] += (
            original["completion_token_ids"] == row["completion_token_ids"]
        )
        equality["text_identical_completions"] += (
            original["completion_text"] == row["completion_text"]
        )
        equality["same_correctness"] += original["reward"] == row["reward"]
    readout = read_json(repo / "artifacts/replay-v1/followup/readout.json")
    contrasts, sensitivity = {}, {}
    for role in ("targeted", "sentinel"):
        old_items = {
            r["task_id"]: r for r in old["item_counts"] if r["panel_role"] == role
        }
        new_items = {
            r["task_id"]: r for r in new["item_counts"] if r["panel_role"] == role
        }
        assert set(old_items) == set(new_items)
        delta = [
            (new_items[k]["successes"] - old_items[k]["successes"]) / 4
            for k in sorted(old_items)
        ]
        contrasts[role] = paired_interval(
            delta,
            f"u10-recheck-{role}-new-minus-original",
            namespace="duraseed-publication-checks-20260909|",
        )
        curves = readout["blocks"]["11"]["arms"]
        original_curve = curves["R-P"]["curves"][role]["raw"][:6]
        recheck_curve = list(original_curve)
        recheck_curve[4] = new_stats[role]["unconditional_success"]
        solver_auc = float(auc(curves["R-S"]["curves"][role]["raw"][:6], GRID[:6]))
        sensitivity[role] = {
            "original_R_S_auc": solver_auc,
            "original_R_P_auc": float(auc(original_curve, GRID[:6])),
            "recheck_u10_only_R_P_auc": float(auc(recheck_curve, GRID[:6])),
            "recheck_u10_only_R_P_minus_R_S": float(auc(recheck_curve, GRID[:6]))
            - solver_auc,
            "u10_weight": 0.375,
            "status": "post-hoc single-point sensitivity only; original primary AUC is preserved and not replaced",
        }
    value = {
        "date": "2026-09-09",
        "source": str(SOURCE / POINT / "a_monitor"),
        "recheck": str(recheck.relative_to(repo)),
        "rows": 1536,
        "matched_prompt_and_seed_records": 1536,
        "original": old_stats,
        "recheck_results": new_stats,
        "equality": dict(equality),
        "paired_recheck_minus_original": contrasts,
        "auc_sensitivity": sensitivity,
        "stored_adapter_difference": tensor_difference(archive),
        "billing": read_json(recheck / "billing.json"),
    }
    write_json(output / "recheck-comparison.json", value)
    uncertainty = read_json(output / "half-life-uncertainty.json")
    geometry = read_json(output / "geometry.json")
    lines = [
        "# DuraSeed computational handoff — 9 September 2026",
        "",
        "Completed computations only. The paper, frozen protocol, selection records, and original observations have not been edited. No new training was performed.",
        "",
        "R-S and R-P are the shared-prompt SFT replay arms using solver traces and archived correct policy traces, respectively. B-S and B-G are the original Pilot SFT and RL procedures.",
        "",
        "## Half-life uncertainty",
        "",
        uncertainty["bootstrap"],
        "",
        uncertainty["scope"],
        "",
        "Half-life uses each resampled arm's own update-0 raw Pass@1 and the existing first-downward-crossing definition. Interpolation is unchanged. All 50,000 replicates were finite in every reported comparison; none were discarded.",
        "",
    ]

    def number(v):
        return f"{v:.6f}"

    def estimate(v):
        lo, hi = v["ci95"]
        return f"{number(v['estimate'])} [{number(lo)}, {number(hi)}]"

    for role in ("targeted", "sentinel"):
        lines += [
            f"### {role.title()}",
            "",
            "| Comparison | First arm: updates [95% CI] | Second arm: updates [95% CI] | Second − first [95% CI] |",
            "|---|---:|---:|---:|",
        ]
        for r in uncertainty["results"]:
            if r["label"].endswith(role):
                names = (
                    ("R-S", "R-P")
                    if r["label"].startswith("replay")
                    else ("B-S", "B-G")
                )
                lines += [
                    f"| {r['label']} ({names[0]}, {names[1]}) | {estimate(r['arms'][names[0]])} | {estimate(r['arms'][names[1]])} | {estimate(r['difference'])} |"
                ]
        lines += [""]
    lines += [
        "## Update-10 re-evaluation",
        "",
        "Same saved sampler path, 384 monitor items, four draws per item, recorded seeds, role-colon rendering, temperature 1, top-p 0.95, and 4,096-token cap. A new client session was used. Both roles contain 768 completions.",
        "",
        "| Role | Original correct / 768 | Recheck correct / 768 | Recheck − original Pass@1 [paired item-bootstrap 95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for role in ("targeted", "sentinel"):
        lines += [
            f"| {role} | {old_stats[role]['successes']} | {new_stats[role]['successes']} | {estimate(contrasts[role])} |"
        ]
    lines += [
        "",
        "| Role / observation | Mean tokens | Median tokens | Cap reached / 768 | Valid output / 768 |",
        "|---|---:|---:|---:|---:|",
    ]
    for role in ("targeted", "sentinel"):
        for label, stats in (("original", old_stats), ("recheck", new_stats)):
            r = stats[role]
            lines += [
                f"| {role} / {label} | {r['tokens_mean']:.3f} | {r['tokens_median']} | {r['overlapping_indicators']['cap_reached']} | {r['overlapping_indicators']['output_valid']} |"
            ]
    lines += [
        "",
        "Valid output means valid answer tag, syntax, and lexing together. Invalid outputs remain failures. Full failure-code counts are in `recheck-comparison.json`.",
        "",
        f"Matched prompts/settings/seeds: 1,536/1,536. Identical completion token sequences: {equality['token_identical_completions']}/1,536; identical completion text: {equality['text_identical_completions']}/1,536; identical correctness labels: {equality['same_correctness']}/1,536.",
        "",
        "### Single-point AUC sensitivity (not a replacement primary result)",
        "",
        "| Role | Original R-S AUC | Original R-P AUC | R-P AUC using recheck at u10 only | Difference from R-S |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, r in sensitivity.items():
        lines += [
            f"| {role} | {number(r['original_R_S_auc'])} | {number(r['original_R_P_auc'])} | {number(r['recheck_u10_only_R_P_auc'])} | {number(r['recheck_u10_only_R_P_minus_R_S'])} |"
        ]
    lines += [
        "",
        "The existing 0–20 grid and raw-Pass@1 normalization are unchanged; u10 has weight 0.375. This explicitly post-hoc calculation changes only the u10 observation. The original registered results remain available and unchanged.",
        "",
        "## Replay adapter geometry",
        "",
        "Existing Pilot geometry reducer, 249 adapted modules per checkpoint, rank 32 and alpha 32 (scale 1). These are the full saved LoRA factors, with no M0 subtraction. Frobenius norms aggregate squared module norms; BA spectra are block-diagonal unions of module spectra, not a full-network operator.",
        "",
        "| Checkpoint | ‖A‖F | ‖B‖F | ‖BA‖F | σ1 | Stable rank | Entropy effective rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in geometry["checkpoints"]:
        a = r["aggregate"]
        keys = "A_frobenius B_frobenius BA_frobenius sigma1 stable_rank entropy_effective_rank".split()
        cells = " | ".join(number(a[k]) for k in keys)
        lines += [f"| {r['name']} | {cells} |"]
    lines += [
        "",
        f"Selected replay R-S/R-P B-factor norm ratio: {geometry['selected_R_S_over_R_P_B_norm']:.6f}. Per-module/per-layer norms and spectra are in `geometry.json`. Downloaded adapters are retained locally.",
        "",
        "The downloaded stored u10 adapter differs from the stored selected R-P adapter; parameter differences are in `recheck-comparison.json`. Download metadata and current tensor identity do not retrospectively prove which weights a historical request served.",
        "",
        "## Usage and files",
        "",
        f"Recheck: {value['billing']['observed']['prefill']:,} observed prefill tokens, {value['billing']['observed']['sample']:,} sampled tokens, zero training tokens. Local token-priced cost: ${value['billing']['observed_token_cost_plus_storage_reservation_usd']:.8f}. This is not a settled provider invoice; checkpoint download/analysis generated no new model tokens or retained remote checkpoints.",
        "",
        "Machine-readable outputs: `half-life-uncertainty.json`, `geometry.json`, `recheck-comparison.json`. All confidence intervals here describe evaluation-item uncertainty, not replication across training seeds.",
        "",
        "Reproduction from the repository root (local analysis only):",
        "",
        "```sh",
        "PYTHONPATH=src .venv/bin/python tools/replay_half_life_uncertainty.py --output artifacts/replay-v1/publication-checks-20260909",
        f"PYTHONPATH=src .venv/bin/python tools/replay_publication_summary.py --recheck {recheck.relative_to(repo)} --output {output.relative_to(repo)} --geometry-archive {archive.relative_to(repo)}",
        "```",
        "",
    ]
    (output / "COMPUTATIONAL-HANDOFF.md").write_text("\n".join(lines))
    print(output / "COMPUTATIONAL-HANDOFF.md")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--recheck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-archive", type=Path, required=True)
    args = parser.parse_args()
    summarize(
        args.repo.resolve(),
        args.recheck.resolve(),
        args.output.resolve(),
        args.geometry_archive.resolve(),
    )


if __name__ == "__main__":
    main()
