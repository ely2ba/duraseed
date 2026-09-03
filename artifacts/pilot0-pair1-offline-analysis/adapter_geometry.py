"""Offline descriptive geometry of the 34 existing pair-1 Stage-A archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def summary(modules):
    singular = sorted(
        (s for module in modules for s in module["ba_singular_values"]), reverse=True
    )
    total = sum(singular)
    energy = sum(s * s for s in singular)
    return {
        "module_count": len(modules),
        "a_frobenius_norm": math.sqrt(
            sum(m["a_frobenius_norm"] ** 2 for m in modules)
        ),
        "b_frobenius_norm": math.sqrt(
            sum(m["b_frobenius_norm"] ** 2 for m in modules)
        ),
        "block_diagonal_ba_frobenius_norm": math.sqrt(energy),
        "largest_singular_value": singular[0],
        "smallest_retained_singular_value": singular[-1],
        "entropy_effective_rank": math.exp(
            -sum((s / total) * math.log(s / total) for s in singular if s > 0)
        ) if total else 0.0,
        "stable_rank": energy / singular[0] ** 2 if singular[0] else 0.0,
        "numerical_rank_relative_1e-6": sum(s > singular[0] * 1e-6 for s in singular),
        "block_diagonal_ba_singular_values": singular,
    }


def selected_spot_check(adapter, modules):
    """Recompute first/middle/unassigned module metrics, not any remote path."""
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    selected = [modules[0], modules[len(modules) // 2], modules[-1]]
    rows = []
    with safe_open(str(adapter), framework="pt", device="cpu") as handle:
        for module in selected:
            stem = module["module"]
            a = handle.get_tensor(stem + ".lora_A.weight")
            b = handle.get_tensor(stem + ".lora_B.weight")
            a_norm = float(torch.linalg.vector_norm(a.float()))
            b_norm = float(torch.linalg.vector_norm(b.float()))
            _, ra = torch.linalg.qr(a.double().T, mode="reduced")
            _, rb = torch.linalg.qr(b.double(), mode="reduced")
            spectrum = torch.linalg.svdvals(rb @ ra.T).tolist()
            for cached, measured in zip(module["ba_singular_values"], spectrum, strict=True):
                assert math.isclose(cached, measured, rel_tol=1e-9, abs_tol=1e-12)
            for key, value in (("a_frobenius_norm", a_norm), ("b_frobenius_norm", b_norm)):
                assert math.isclose(module[key], value, rel_tol=2e-6, abs_tol=1e-8)
            rows.append({
                "module": stem,
                "a_frobenius_norm_recomputed": a_norm,
                "b_frobenius_norm_recomputed": b_norm,
                "spectrum_max_absolute_difference": max(
                    abs(old - new) for old, new in zip(module["ba_singular_values"], spectrum)
                ),
                "passed": True,
            })
    return rows


def build(repo):
    archive = repo / "artifacts/pilot0-pair1-lora-geometry"
    manifest_path = archive / "archive-manifest.json"
    manifest = read(manifest_path)
    run = repo / manifest["run_root"]
    expected = {}
    for segment_path in sorted(run.glob("seed-11/B-*/steps-*/segment.json")):
        segment = read(segment_path)
        if segment.get("checkpoint_retained"):
            expected[segment["state_path"]] = segment
    matching_path = run / "seed-11/matching.json"
    matching = read(matching_path)
    selected_paths = {matching[method]["state_path"] for method in ("B-S", "B-G")}
    rows = manifest["checkpoints"]
    assert len(rows) == len(expected) == 34
    assert {row["state_path"] for row in rows} == set(expected)
    assert {row["state_path"] for row in rows if row["selected"]} == selected_paths
    results, checks = [], []
    total_bytes = 0
    for row in sorted(rows, key=lambda row: (row["method"], row["step"])):
        directory = repo / row["directory"]
        adapter = directory / "adapter_model.safetensors"
        geometry_path = directory / "geometry.json"
        config_path = directory / "adapter_config.json"
        assert sha256(adapter) == row["adapter_sha256"]
        assert sha256(geometry_path) == row["geometry_sha256"]
        geometry = read(geometry_path)
        assert geometry["adapter_sha256"] == row["adapter_sha256"]
        assert geometry["checkpoint"]["state_path"] == row["state_path"]
        assert expected[row["state_path"]]["sampler_path"] == row["sampler_path"]
        config = read(config_path)
        assert (config["r"], config["lora_alpha"], config["use_rslora"], config["use_dora"]) == (32, 32, False, False)
        assert not config["alpha_pattern"] and not config["rank_pattern"]
        modules = geometry["modules"]
        assert len(modules) == 249
        layer_results = []
        for layer in geometry["layers"]:
            key = layer["layer"]
            members = [
                module for module in modules
                if ("unassigned" if module["layer"] is None else str(module["layer"])) == key
            ]
            pooled = summary(members)
            assert pooled["block_diagonal_ba_singular_values"] == layer["block_diagonal_ba_singular_values"]
            for key_old, key_new in (("factor_a_frobenius_norm", "a_frobenius_norm"), ("factor_b_frobenius_norm", "b_frobenius_norm"), ("entropy_effective_rank", "entropy_effective_rank"), ("stable_rank", "stable_rank")):
                assert math.isclose(layer[key_old], pooled[key_new], rel_tol=1e-12, abs_tol=1e-12)
            layer_results.append({"layer": key, **pooled})
        spot_checks = selected_spot_check(adapter, modules) if row["selected"] else []
        total_bytes += adapter.stat().st_size
        checks.append({
            **row,
            "adapter_bytes": adapter.stat().st_size,
            "adapter_config_sha256": sha256(config_path),
            "payload_and_geometry_hashes_verified": True,
            "selected_tensor_spot_checks": spot_checks,
        })
        results.append({
            "method": row["method"], "step": row["step"], "selected": row["selected"],
            "global_block_diagonal_summary": summary(modules),
            "layers": layer_results, "modules": modules,
        })
        print(f"Verified {row['method']} u{row['step']}: 249 modules", flush=True)
    return {
        "role": "Supplementary post-hoc descriptive analysis; no decision or gate use",
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_manifest_sha256": sha256(manifest_path),
        "matching_record": str(matching_path.relative_to(repo)),
        "matching_record_sha256": sha256(matching_path),
        "definitions": {
            "factor_norms": "Frobenius norms of A and B, pooled by root-sum-of-squares across modules",
            "spectrum": "Sorted union of each module's unscaled BA singular values; a block-diagonal bookkeeping operator, not a composed transformer-layer map",
            "entropy_effective_rank": "exp(-sum(p_i log p_i)), p_i=s_i/sum(s)",
            "stable_rank": "sum(s_i^2)/max(s_i)^2",
            "numerical_rank": "count of s_i strictly greater than 1e-6 times max(s)",
            "scaling": "All archived configs specify ordinary LoRA r=32, alpha=32, so alpha/r=1; no rsLoRA or DoRA",
            "gauge": "A and B norms depend on factorization gauge; BA and its spectrum do not under A->CA, B->BC^-1",
            "reference": "Full retained adapter values, not differences from M0 or between adjacent checkpoints",
        },
        "audit": {"checkpoints": len(rows), "adapter_bytes_hashed": total_bytes,
                  "modules_per_checkpoint": 249, "transformer_layers": 32,
                  "unassigned_modules": ["base_model.model.model.unembed_tokens"],
                  "selected_tensor_spot_checks": sum(len(r["selected_tensor_spot_checks"]) for r in checks),
                  "checks": checks},
        "checkpoints": results,
    }


def markdown(data):
    selected = {r["method"]: r for r in data["checkpoints"] if r["selected"]}
    bs, bg = selected["B-S"], selected["B-G"]
    lines = [
        "# Pair-1 adapter geometry — descriptive only", "",
        "Local archived tensors and their hash-bound geometry caches only. No remote calls, new sampling, pair-2 interaction, or gate input.", "",
        "## Definitions and provenance", "",
        f"All **{data['audit']['checkpoints']}** retained Stage-A checkpoints were checked against local segment and matching records; all 34 adapter payload hashes and all 34 cached-geometry hashes matched the archive manifest. Total adapter bytes hashed: **{data['audit']['adapter_bytes_hashed']:,}**.", "",
        "Each checkpoint contains **249 adapted modules** across transformer layers **0–31**, plus one unassigned output projection (`base_model.model.model.unembed_tokens`). B-S cadence checkpoints: u10–u290 every 10 updates; B-G: u10–u50 every 10 updates. The selected B-S u140 and B-G u30 are members of those 34 checkpoints. No u0/M0 or B-S u294 adapter is present in this archive.", "",
        "All cached module spectra are recombined locally to validate every layer-level cache. For each selected checkpoint, the first, middle, and last module's factor norms and float64 thin-QR/SVD spectrum were independently recomputed from its tensor file (six modules total; one CPU thread). All checks passed. The other 8,460 cached module spectra were provenance-verified, not recomputed.", "",
        "`||A||F` and `||B||F` are Frobenius norms; layer/model aggregation is the root-sum-of-squares across factors. `||BA||F`, singular spectra, and rank summaries use the **block-diagonal union of module BA matrices**. This is a bookkeeping aggregate, not the composition, Jacobian, or effective rank of the whole transformer layer/model. Full ordered spectra for every module and layer at every cadence point are in [geometry.json](geometry.json).", "",
        "Entropy effective rank is `exp(-Σ p_i log p_i)`, with `p_i = σ_i/Σσ`; stable rank is `Σσ_i²/σ₁²`. Numerical rank counts singular values above `10⁻⁶ σ₁`. A/B norms are gauge-dependent: an invertible factor change can alter them without changing BA. These are full adapter values, not changes from M0. All 34 archived configs specify ordinary rank-32 LoRA with alpha 32 (alpha/r = 1), no rsLoRA/DoRA; the reported unscaled BA also has the configured scale of 1.", "",
        f"Source manifest: `{data['source_manifest']}` (`{data['source_manifest_sha256']}`). Individual source hashes and selected spot-check residuals are in the JSON audit.", "",
        "## Selected checkpoint: per-layer factor and BA norms", "",
        "| Layer | Modules | B-S u140 ‖A‖F | B-G u30 ‖A‖F | B-S ‖B‖F | B-G ‖B‖F | B-S ‖BA‖F | B-G ‖BA‖F |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for left, right in zip(bs["layers"], bg["layers"], strict=True):
        assert left["layer"] == right["layer"]
        nums = [left[k] for k in ("a_frobenius_norm",)] + [right["a_frobenius_norm"], left["b_frobenius_norm"], right["b_frobenius_norm"], left["block_diagonal_ba_frobenius_norm"], right["block_diagonal_ba_frobenius_norm"]]
        lines.append(f"| {left['layer']} | {left['module_count']} | " + " | ".join(f"{n:.6f}" for n in nums) + " |")
    lines += ["", "## Selected checkpoint: per-layer spectrum and ranks", "",
              "Full spectra, including small singular values, are retained in JSON; this table shows their largest value and two rank summaries.", "",
              "| Layer | B-S σ1 | B-G σ1 | B-S entropy rank | B-G entropy rank | B-S stable rank | B-G stable rank |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for left, right in zip(bs["layers"], bg["layers"], strict=True):
        nums = [row[key] for key in ("largest_singular_value", "entropy_effective_rank", "stable_rank") for row in (left, right)]
        lines.append(f"| {left['layer']} | " + " | ".join(f"{n:.6f}" for n in nums) + " |")
    lines += ["", "## Cadence trajectories: all-module block-diagonal aggregate", "",
              "The JSON also gives the full per-layer cadence trajectories and spectra. No weighting or normalization for different update counts is applied.", "",
              "| Arm | Update | Selected | ‖A‖F | ‖B‖F | ‖BA‖F | σ1 | Entropy rank | Stable rank |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in data["checkpoints"]:
        stats = row["global_block_diagonal_summary"]
        nums = [stats[key] for key in ("a_frobenius_norm", "b_frobenius_norm", "block_diagonal_ba_frobenius_norm", "largest_singular_value", "entropy_effective_rank", "stable_rank")]
        lines.append(f"| {row['method']} | {row['step']} | {'yes' if row['selected'] else ''} | " + " | ".join(f"{n:.6f}" for n in nums) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    data = build(args.repo)
    (output / "geometry.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (output / "geometry.md").write_text(markdown(data))
    print(json.dumps({"checkpoints_verified": data["audit"]["checkpoints"],
                      "selected": [{"method": row["method"], "step": row["step"],
                                    "summary": {k: v for k, v in row["global_block_diagonal_summary"].items() if k != "block_diagonal_ba_singular_values"}}
                                   for row in data["checkpoints"] if row["selected"]]}, indent=2))


if __name__ == "__main__":
    main()
