#!/usr/bin/env python3
"""Download three existing replay adapters and reuse the Pilot geometry reducer."""

import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import subprocess

import torch
from tinker import ServiceClient
from tinker_cookbook import weights

from archive_pilot_lora_geometry import (
    _rank_summary,
    _remote_metadata,
    _sha256,
    analyze_adapter,
)
from duraseed.replay_inputs import read_json
from duraseed.replay_remote import write_json
from recheck_replay_u10 import SOURCE


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    os.environ["TINKER_API_KEY"] = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-s",
            "duraseed-tinker-api-key",
            "-w",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    project = read_json(repo / SOURCE / "config.json")["project_id"]
    rest = ServiceClient(project_id=project).create_rest_client()
    torch.set_num_threads(2)
    rows, cache = [], {}
    points = (("R-S", "stage_a", 220), ("R-P", "stage_a", 20), ("R-P", "stage_b", 10))
    for arm, stage, update in points:
        name = f"seed-11-{arm}-{stage}-u{update}"
        source = SOURCE / "seed-11" / arm / stage / f"u{update}" / "checkpoint.json"
        checkpoint = read_json(repo / source)
        metadata = _remote_metadata(rest, checkpoint["sampler_path"], cache)
        destination = args.output / name
        if not destination.exists():
            weights.download(
                tinker_path=checkpoint["sampler_path"], output_dir=str(destination)
            )
        adapters = list(destination.rglob("adapter_model.safetensors"))
        if len(adapters) != 1:
            raise ValueError("download did not yield exactly one adapter")
        adapter = adapters[0]
        config = read_json(adapter.with_name("adapter_config.json"))
        geometry = analyze_adapter(adapter)
        singular = sorted(
            [s for row in geometry["modules"] for s in row["ba_singular_values"]],
            reverse=True,
        )
        aggregate = {
            "A_frobenius": math.sqrt(
                sum(row["a_frobenius_norm"] ** 2 for row in geometry["modules"])
            ),
            "B_frobenius": math.sqrt(
                sum(row["b_frobenius_norm"] ** 2 for row in geometry["modules"])
            ),
            "BA_frobenius": math.sqrt(sum(s * s for s in singular)),
            "sigma1": singular[0],
            **_rank_summary(singular),
        }
        result = {
            "name": name,
            "seed": 11,
            "arm": arm,
            "stage": stage,
            "update": update,
            "module_count": len(geometry["modules"]),
            "aggregate": aggregate,
            "spectrum_scope": "block-diagonal union of the module BA spectra, not a full-network operator",
            "lora_rank": config["r"],
            "lora_alpha": config["lora_alpha"],
            "adapter_sha256": _sha256(adapter),
            "remote_metadata": metadata,
            **geometry,
        }
        write_json(destination / "geometry.json", result)
        rows.append(result)
        print(
            json.dumps({k: result[k] for k in ("name", "module_count", "aggregate")}),
            flush=True,
        )
    write_json(
        args.summary,
        {
            "date": datetime.now(UTC).isoformat(),
            "scientific_role": "post-hoc descriptive geometry; no intervention, sampling, training or decision use",
            "method": "existing Pilot reducer: float64 thin QR/SVD; factor norms as in the banked Pilot geometry",
            "checkpoints": rows,
            "selected_R_S_over_R_P_B_norm": rows[0]["aggregate"]["B_frobenius"]
            / rows[1]["aggregate"]["B_frobenius"],
            "u10_sampler_file_differs_from_selected_R_P": rows[2]["adapter_sha256"]
            != rows[1]["adapter_sha256"],
        },
    )


if __name__ == "__main__":
    main()
