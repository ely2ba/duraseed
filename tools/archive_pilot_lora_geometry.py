#!/usr/bin/env python3
"""Archive Pilot Stage-A LoRA tensors and compute supplementary geometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
from safetensors import safe_open
from tinker import ParsedCheckpointTinkerPath, ServiceClient
from tinker_cookbook import weights


@dataclass(frozen=True)
class RetainedCheckpoint:
    seed: int
    method: str
    step: int
    sampler_path: str
    state_path: str
    selected: bool


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def discover_checkpoints(run_root: Path) -> tuple[RetainedCheckpoint, ...]:
    selected_paths: set[str] = set()
    for matching in sorted(run_root.glob("seed-*/matching.json")):
        value = _json(matching)
        for method in ("B-S", "B-G"):
            row = value.get(method)
            if isinstance(row, dict) and isinstance(row.get("state_path"), str):
                selected_paths.add(row["state_path"])

    rows: dict[str, RetainedCheckpoint] = {}
    for path in sorted(run_root.glob("seed-*/B-*/steps-*/segment.json")):
        value = _json(path)
        sampler_path = value.get("sampler_path")
        state_path = value.get("state_path")
        if (
            value.get("checkpoint_retained") is not True
            or not isinstance(sampler_path, str)
            or not isinstance(state_path, str)
        ):
            continue
        method = str(value.get("method"))
        seed = int(value.get("seed"))
        step = int(value.get("step"))
        rows[state_path] = RetainedCheckpoint(
            seed=seed,
            method=method,
            step=step,
            sampler_path=sampler_path,
            state_path=state_path,
            selected=state_path in selected_paths,
        )
    return tuple(
        sorted(rows.values(), key=lambda row: (row.seed, row.method, row.step))
    )


def _factor_pairs(adapter: Path) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    with safe_open(str(adapter), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        pairs = []
        for a_key in sorted(key for key in keys if key.endswith(".lora_A.weight")):
            stem = a_key.removesuffix(".lora_A.weight")
            b_key = f"{stem}.lora_B.weight"
            if b_key not in keys:
                raise ValueError(f"missing LoRA B factor for {a_key}")
            pairs.append((stem, handle.get_tensor(a_key), handle.get_tensor(b_key)))
    if not pairs:
        raise ValueError(f"no LoRA A/B factors found in {adapter}")
    return tuple(pairs)


def _spectrum(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError(
            f"unsupported LoRA factor shapes: A={tuple(a.shape)}, B={tuple(b.shape)}"
        )
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    _qa, ra = torch.linalg.qr(a64.T, mode="reduced")
    _qb, rb = torch.linalg.qr(b64, mode="reduced")
    return [float(value) for value in torch.linalg.svdvals(rb @ ra.T)]


def _rank_summary(values: list[float]) -> dict[str, Any]:
    positive = [value for value in values if value > 0.0]
    total = sum(positive)
    if not positive or total == 0.0:
        return {
            "numerical_rank_relative_1e-6": 0,
            "entropy_effective_rank": 0.0,
            "stable_rank": 0.0,
        }
    cutoff = positive[0] * 1e-6
    probabilities = [value / total for value in positive]
    entropy = -sum(value * math.log(value) for value in probabilities)
    squares = [value * value for value in positive]
    return {
        "numerical_rank_relative_1e-6": sum(value > cutoff for value in positive),
        "entropy_effective_rank": math.exp(entropy),
        "stable_rank": sum(squares) / max(squares),
    }


def analyze_adapter(adapter: Path) -> dict[str, Any]:
    modules, layers = [], {}
    layer_pattern = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")
    for stem, a, b in _factor_pairs(adapter):
        singular_values = _spectrum(a, b)
        match = layer_pattern.search(stem)
        layer = int(match.group(1)) if match else None
        record = {
            "module": stem,
            "layer": layer,
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
            "a_frobenius_norm": float(torch.linalg.vector_norm(a.float())),
            "b_frobenius_norm": float(torch.linalg.vector_norm(b.float())),
            "ba_singular_values": singular_values,
            **_rank_summary(singular_values),
        }
        modules.append(record)
        key = "unassigned" if layer is None else str(layer)
        bucket = layers.setdefault(key, {"a_squared": 0.0, "b_squared": 0.0, "s": []})
        bucket["a_squared"] += record["a_frobenius_norm"] ** 2
        bucket["b_squared"] += record["b_frobenius_norm"] ** 2
        bucket["s"].extend(singular_values)

    layer_rows = []
    for layer, bucket in sorted(
        layers.items(),
        key=lambda item: (
            item[0] == "unassigned",
            int(item[0]) if item[0].isdigit() else 0,
        ),
    ):
        singular_values = sorted(bucket["s"], reverse=True)
        layer_rows.append(
            {
                "layer": layer,
                "factor_a_frobenius_norm": math.sqrt(bucket["a_squared"]),
                "factor_b_frobenius_norm": math.sqrt(bucket["b_squared"]),
                "block_diagonal_ba_singular_values": singular_values,
                **_rank_summary(singular_values),
            }
        )
    return {
        "definitions": {
            "ba_singular_values": "exact nonzero singular values from thin QR of B and A.T",
            "layer_spectrum": "union of adapted-module BA spectra within a transformer layer",
            "entropy_effective_rank": "exp(Shannon entropy of normalized singular values)",
            "numerical_rank": "count above 1e-6 times the largest singular value",
        },
        "modules": modules,
        "layers": layer_rows,
    }


def _remote_metadata(
    rest: Any, path: str, cache: dict[str, tuple[Any, ...]]
) -> dict[str, Any]:
    parsed = ParsedCheckpointTinkerPath.from_tinker_path(path)
    checkpoints = cache.get(parsed.training_run_id)
    if checkpoints is None:
        checkpoints = tuple(
            rest.list_checkpoints(parsed.training_run_id).result().checkpoints
        )
        cache[parsed.training_run_id] = checkpoints
    checkpoint = next((row for row in checkpoints if row.tinker_path == path), None)
    if checkpoint is None:
        raise ValueError(f"retained checkpoint is absent or expired: {path}")
    return {
        "created_at": checkpoint.time.astimezone(UTC).isoformat(),
        "expires_at": (
            checkpoint.expires_at.astimezone(UTC).isoformat()
            if checkpoint.expires_at is not None
            else None
        ),
        "size_bytes": checkpoint.size_bytes,
    }


def archive(run_root: Path, output: Path) -> dict[str, Any]:
    retained = discover_checkpoints(run_root)
    if not retained:
        return {"status": "no_retained_checkpoints_yet", "checkpoints": []}
    rest = ServiceClient().create_rest_client()
    cache: dict[str, tuple[Any, ...]] = {}
    manifest_rows = []
    for row in retained:
        name = f"seed-{row.seed}-{row.method}-step-{row.step}"
        destination = output / name
        existing = tuple(destination.rglob("adapter_model.safetensors"))
        if len(existing) > 1:
            raise ValueError(f"archive contains multiple adapters: {destination}")
        if not existing:
            if destination.exists():
                raise ValueError(f"incomplete archive already exists: {destination}")
            output.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=output) as temporary:
                weights.download(tinker_path=row.sampler_path, output_dir=temporary)
                temporary_path = Path(temporary)
                downloaded = tuple(temporary_path.rglob("adapter_model.safetensors"))
                if len(downloaded) != 1:
                    raise ValueError(
                        f"checkpoint archive has {len(downloaded)} adapters"
                    )
                os.replace(temporary_path, destination)
        adapter = next(iter(destination.rglob("adapter_model.safetensors")))
        geometry = {
            "schema_version": "duraseed-pilot0-lora-geometry-v1",
            "scientific_role": "supplementary-post-hoc-no-decision-use",
            "checkpoint": asdict(row),
            "remote": _remote_metadata(rest, row.state_path, cache),
            "adapter_sha256": _sha256(adapter),
            **analyze_adapter(adapter),
        }
        _write_json(destination / "geometry.json", geometry)
        manifest_rows.append(
            {
                **asdict(row),
                **geometry["remote"],
                "directory": str(destination),
                "adapter_sha256": geometry["adapter_sha256"],
                "geometry_sha256": _sha256(destination / "geometry.json"),
            }
        )
    expiries = [row["expires_at"] for row in manifest_rows if row["expires_at"]]
    result = {
        "schema_version": "duraseed-pilot0-lora-archive-v1",
        "status": "archived_available_checkpoints",
        "run_root": str(run_root),
        "generated_at": datetime.now(UTC).isoformat(),
        "earliest_expires_at": min(expiries) if expiries else None,
        "checkpoints": manifest_rows,
    }
    _write_json(output / "archive-manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    if args.discover_only:
        value = {
            "checkpoints": [asdict(row) for row in discover_checkpoints(args.run_root)]
        }
    else:
        value = archive(args.run_root, args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
