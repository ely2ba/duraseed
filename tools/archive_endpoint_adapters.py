#!/usr/bin/env python3
"""Download retained endpoint-clone adapters; supplementary, no runner mutation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile

import archive_pilot_lora_geometry as pilot


def discover(run: Path) -> list[dict]:
    selected = None
    for name in ("matching.json", "selection.json"):
        if (run / name).exists():
            selected = pilot._json(run / name).get("selected_update")
            break
    found = {}

    def add(value, arm):
        if not value.get("state_path") or not value.get("sampler_path"):
            return
        update = int(value["update"])
        row = {
            "arm": arm,
            "update": update,
            "state_path": value["state_path"],
            "sampler_path": value["sampler_path"],
            "selected": update == (30 if arm == "T" else selected),
        }
        prior = found.get(row["state_path"])
        if prior is not None and prior != row:
            raise ValueError("one retained state has conflicting checkpoint identities")
        found[row["state_path"]] = row

    for arm in ("T", "S"):
        for path in sorted(
            (run / "acquisition" / arm / "stage_a").glob("u*/checkpoint.json")
        ):
            value = pilot._json(path)
            if value.get("stage") != "stage_a" or value.get("arm") != arm:
                raise ValueError(
                    f"checkpoint outside its acquisition identity: {path.name}"
                )
            add(value, arm)
        summary = run / ("teacher.json" if arm == "T" else "student.json")
        if summary.exists():
            value = pilot._json(summary)
            if value.get("checkpoint"):
                add(value["checkpoint"], arm)
            checkpoints = value.get("checkpoints", [])
            for checkpoint in (
                checkpoints.values() if isinstance(checkpoints, dict) else checkpoints
            ):
                add(checkpoint, arm)
    rows = sorted(found.values(), key=lambda row: (row["arm"], row["update"]))
    if len({(row["arm"], row["update"]) for row in rows}) != len(rows):
        raise ValueError("multiple retained states claim the same arm and update")
    return rows


def aggregate(geometry: dict) -> dict:
    modules = geometry["modules"]
    singular = [s for module in modules for s in module["ba_singular_values"]]
    squared = math.fsum(s * s for s in singular)
    sigma1 = max(singular, default=0.0)
    return {
        "factor_a_frobenius_norm": math.sqrt(
            math.fsum(m["a_frobenius_norm"] ** 2 for m in modules)
        ),
        "factor_b_frobenius_norm": math.sqrt(
            math.fsum(m["b_frobenius_norm"] ** 2 for m in modules)
        ),
        "ba_frobenius_norm": math.sqrt(squared),
        "sigma1": sigma1,
        "stable_rank": squared / sigma1**2 if sigma1 else 0.0,
    }


def archive(run: Path, output: Path) -> dict:
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError("supplementary archive must be outside the live run directory")
    retained = discover(run)
    existing_manifest = output / "archive-manifest.json"
    old = pilot._json(existing_manifest) if existing_manifest.exists() else {}
    prior = {row["state_path"]: row for row in old.get("checkpoints", [])}
    if old and old.get("run_id") != run.name:
        raise ValueError("archive destination belongs to another endpoint run")
    rest, cache, rows, errors, new_count = None, {}, [], [], 0
    known_expiries = {
        row["state_path"]: prior[row["state_path"]].get("expires_at")
        for row in retained
        if row["state_path"] in prior
    }
    for checkpoint in retained:
        try:
            name = f"{checkpoint['arm']}-u{checkpoint['update']}"
            directory = output / name
            geometry_path = directory / "geometry.json"
            expected = prior.get(checkpoint["state_path"])
            if geometry_path.exists():
                geometry = pilot._json(geometry_path)
                identity = {
                    key: value for key, value in checkpoint.items() if key != "selected"
                }
                if geometry["checkpoint"] != identity:
                    raise ValueError(
                        "existing adapter archive changed checkpoint identity"
                    )
                adapters = tuple(directory.rglob("adapter_model.safetensors"))
                if (
                    len(adapters) != 1
                    or pilot._sha256(adapters[0]) != geometry["adapter_sha256"]
                ):
                    raise ValueError(
                        "existing adapter archive is incomplete or its tensor hash changed"
                    )
                if (
                    expected
                    and pilot._sha256(geometry_path) != expected["geometry_sha256"]
                ):
                    raise ValueError("existing adapter geometry hash changed")
            else:
                if directory.exists():
                    raise ValueError(
                        "incomplete archive exists; inspect it before another download"
                    )
                if rest is None:
                    rest = pilot.ServiceClient().create_rest_client()
                metadata = pilot._remote_metadata(rest, checkpoint["state_path"], cache)
                sampler_metadata = pilot._remote_metadata(
                    rest, checkpoint["sampler_path"], cache
                )
                pair_expiries = [
                    m["expires_at"]
                    for m in (metadata, sampler_metadata)
                    if m["expires_at"]
                ]
                metadata = {
                    **metadata,
                    "state_expires_at": metadata["expires_at"],
                    "sampler_expires_at": sampler_metadata["expires_at"],
                    "expires_at": min(pair_expiries) if pair_expiries else None,
                }
                known_expiries[checkpoint["state_path"]] = metadata["expires_at"]
                output.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=output) as temporary:
                    temporary_path = Path(temporary)
                    pilot.weights.download(
                        tinker_path=checkpoint["sampler_path"], output_dir=temporary
                    )
                    adapters = tuple(temporary_path.rglob("adapter_model.safetensors"))
                    if len(adapters) != 1:
                        raise ValueError(f"download contains {len(adapters)} adapters")
                    geometry = {
                        "schema_version": "duraseed-endpoint-lora-geometry-v1",
                        "scientific_role": "supplementary-descriptive-no-decision-use",
                        "checkpoint": {
                            key: value
                            for key, value in checkpoint.items()
                            if key != "selected"
                        },
                        "remote": metadata,
                        "adapter_sha256": pilot._sha256(adapters[0]),
                        **pilot.analyze_adapter(adapters[0]),
                    }
                    geometry["aggregate"] = aggregate(geometry)
                    pilot._write_json(temporary_path / "geometry.json", geometry)
                    os.replace(temporary_path, directory)
                new_count += 1
            rows.append(
                {
                    **checkpoint,
                    **geometry["remote"],
                    "directory": name,
                    "adapter_sha256": geometry["adapter_sha256"],
                    "geometry_sha256": pilot._sha256(geometry_path),
                    "aggregate": geometry["aggregate"],
                }
            )
            known_expiries[checkpoint["state_path"]] = geometry["remote"]["expires_at"]
        except Exception as error:
            errors.append(
                {
                    "arm": checkpoint["arm"],
                    "update": checkpoint["update"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    expiries = [value for value in known_expiries.values() if value]
    result = {
        "schema_version": "duraseed-endpoint-lora-archive-v1",
        "run_id": run.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ARCHIVE_ERRORS"
        if errors
        else "ARCHIVED"
        if retained
        else "NO_RETAINED_CHECKPOINTS",
        "retained_count": len(retained),
        "archived_count": len(rows),
        "new_archive_count": new_count,
        "selected_count": sum(row["selected"] for row in rows),
        "earliest_expires_at": min(expiries) if expiries else None,
        "expiry_unknown_count": len(retained) - len(known_expiries),
        "aggregate_definition": "Frobenius factor concatenations; BA spectrum is the block-diagonal union of adapted-module spectra, not a composed model Jacobian",
        "checkpoints": rows,
        "errors": errors,
    }
    pilot._write_json(existing_manifest, result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error("run directory does not exist")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".archive.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = archive(args.run, args.output)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("checkpoints", "aggregate_definition")
            },
            indent=2,
        )
    )
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
