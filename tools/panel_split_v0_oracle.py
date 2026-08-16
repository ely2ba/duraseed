#!/usr/bin/env python3
"""Run only the archived v0 panel-split builder on frozen production inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from duraseed.config import load_pilot_config
from duraseed.data.manifests import read_manifest
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.data.sealing import ExecutionContext
from duraseed.provenance import canonical_json_bytes
from duraseed.tinker_teacher_dose import _build_panel_split_manifest


_BUILDER = Path("src/duraseed/tinker_teacher_dose.py")
_EXPECTED_SHA256 = "5e461b768f7b00deac02a101a272242bd3fff910e222f8c8c8a2426a6827944e"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.source_root / _BUILDER) != _EXPECTED_SHA256:
        raise RuntimeError("archived v0 panel-split builder changed")
    config = load_pilot_config(args.source_root / "duraseed_pilot_config.yaml")
    panel = FamilyPanelArtifact.model_validate_json(
        (args.boundary / "target_sentinel_panels.json").read_bytes()
    )
    broad = read_manifest(
        args.boundary / "a_candidate_manifest.json",
        context=ExecutionContext.SELECTION,
    )
    confirmation = read_manifest(
        args.boundary / "a_candidate_confirmation_manifest.json",
        context=ExecutionContext.SELECTION,
    )
    rl_train = read_manifest(
        args.prompts / "a_rl_train_manifest.json",
        context=ExecutionContext.TRAINING,
    )
    monitor = read_manifest(
        args.prompts / "a_monitor_manifest.json",
        context=ExecutionContext.SELECTION,
    )
    forbidden = tuple(
        record
        for manifest in (broad, confirmation, rl_train, monitor)
        for record in manifest.records
    )
    train = _build_panel_split_manifest(
        config,
        artifact=panel,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
        split="a_seed_train",
        items_per_family=16,
        forbidden_records=forbidden,
    )
    gate = _build_panel_split_manifest(
        config,
        artifact=panel,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
        split="a_seed_gate",
        items_per_family=config.teacher_dose.gate_items_per_family,
        forbidden_records=(*forbidden, *train.records),
    )
    args.output.write_bytes(
        canonical_json_bytes({"a_seed_train": train, "a_seed_gate": gate}) + b"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
