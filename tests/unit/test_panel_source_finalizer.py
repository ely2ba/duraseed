from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import duraseed.panel_source_finalizer as finalizer
from duraseed.data.manifests import write_manifest
from duraseed.data.panel_split_production import (
    PanelSplitManifests,
    build_production_panel_splits,
)
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from tests.unit.test_panel_split_manifest import _fixture


def _write_boundary_gate(
    directory: Path,
    *,
    tamper: str | None = None,
    broad_id: str = "sha256:" + "a" * 64,
    confirmation_id: str = "sha256:" + "b" * 64,
) -> dict:
    scientific = "sha256:" + "c" * 64
    equivalence = {
        "schema_version": "duraseed-three-cohort-freeze-equivalence-v1",
        "status": "passed",
        "old_new_identical": True,
        "token_count_map_identical": True,
        "source_run_id": "boundary-run",
        "scientific_outputs_sha256": scientific,
        "archived_v0": {"scientific_outputs_sha256": scientific},
        "current_v1": {"scientific_outputs_sha256": scientific},
    }
    if tamper in {"schema_version", "status", "old_new_identical"}:
        equivalence[tamper] = {
            "schema_version": "wrong",
            "status": "pending",
            "old_new_identical": False,
        }[tamper]
    elif tamper == "token_count_map_identical":
        equivalence[tamper] = False
    elif tamper in {"archived_v0", "current_v1"}:
        equivalence[tamper]["scientific_outputs_sha256"] = "sha256:" + "d" * 64
    elif tamper == "source_run_id":
        equivalence[tamper] = "other-run"
    raw = canonical_json_bytes(equivalence) + b"\n"
    (directory / "three_cohort_equivalence.json").write_bytes(raw)
    source = {
        "equivalence_sha256": sha256_bytes(raw),
        "consolidated_run_directory": "/runs/boundary-run",
        "combined_broad_manifest_id": broad_id,
        "combined_confirmation_manifest_id": confirmation_id,
    }
    if tamper == "equivalence_sha256":
        source["equivalence_sha256"] = "sha256:" + "e" * 64
    (directory / "preflight.json").write_bytes(
        canonical_json_bytes({"source": source}) + b"\n"
    )
    return source


def test_archived_builder_runs_isolated_without_remote_credentials(
    tmp_path, monkeypatch
) -> None:
    config, panel, broad, confirmation, _ = _fixture()
    pair = build_production_panel_splits(
        config,
        artifact=panel,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
    )
    v0 = tmp_path / "v0"
    python = v0 / ".venv/bin/python"
    builder = v0 / finalizer._V0_BUILDER
    python.parent.mkdir(parents=True)
    builder.parent.mkdir(parents=True)
    python.write_text("")
    builder.write_text("")
    captured = {}

    class Process:
        returncode = 0

        def communicate(self):
            return "", ""

    def popen(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(
            canonical_json_bytes({"a_seed_train": pair.train, "a_seed_gate": pair.gate})
            + b"\n"
        )
        return Process()

    monkeypatch.setattr(
        finalizer, "_sha256", lambda _path: finalizer._V0_BUILDER_SHA256
    )
    monkeypatch.setattr(finalizer.subprocess, "Popen", popen)
    monkeypatch.setenv("TINKER_API_KEY", "secret")
    process = finalizer._start_archived_builder(
        v0_source_root=v0,
        boundary_directory=tmp_path / "boundary",
        prompt_directory=tmp_path / "prompts",
        output_path=tmp_path / "old.json",
    )
    observed = finalizer._finish_archived_builder(process, tmp_path / "old.json")

    assert observed == pair
    assert captured["command"][:2] == [
        str(python),
        str(finalizer._REPO / "tools/panel_split_v0_oracle.py"),
    ]
    assert captured["environment"]["PYTHONPATH"] == str(v0 / "src")
    assert "TINKER_API_KEY" not in captured["environment"]


@pytest.mark.parametrize(
    "tamper",
    [
        "schema_version",
        "status",
        "old_new_identical",
        "token_count_map_identical",
        "archived_v0",
        "current_v1",
        "source_run_id",
        "equivalence_sha256",
    ],
)
def test_boundary_equivalence_must_be_exactly_bound(tmp_path, tamper) -> None:
    _write_boundary_gate(tmp_path, tamper=tamper)

    with pytest.raises(finalizer.PanelSourceFinalizationError, match="did not pass"):
        finalizer._require_boundary_equivalence(tmp_path)


def test_boundary_equivalence_passes_before_source_build(tmp_path) -> None:
    source = _write_boundary_gate(tmp_path)

    assert finalizer._require_boundary_equivalence(tmp_path) == source


def test_boundary_equivalence_binds_combined_manifest_ids(tmp_path) -> None:
    _config, panel, broad, confirmation, _ = _fixture()
    _write_boundary_gate(
        tmp_path,
        broad_id="sha256:" + "f" * 64,
        confirmation_id=confirmation.manifest_id,
    )
    write_manifest(tmp_path / "a_candidate_manifest.json", broad)
    write_manifest(tmp_path / "a_candidate_confirmation_manifest.json", confirmation)
    (tmp_path / "target_sentinel_panels.json").write_bytes(
        canonical_json_bytes(panel) + b"\n"
    )

    with pytest.raises(finalizer.PanelSourceFinalizationError, match="manifest IDs"):
        finalizer._boundary_objects(tmp_path)


def test_finalizer_orders_sources_and_binds_authorization(
    tmp_path, monkeypatch
) -> None:
    config, panel, broad, confirmation, _ = _fixture()
    pair = PanelSplitManifests(
        SimpleNamespace(split="a_seed_train"),
        SimpleNamespace(split="a_seed_gate"),
    )
    rl, monitor = object(), object()
    prompts = SimpleNamespace(
        a_rl_train_manifest=rl,
        a_monitor_manifest=monitor,
    )
    fake_config = SimpleNamespace(teacher_dose=SimpleNamespace(calibration_updates=16))
    events = []
    monkeypatch.setattr(
        finalizer, "_sha256", lambda _path: finalizer._V1_BUILDER_SHA256
    )
    monkeypatch.setattr(finalizer, "load_pilot_config", lambda _path: fake_config)
    monkeypatch.setattr(
        finalizer,
        "_boundary_objects",
        lambda _path: (panel, broad, confirmation),
    )

    def build_prompts(*_args, **_kwargs):
        events.append("prompt-build")
        return prompts

    def write_prompts(path, _prompts):
        events.append("prompt-write")
        for name in (
            "stage_a_prompt_pools.json",
            "a_rl_train_manifest.json",
            "a_monitor_manifest.json",
        ):
            (path / name).write_text("prompt")

    def start_archived(**kwargs):
        events.append("v0-start")
        kwargs["output_path"].write_text("temporary")
        return object()

    def finish_archived(_process, _path):
        events.append("v0-finish")
        return pair

    def build_v1(_config, **kwargs):
        events.append("v1")
        assert kwargs["prior_manifests"] == (rl, monitor)
        return pair

    monkeypatch.setattr(finalizer, "build_stage_a_prompt_pools", build_prompts)
    monkeypatch.setattr(finalizer, "write_stage_a_prompt_pool_bundle", write_prompts)
    monkeypatch.setattr(finalizer, "_start_archived_builder", start_archived)
    monkeypatch.setattr(finalizer, "_finish_archived_builder", finish_archived)
    monkeypatch.setattr(finalizer, "build_production_panel_splits", build_v1)
    monkeypatch.setattr(
        finalizer, "validate_teacher_allocation_base_sources", lambda source: source
    )
    monkeypatch.setattr(finalizer, "production_forbidden_records", lambda *_: ())

    def write_equivalence(path, **_kwargs):
        events.append("equivalence")
        (path / "panel_split_equivalence.json").write_bytes(b"equivalent")
        return {"status": "passed"}

    def write_sources(path, _sources, _prompts):
        events.append("sources")
        for name in (
            "stage_a_prompt_pools.json",
            "a_rl_train_manifest.json",
            "a_monitor_manifest.json",
            "a_seed_train_manifest.json",
            "a_seed_gate_manifest.json",
            "source-identities.json",
        ):
            (path / name).write_text("source")

    monkeypatch.setattr(finalizer, "write_equivalent_panel_splits", write_equivalence)
    monkeypatch.setattr(finalizer, "write_calibration_sources", write_sources)
    output = finalizer.finalize_panel_sources(
        run_id="sources",
        boundary_directory=tmp_path / "boundary",
        output_root=tmp_path / "output",
        config_path=tmp_path / "config.yaml",
        v0_source_root=tmp_path / "v0",
        authorizer="Ely",
        accepted_at=datetime(2026, 8, 15, 4, 0, tzinfo=UTC),
    )

    assert events[:5] == [
        "prompt-build",
        "prompt-write",
        "v0-start",
        "v1",
        "v0-finish",
    ]
    assert {path.name for path in output.iterdir()} == {
        "stage_a_prompt_pools.json",
        "a_rl_train_manifest.json",
        "a_monitor_manifest.json",
        "a_seed_train_manifest.json",
        "a_seed_gate_manifest.json",
        "source-identities.json",
        "panel_split_equivalence.json",
        "panel_split_authorization.json",
    }
    authorization_raw = (output / "panel_split_authorization.json").read_bytes()
    authorization = json.loads(authorization_raw)
    assert authorization_raw == canonical_json_bytes(authorization) + b"\n"
    assert authorization == {
        "schema_version": "duraseed-calibration-panel-split-authorization-v1",
        "status": "accepted",
        "equivalence_sha256": sha256_bytes(b"equivalent"),
        "authorizer": "Ely",
        "accepted_at_utc": "2026-08-15T04:00:00Z",
    }


def test_current_build_failure_terminates_and_reaps_archived_builder(
    tmp_path, monkeypatch
) -> None:
    _config, panel, broad, confirmation, _ = _fixture()
    prompts = SimpleNamespace(
        a_rl_train_manifest=object(),
        a_monitor_manifest=object(),
    )

    class Process:
        terminated = False
        reaped = False

        def terminate(self):
            self.terminated = True

        def communicate(self):
            self.reaped = True
            return "", ""

    process = Process()
    monkeypatch.setattr(
        finalizer, "_sha256", lambda _path: finalizer._V1_BUILDER_SHA256
    )
    monkeypatch.setattr(finalizer, "load_pilot_config", lambda _path: object())
    monkeypatch.setattr(
        finalizer,
        "_boundary_objects",
        lambda _path: (panel, broad, confirmation),
    )
    monkeypatch.setattr(
        finalizer, "build_stage_a_prompt_pools", lambda *_args, **_kwargs: prompts
    )
    monkeypatch.setattr(
        finalizer, "write_stage_a_prompt_pool_bundle", lambda *_args: None
    )
    monkeypatch.setattr(finalizer, "_start_archived_builder", lambda **_kwargs: process)
    monkeypatch.setattr(
        finalizer,
        "build_production_panel_splits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("current failed")),
    )

    with pytest.raises(RuntimeError, match="current failed"):
        finalizer.finalize_panel_sources(
            run_id="sources",
            boundary_directory=tmp_path / "boundary",
            output_root=tmp_path / "output",
            config_path=tmp_path / "config.yaml",
            v0_source_root=tmp_path / "v0",
            authorizer="Ely",
        )

    assert process.terminated is True
    assert process.reaped is True
    assert not (tmp_path / "output/sources").exists()
