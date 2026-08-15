from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import duraseed.boundary_freeze_finalizer as finalizer
from duraseed.boundary_freeze_finalizer import (
    BoundaryFreezeFinalizationError,
    _consolidated_source_identity,
)
from duraseed.run_records import RunStatus


def _write_source(tmp_path, **updates):
    source = {
        "sampler_checkpoint_path": "tinker://m0/sampler",
        "state_checkpoint_path": "tinker://m0/state",
        "training_step": 2,
        **updates,
    }
    (tmp_path / "preflight.json").write_text(json.dumps({"source": source}))
    return source


def test_consolidated_checkpoint_identity_is_carried_to_freeze(tmp_path) -> None:
    expected = _write_source(tmp_path)
    run = SimpleNamespace(parent_tinker_checkpoint_path="tinker://m0/state")

    assert (
        _consolidated_source_identity(tmp_path, run, "tinker://m0/sampler") == expected
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"sampler_checkpoint_path": "tinker://wrong"},
        {"state_checkpoint_path": "tinker://wrong"},
        {"training_step": True},
        {"training_step": 1},
    ],
)
def test_consolidated_checkpoint_identity_fails_closed(tmp_path, updates) -> None:
    _write_source(tmp_path, **updates)
    run = SimpleNamespace(parent_tinker_checkpoint_path="tinker://m0/state")

    with pytest.raises(BoundaryFreezeFinalizationError, match="identity changed"):
        _consolidated_source_identity(tmp_path, run, "tinker://m0/sampler")


def test_published_preflight_uses_shared_run_and_authenticated_step(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    consolidated = tmp_path / "consolidated"
    consolidated.mkdir()
    _write_source(consolidated)
    sha_a = "sha256:" + "a" * 64
    sha_b = "sha256:" + "b" * 64
    completed = SimpleNamespace(
        status=RunStatus.COMPLETED,
        parent_tinker_checkpoint_path="tinker://m0/state",
        model_id="model",
        renderer="role_colon",
        lora_rank=32,
        project_id="project",
        tinker_sdk_version="0.25.0",
        tinker_cookbook_version="0.5.3",
    )
    cohorts = tuple(
        SimpleNamespace(
            cohort_id=cohort_id,
            sampler_checkpoint_path="tinker://m0/sampler",
        )
        for cohort_id in ("initial", "extension_1", "extension_2")
    )
    settings = SimpleNamespace(
        projection_sha256=sha_a,
        allocation_seed=11,
    )
    result = SimpleNamespace(
        teacher_trace_token_counts={"family": {"task": 3}},
        combined_broad_manifest=SimpleNamespace(manifest_id=sha_a, root_seed=5),
        combined_confirmation_manifest=SimpleNamespace(manifest_id=sha_b),
        intermediate_family_ids=("family",),
    )
    config = SimpleNamespace(
        protocol=SimpleNamespace(version="protocol"),
        resolved_config_hash=lambda: "config-hash",
    )
    captured = {}
    monkeypatch.setattr(finalizer, "load_pilot_config", lambda _path: config)
    monkeypatch.setattr(finalizer, "read_run_record", lambda _path: completed)
    monkeypatch.setattr(finalizer, "load_three_cohort_inputs", lambda *_args: cohorts)
    monkeypatch.setattr(finalizer, "freeze_settings_from_config", lambda _: settings)
    monkeypatch.setattr(finalizer, "_source_payload", lambda *_args: {"source": 1})
    monkeypatch.setattr(
        finalizer,
        "_run_reducers",
        lambda **_kwargs: (result, b"identical", b"identical"),
    )
    monkeypatch.setattr(finalizer, "_scientific_payload", lambda _: {"field": 1})
    monkeypatch.setattr(finalizer, "_write_scientific_artifacts", lambda *_: None)
    monkeypatch.setattr(finalizer, "_git_commit", lambda _: "commit")
    monkeypatch.setattr(finalizer, "_source_hashes", lambda *_: {})
    monkeypatch.setattr(finalizer, "write_run_record", lambda *_: None)
    monkeypatch.setattr(finalizer.os, "replace", lambda *_: None)

    def capture(path, value):
        captured[path.name] = value
        return b"canonical\n"

    monkeypatch.setattr(finalizer, "_json", capture)
    finalizer.finalize_three_cohort_freeze(
        run_id="freeze",
        source_root=source_root,
        consolidated_run=consolidated,
        output_root=tmp_path / "output",
        config_path=tmp_path / "config.yaml",
        v0_source_root=tmp_path / "v0",
    )

    source = captured["preflight.json"]["source"]
    assert source["confirmation_run_directories"] == [
        str(source_root / "boundary-confirm-20260810T144517Z"),
        str(consolidated),
        str(consolidated),
    ]
    assert source["sampler_checkpoint_path"] == "tinker://m0/sampler"
    assert source["state_checkpoint_path"] == "tinker://m0/state"
    assert source["training_step"] == 2
