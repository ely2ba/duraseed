import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")


@pytest.fixture
def archiver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    module = importlib.import_module("archive_endpoint_adapters")
    calls = {"clients": 0, "downloads": 0, "analyses": 0}

    def client():
        calls["clients"] += 1
        return SimpleNamespace(create_rest_client=lambda: object())

    def download(*, tinker_path, output_dir):
        calls["downloads"] += 1
        (Path(output_dir) / "adapter_model.safetensors").write_bytes(
            tinker_path.encode()
        )

    def analyze(adapter):
        calls["analyses"] += 1
        return {
            "modules": [
                {
                    "a_frobenius_norm": 3.0,
                    "b_frobenius_norm": 4.0,
                    "ba_singular_values": [3.0, 1.0],
                },
                {
                    "a_frobenius_norm": 4.0,
                    "b_frobenius_norm": 3.0,
                    "ba_singular_values": [2.0, 0.0],
                },
            ],
            "layers": [],
        }

    monkeypatch.setattr(module.pilot, "ServiceClient", client)
    monkeypatch.setattr(module.pilot.weights, "download", download)
    monkeypatch.setattr(module.pilot, "analyze_adapter", analyze)
    monkeypatch.setattr(
        module.pilot,
        "_remote_metadata",
        lambda rest, path, cache: {
            "created_at": "2026-09-10T17:00:00+00:00",
            "expires_at": "2026-10-10T17:00:01+00:00"
            if path.endswith("state")
            else "2026-10-10T17:00:00+00:00",
            "size_bytes": 100,
        },
    )
    return module, calls


def retain(module, run, arm, update):
    checkpoint = {
        "arm": arm,
        "stage": "stage_a",
        "update": update,
        "sampler_path": f"tinker://example/sampler_weights/{arm}-u{update}-sampler",
        "state_path": f"tinker://example/weights/{arm}-u{update}-state",
    }
    module.pilot._write_json(
        run / "acquisition" / arm / "stage_a" / f"u{update}" / "checkpoint.json",
        checkpoint,
    )
    return checkpoint


def test_empty_run_does_not_connect(archiver, tmp_path):
    module, calls = archiver
    run = tmp_path / "run"
    run.mkdir()
    result = module.archive(run, tmp_path / "out")
    assert result["retained_count"] == 0
    assert result["status"] == "NO_RETAINED_CHECKPOINTS"
    assert calls == {"clients": 0, "downloads": 0, "analyses": 0}


def test_archive_then_verify_and_skip_without_remote_access(archiver, tmp_path):
    module, calls = archiver
    run, out = tmp_path / "run", tmp_path / "out"
    retain(module, run, "T", 30)
    first = module.archive(run, out)
    assert first["new_archive_count"] == 1
    assert first["selected_count"] == 1
    assert first["earliest_expires_at"] == "2026-10-10T17:00:00+00:00"
    norms = first["checkpoints"][0]["aggregate"]
    assert norms["factor_a_frobenius_norm"] == 5.0
    assert norms["factor_b_frobenius_norm"] == 5.0
    assert norms["ba_frobenius_norm"] == pytest.approx(14**0.5)
    assert norms["sigma1"] == 3.0
    assert norms["stable_rank"] == pytest.approx(14 / 9)
    second = module.archive(run, out)
    assert second["new_archive_count"] == 0
    assert second["archived_count"] == 1
    assert calls == {"clients": 1, "downloads": 1, "analyses": 1}


def test_later_matching_marks_selected_student_without_redownload(archiver, tmp_path):
    module, calls = archiver
    run, out = tmp_path / "run", tmp_path / "out"
    retain(module, run, "S", 10)
    module.archive(run, out)
    module.pilot._write_json(
        run / "matching.json", {"selected_update": 10, "status": "MATCHED"}
    )
    second = module.archive(run, out)
    assert second["selected_count"] == 1
    assert second["new_archive_count"] == 0
    assert calls["downloads"] == 1


def test_summary_selected_checkpoint_is_discovered_without_cadence_file(
    archiver, tmp_path
):
    module, _ = archiver
    run = tmp_path / "run"
    checkpoint = retain(module, run, "S", 20)
    module.pilot._write_json(run / "student.json", {"checkpoints": {"20": checkpoint}})
    module.pilot._write_json(run / "selection.json", {"selected_update": 20})
    (run / "acquisition/S/stage_a/u20/checkpoint.json").unlink()
    assert module.discover(run) == [
        {
            "arm": "S",
            "update": 20,
            "selected": True,
            "state_path": checkpoint["state_path"],
            "sampler_path": checkpoint["sampler_path"],
        }
    ]


def test_corrupt_archive_is_reported_not_overwritten(archiver, tmp_path):
    module, calls = archiver
    run, out = tmp_path / "run", tmp_path / "out"
    retain(module, run, "T", 30)
    module.archive(run, out)
    adapter = out / "T-u30/adapter_model.safetensors"
    adapter.write_bytes(b"changed")
    result = module.archive(run, out)
    assert result["status"] == "ARCHIVE_ERRORS"
    assert result["archived_count"] == 0
    assert len(result["errors"]) == 1
    assert adapter.read_bytes() == b"changed"
    assert calls["downloads"] == 1
