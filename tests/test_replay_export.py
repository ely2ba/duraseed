"""Synthetic privacy and count-preservation checks for the local public export."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "replay_public_export",
    Path(__file__).resolve().parents[1] / "tools/export_replay_v1.py",
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_private_locators_are_aliased_without_changing_counts():
    record = {
        "project_id": "private-project",
        "billing": {"session_ids": ["private-session"], "observed": {"train": 42}},
        "source": "tinker://11111111-2222-3333-4444-555555555555/weights/origin",
        "local": "/Users/synthetic/private.json",
        "successes": [[1, 2], [0, 1]],
        "trials": [[4, 4], [4, 4]],
        "task_id": "sha256:synthetic-task",
    }
    original = deepcopy(record)
    result = exporter.clean(record)
    assert record == original
    assert result["successes"] == record["successes"]
    assert result["trials"] == record["trials"]
    assert result["task_id"] == record["task_id"]
    assert result["source"] == "opaque-" + exporter.digest(record["source"].encode())
    assert exporter.clean(record)["source"] == result["source"]
    assert "project_id" not in result and "session_ids" not in result["billing"]
    exporter.privacy_check(json.dumps(result))


@pytest.mark.parametrize(
    "private",
    [
        "/Users/synthetic/data",
        "/home/synthetic/data",
        "tinker://checkpoint",
        "ephemeral:sampler",
        "11111111-2222-3333-4444-555555555555",
        '{"api_key": "synthetic"}',
    ],
)
def test_privacy_check_rejects_unprojected_private_values(private):
    with pytest.raises(ValueError, match="private identifier"):
        exporter.privacy_check(private)


def test_dump_keeps_missing_block_and_exact_numerical_values(tmp_path):
    record = {
        "blocks": {
            "11": {"raw": [0.0, 0.3125], "trials": [4, 16]},
            "29": {"status": "NO_MATCH", "arms": {}},
        }
    }
    written = []
    exporter.dump(tmp_path, "readout.json", record, written)
    assert exporter.read(tmp_path / "readout.json") == record
    assert written == ["readout.json"]
    with pytest.raises(ValueError):
        exporter.dump(tmp_path, "invalid.json", '{"secret":"synthetic"}', written)
    assert not (tmp_path / "invalid.json").exists()
