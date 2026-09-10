"""Synthetic item-cluster reduction and stopped-cloning readouts; local only."""

import importlib.util
from pathlib import Path
import sys

import pytest

from duraseed.endpoint_continuation import DENSE_GRID, MAPS_GRID, RUN_STOPS
from duraseed.replay_remote import write_json

np = pytest.importorskip("numpy")
tools = Path(__file__).parents[1] / "tools"
spec = importlib.util.spec_from_file_location(
    "endpoint_analysis", tools / "analyze_endpoint_clone.py"
)
analysis = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(tools))
try:
    spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def observations(grid, *, role="targeted", successes=2, trials=4, items=3):
    return [
        {
            "update": u,
            "manifest_id": "synthetic-panel",
            "item_counts": [
                {
                    "task_id": f"{role}-{i}",
                    "panel_role": role,
                    "successes": successes,
                    "trials": trials,
                }
                for i in range(items)
            ],
        }
        for u in grid
    ]


def test_dense_auc_half_life_and_item_pairing():
    rows = observations(DENSE_GRID)
    for row in rows:
        for item in row["item_counts"]:
            item["successes"] = 4 if row["update"] < 3 else 0
    curve = analysis.curve(rows, "targeted", DENSE_GRID)
    assert curve["raw_auc_0_20"] == 2.5 / 20
    assert curve["half_life"]["update"] == 2.5
    rows[3]["item_counts"][0]["task_id"] = "other-item"
    with pytest.raises(ValueError, match="paired items"):
        analysis.curve(rows, "targeted", DENSE_GRID)


def test_maps_gain_auc_uses_own_baseline_not_absolute_auc():
    rows = observations(MAPS_GRID, role="stage-b", successes=4, trials=16)
    for row in rows[1:]:
        for item in row["item_counts"]:
            item["successes"] = 8
    curve = analysis.curve(rows, "stage-b", MAPS_GRID)
    assert curve["summary"]["baseline"] == 0.25
    assert curve["summary"]["endpoint480"] == 0.5
    assert curve["summary"]["gain_auc_0_40"] == pytest.approx(0.246875)
    assert curve["summary"]["absolute_auc_0_40"] == pytest.approx(0.496875)


def test_paired_linear_contrasts_keep_all_repetitions_and_windows(monkeypatch):
    runs, captured = {}, {}
    for label in RUN_STOPS:
        curves = {
            role: analysis.curve(
                observations(
                    DENSE_GRID, role=role, successes=1 if label.startswith("T") else 2
                ),
                role,
                DENSE_GRID,
            )
            for role in ("targeted", "sentinel")
        }
        curves["maps"] = analysis.curve(
            observations(
                MAPS_GRID,
                role="stage-b",
                successes=2 if label.startswith("T") else 6,
                trials=16,
            ),
            "stage-b",
            MAPS_GRID,
        )
        runs[label] = {"curves": curves}

    def interval(values, name, namespace):
        captured[name] = np.asarray(values)
        return {"estimate": float(np.mean(values))}

    monkeypatch.setattr(analysis, "paired_interval", interval)
    contrasts = analysis.contrast_intervals(runs)
    assert len(contrasts) == 8
    for name, values in captured.items():
        assert values.shape == (3,)
        assert np.allclose(values, 0 if "gain-AUC" in name else 0.25)
    runs["S2"]["curves"]["targeted"]["task_ids"][0] = "unpaired"
    with pytest.raises(ValueError, match="shared evaluation items"):
        analysis.contrast_intervals(runs)


def unavailable(root):
    write_json(
        root / "matching.json",
        {"status": "NO_MATCH", "stage_b_allowed": False, "assessments": []},
    )
    write_json(
        root / "result.json",
        {
            "status": "CLONE_UNAVAILABLE",
            "stage_b_started": False,
            "acquisition_billing": {"synthetic": True},
        },
    )


def test_unavailable_clone_reports_without_continuation_or_plot(tmp_path, monkeypatch):
    root, output = tmp_path / "synthetic-run", tmp_path / "readout"
    unavailable(root)
    monkeypatch.setattr(
        analysis, "plot", lambda *args: pytest.fail("no trajectories exist")
    )
    result = analysis.analyze(root, output)
    assert result["status"] == "CLONE_UNAVAILABLE" and result["runs"] == {}
    assert "No Stage-B continuation was started" in (output / "readout.md").read_text()
    assert (output / "readout.json").exists()
    (root / "continuations/T1").mkdir(parents=True)
    with pytest.raises(ValueError, match="continuation artifacts"):
        analysis.reduce(root)


def test_censored_half_life_is_retained():
    result = analysis.curve(observations(DENSE_GRID), "targeted", DENSE_GRID)
    assert result["half_life"] == {
        "status": "not_reached",
        "update": None,
        "right_censored_after": 20,
    }


def test_evaluation_uses_full_configured_run_identity(tmp_path, monkeypatch):
    root = tmp_path / "T1"
    run_id = "synthetic-endpoint-package-T1"
    write_json(root / "config.json", {"run_id": run_id})
    checkpoint = {
        "replay_arm": "T1",
        "seed": 11,
        "stage": "stage_b",
        "update": 0,
        "state_path": "synthetic-state",
        "sampler_path": "synthetic-sampler",
    }
    point = root / "stage_b/u0"
    write_json(point / "checkpoint.json", checkpoint)
    namespace = "endpoint_clone.T1.a_monitor.0"
    write_json(
        point / "a_monitor/replay-identity.json",
        {
            "seed": 11,
            "replay_arm": "T1",
            "stage": "stage_b",
            "update": 0,
            "purpose": "a_monitor",
            "draws": 4,
            "cap": 4096,
            "sampler_path": "synthetic-sampler",
            "origin_sampler_path": "synthetic-sampler",
            "seed_namespace": namespace,
            "manifest_id": "synthetic-manifest",
        },
    )
    result = {
        "manifest_id": "synthetic-manifest",
        "item_count": 384,
        "row_count": 1536,
        "samples_per_item": 4,
        "sampler_path": "synthetic-sampler",
        "coordinates": {
            "run_id": run_id,
            "seed_namespace": namespace,
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
            "training_step": 0,
            "origin_sampler_path": "synthetic-sampler",
        },
    }
    monkeypatch.setattr(analysis, "read_evaluation", lambda directory: result)
    plan = {"seed": 11, "origin": checkpoint}
    assert analysis.evaluation(root, "T1", 0, "a_monitor", plan)["update"] == 0
    result["coordinates"]["run_id"] = root.name
    with pytest.raises(ValueError, match="coordinates differ"):
        analysis.evaluation(root, "T1", 0, "a_monitor", plan)
