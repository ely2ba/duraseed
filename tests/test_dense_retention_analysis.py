"""Synthetic numerical checks only; no remote access or scientific output paths."""

import importlib.util
from pathlib import Path
import sys

import pytest

from duraseed.replay_analysis import auc, half_life

np = pytest.importorskip("numpy")
tools = Path(__file__).parents[1] / "tools"
spec = importlib.util.spec_from_file_location(
    "dense_analysis", tools / "analyze_dense_retention.py"
)
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(tools))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_dense_half_life_and_auc_reuse_existing_definitions_with_shared_resampling():
    curve = np.linspace(1, 0, 21)
    curve[11] = 0.9  # A rebound does not change the first crossing.
    values = np.outer(np.linspace(0.1, 1, 12), curve)
    result = module.uncertainty([values, values], "test", replicates=250)
    expected = half_life(values.mean(axis=0), module.GRID)
    for name in module.ARMS:
        assert result["arms"][name]["half_life"]["crossing"] == expected
        assert result["arms"][name]["auc_0_20"]["estimate"] == pytest.approx(
            auc(values.mean(axis=0), module.GRID)
        )
    for metric in ("half_life", "auc_0_20"):
        assert result["R-P_minus_R-S"][metric]["ci95"] == [0, 0]


def test_censored_half_lives_do_not_drop_replicates_or_invent_difference():
    values = np.ones((2, 12, 21))
    result = module.uncertainty(values, "censored", replicates=100)
    assert result["arms"]["R-S"]["half_life"]["censored_replicates"] == 100
    assert result["R-P_minus_R-S"]["half_life"]["estimate"] is None
    assert result["R-P_minus_R-S"]["half_life"]["ci95"] is None


def test_per_item_matrix_rejects_missing_items():
    with pytest.raises(ValueError, match="shared exactly"):
        module.matrices([], "targeted")


def test_report_and_plot_render_synthetic_results_only_in_test_directory(tmp_path):
    pytest.importorskip("matplotlib")
    curves = [
        {
            "arm": arm,
            "role": role,
            "update": step,
            "successes": 768 - step * 30,
            "unconditional_success": (768 - step * 30) / 768,
            "tokens_mean": 20.0,
            "tokens_median": 20,
            "overlapping_indicators": {"output_valid": 768, "cap_reached": 0},
        }
        for arm in module.ARMS
        for role in module.ROLES
        for step in module.GRID
    ]
    raw = np.tile(np.linspace(1, 0, 21), (2, 12, 1))
    value = dict.fromkeys(
        ("scope", "baseline", "definitions", "bootstrap"), "Synthetic test"
    )
    value.update(
        summaries=[module.uncertainty(raw, role, 100) for role in module.ROLES],
        trajectories=curves,
        original_coarse_anchors=[r for r in curves if r["update"] in module.ANCHORS],
    )
    module.plot(curves, tmp_path)
    module.report(value, tmp_path)
    assert (tmp_path / "dense-retention.png").stat().st_size > 1000
    assert (tmp_path / "dense-retention.svg").stat().st_size > 1000
    text = (tmp_path / "README.md").read_text()
    assert "R-P_minus_R-S" in text
    assert "Original frozen results are not replaced" in text
