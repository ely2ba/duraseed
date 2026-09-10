"""Numerical crossing equivalence and paired resampling behavior."""

import importlib.util
from pathlib import Path

import pytest

from duraseed.replay_analysis import half_life

np = pytest.importorskip("numpy")

spec = importlib.util.spec_from_file_location(
    "publication_uncertainty",
    Path(__file__).parents[1] / "tools/replay_half_life_uncertainty.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_batch_crossing_matches_original_including_rebounds_and_zero_baselines():
    rng = np.random.default_rng(910)
    curves = rng.uniform(0, 1, (300, 11))
    curves[0] = 0
    curves[1] = 1
    batch = module.crossing_batch(curves)
    for row, observed in zip(curves, batch, strict=True):
        original = half_life(row)
        if original["status"] == "undefined_zero_baseline":
            assert np.isnan(observed)
        elif original["status"] == "not_reached":
            assert np.isposinf(observed)
        else:
            assert observed == pytest.approx(original["update"], abs=1e-14)


def test_shared_item_resampling_gives_zero_difference_for_identical_arms():
    values = np.outer(
        np.linspace(0.1, 1, 20), [1, 0.8, 0.3, 0.7, 0.2, 0.1, 0, 0, 0, 0, 0]
    )
    result = module.paired_half_lives(
        [values, values], ("left", "right"), "test", replicates=1000
    )
    assert result["difference"]["estimate"] == 0
    assert result["difference"]["ci95"] == [0, 0]
    assert result["difference"]["undefined_replicates"] == 0
