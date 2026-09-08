"""Synthetic known trajectories and authoritative failure accounting."""

from hashlib import sha256

import pytest

from duraseed.replay_analysis import (
    attainment,
    auc,
    bootstrap_seed,
    failure_summary,
    half_life,
    paired_interval,
)


def test_auc_and_gain_decomposition_on_nonuniform_grid():
    pytest.importorskip("numpy")
    grid = (0, 1, 2, 5, 10, 20)
    s, p = [0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.3] * 6
    assert auc(s, grid) == pytest.approx(0.36)
    assert auc(p, grid) == pytest.approx(0.3)
    absolute, baseline = auc(p, grid) - auc(s, grid), p[0] - s[0]
    gain = (auc(p, grid) - p[0]) - (auc(s, grid) - s[0])
    assert gain == pytest.approx(absolute - baseline)
    assert absolute < 0 and gain < absolute


def test_first_attainment_keeps_reversals_and_brackets():
    grid = (0, 1, 2, 5)
    maps, retention = [0, 0.1, 0.02, 0.2], [1, 0.8, 0.4, 0.2]
    c = attainment(maps, retention, 0.06, grid)
    assert c["status"] == "crossed" and c["bracket"] == [0, 1]
    assert c["update"] == pytest.approx(0.6)
    assert c["retention"] == pytest.approx(0.88)
    assert (
        attainment([0.07, 0, 0.1, 0.2], retention, 0.07, grid)["status"]
        == "baseline_exceeded"
    )
    assert attainment(maps, retention, 0.3, grid) == {
        "status": "not_reached",
        "update": None,
        "retention": None,
        "bracket": None,
    }


def test_half_life_is_first_linear_crossing_or_undefined():
    assert half_life([1, 0.8, 0.2, 0.9], (0, 1, 2, 5))["update"] == pytest.approx(1.5)
    assert half_life([0, 0, 0], (0, 1, 2))["status"] == "undefined_zero_baseline"
    assert half_life([1, 0.9, 0.8], (0, 1, 2))["right_censored_after"] == 2


def record(code=None, *, bad_tag=False, bad_syntax=False, capped=False):
    return {
        "sampled_tokens": 4096 if capped else 20,
        "sampling_max_tokens": 4096,
        "verification": {
            "reward": 1 if code is None else 0,
            "failure_code": code,
            "valid_answer_tag": not bad_tag,
            "valid_syntax": not bad_syntax,
            "valid_lexing": not bad_syntax,
        },
    }


def test_disjoint_failures_and_overlapping_indicators_have_distinct_denominators():
    result = failure_summary(
        [
            record(),
            record("missing_answer_tag", bad_tag=True, bad_syntax=True, capped=True),
            record("wrong_target"),
            record("operand_multiset_mismatch"),
        ]
    )
    assert sum(result["disjoint_authoritative_codes"].values()) == 4
    assert sum(result["disjoint_groups"].values()) == 4
    assert result["overlapping_indicators"]["cap_or_parse_failure"] == 1
    assert result["overlapping_indicators"]["invalid_tag"] == 1
    assert result["overlapping_indicators"]["invalid_syntax"] == 1
    assert result["conditional_output_valid"] == {
        "successes": 1,
        "denominator": 3,
        "accuracy": 1 / 3,
    }
    assert result["unconditional_success"] == 0.25
    assert result["disjoint_groups"]["executable_wrong_target"] == 1


def test_no_valid_outputs_is_undefined_not_zero():
    result = failure_summary([record("missing_answer_tag", bad_tag=True)])
    assert result["conditional_output_valid"]["accuracy"] is None


def test_paired_bootstrap_is_deterministic_and_keeps_item_vectors():
    pytest.importorskip("numpy")
    name = "synthetic-known-contrast"
    expected = int.from_bytes(
        sha256(("duraseed-replay-v1|pilot-audit|" + name).encode()).digest()[:8], "big"
    )
    assert bootstrap_seed(name) == expected
    r = paired_interval([0.25] * 12, name, replicates=1000)
    assert r["estimate"] == 0.25 and r["ci95"] == [0.25, 0.25]
    assert paired_interval([0.25] * 12, name, replicates=1000) == r
    assert r["undefined_replicates"] == 0
