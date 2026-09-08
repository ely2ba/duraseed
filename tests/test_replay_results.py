"""Synthetic offline follow-up reduction tests; never write scientific fixtures."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from duraseed import replay_results as report
from duraseed.replay_analysis import GRID, auc, bootstrap_seed, paired_interval


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def synthetic_evaluation(root, seed, arm, step, panel, selected):
    index = GRID.index(step)
    if panel == "b_validation":
        roles, draws = ("stage-b",), 16
        successes = 4 + min(index, 6) if arm == "R-S" else 8 + min(index, 3)
    else:
        roles, draws = ("targeted", "sentinel"), 4 if panel == "a_monitor" else 16
        successes = max(0, 2 - index) + int(arm == "R-P")
    # Small complete item populations suffice for pure reducer fixtures.
    items = [
        {
            "task_id": f"synthetic-{role}-{i}",
            "panel_role": role,
            "successes": successes,
            "trials": draws,
        }
        for role in roles
        for i in range(3)
    ]
    return {
        "source": f"synthetic-{seed}-{arm}-{step}-{panel}",
        "failures": {},
        "result": {"item_counts": items, "manifest_id": f"synthetic-{panel}"},
    }


def fixture_run(tmp_path, monkeypatch, matched=True):
    root = tmp_path / "synthetic-replay-run"
    no_match = {"status": "NO_MATCH", "selected": None, "stage_b_allowed": False}
    matching = {"11": deepcopy(no_match), "29": deepcopy(no_match)}
    if matched:
        matching["11"] = {
            "status": "MATCHED",
            "selected": {"R-S": 20, "R-P": 30},
            "stage_b_allowed": True,
        }
        for arm, update in (("R-S", 20), ("R-P", 30)):
            save(
                root / "seed-11" / arm / "pre-b" / "profile.json",
                {"seed": 11, "replay_arm": arm, "stage_a_update": update},
            )
    save(root / "matching.json", matching)
    save(
        root / "result.json",
        {"status": "COMPLETED" if matched else "NO_MATCH", "matching": matching},
    )
    monkeypatch.setattr(report, "_evaluation", synthetic_evaluation)
    return root


def test_new_followup_raw_posterior_trajectories_and_item_intervals(
    tmp_path, monkeypatch
):
    pytest.importorskip("numpy")
    root = fixture_run(tmp_path, monkeypatch)
    result = report.write_report(root)
    block = result["blocks"]["11"]
    primary = block["contrasts"]["targeted/raw-absolute-AUC-0-20"]
    assert primary["estimate"] == 0.25
    assert primary["ci95"] == [0.25, 0.25]
    assert primary["replicates"] == 50_000
    contrast = "seed-11|targeted|raw-absolute-AUC-0-20|R-P-minus-R-S"
    expected_seed = int.from_bytes(
        sha256((report.BOOTSTRAP_NAMESPACE + contrast).encode()).digest()[:8], "big"
    )
    assert primary["contrast"] == contrast and primary["seed"] == expected_seed
    rs, rp = (block["arms"][arm] for arm in ("R-S", "R-P"))
    target = rs["curves"]["targeted"]
    assert target["raw"][:3] == [0.5, 0.25, 0]
    assert target["posterior"][2] == pytest.approx(0.1)
    assert target["relative_retention"][:3] == [1, 0.5, 0]
    assert target["half_life"]["update"] == 1
    assert rp["curves"]["targeted"]["half_life"]["update"] == 1.5
    maps_rs, maps_rp = rs["curves"]["stage-b"]["raw"], rp["curves"]["stage-b"]["raw"]
    d = block["gain_auc_decomposition"]
    assert d["baseline_difference"] == 0.25
    assert d["absolute_auc_difference"] == pytest.approx(
        auc(maps_rp[:7], GRID[:7]) - auc(maps_rs[:7], GRID[:7])
    )
    assert d["gain_auc_difference"] == pytest.approx(
        d["absolute_auc_difference"] - d["baseline_difference"]
    )
    assert block["contrasts"]["stage-b/raw-absolute-AUC-0-480"][
        "estimate"
    ] == pytest.approx(auc(maps_rp) - auc(maps_rs))
    assert block["contrasts"]["stage-b/raw-endpoint480"]["estimate"] == 1 / 16
    assert rs["first_attainment"]["0.2"]["targeted"]["status"] == "baseline_exceeded"
    assert len(rs["failures"]["a_monitor"]) == 7
    assert result["blocks"]["29"]["status"] == "NO_MATCH"
    assert "Source block 29: NO_MATCH" in (root / "analysis/readout.md").read_text()
    assert json.loads((root / "analysis/readout.json").read_text()) == result


def test_bootstrap_optional_namespace_preserves_pilot_default():
    pytest.importorskip("numpy")
    pilot = bootstrap_seed("synthetic-contrast")
    expected = int.from_bytes(
        sha256(b"duraseed-replay-v1|pilot-audit|synthetic-contrast").digest()[:8], "big"
    )
    assert pilot == expected
    followup = paired_interval(
        [1, 1],
        "synthetic-contrast",
        replicates=100,
        namespace=report.BOOTSTRAP_NAMESPACE,
    )
    assert followup["seed"] != pilot and followup["ci95"] == [1, 1]


def test_no_match_never_opens_outcomes_and_rejects_stage_b(tmp_path, monkeypatch):
    root = fixture_run(tmp_path, monkeypatch, matched=False)
    monkeypatch.setattr(
        report, "_evaluation", lambda *args: pytest.fail("read unavailable outcome")
    )
    assert all(
        row["status"] == "NO_MATCH"
        for row in report.reduce_results(root)["blocks"].values()
    )
    (root / "seed-29/R-P/stage_b").mkdir(parents=True)
    with pytest.raises(ValueError, match="unmatched"):
        report.reduce_results(root)


def test_incomplete_or_unpaired_evidence_is_not_imputed(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    root = fixture_run(tmp_path, monkeypatch)
    original = synthetic_evaluation

    def changed(*args):
        result = original(*args)
        if args[2] == "R-P" and args[3] == 0:
            result["result"]["item_counts"][0]["task_id"] = "different-item"
        return result

    monkeypatch.setattr(report, "_evaluation", changed)
    with pytest.raises(ValueError, match="population"):
        report.reduce_results(root)
    save(root / "result.json", {"status": "RUNNING"})
    with pytest.raises(ValueError, match="not terminal"):
        report.reduce_results(root)


def test_zero_baseline_ratios_and_half_life_remain_undefined():
    pytest.importorskip("numpy")
    rows = [synthetic_evaluation(None, 11, "R-S", s, "a_monitor", 20) for s in GRID]
    for item in rows[0]["result"]["item_counts"]:
        item["successes"] = 0
    curve = report._curve(rows, "targeted")
    assert curve["relative_retention"] == [None] * len(GRID)
    assert curve["relative_retention_status"] == "undefined_zero_baseline"
    assert curve["half_life"]["status"] == "undefined_zero_baseline"


def test_explicit_identity_blocks_wrong_arm_before_legacy_reader(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-run"
    base = root / "seed-11/R-S"
    save(
        base / "stage_a/u20/checkpoint.json",
        {
            "seed": 11,
            "replay_arm": "R-S",
            "update": 20,
            "stage": "stage_a",
            "sampler_path": "synthetic-sampler",
        },
    )
    save(base / "pre-b/a_monitor/replay-identity.json", {"replay_arm": "B-S"})
    monkeypatch.setattr(
        report, "read_evaluation", lambda *args: pytest.fail("read before arm check")
    )
    with pytest.raises(ValueError, match="evaluation identity"):
        report._evaluation(root, 11, "R-S", 0, "a_monitor", 20)
