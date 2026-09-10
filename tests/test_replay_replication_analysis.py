"""Offline synthetic readout checks; no remote access or scientific output paths."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from duraseed.replay_analysis import GRID, auc, half_life

np = pytest.importorskip("numpy")
tools = Path(__file__).parents[1] / "tools"
spec = importlib.util.spec_from_file_location(
    "replication_analysis", tools / "analyze_replay_replication.py"
)
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(tools))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def terminal(root, *, matched):
    selected = {"R-S": 50, "R-P": 20} if matched else None
    matching = {
        "47": {
            "selected": selected,
            "status": "MATCHED" if matched else "NO_MATCH",
            "stage_b_allowed": matched,
        }
    }
    result = {
        "status": "COMPLETED" if matched else "NO_MATCH",
        "matching": matching,
        "source_block": 11,
        "acquisition_order_seed": 47,
        "billing": {"session_ids": ["synthetic"]},
    }
    write(root / "result.json", result)
    write(root / "matching.json", matching)
    write(root / "billing.json", result["billing"])
    write(root / "remote/remote-call-state.json", {"pending": None})
    return result


def curve(role, grid=GRID, *, constant=False):
    trials = 16 if role == "stage-b" else 4
    counts = (
        [trials] * len(grid)
        if constant
        else [max(trials - i, 0) for i in range(len(grid))]
    )
    successes = np.tile(counts, (12, 1))
    budget = np.full_like(successes, trials)
    raw = successes.mean(axis=0) / trials
    return {
        "grid": list(grid),
        "manifest_id": f"synthetic-{role}",
        "task_ids": [f"{role}-{i}" for i in range(12)],
        "items": 12,
        "successes": successes.tolist(),
        "trials": budget.tolist(),
        "raw": raw.tolist(),
        "posterior": ((successes + 0.5) / (budget + 1)).mean(axis=0).tolist(),
        "own_baseline_gain": (raw - raw[0]).tolist(),
        "relative_retention": (raw / raw[0]).tolist(),
        "half_life": half_life(raw, grid),
        "sources": ["synthetic"] * len(grid),
    }


def arm(root, seed, name, selected):
    curves = {role: curve(role) for role in ("targeted", "sentinel", "stage-b")}
    profile = f"seed-{seed}/{name}/pre-b/profile.json"
    write(
        root / profile, {"seed": seed, "replay_arm": name, "stage_a_update": selected}
    )
    return {
        "selected_update": selected,
        "curves": curves,
        "validation": {
            role: curve(role, (0, 480)) for role in ("targeted", "sentinel")
        },
        "F3_profile": profile,
        "failures": {},
        "first_attainment": {},
        "summary": {
            "targeted_raw_retention_auc_0_20": float(
                auc(curves["targeted"]["raw"][:6], GRID[:6])
            )
        },
    }


def test_no_match_outputs_are_explicit_and_stage_b_is_forbidden(tmp_path):
    root, output = tmp_path / "run", tmp_path / "analysis"
    terminal(root, matched=False)
    result = module.analyze(root, output)
    assert result["blocks"]["47"]["arms"] == {}
    assert (output / "per-item-counts.jsonl").read_text() == ""
    assert (
        "Source block 11 / order seed 47: NO_MATCH"
        in (output / "readout.md").read_text()
    )
    (root / "seed-47/R-P/stage_b").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unmatched replication contains Stage-B"):
        module.reduce(root)


@pytest.mark.parametrize(
    "violation", ["matching", "pending", "billing", "source_block"]
)
def test_terminal_integrity_is_required_before_arm_loading(
    tmp_path, monkeypatch, violation
):
    terminal(tmp_path, matched=False)
    monkeypatch.setattr(
        module, "_arm", lambda *args: pytest.fail("must not read arm evidence")
    )
    if violation == "matching":
        write(tmp_path / "matching.json", {"47": {}})
    elif violation == "pending":
        write(
            tmp_path / "nested/remote-call-state.json",
            {"pending": {"operation": "sample"}},
        )
    elif violation == "billing":
        write(tmp_path / "billing.json", {"session_ids": ["different"]})
    else:
        data = json.loads((tmp_path / "result.json").read_text())
        data["source_block"] = 47
        write(tmp_path / "result.json", data)
    with pytest.raises(ValueError):
        module.reduce(tmp_path)


def test_half_life_resamples_paired_trajectories_and_handles_non_crossing():
    arms = {name: {"curves": {"targeted": curve("targeted")}} for name in module.ARMS}
    result = module.half_life_intervals(arms, "targeted", replicates=100)
    assert result["R-P_minus_R-S"]["estimate"] == 0
    assert result["R-P_minus_R-S"]["ci95"] == [0, 0]
    for name in module.ARMS:
        arms[name]["curves"]["targeted"] = curve("targeted", constant=True)
    result = module.half_life_intervals(arms, "targeted", replicates=100)
    assert result["R-P_minus_R-S"]["estimate"] is None
    assert result["R-P_minus_R-S"]["ci95"] is None
    assert result["arms"]["R-S"]["censored_replicates"] == 100


def test_complete_readout_uses_generic_seed_loader_and_portable_outputs(
    tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    root, output = tmp_path / "run", tmp_path / "analysis"
    terminal(root, matched=True)
    calls = []

    def load(*args):
        calls.append(args)
        return arm(*args)

    uncertainty = module.half_life_intervals
    monkeypatch.setattr(module, "_arm", load)
    monkeypatch.setattr(
        module,
        "half_life_intervals",
        lambda arms, role: uncertainty(arms, role, replicates=100),
    )
    result = module.analyze(root, output)
    assert calls == [(root, 47, "R-S", 50), (root, 47, "R-P", 20)]
    assert result["source_block"] == 11 and result["acquisition_order_seed"] == 47
    block = result["blocks"]["47"]
    assert block["contrasts"]["targeted/raw-absolute-AUC-0-20"]["estimate"] == 0
    assert block["arms"]["R-S"]["summary"][
        "targeted_raw_retention_auc_0_20"
    ] == pytest.approx(0.1625)
    for name in module.ARMS:
        assert (output / f"{name}-profile.json").read_bytes() == (
            root / f"seed-47/{name}/pre-b/profile.json"
        ).read_bytes()
    text = (output / "readout.md").read_text()
    assert "[R-S-profile.json](R-S-profile.json)" in text
    assert "Source block 47" not in text
    assert "Descriptive first-crossing half-lives" in text
    rows = [
        json.loads(line)
        for line in (output / "per-item-counts.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 * 12 * (3 * len(GRID) + 2 * 2)
    assert {row["source_block"] for row in rows} == {11}
    assert {row["acquisition_order_seed"] for row in rows} == {47}
    assert {row["panel"] for row in rows} == {
        "a_monitor",
        "b_validation",
        "a_validation",
    }
    assert (output / "trajectories.svg").stat().st_size > 1000
    assert (output / "trajectories.png").stat().st_size > 1000
