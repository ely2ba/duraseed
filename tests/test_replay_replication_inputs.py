from types import SimpleNamespace

import pytest

from duraseed import replay_replication_inputs as inputs
from duraseed.replay_data import order_key


def record(task_id, arm):
    return SimpleNamespace(task_id=task_id, prompt_text=f"prompt-{task_id}", arm=arm)


def test_order_preserves_paired_content_and_uses_new_seed():
    original = {
        arm: tuple(record(str(i), arm) for i in range(40)) for arm in inputs.ARMS
    }
    original["R-P"] = tuple(reversed(original["R-P"]))
    ordered = inputs.ordered_records(original)
    expected = sorted(
        (str(i) for i in range(40)), key=lambda key: (order_key(47, key), key)
    )
    for arm in inputs.ARMS:
        assert [row.task_id for row in ordered[arm]] == expected
        assert {id(row) for row in ordered[arm]} == {id(row) for row in original[arm]}
    assert [row.task_id for row in ordered["R-S"]] != [
        row.task_id for row in original["R-S"]
    ]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "different_prompt"])
def test_unpaired_corpora_are_rejected(defect):
    rows = {arm: [record(str(i), arm) for i in range(32)] for arm in inputs.ARMS}
    if defect == "missing":
        rows["R-P"].pop()
    elif defect == "duplicate":
        rows["R-P"][-1] = rows["R-P"][0]
    else:
        rows["R-P"][0].prompt_text = "different"
    with pytest.raises(ValueError):
        inputs.ordered_records(rows)


def test_preflight_prices_exact_fixed_schedule(monkeypatch):
    def manifest(count):
        return SimpleNamespace(record_count=count, records=tuple(range(count)))

    source = SimpleNamespace(
        a_cadence=manifest(192),
        a_validation=manifest(512),
        b_validation=manifest(512),
        prompt_pools=SimpleNamespace(a_monitor_manifest=manifest(384)),
    )
    renderer = SimpleNamespace(
        build_generation_prompt=lambda *a, **k: SimpleNamespace(length=7)
    )
    runtime = SimpleNamespace(renderer=renderer)
    monkeypatch.setattr(inputs, "candidate_manifest", lambda s: manifest(256))
    monkeypatch.setattr(inputs, "stage_b_sources", lambda s: tuple(range(4096)))
    monkeypatch.setattr(inputs, "_prompt", str)
    monkeypatch.setattr(
        inputs,
        "sft_datum",
        lambda *a, **k: SimpleNamespace(model_input=SimpleNamespace(length=11)),
    )
    config = {
        "max_length": 4217,
        "storage_per_pair_usd": 0.25,
        "checkpoint_ttl_seconds": 2592000,
    }
    report = inputs.build_preflight(
        config, source, {arm: tuple(range(579)) for arm in inputs.ARMS}, runtime
    )
    components = {row["component"]: row for row in report["components"]}
    assert len(components) == 8
    assert report["storage"]["pairs"] == 80
    assert report["storage"]["reserved_usd"] == 20
    assert report["token_budget"]["train"] == (2 * 294 + 2 * 480) * 32 * 11
    assert report["token_budget"]["sample"] == 443547648
    assert sum(row.get("completions", 0) for row in components.values()) == 282880
    assert report["token_budget"]["prefill"] == 282880 * 7
    assert components["candidate_assessments"]["points"] == 3
    assert components["pre_b_and_final_validation"]["points"] == 2
