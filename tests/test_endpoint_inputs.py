from decimal import Decimal
from types import SimpleNamespace

from duraseed import endpoint_inputs as inputs
from duraseed.endpoint_confirmation import _generate_family
from duraseed.data.manifests import build_tces_record
from duraseed.data.splits import tces_numeric_key
from duraseed.replay_data import order_key
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig


def test_family_confirmation_is_exact_and_excludes_numeric_content():
    config = TCESGeneratorConfig(n_operands=3, operand_max=12, max_attempts=256)
    template = TCESGenerator(9, config).generate(0)
    # Manifest conversion requires an explicit split.
    from dataclasses import replace

    config = replace(config, split="a_validation")
    original = build_tces_record(TCESGenerator(9, config).generate(0))
    job = (
        original,
        2,
        config,
        frozenset(),
        frozenset({tces_numeric_key(original)}),
        frozenset({original.content_hash}),
    )
    first = _generate_family(job)
    second = _generate_family(job)
    assert first == second
    assert len({row.content_hash for row in first}) == 2
    assert {row.intended_family for row in first} == {template.intended_family}
    assert all(row.split == "a_validation" for row in first)
    assert all(tces_numeric_key(row) != tces_numeric_key(original) for row in first)


def test_corpus_permutation_preserves_every_draw():
    manifest = SimpleNamespace(
        records=[SimpleNamespace(task_id=f"task-{i}") for i in range(800)]
    )
    order = inputs.corpus_coordinates(manifest)
    assert len(order) == len(set(order)) == 6400
    assert set(order) == {(f"task-{i}", draw) for i in range(800) for draw in range(8)}
    assert order == tuple(
        sorted(order, key=lambda pair: (order_key(11, f"{pair[0]}:{pair[1]}"), pair))
    )


def test_preflight_partitions_one_finite_package(monkeypatch):
    def manifest(name, count):
        return SimpleNamespace(
            records=tuple(SimpleNamespace(task_id=f"{name}-{i}") for i in range(count))
        )

    source = SimpleNamespace(
        a_cadence=manifest("cadence", 192),
        a_validation=manifest("selection", 512),
        b_validation=manifest("maps", 512),
        prompt_pools=SimpleNamespace(a_monitor_manifest=manifest("monitor", 384)),
    )
    clone, confirmation = manifest("clone", 800), manifest("confirm", 512)
    monkeypatch.setattr(
        inputs, "teacher_records", lambda _: manifest("teacher", 480).records
    )
    monkeypatch.setattr(inputs, "stage_b_sources", lambda _: range(4096))
    monkeypatch.setattr(
        inputs,
        "sft_datum",
        lambda *a, **k: SimpleNamespace(model_input=SimpleNamespace(length=11)),
    )
    monkeypatch.setattr(inputs, "_prompt", lambda row: row.task_id)
    runtime = SimpleNamespace(
        renderer=SimpleNamespace(
            build_generation_prompt=lambda *a, **k: SimpleNamespace(length=7)
        )
    )
    config = {
        "checkpoint_ttl_seconds": 2592000,
        "runs": {
            name: {"stop": stop}
            for name, stop in (("T1", 480), ("S1", 480), ("T2", 20), ("S2", 20))
        },
    }
    report = inputs.build_preflight(config, source, clone, confirmation, runtime)
    workers = report["workers"]
    assert set(workers) == {"acquisition", "T1", "T2", "S1", "S2"}
    assert report["storage"] == {"pairs": 123, "reserved_usd": 30.75}
    arithmetic_draws = (
        480 * 8 + 800 * 8 + 33 * 192 + 7 * 512 * 16 + 94 * 384 * 4 + 2 * 512 * 16
    )
    maps_draws = 34 * 512 * 16
    assert report["token_budget"] == {
        "prefill": (arithmetic_draws + maps_draws) * 7,
        "sample": arithmetic_draws * 4096 + maps_draws * 128,
        "train": (480 * 8 + 294 * 32) * (7 + 4096 - 1) + 1000 * 32 * 11,
    }
    assert sum(Decimal(w["main_usd"]) for w in workers.values()) == Decimal(
        report["main_usd"]
    )
    assert workers["acquisition"]["ephemeral_sampler_reserve_usd"] == 1.5
    for field in ("prefill", "sample", "train"):
        assert (
            sum(w["token_budget"][field] for w in workers.values())
            == report["token_budget"][field]
        )
    assert workers["T1"] == workers["S1"]
    assert workers["T2"] == workers["S2"]
