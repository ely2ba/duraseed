from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.data.manifests import build_manifest, build_tces_record
from duraseed.pilot0_analysis import (
    paired_primary_aggregate,
    summarize_method,
    summarize_selected_method,
)
from duraseed.pilot0_budget import Pilot0Budget, calculate_pilot0_budget
from duraseed.pilot0_contract import STAGE_B_GRID
from duraseed.runners import RunnerGateError
from duraseed.runners import pilot0_live
from duraseed.runners.pilot0_live import run_pilot0
from duraseed.runners.pilot0_sampling import evaluate_manifest
from duraseed.runtime import RuntimeBundle, SDKBundle, TokenBudget, TokenLedger
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig, enumerate_task
from duraseed.training.capability_dose_evidence import EPOCH_UPDATES


class _Input:
    def __init__(self, values: list[int], text: str = "") -> None:
        self.values, self.text, self.length = values, text, len(values)

    def to_ints(self) -> list[int]:
        return list(self.values)


class _SamplingParams:
    def __init__(self, **values) -> None:  # type: ignore[no-untyped-def]
        self.values = values


class _Renderer:
    def build_generation_prompt(self, messages, *, role):  # type: ignore[no-untyped-def]
        return _Input([1, 2], messages[0]["content"])

    def get_stop_sequences(self) -> list[str]:
        return ["</answer>"]

    def parse_response(self, tokens):  # type: ignore[no-untyped-def]
        return None, "stop_sequence"


class _Tokenizer:
    eos_token_id = None

    def decode(self, tokens) -> str:  # type: ignore[no-untyped-def]
        return "".join(chr(value) for value in tokens)


class _Sampler:
    def __init__(self, answer: str) -> None:
        self.answer, self.calls = answer, 0

    async def sample_async(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        tokens = [ord(value) for value in self.answer]
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=tokens,
                    logprobs=[-0.1] * len(tokens),
                    stop_reason="stop",
                )
            ]
        )


def _runtime() -> RuntimeBundle:
    sdk = SDKBundle(
        SimpleNamespace(SamplingParams=_SamplingParams),
        None,
        None,
        None,
        "0.25.0",
        "0.5.3",
    )
    return RuntimeBundle(sdk, object(), object(), _Renderer(), _Tokenizer())


def _manifest(split: str = "a_validation"):
    config = TCESGeneratorConfig(
        n_operands=3,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_valid_expressions=1_000,
        split=split,
    )
    from duraseed.provenance import derive_namespaced_seed

    seed = 11
    generated = TCESGenerator(
        derive_namespaced_seed(seed, "dataset.tces.split", split), config
    ).generate()
    record = build_tces_record(generated)
    return build_manifest(
        name=f"pilot-test-{split}",
        split=split,
        generator_version=record.generator_version,
        root_seed=seed,
        records=[record],
    )


def test_fake_runtime_exercises_sampling_join_and_completed_resume(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    record = manifest.records[0]
    expression = enumerate_task(record.to_task()).expressions[0].canonical_expression
    sampler = _Sampler(f"<answer>{expression}</answer>")
    ledger = TokenLedger(TokenBudget(100, 10_000, 0), 25.0)
    inputs = SimpleNamespace(
        runtime=_runtime(),
        ledger=ledger,
        run_id="pilot-fake",
        config=SimpleNamespace(evaluation={"temperature": 1.0, "top_p": 0.95}),
    )
    source = SimpleNamespace(
        seed=11,
        prompt_pools=SimpleNamespace(
            artifact=SimpleNamespace(
                boundary_family_ids=(record.intended_family,),
                sentinel_family_ids=("unused",),
            )
        ),
    )
    arguments = dict(
        manifest=manifest,
        sampler=sampler,
        sampler_path="tinker://fake/stage-a",
        origin_sampler_path="tinker://fake/origin",
        method="B-S",
        checkpoint_stage="stage_a",
        training_step=50,
        label="pilot-fake-stage-a",
        samples_per_item=2,
        max_tokens=128,
        seed_namespace="pilot0.test.validation",
        output=tmp_path / "evaluation",
    )
    result = asyncio.run(evaluate_manifest(inputs, source, **arguments))
    assert result["row_count"] == 2
    assert result["item_counts"][0]["successes"] == 2
    assert result["generation_sha256"] != result["reward_sha256"]
    assert sampler.calls == 2

    resumed = asyncio.run(evaluate_manifest(inputs, source, **arguments))
    assert resumed == result
    assert sampler.calls == 2
    changed = {**arguments, "training_step": 25}
    with pytest.raises(RunnerGateError, match="row mismatch"):
        asyncio.run(evaluate_manifest(inputs, source, **changed))

    extension = asyncio.run(
        evaluate_manifest(
            inputs,
            source,
            **{
                **arguments,
                "output": tmp_path / "extension",
                "sample_index_start": 2,
            },
        )
    )
    rows = [
        json.loads(row)
        for row in (tmp_path / "extension" / "generations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["sample_index"] for row in rows] == [2, 3]
    assert {row["sampling_seed"] for row in rows}.isdisjoint(
        json.loads(row)["sampling_seed"]
        for row in (tmp_path / "evaluation" / "generations.jsonl")
        .read_text()
        .splitlines()
    )
    assert extension["samples_per_item"] == 2


def _counts(target: tuple[int, int], sentinel: tuple[int, int]) -> dict:
    return {
        "item_counts": [
            {
                "task_id": "target",
                "panel_role": "targeted",
                "successes": target[0],
                "trials": target[1],
            },
            {
                "task_id": "sentinel",
                "panel_role": "sentinel",
                "successes": sentinel[0],
                "trials": sentinel[1],
            },
        ]
    }


def test_primary_analysis_uses_raw_gain_auc_and_fixed_budget_retention() -> None:
    maps = tuple(
        {
            "item_counts": [
                {
                    "task_id": "maps",
                    "panel_role": "stage-b",
                    "successes": index,
                    "trials": 16,
                }
            ]
        }
        for index in range(len(STAGE_B_GRID))
    )
    result = summarize_method(
        seed=11,
        method="B-S",
        m0_validation=_counts((2, 16), (2, 16)),
        m0_monitor=_counts((1, 4), (1, 4)),
        stage_a_final=_counts((12, 16), (3, 16)),
        stage_b_maps=maps,
        stage_b_retention=tuple(_counts((3, 4), (1, 4)) for _ in STAGE_B_GRID),
        stage_b_final_retention=_counts((9, 16), (2, 16)),
    )
    assert result.targeted_acquisition_gain > result.sentinel_acquisition_gain
    assert result.maps_auc.raw_gain_auc > 0
    assert result.fixed_budget_targeted_retention < result.pre_b_targeted_score
    assert result.targeted_monitor_retention_absolute_auc == pytest.approx(0.7)
    assert result.sentinel_monitor_retention_absolute_auc == pytest.approx(0.3)
    assert len(result.targeted_retention_curve) == len(STAGE_B_GRID)
    assert len(result.targeted_cover_curve) == 7
    assert next(row for row in result.targeted_cover_curve if row["tau"] == 0.25)
    assert result.matched_a_selection == "pending_post_pilot_target_freeze"
    other = replace(result, method="B-G")
    aggregate = paired_primary_aggregate(
        (result, other, replace(result, seed=29), replace(other, seed=29))
    )
    assert aggregate["degrees_of_freedom"] == 1
    assert aggregate["mean_raw_gain_stage_b_auc_difference"] == 0
    assert aggregate["mean_targeted_monitor_retention_absolute_auc_difference"] == 0


def test_selected_summary_reports_the_frozen_pass_at_k_schedule() -> None:
    maps = tuple(_counts((8, 16), (8, 16)) for _ in STAGE_B_GRID)
    for result in maps:
        for row in result["item_counts"]:
            row["panel_role"] = "stage-b"
    result = summarize_selected_method(
        seed=11,
        method="B-S",
        stage_a_selected=_counts((8, 16), (4, 16)),
        stage_b_maps=maps,
        stage_b_retention=tuple(_counts((2, 4), (1, 4)) for _ in STAGE_B_GRID),
        stage_b_final_retention=_counts((6, 16), (3, 16)),
    )
    assert set(result["F1_retention"]["pre_b_pass_at_k"]["targeted"]) == {
        "1",
        "4",
        "16",
    }
    assert set(result["monitor_retention"]["targeted_pass_at_k_curve"][0]) == {
        "1",
        "4",
    }
    assert set(result["F2_stage_b_learning"]["maps_pass_at_k_curve"][0]) == {
        "1",
        "4",
        "16",
    }


def test_budget_preflight_counts_one_frozen_pair(monkeypatch) -> None:
    record = SimpleNamespace(task_family="tces", to_task=lambda: object(), task_id="x")
    manifest = SimpleNamespace(records=(record,), record_count=512)
    monitor = SimpleNamespace(records=(record,), record_count=384)
    datum = SimpleNamespace(model_input=SimpleNamespace(length=100))
    runtime = _runtime()
    source = SimpleNamespace(
        seed=11,
        a_cadence=SimpleNamespace(records=(record,), record_count=192),
        a_validation=manifest,
        b_validation=manifest,
        prompt_pools=SimpleNamespace(
            a_monitor_manifest=monitor,
            artifact=SimpleNamespace(bs_slot_order=("x",), bg_group_order=("x",)),
        ),
    )
    inputs = SimpleNamespace(
        runtime=runtime,
        ledger=TokenLedger(TokenBudget(10**12, 10**12, 10**12), 600.0),
        acquisition=SimpleNamespace(selected_max_tokens=4096),
        source=source,
    )
    monkeypatch.setattr("duraseed.pilot0_budget.sft_datum", lambda *a, **k: datum)
    monkeypatch.setattr(
        "duraseed.pilot0_budget.ordered_stage_a_pools", lambda *a: {"x": (record,)}
    )
    monkeypatch.setattr(
        "duraseed.pilot0_budget.stage_a_solver_sources", lambda *a: {"x": object()}
    )
    schedule_steps = []

    def scheduled(*args):  # type: ignore[no-untyped-def]
        schedule_steps.append(args[2])
        return (record,)

    monkeypatch.setattr("duraseed.pilot0_budget.scheduled_stage_a_records", scheduled)
    monkeypatch.setattr(
        "duraseed.pilot0_budget.stage_b_sources", lambda *a: (object(),)
    )
    monkeypatch.setattr("duraseed.pilot0_budget._prompt_length", lambda *a: 100)
    budget = calculate_pilot0_budget(inputs)
    assert budget.fixed_storage_usd == pytest.approx(9.7)
    assert budget.rerun_policy == "no_rerun_without_new_authorization"
    assert budget.passed is False
    assert schedule_steps[:294] == [
        ((step - 1) % EPOCH_UPDATES) + 1 for step in range(1, 295)
    ]
    assert schedule_steps[294:] == list(range(1, 51))


def test_main_hard_stops_before_orchestration_when_budget_fails(
    monkeypatch, tmp_path: Path
) -> None:
    fake = SimpleNamespace(
        output_root=tmp_path, run_id="pilot-budget-stop", source=object()
    )
    budget = Pilot0Budget(TokenBudget(1, 2, 3), 0.0, 601.0, Decimal("601.00"), False)
    monkeypatch.setattr(pilot0_live, "validate_pilot0_inputs", lambda value: value)
    monkeypatch.setattr(pilot0_live, "write_pilot_seed_sources", lambda *a: None)
    monkeypatch.setattr(
        pilot0_live,
        "_preflight",
        lambda inputs, root: {
            "budget_gate_passed": False,
            "authorized_usd": 600.0,
            "upper_bound_usd": budget.upper_bound_usd,
        },
    )

    async def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("paid orchestration must not start")

    monkeypatch.setattr(pilot0_live, "run_stage_a_seed", forbidden)
    with pytest.raises(RunnerGateError, match="exact authorization"):
        asyncio.run(run_pilot0(fake))
    state = json.loads((tmp_path / fake.run_id / "run.json").read_text())
    assert state["status"] == "blocked_budget"
