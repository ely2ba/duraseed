from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.data.manifests import build_manifest, build_tces_record
from duraseed.pilot0_analysis import paired_primary_aggregate, summarize_method
from duraseed.pilot0_budget import Pilot0Budget, calculate_pilot0_budget
from duraseed.pilot0_contract import STAGE_B_GRID
from duraseed.runners import RunnerGateError
from duraseed.runners import pilot0_live
from duraseed.runners.pilot0_live import run_pilot0
from duraseed.runners.pilot0_sampling import evaluate_manifest
from duraseed.runtime import RuntimeBundle, SDKBundle, TokenBudget, TokenLedger
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig, enumerate_task


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


def test_budget_preflight_counts_frozen_full_path_and_exceeds_cap(monkeypatch) -> None:
    record = SimpleNamespace(task_family="tces", to_task=lambda: object(), task_id="x")
    manifest = SimpleNamespace(records=(record,), record_count=512)
    monitor = SimpleNamespace(records=(record,), record_count=384)
    datum = SimpleNamespace(model_input=SimpleNamespace(length=100))
    runtime = _runtime()
    source = SimpleNamespace(
        seed=11,
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
        config=SimpleNamespace(
            teacher_dose=SimpleNamespace(calibration_updates=1),
            stage_a=SimpleNamespace(monitor_samples_per_item=4),
            evaluation={"pilot_samples_per_item": 16},
        ),
        seed_sources=(source, source),
    )
    monkeypatch.setattr("duraseed.pilot0_budget.sft_datum", lambda *a, **k: datum)
    monkeypatch.setattr(
        "duraseed.pilot0_budget.boundary_teacher_sources", lambda *a: (object(),)
    )
    monkeypatch.setattr(
        "duraseed.pilot0_budget.ordered_stage_a_pools", lambda *a: {"x": (record,)}
    )
    monkeypatch.setattr(
        "duraseed.pilot0_budget.stage_a_solver_sources", lambda *a: {"x": object()}
    )
    monkeypatch.setattr(
        "duraseed.pilot0_budget.scheduled_stage_a_records", lambda *a: (record,)
    )
    monkeypatch.setattr(
        "duraseed.pilot0_budget.stage_b_sources", lambda *a: (object(),)
    )
    monkeypatch.setattr("duraseed.pilot0_budget._prompt_length", lambda *a: 100)
    budget = calculate_pilot0_budget(inputs)
    assert budget.fixed_storage_usd == pytest.approx(19.4)
    assert budget.rerun_reservation_usd == 0
    assert budget.rerun_policy == "no_rerun_without_new_authorization"
    assert budget.upper_bound_usd > 600
    assert budget.passed is False


def test_main_hard_stops_before_orchestration_when_budget_fails(
    monkeypatch, tmp_path: Path
) -> None:
    fake = SimpleNamespace(output_root=tmp_path, run_id="pilot-budget-stop")
    budget = Pilot0Budget(TokenBudget(1, 2, 3), 0.0, 600.0, 601.0, False)
    monkeypatch.setattr(pilot0_live, "validate_pilot0_inputs", lambda value: value)
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
    with pytest.raises(RunnerGateError, match=r"\$600"):
        asyncio.run(run_pilot0(fake))
    state = json.loads((tmp_path / fake.run_id / "run.json").read_text())
    assert state["status"] == "blocked_budget"


def test_main_rejects_ledger_not_equal_to_full_path_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    ledger = TokenLedger(TokenBudget(9, 9, 9), 600.0)
    fake = SimpleNamespace(output_root=tmp_path, run_id="pilot-ledger", ledger=ledger)
    monkeypatch.setattr(pilot0_live, "validate_pilot0_inputs", lambda value: value)

    def preflight(inputs, root):  # type: ignore[no-untyped-def]
        (root / "preflight.json").write_text("{}")
        return {
            "budget_gate_passed": True,
            "token_budget": {"prefill": 1, "sample": 2, "train": 3},
        }

    monkeypatch.setattr(pilot0_live, "_preflight", preflight)
    with pytest.raises(RunnerGateError, match="equal the full-path"):
        asyncio.run(run_pilot0(fake))
