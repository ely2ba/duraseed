from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.config import load_pilot_config
from duraseed.provenance import canonical_json_hash, derive_namespaced_seed
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners.stage_a_updates import (
    Branch,
    _explicit_group_seeds,
    grouped_rl_update,
    supervised_update,
)
from duraseed.runners.teacher_dose_sampling import sample_gate
from duraseed.run_records import TrainingMetricRecord
from duraseed.runtime import TokenBudget, TokenLedger
from tests.unit.teacher_allocation_fixtures import freezer_sources


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_pilot_config(ROOT / "duraseed_pilot_config.yaml")


def _inputs(**changes):  # type: ignore[no-untyped-def]
    source = freezer_sources(CONFIG)
    values = {
        "config": CONFIG,
        "runtime": SimpleNamespace(
            renderer=SimpleNamespace(
                build_generation_prompt=lambda *_args, **_kwargs: SimpleNamespace(
                    length=3
                )
            )
        ),
        "teacher_ledger": TokenLedger(TokenBudget(100_000, 100_000, 100_000), 150),
        "stage_a_ledger": TokenLedger(TokenBudget(100_000, 100_000, 100_000), 150),
        "teacher_sources": source,
        "m0_training_step": 2,
        "m0_sampler_path": "tinker://m0/sampler",
        "m0_state_path": "tinker://m0/state",
        "run_id": "calibration-fake",
        "project_id": "project-fake",
        "tinker_session_id": "session-fake",
        "reconciled_restarts": (),
        "smoke": SimpleNamespace(protocol_max_tokens=4096),
        "max_tokens": SimpleNamespace(selected_max_tokens=256),
        "prompt_pools": SimpleNamespace(
            artifact=SimpleNamespace(
                bs_slot_order=("boundary",), bg_group_order=("boundary",)
            ),
            a_rl_train_manifest=source.a_rl_train_manifest,
            a_monitor_manifest=source.a_monitor_manifest,
        ),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_teacher_gate_uses_archived_coordinates_real_config_and_m0_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from duraseed.runners import teacher_dose_sampling as sampling

    inputs = _inputs()
    captured = []

    async def fake_sample(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append((args[3], kwargs))
        return ()

    monkeypatch.setattr(sampling, "sample_seeded", fake_sample)
    journal = RemoteJournal(tmp_path)
    record = inputs.teacher_sources.gate_manifest.records[0]
    asyncio.run(
        sample_gate(
            inputs,
            object(),
            (record,),
            seed=17,
            group_size=1,
            sampler_path=inputs.m0_sampler_path,
            checkpoint_stage="m0",
            role="sentinel",
            output_directory=tmp_path,
            journal=journal,
        )
    )

    coordinates, settings = captured[0]
    assert coordinates.seed_namespace == "tinker.teacher_dose.seed-17.gate"
    assert coordinates.training_step == 2
    assert settings["temperature"] == CONFIG.evaluation["temperature"]
    assert settings["top_p"] == CONFIG.evaluation["top_p"]


def test_stage_a_monitor_uses_one_archived_seed_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from duraseed.runners import stage_a_evidence as evidence

    inputs = _inputs()
    record = inputs.teacher_sources.gate_manifest.records[0]
    captured = []

    async def fake_sample(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append((args[3], kwargs))
        return ()

    monkeypatch.setattr(evidence, "_monitor_records", lambda *args: (record,))
    monkeypatch.setattr(evidence, "sample_seeded", fake_sample)
    asyncio.run(
        evidence.evaluate_panel(
            inputs,
            object(),
            tmp_path,
            role="targeted",
            samples_per_item=1,
            sampler_path="tinker://candidate",
            training_step=10,
            label="screen",
            origin_sampler_path="tinker://origin",
            journal=RemoteJournal(tmp_path / "monitor"),
            method="B-S",
        )
    )

    coordinates, settings = captured[0]
    assert coordinates.seed_namespace == "tinker.stage_a.seed-17.monitor"
    assert coordinates.experiment_seed == 17
    assert settings["temperature"] == CONFIG.evaluation["temperature"]


def test_teacher_collector_reuses_one_m0_baseline_for_every_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from duraseed.runners import teacher_dose_live as live
    from duraseed.training.acquisition_freeze import TeacherDoseArmEvidence
    from tests.unit.teacher_dose_fixtures import _assessment

    inputs = _inputs()
    baselines: dict[int, object] = {}
    seen: list[tuple[int, int, float, object]] = []

    async def fake_baseline(inputs, attempts, *, seed):  # type: ignore[no-untyped-def]
        del inputs, attempts
        return baselines.setdefault(seed, object())

    async def fake_arm(inputs, attempts, baseline, **values):  # type: ignore[no-untyped-def]
        del inputs, attempts
        seen.append((values["seed"], values["dose"], values["learning_rate"], baseline))
        return TeacherDoseArmEvidence(
            values["learning_rate"], _assessment(values["dose"], seed=values["seed"])
        )

    monkeypatch.setattr(
        live, "validate_teacher_allocation_base_sources", lambda value: value
    )
    monkeypatch.setattr(
        live,
        "teacher_dose_budget",
        lambda *args: SimpleNamespace(
            tokens=TokenBudget(0, 0, 0), fixed_storage_usd=0.0, upper_bound_usd=0.0
        ),
    )
    monkeypatch.setattr(live, "baseline_attempt", fake_baseline)
    monkeypatch.setattr(live, "teacher_arm_attempt", fake_arm)
    result = asyncio.run(
        live.collect_teacher_dose(
            inputs, tmp_path, preflight_sha256="sha256:" + "a" * 64
        )
    )

    assert len(result.calibration_arms) == 3 and len(seen) == 4
    assert {id(row[3]) for row in seen if row[0] == 17} == {id(baselines[17])}
    assert {id(row[3]) for row in seen if row[0] == 37} == {id(baselines[37])}


def test_stage_a_updates_cross_real_sft_and_grouped_rl_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from duraseed.runners import stage_a_updates as updates
    from duraseed.runtime import SampleObservation

    inputs = _inputs()
    record = SimpleNamespace(task_id="task-1", item_index=1, intended_family="family")
    record.to_task = lambda: object()
    pools = {"boundary": (record,)}
    source = object()
    calls: list[tuple[object, ...]] = []

    async def fake_update(runtime, datums, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(
            ("update", kwargs["loss_fn"], kwargs["learning_rate"], len(datums))
        )
        return {"loss": 0.5}

    def fake_datum(runtime, value):  # type: ignore[no-untyped-def]
        del runtime, value
        return SimpleNamespace(model_input=SimpleNamespace(length=4))

    def scheduled(_pools, order, step):  # type: ignore[no-untyped-def]
        return tuple(record for _ in order)

    monkeypatch.setattr(updates, "scheduled_records", scheduled)
    monkeypatch.setattr(updates, "sft_datum", fake_datum)
    monkeypatch.setattr(updates, "apply_update", fake_update)
    sft_branch = Branch("B-S", 1e-4, SimpleNamespace())
    asyncio.run(
        supervised_update(
            inputs,
            sft_branch,
            1,
            pools,  # type: ignore[arg-type]
            {"task-1": source},
            tmp_path,
            RemoteJournal(tmp_path / "sft"),
        )
    )

    class Model:
        async def save_weights_and_get_sampling_client_async(self):
            assert inputs.stage_a_ledger.has_pending_call
            calls.append(("ephemeral",))
            return SimpleNamespace(model_id="ephemeral:fake")

    async def fake_sample(*args, **kwargs):  # type: ignore[no-untyped-def]
        coordinates = args[3]
        explicit = kwargs["explicit_seeds"]
        calls.append(("seeds", coordinates.seed_namespace, explicit))
        rows = []
        for index, seed in enumerate(explicit):
            reward = float(index % 2)
            generation = SimpleNamespace(
                model_copy=lambda **_kwargs: generation,
                advantage=None,
                completion_text=f"completion-{index}",
            )
            verification = SimpleNamespace(strategy_family_id="family")
            rows.append(
                SampleObservation(
                    generation,
                    SimpleNamespace(reward=reward, exact_verification=verification),
                    SimpleNamespace(length=2),
                    (1, 2),
                    (-0.2, -0.2),
                )
            )
        return tuple(rows)

    monkeypatch.setattr(updates, "render_prompt", lambda _task: "prompt")
    monkeypatch.setattr(updates, "sample_seeded", fake_sample)
    monkeypatch.setattr(updates, "append_jsonl", lambda *args: None)
    monkeypatch.setattr(
        updates,
        "rl_datums",
        lambda runtime, rows, advantages: [
            SimpleNamespace(model_input=SimpleNamespace(length=4)) for _ in rows
        ],
    )
    bg_runtime = SimpleNamespace(
        model=Model(),
        renderer=inputs.runtime.renderer,
    )
    bg_branch = Branch("B-G", 1e-5, bg_runtime)
    asyncio.run(
        grouped_rl_update(
            inputs,
            bg_branch,
            4,
            pools,  # type: ignore[arg-type]
            "tinker://boundary/sampler",
            tmp_path,
            RemoteJournal(tmp_path / "bg"),
        )
    )

    expected = tuple(
        derive_namespaced_seed(
            derive_namespaced_seed(17, "tinker.stage_a.bg_rollout", 4, 0, "task-1"),
            "tinker.smoke.group_sample",
            index,
        )
        for index in range(8)
    )
    assert ("seeds", "tinker.stage_a.bg_rollout", expected) in calls
    assert {row[1] for row in calls if row[0] == "update"} == {
        "cross_entropy",
        "importance_sampling",
    }
    assert not inputs.stage_a_ledger.has_pending_call
    assert inputs.stage_a_ledger.committed_fixed_usd == 0.05


def test_explicit_group_seed_helper_matches_archived_nested_contract() -> None:
    root = derive_namespaced_seed(17, "tinker.stage_a.bg_rollout", 9, 3, "task-z")
    assert _explicit_group_seeds(9, 3, "task-z") == tuple(
        derive_namespaced_seed(root, "tinker.smoke.group_sample", index)
        for index in range(8)
    )


def test_stage_a_collector_runs_all_six_screens_and_both_continuations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from duraseed.runners import stage_a_live as live
    from duraseed.runners.stage_a_updates import Branch
    from tests.unit.test_stage_a_calibration_decision import _final, _screen

    sources = freezer_sources(CONFIG)
    artifact = SimpleNamespace(
        family_panel_artifact_id=canonical_json_hash(sources.panel),
        targeted_panel=SimpleNamespace(value="frozen-target"),
        sentinel_panel=SimpleNamespace(value="frozen-sentinel"),
    )
    inputs = _inputs(
        prompt_pools=SimpleNamespace(
            a_rl_train_manifest=sources.a_rl_train_manifest,
            a_monitor_manifest=sources.a_monitor_manifest,
            artifact=artifact,
        )
    )
    branches: list[Branch] = []
    updates: list[tuple[str, float, int]] = []

    async def origin(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return SimpleNamespace(boundary_state_path="state")

    async def screen(
        inputs, origin, output, journal, *, method, learning_rate, **kwargs
    ):  # type: ignore[no-untyped-def]
        del inputs, origin, output, journal, kwargs
        branch = Branch(method, learning_rate, SimpleNamespace())
        for step in range(1, 11):
            values = {"loss": 0.5}
            if method == "B-G":
                values["mixed_group_rate"] = 0.25
            branch.metrics.append(
                TrainingMetricRecord(
                    phase="stage_a", training_step=step, metrics=values
                )
            )
            updates.append((method, learning_rate, step))
        branches.append(branch)
        return _screen(method, learning_rate, set(range(40))), branch

    async def final(
        inputs, origin, output, journal, *, method, learning_rate, branch, **kwargs
    ):  # type: ignore[no-untyped-def]
        del inputs, origin, output, journal, kwargs
        for step in range(11, 51):
            values = {"loss": 0.5}
            if method == "B-G":
                values["mixed_group_rate"] = 0.25
                branch.surprisal_by_step[step] = 0.5
                branch.unique_completions_by_step[step] = {"a", "b"}
                branch.successful_completions_by_step[step] = {"a", "b"}
                branch.valid_families_by_step[step] = {"f1", "f2"}
            branch.metrics.append(
                TrainingMetricRecord(
                    phase="stage_a", training_step=step, metrics=values
                )
            )
            updates.append((method, learning_rate, step))
        from duraseed.runners.stage_a_result import final_envelope

        return final_envelope(branch, _final(method, learning_rate))

    monkeypatch.setattr(live, "build_origin", origin)
    monkeypatch.setattr(live, "run_screen", screen)
    monkeypatch.setattr(live, "run_final", final)
    monkeypatch.setattr(live, "ordered_pools", lambda bundle: {})
    monkeypatch.setattr(live, "solver_sources", lambda manifest: ())
    monkeypatch.setattr(
        live,
        "stage_a_budget",
        lambda *args, **kwargs: SimpleNamespace(
            tokens=TokenBudget(0, 0, 0), fixed_storage_usd=0.0, upper_bound_usd=0.0
        ),
    )
    evidence, common = asyncio.run(
        live.collect_stage_a(
            inputs,
            tmp_path,
            selected_dose=1,
            teacher_learning_rate=1e-4,
            preflight_sha256="sha256:" + "a" * 64,
        )
    )

    assert len(evidence.bs_screens) == len(evidence.bg_screens) == 3
    assert {len(branch.metrics) for branch in branches} == {10, 50}
    assert len(updates) == 140
    assert common.entropy_collapse_gate_passed
    assert common.final_ten_mixed_group_rate == 0.25
    assert (tmp_path / "complete-bounded-stage-a" / "completed.json").exists()
