from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.calibration_stage_a_terminal import StageAScientificFailure
from duraseed.provenance import canonical_json_hash
from duraseed.run_records import TrainingMetricRecord
from duraseed.runners.stage_a_amended_arms import paired_answer_tag_items
from duraseed.runners.stage_a_result import final_envelope
from duraseed.runners.stage_a_updates import Branch
from duraseed.runtime import TokenBudget
from duraseed.training.stage_a_amended import (
    AMENDED_STAGE_A_LEARNING_RATES,
    AmendedStageAFinalEvidence,
    AmendedStageAScreenEvidence,
    StageAAnswerTagPairedItemEvidence,
    assess_amended_stage_a_final,
    assess_amended_stage_a_screen,
    decide_amended_stage_a_duration,
    decide_amended_stage_a_screen,
)
from duraseed.training.stage_a_calibration import StageADurationDecisionStatus
from tests.unit.teacher_allocation_fixtures import freezer_sources
from tests.unit.test_calibration_live_collectors import CONFIG, _inputs


def _items(
    prefix: str,
    *,
    samples: int,
    successes: set[int] | None = None,
    origin_successes: set[int] | None = None,
    tag_failures: set[int] | None = None,
    origin_tag_failures: set[int] | None = None,
    length_stops: set[int] | None = None,
) -> tuple[StageAAnswerTagPairedItemEvidence, ...]:
    successes = successes or set()
    origin_successes = origin_successes or set()
    tag_failures = tag_failures or set()
    origin_tag_failures = origin_tag_failures or set()
    length_stops = length_stops or set()
    return tuple(
        StageAAnswerTagPairedItemEvidence(
            f"{prefix}-{index}",
            f"family-{index // 8}",
            tuple(1000 + index * samples + sample for sample in range(samples)),
            (index in origin_successes,) * samples,
            (index in successes,) * samples,
            (index not in origin_tag_failures,) * samples,
            (index not in tag_failures,) * samples,
            (False,) * samples,
            (index in length_stops,) * samples,
        )
        for index in range(96)
    )


def _metrics(method: str, final_step: int) -> tuple[TrainingMetricRecord, ...]:
    return tuple(
        TrainingMetricRecord(
            phase="stage_a",
            training_step=step,
            metrics={
                "loss": 0.5,
                **({"mixed_group_rate": 0.25} if method == "B-G" else {}),
            },
        )
        for step in range(1, final_step + 1)
    )


def _screen(
    method: str,
    *,
    tag_failures: set[int] | None = None,
    sentinel_tag_failures: set[int] | None = None,
) -> AmendedStageAScreenEvidence:
    rate = AMENDED_STAGE_A_LEARNING_RATES[method]  # type: ignore[index]
    return AmendedStageAScreenEvidence(
        method,  # type: ignore[arg-type]
        rate,
        "target-panel",
        "sentinel-panel",
        "tinker://m0",
        f"tinker://{method}/step-10",
        _items("target", samples=1, tag_failures=tag_failures),
        _items("sentinel", samples=1, tag_failures=sentinel_tag_failures),
        _metrics(method, 10),
        True,
    )


def _final(method: str) -> AmendedStageAFinalEvidence:
    rate = AMENDED_STAGE_A_LEARNING_RATES[method]  # type: ignore[index]
    return AmendedStageAFinalEvidence(
        method,  # type: ignore[arg-type]
        rate,
        "target-panel",
        "sentinel-panel",
        "tinker://m0",
        f"tinker://{method}/step-50",
        _items(
            "target",
            samples=2,
            successes={index for index in range(96) if index % 8 < 4},
        ),
        _items("sentinel", samples=1),
        _metrics(method, 50),
        True,
    )


def test_valid_answer_tag_gate_is_paired_to_m0_and_sentinel_is_descriptive() -> None:
    passing = assess_amended_stage_a_screen(
        _screen(
            "B-S",
            tag_failures=set(range(9)),
            sentinel_tag_failures=set(range(96)),
        )
    )
    failing = assess_amended_stage_a_screen(_screen("B-S", tag_failures=set(range(10))))

    assert passing.target_health.origin_valid_answer_tag_rate == 1.0
    assert passing.target_health.current_valid_answer_tag_rate == 87 / 96
    assert passing.target_health.valid_answer_tag_retention_passed
    assert passing.sentinel_health.current_valid_answer_tag_rate == 0.0
    assert passing.eligible
    assert not failing.target_health.valid_answer_tag_retention_passed
    assert not failing.eligible


def test_amended_final_keeps_capability_sentinel_and_rollout_gates() -> None:
    screens = tuple(
        decide_amended_stage_a_screen(method, (_screen(method),))
        for method in ("B-S", "B-G")
    )
    finals = tuple(_final(method) for method in ("B-S", "B-G"))

    decision = decide_amended_stage_a_duration(screens, finals)
    bg = assess_amended_stage_a_final(finals[1])

    assert decision.status is StageADurationDecisionStatus.FROZEN
    assert decision.selected_max_updates == 50
    assert bg.target_paired_gain.mean_change == 0.5
    assert bg.sentinel_paired_change.mean_change == 0.0
    assert bg.final_ten_mixed_group_rate == 0.25
    assert {row.name for row in bg.criteria} >= {
        "positive_target_paired_gain",
        "target_gain_approximate_95_lower_bound",
        "catastrophic_sentinel_drop",
        "target_valid_answer_tag_retention",
    }


def test_pairing_uses_verifier_valid_answer_tag_not_outer_wrapper() -> None:
    def observation(tag: bool, *, sample_index: int = 0):
        return SimpleNamespace(
            generation=SimpleNamespace(
                task_id="task-1",
                sampling_seed=17,
                sample_index=sample_index,
                assigned_family_id="family-1",
                stop_reason="stop",
                completion_text="derivation\n<answer>1</answer>",
            ),
            reward=SimpleNamespace(
                reward=1.0,
                exact_verification=SimpleNamespace(valid_answer_tag=tag),
            ),
        )

    item = paired_answer_tag_items((observation(True),), (observation(False),))[0]

    assert item.origin_valid_answer_tags == (True,)
    assert item.current_valid_answer_tags == (False,)


def test_two_selected_arms_continue_the_same_branches_to_step_50(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from duraseed.runners import stage_a_amended_live as live

    sources = freezer_sources(CONFIG)
    artifact = SimpleNamespace(
        family_panel_artifact_id=canonical_json_hash(sources.panel),
        targeted_panel=SimpleNamespace(value="target-panel"),
        sentinel_panel=SimpleNamespace(value="sentinel-panel"),
    )
    inputs = _inputs(
        prompt_pools=SimpleNamespace(
            a_rl_train_manifest=sources.a_rl_train_manifest,
            a_monitor_manifest=sources.a_monitor_manifest,
            artifact=artifact,
        )
    )
    branches: dict[str, Branch] = {}

    async def origin(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(boundary_state_path="state")

    async def screen(*_args, method, learning_rate, **_kwargs):  # type: ignore[no-untyped-def]
        assert learning_rate == AMENDED_STAGE_A_LEARNING_RATES[method]
        branch = Branch(method, learning_rate, SimpleNamespace())
        branch.metrics.extend(_metrics(method, 10))
        branches[method] = branch
        return _screen(method), branch

    async def final(*_args, method, learning_rate, branch, **_kwargs):  # type: ignore[no-untyped-def]
        assert branch is branches[method]
        assert len(branch.metrics) == 10
        branch.metrics.extend(_metrics(method, 50)[10:])
        if method == "B-G":
            for step in range(41, 51):
                branch.surprisal_by_step[step] = 0.5
                branch.unique_completions_by_step[step] = {"a"}
                branch.successful_completions_by_step[step] = {"a"}
                branch.valid_families_by_step[step] = {"family"}
        return final_envelope(branch, _final(method))  # type: ignore[arg-type]

    monkeypatch.setattr(live, "build_origin", origin)
    monkeypatch.setattr(live, "run_amended_screen", screen)
    monkeypatch.setattr(live, "run_amended_final", final)
    monkeypatch.setattr(live, "ordered_pools", lambda _bundle: {})
    monkeypatch.setattr(live, "solver_sources", lambda _manifest: ())
    monkeypatch.setattr(
        live,
        "stage_a_budget",
        lambda *args, **kwargs: SimpleNamespace(
            tokens=TokenBudget(0, 0, 0),
            fixed_storage_usd=0.0,
            upper_bound_usd=0.0,
        ),
    )

    evidence, common = asyncio.run(
        live.collect_amended_stage_a(
            inputs, tmp_path, preflight_sha256="sha256:" + "a" * 64
        )
    )

    assert len(evidence.bs_screens) == len(evidence.bg_screens) == 1
    assert {len(branch.metrics) for branch in branches.values()} == {50}
    assert common.final_ten_mixed_group_rate == 0.25


def test_failed_step10_gate_is_terminal_before_any_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from duraseed.runners import stage_a_amended_live as live

    sources = freezer_sources(CONFIG)
    artifact = SimpleNamespace(
        family_panel_artifact_id=canonical_json_hash(sources.panel),
        targeted_panel=SimpleNamespace(value="target-panel"),
        sentinel_panel=SimpleNamespace(value="sentinel-panel"),
    )
    inputs = _inputs(
        prompt_pools=SimpleNamespace(
            a_rl_train_manifest=sources.a_rl_train_manifest,
            a_monitor_manifest=sources.a_monitor_manifest,
            artifact=artifact,
        )
    )

    async def screen(*_args, method, learning_rate, **_kwargs):  # type: ignore[no-untyped-def]
        branch = Branch(method, learning_rate, SimpleNamespace())
        branch.metrics.extend(_metrics(method, 10))
        failures = set(range(10)) if method == "B-S" else set()
        return _screen(method, tag_failures=failures), branch

    async def no_final(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("step-10 failure must stop before continuation")

    async def origin(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(boundary_state_path="state")

    monkeypatch.setattr(live, "build_origin", origin)
    monkeypatch.setattr(live, "run_amended_screen", screen)
    monkeypatch.setattr(live, "run_amended_final", no_final)
    monkeypatch.setattr(live, "ordered_pools", lambda _bundle: {})
    monkeypatch.setattr(live, "solver_sources", lambda _manifest: ())
    monkeypatch.setattr(
        live,
        "stage_a_budget",
        lambda *args, **kwargs: SimpleNamespace(
            tokens=TokenBudget(0, 0, 0),
            fixed_storage_usd=0.0,
            upper_bound_usd=0.0,
        ),
    )

    with pytest.raises(StageAScientificFailure) as failure:
        asyncio.run(
            live.collect_amended_stage_a(
                inputs, tmp_path, preflight_sha256="sha256:" + "b" * 64
            )
        )

    assert failure.value.status == "no_eligible_learning_rate"
    assert failure.value.screen_only
