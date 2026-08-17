from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.calibration_provenance import CANDIDATE_TTL_SECONDS, _ttl_paths
from duraseed.calibration_stage_a_terminal import StageAScientificFailure
from duraseed.calibration_sources import (
    ACCEPTED_BOUNDARY_CONFIG_SHA256,
    ACCEPTED_NONPROTOCOL_CONFIG_SHA256,
    _nonprotocol_config_hash,
)
from duraseed.config import load_pilot_config
from duraseed.run_records import (
    RunStatus,
    TrainingMetricRecord,
    append_jsonl,
)
from duraseed.runners import RunnerGateError
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners.stage_a_setup import build_origin
from duraseed.runtime import TokenBudget, TokenLedger
from duraseed.training.stage_a_calibration import (
    STAGE_A_LEARNING_RATE_GRIDS,
    StageALearningRateDecisionStatus,
)
from duraseed.training.stage_a_direct import (
    screen_mean_mixed_group_rate,
    select_direct_m0_learning_rate,
)
from tests.unit.test_stage_a_calibration_decision import _screen


def _bg_screen(learning_rate: float, successes: int, mixed: float):
    evidence = _screen("B-G", learning_rate, set(range(successes)))
    metrics = tuple(
        TrainingMetricRecord(
            phase="stage_a",
            training_step=step,
            metrics={"loss": 0.5, "mixed_group_rate": mixed},
        )
        for step in range(1, 11)
    )
    return replace(evidence, metrics=metrics)


def test_build_origin_uses_the_frozen_m0_pair_directly(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from duraseed.runners import stage_a_setup as setup

    sampler = object()
    calls = []

    async def fake_sampler(inputs, path, output, journal):  # type: ignore[no-untyped-def]
        calls.append(("sampler", path, output, journal))
        return sampler

    async def fake_evaluate(inputs, observed_sampler, output, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(("evaluate", observed_sampler, kwargs))
        row = SimpleNamespace(
            generation=SimpleNamespace(sample_id=f"sample-{kwargs['role']}"),
            reward=SimpleNamespace(sample_id=f"sample-{kwargs['role']}"),
        )
        return (row,)

    monkeypatch.setattr(setup, "create_recorded_sampler", fake_sampler)
    monkeypatch.setattr(setup, "evaluate_panel", fake_evaluate)
    inputs = SimpleNamespace(
        m0_sampler_path="tinker://m0/sampler",
        m0_state_path="tinker://m0/state",
        m0_training_step=2,
    )
    journal = RemoteJournal(tmp_path / "journal")

    result = asyncio.run(build_origin(inputs, tmp_path, journal))

    assert result.boundary_sampler_path == inputs.m0_sampler_path
    assert result.boundary_state_path == inputs.m0_state_path
    assert calls[0][0:2] == ("sampler", inputs.m0_sampler_path)
    evaluations = [row[2] for row in calls if row[0] == "evaluate"]
    assert [row["role"] for row in evaluations] == ["targeted", "sentinel"]
    assert {row["sampler_path"] for row in evaluations} == {inputs.m0_sampler_path}
    assert {row["origin_sampler_path"] for row in evaluations} == {
        inputs.m0_sampler_path
    }
    assert {row["checkpoint_stage"] for row in evaluations} == {"m0"}
    assert {row["training_step"] for row in evaluations} == {2}


def test_direct_m0_bg_screen_excludes_low_mixed_candidate() -> None:
    rates = STAGE_A_LEARNING_RATE_GRIDS["B-G"]
    screens = (
        _bg_screen(rates[0], 60, 0.19),
        _bg_screen(rates[1], 40, 0.20),
        _bg_screen(rates[2], 10, 0.30),
    )

    decision = select_direct_m0_learning_rate("B-G", screens)

    assert decision.status is StageALearningRateDecisionStatus.SELECTED
    assert decision.selected_learning_rate == rates[1]
    assert screen_mean_mixed_group_rate(screens[1]) == 0.20


def test_direct_m0_bg_screen_stops_when_no_lr_has_healthy_rollouts() -> None:
    screens = tuple(
        _bg_screen(rate, 40 - index, 0.19)
        for index, rate in enumerate(STAGE_A_LEARNING_RATE_GRIDS["B-G"])
    )

    decision = select_direct_m0_learning_rate("B-G", screens)

    assert decision.status is StageALearningRateDecisionStatus.NO_ELIGIBLE_CANDIDATE
    assert decision.selected_learning_rate is None


def test_direct_m0_selection_leaves_bs_rule_unchanged() -> None:
    rates = STAGE_A_LEARNING_RATE_GRIDS["B-S"]
    screens = tuple(
        _screen("B-S", rate, set(range(40 - index * 10)))
        for index, rate in enumerate(rates)
    )

    decision = select_direct_m0_learning_rate("B-S", screens)

    assert decision.status is StageALearningRateDecisionStatus.SELECTED
    assert decision.selected_learning_rate == rates[0]


def test_complete_unhealthy_screen_is_a_scientific_terminal() -> None:
    from duraseed.runners.stage_a_live import _resolve_completed
    from duraseed.training.acquisition_freeze import StageALiveEvidence

    bs = tuple(
        _screen("B-S", rate, set(range(30)))
        for rate in STAGE_A_LEARNING_RATE_GRIDS["B-S"]
    )
    bg = tuple(
        _bg_screen(rate, 30, 0.19) for rate in STAGE_A_LEARNING_RATE_GRIDS["B-G"]
    )
    with pytest.raises(StageAScientificFailure) as observed:
        _resolve_completed(StageALiveEvidence(bs, bg, ()), None)

    assert observed.value.status == "no_eligible_learning_rate"
    assert observed.value.screen_only is True


def test_direct_m0_ttl_lineage_contains_only_eight_stage_a_candidates(
    tmp_path: Path,
) -> None:
    arm = tmp_path / "complete-bounded-stage-a"
    attempt = arm / "attempt-0001"
    attempt.mkdir(parents=True)
    (arm / "completed.json").write_text('{"attempt":1}')
    for index in range(8):
        append_jsonl(
            attempt / "checkpoints.jsonl",
            {"method": "B-S" if index < 4 else "B-G", "sampler": f"path-{index}"},
        )

    paths = _ttl_paths("stage-a", tmp_path, SimpleNamespace())

    assert paths == {
        CANDIDATE_TTL_SECONDS: tuple(f"path-{index}" for index in range(8))
    }
    (attempt / "checkpoints.jsonl").write_text("")
    for index in range(6):
        append_jsonl(
            attempt / "checkpoints.jsonl",
            {"method": "B-S" if index < 3 else "B-G", "sampler": f"path-{index}"},
        )
    assert _ttl_paths(
        "stage-a", tmp_path, SimpleNamespace(), stage_a_screen_only=True
    ) == {CANDIDATE_TTL_SECONDS: tuple(f"path-{index}" for index in range(6))}
    (attempt / "checkpoints.jsonl").write_text("")
    with pytest.raises(RunnerGateError, match="omits a candidate"):
        _ttl_paths("stage-a", tmp_path, SimpleNamespace())


def test_live_calibration_executes_and_commits_only_stage_a(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from duraseed.runners import calibration_live as live
    from duraseed.runners import stage_a_live

    events = []
    monkeypatch.setattr(live, "calibration_preflight", lambda *_args: {"run": "new"})
    monkeypatch.setattr(live, "validate_restart_reconciliations", lambda *_args: None)
    monkeypatch.setattr(live, "read_calibration_session_ids", lambda _root: ())
    monkeypatch.setattr(live, "write_calibration_sources", lambda *_args: None)
    monkeypatch.setattr(
        live,
        "start_calibration_run",
        lambda *_args: SimpleNamespace(status=RunStatus.RUNNING),
    )

    async def collect(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        events.append("stage-a")
        return object(), object()

    async def ttl(_action, root, _inputs):  # type: ignore[no-untyped-def]
        root.mkdir(parents=True, exist_ok=True)
        (root / "checkpoint-ttl-audit.json").write_text("{}")

    monkeypatch.setattr(stage_a_live, "collect_stage_a", collect)
    monkeypatch.setattr(live, "seal_calibration_action", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(live, "verify_action_ttls", ttl)
    monkeypatch.setattr(live, "freeze_acquisition", lambda *_args: {"frozen": True})
    monkeypatch.setattr(
        live,
        "finish_calibration_run",
        lambda _inputs, _root, status, **_kwargs: events.append(status.value),
    )
    inputs = SimpleNamespace(
        smoke=SimpleNamespace(protocol_max_tokens=4096),
        config=SimpleNamespace(
            tinker=SimpleNamespace(max_sampled_tokens=4096),
            stage_a=SimpleNamespace(provisional_max_tokens=256),
        ),
        max_tokens=SimpleNamespace(selected_max_tokens=4096),
        run_id="direct-m0-test",
        project_id="project",
        tinker_session_id="session",
        git_commit="commit",
        output_root=tmp_path,
        reconciled_restarts=(),
        teacher_sources=object(),
        prompt_pools=object(),
        stage_a_ledger=TokenLedger(TokenBudget(0, 0, 0), 153.32),
    )

    result = asyncio.run(live.run_live_calibration(inputs))

    assert events == ["stage-a", "completed"]
    assert result["state"]["completed_actions"] == ["stage-a"]
    assert set(result["artifacts"]) == {"stage-a"}


def test_source_config_allows_only_protocol_metadata_drift() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_pilot_config(root / "duraseed_pilot_config.yaml")
    relabeled = config.model_copy(
        update={"protocol": {**config.protocol, "version": "metadata-only-change"}}
    )
    changed_science = config.model_copy(update={"seed": config.seed + 1})

    assert ACCEPTED_BOUNDARY_CONFIG_SHA256 == (
        "sha256:6d0caf9912e1cbafecb1103fe9e4999f62ab9fae4f9d2ee71f34b86f177748c1"
    )
    assert _nonprotocol_config_hash(config) == ACCEPTED_NONPROTOCOL_CONFIG_SHA256
    assert _nonprotocol_config_hash(relabeled) == ACCEPTED_NONPROTOCOL_CONFIG_SHA256
    assert (
        _nonprotocol_config_hash(changed_science) != ACCEPTED_NONPROTOCOL_CONFIG_SHA256
    )
