from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.calibration_attempts import ArmAttempts
from duraseed.calibration_provenance import CANDIDATE_TTL_SECONDS, _ttl_paths
from duraseed.calibration_stage_a_terminal import (
    StageAScientificFailure,
    _validate_failure,
)
from duraseed.run_records import append_jsonl
from duraseed.runners import RunnerGateError
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners.stage_a_amended_live import (
    _completed_update_health_failure,
    _terminalize_update_health_failure,
)
from duraseed.runners.stage_a_update_failure import (
    apply_grouped_update_or_fail,
    update_health_failure,
)
from duraseed.runtime import TokenBudget, TokenLedger
from duraseed.training.stage_a_update_health import StageAUpdateHealthFailure
from duraseed.training_metric_errors import NonFiniteTrainingMetricError


def _branch() -> SimpleNamespace:
    return SimpleNamespace(learning_rate=1e-5, runtime=object())


def test_zero_mixed_failure_is_exact_and_completes_the_attempt(tmp_path: Path) -> None:
    ledger = TokenLedger(TokenBudget(0, 0, 0), 1)
    attempts = ArmAttempts(
        tmp_path,
        ledger,
        run_id="run",
        action="stage-a",
        project_id="project",
        preflight_sha256="sha256:" + "a" * 64,
    )
    attempt = attempts.open("complete-bounded-stage-a")
    failure = update_health_failure(
        attempt.directory,
        _branch(),
        7,
        reason="zero_mixed_group",
        mixed=0,
        all_zero=12,
        all_one=4,
        optimizer_update_completed=False,
    )

    with pytest.raises(StageAScientificFailure) as terminal:
        _terminalize_update_health_failure(attempts, attempt, failure)

    assert terminal.value.status == "update_health_failed"
    assert terminal.value.update_health_failure == failure.evidence
    _validate_failure(terminal.value)
    completed = json.loads(
        (tmp_path / "complete-bounded-stage-a/completed.json").read_bytes()
    )
    assert _completed_update_health_failure(completed["evidence"]) == failure.evidence


def test_nonfinite_metric_is_counted_but_remote_error_remains_pending(
    tmp_path: Path,
) -> None:
    inputs = SimpleNamespace(stage_a_ledger=object())
    datums = [SimpleNamespace(model_input=SimpleNamespace(length=4))]

    async def nonfinite(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise NonFiniteTrainingMetricError("optimizer.loss")

    counted_journal = RemoteJournal(tmp_path / "counted")
    with pytest.raises(StageAUpdateHealthFailure) as counted:
        asyncio.run(
            apply_grouped_update_or_fail(
                inputs,
                _branch(),
                11,
                tmp_path / "counted",
                counted_journal,
                datums,
                nonfinite,
                mixed=8,
                all_zero=4,
                all_one=4,
            )
        )
    state = json.loads((tmp_path / "counted/remote-call-state.json").read_bytes())
    assert state["pending"] is None
    assert counted.value.evidence.optimizer_update_completed
    assert counted.value.evidence.metric_name == "optimizer.loss"

    async def remote_error(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("network failure")

    interrupted_journal = RemoteJournal(tmp_path / "interrupted")
    with pytest.raises(RuntimeError, match="network failure"):
        asyncio.run(
            apply_grouped_update_or_fail(
                inputs,
                _branch(),
                11,
                tmp_path / "interrupted",
                interrupted_journal,
                datums,
                remote_error,
                mixed=8,
                all_zero=4,
                all_one=4,
            )
        )
    state = json.loads((tmp_path / "interrupted/remote-call-state.json").read_bytes())
    assert state["pending"] is not None
    assert not (tmp_path / "interrupted/update-health-failure.json").exists()


@pytest.mark.parametrize(
    ("step", "coordinates"),
    (
        (7, (("B-S", 1e-4, 10),)),
        (11, (("B-S", 1e-4, 10), ("B-G", 1e-5, 10), ("B-S", 1e-4, 50))),
    ),
)
def test_update_health_ttl_retains_exactly_the_already_saved_candidates(
    tmp_path: Path,
    step: int,
    coordinates: tuple[tuple[str, float, int], ...],
) -> None:
    arm = tmp_path / "complete-bounded-stage-a"
    attempt = arm / "attempt-0001"
    attempt.mkdir(parents=True)
    (arm / "completed.json").write_text('{"attempt":1}')
    expected = []
    for method, learning_rate, checkpoint_step in coordinates:
        sampler = f"path-{method}-{checkpoint_step}"
        expected.append(sampler)
        append_jsonl(
            attempt / "checkpoints.jsonl",
            {
                "method": method,
                "learning_rate": learning_rate,
                "step": checkpoint_step,
                "sampler": sampler,
            },
        )
    failure = update_health_failure(
        attempt,
        _branch(),
        step,
        reason="zero_mixed_group",
        mixed=0,
        all_zero=16,
        all_one=0,
        optimizer_update_completed=False,
    ).evidence

    assert _ttl_paths(
        "stage-a",
        tmp_path,
        SimpleNamespace(),
        stage_a_update_health_failure=failure,
    ) == {CANDIDATE_TTL_SECONDS: tuple(expected)}

    append_jsonl(
        attempt / "checkpoints.jsonl",
        {"method": "B-G", "learning_rate": 1e-5, "step": 50, "sampler": "extra"},
    )
    with pytest.raises(RunnerGateError, match="checkpoint lineage differs"):
        _ttl_paths(
            "stage-a",
            tmp_path,
            SimpleNamespace(),
            stage_a_update_health_failure=failure,
        )
