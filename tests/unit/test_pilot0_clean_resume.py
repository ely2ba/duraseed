from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from duraseed.pilot0_recovery import (
    pilot0_session_ids,
    prepare_pilot0_resume,
    reconciled_evaluation,
    recovery_segment,
)
from duraseed.provenance import canonical_json_bytes
from duraseed.runtime import PRICE_SNAPSHOT, TokenBudget, UsageQuantities


def _cost(tokens: TokenBudget) -> float:
    return (
        PRICE_SNAPSHOT.cost(
            UsageQuantities(
                prefill_tokens=tokens.prefill,
                sample_tokens=tokens.sample,
                train_tokens=tokens.train,
            )
        )
        + 0.1
    )


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def test_clean_checkpoint_pause_appends_resume_lineage(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    segment = root / "seed-29/B-S/stage-b/steps-2-5"
    evaluation = segment / "a-retention"
    committed = TokenBudget(100, 200, 30)
    observed = TokenBudget(90, 160, 30)
    _write(
        root / "run.json",
        {
            "run_id": root.name,
            "status": "interrupted",
            "ledger": {
                "committed_tokens": {
                    "prefill": committed.prefill,
                    "sample": committed.sample,
                    "train": committed.train,
                },
                "observed_tokens": {
                    "prefill": observed.prefill,
                    "sample": observed.sample,
                    "train": observed.train,
                },
                "committed_cost_usd": _cost(committed),
                "observed_cost_usd": _cost(observed),
            },
        },
    )
    _write(
        root / "preflight.json",
        {"run_id": root.name, "lineage": {"session_id": "primary-session"}},
    )
    _write(
        root / "infrastructure-recovery.json",
        {
            "schema_version": "duraseed-pilot0-infrastructure-recovery-v1",
            "status": "authorized_resume",
            "run_id": root.name,
            "recovery_session_id": "first-recovery-session",
            "resume_ledger": {
                "committed_tokens": {"prefill": 1, "sample": 2, "train": 3},
                "observed_tokens": {"prefill": 1, "sample": 2, "train": 3},
                "fixed_usd": 0.0,
            },
        },
    )
    _write(
        evaluation / "remote-call-state.json",
        {
            "completed_count": 0,
            "attempt_started_at_utc": "2026-09-02T17:00:00+00:00",
            "local_pause": True,
            "reserved_floor": {
                "prefill_tokens": 0,
                "sample_tokens": 0,
                "train_tokens": 0,
                "fixed_usd": 0.0,
            },
            "pending": {
                "sequence": 0,
                "operation": "pilot0-local-pause",
                "coordinate": {"reason": "migration"},
                "reservation": {
                    "prefill_tokens": 0,
                    "sample_tokens": 0,
                    "train_tokens": 0,
                    "fixed_usd": 0.0,
                },
            },
        },
    )
    (segment / "remote-calls.jsonl").write_text(
        json.dumps(
            {
                "sequence": 4,
                "status": "completed",
                "operation": "pilot0-save-checkpoint-pair",
                "sampler_path": "tinker://session/sampler_weights/stage-b-step-5",
                "state_path": "tinker://session/weights/stage-b-step-5",
            }
        )
        + "\n"
    )

    resume = prepare_pilot0_resume(
        root,
        recovery_session_id="second-recovery-session",
        recovery_git_commit="commit-2",
    )
    assert resume["kind"] == "clean_checkpoint_pause"
    assert resume["paused_session_id"] == "first-recovery-session"
    assert resume["sampler_path"].endswith("stage-b-step-5")
    assert pilot0_session_ids(root, "primary-session") == [
        "primary-session",
        "first-recovery-session",
        "second-recovery-session",
    ]
    state = json.loads((evaluation / "remote-call-state.json").read_text())
    assert state["pending"] is None and state["local_pause"] is False
    inputs = SimpleNamespace(output_root=tmp_path, run_id=root.name)
    assert recovery_segment(inputs, segment) == resume
    assert reconciled_evaluation(inputs, evaluation) is True
