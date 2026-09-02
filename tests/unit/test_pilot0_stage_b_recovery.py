from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from duraseed.provenance import canonical_json_bytes
from duraseed.runners import pilot0_selection, pilot0_stage_b
from duraseed.runtime import TokenBudget, TokenLedger


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def test_stage_b_recovery_skips_retraining_and_reuses_saved_pair(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "pilot"
    output = root / "seed-29/B-S/stage-b/steps-1-2"
    _write(
        root / "infrastructure-recovery.json",
        {
            "schema_version": "duraseed-pilot0-infrastructure-recovery-v1",
            "status": "authorized_resume",
            "run_id": root.name,
            "recovery_session_id": "resume",
            "segment": output.relative_to(root).as_posix(),
            "evaluation": (output / "a-retention").relative_to(root).as_posix(),
            "phase": "stage_b",
            "method": "B-S",
            "start": 1,
            "stop": 2,
            "sampler_path": "tinker://session/sampler_weights/stage-b-step-2",
            "state_path": "tinker://session/weights/stage-b-step-2",
        },
    )
    _write(
        output / "remote-call-state.json",
        {
            "completed_count": 4,
            "attempt_started_at_utc": "2026-09-02T00:00:00+00:00",
            "reserved_floor": {
                "prefill_tokens": 0,
                "sample_tokens": 0,
                "train_tokens": 0,
                "fixed_usd": 0.0,
            },
            "pending": None,
        },
    )
    (output / "remote-calls.jsonl").write_text(
        '{"operation":"saved","sequence":0,"status":"completed"}\n'
    )
    inputs = SimpleNamespace(
        output_root=tmp_path,
        run_id=root.name,
        ledger=TokenLedger(TokenBudget(100, 100, 100), 10),
    )
    source = SimpleNamespace(seed=29)
    monkeypatch.setattr(
        pilot0_stage_b, "stage_b_segment_coordinates", lambda *args, **kwargs: kwargs
    )

    async def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("recovery retrained or resaved the interrupted segment")

    async def evaluate(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {"retention_generation_sha256": "sha256:" + "b" * 64}

    monkeypatch.setattr(pilot0_stage_b, "restore_runtime", forbidden)
    monkeypatch.setattr(pilot0_stage_b, "apply_update", forbidden)
    monkeypatch.setattr(pilot0_stage_b, "save_pair", forbidden)
    monkeypatch.setattr(pilot0_stage_b, "evaluate_stage_b_step", evaluate)
    monkeypatch.setattr(
        pilot0_stage_b, "write_segment", lambda directory, value, **kwargs: value
    )
    result = asyncio.run(
        pilot0_stage_b._train_segment(
            inputs,
            source,
            {},
            {"sampler_path": "step-1-sampler", "state_path": "step-1-state"},
            method="B-S",
            start=1,
            stop=2,
            datums=[],
            output=output,
            preflight_sha256="sha256:" + "c" * 64,
            a_validation_seed_namespace="pilot0.a_validation",
            a_validation_samples_per_item=None,
        )
    )
    assert result["sampler_path"].endswith("stage-b-step-2")
    assert result["state_path"].endswith("stage-b-step-2")


def test_completed_selected_profiles_are_reused_without_remote_calls(
    monkeypatch, tmp_path: Path
) -> None:
    matching = {
        "status": "selected",
        "B-S": {
            "step": 40,
            "sampler_path": "sampler-B-S",
            "state_path": "state-B-S",
            "monitor_generation_sha256": "sha256:" + "a" * 64,
        },
        "B-G": {
            "step": 20,
            "sampler_path": "sampler-B-G",
            "state_path": "state-B-G",
            "monitor_generation_sha256": "sha256:" + "b" * 64,
        },
    }
    inputs = SimpleNamespace(
        config=SimpleNamespace(evaluation={"reliability_tau_report": [0.1]}),
        m0_sampler_path="m0-sampler",
        acquisition=SimpleNamespace(selected_max_tokens=4096),
    )
    source = SimpleNamespace(
        seed=29,
        prompt_pools=SimpleNamespace(a_monitor_manifest=object()),
        a_validation=object(),
    )
    monkeypatch.setattr(pilot0_selection, "_cadence_rows", lambda *args: ())
    monkeypatch.setattr(
        pilot0_selection, "select_paired_cadence", lambda *args: matching
    )

    def evaluation(path: Path) -> dict:
        method = next(part for part in path.parts if part in {"B-S", "B-G"})
        return {
            "sampler_path": f"sampler-{method}",
            "generation_sha256": "sha256:" + "c" * 64,
        }

    async def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("completed selected evidence triggered a remote call")

    monkeypatch.setattr(pilot0_selection, "read_evaluation", evaluation)
    monkeypatch.setattr(pilot0_selection, "sampler_for_path", forbidden)
    monkeypatch.setattr(pilot0_selection, "evaluate_manifest", forbidden)
    monkeypatch.setattr(
        pilot0_selection,
        "pre_b_capability_profile",
        lambda **kwargs: {"method": kwargs["method"]},
    )
    bs, bg, profiles = asyncio.run(
        pilot0_selection.select_and_profile(inputs, source, tmp_path, {}, {})
    )
    assert (bs["selected_step"], bg["selected_step"]) == (40, 20)
    assert profiles == ({"method": "B-S"}, {"method": "B-G"})
