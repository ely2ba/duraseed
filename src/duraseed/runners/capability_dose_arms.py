"""One continuous six-epoch B-S capability-dose trajectory."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes
from duraseed.run_records import append_jsonl
from duraseed.runners import RunnerGateError
from duraseed.runners.capability_dose_evidence import panel_evidence
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners.stage_a_arms import observations
from duraseed.runners.stage_a_evidence import evaluate_panel, ordered_pools
from duraseed.runners.stage_a_setup import StageAOriginEvidence
from duraseed.runners.stage_a_updates import restore_branch, supervised_update
from duraseed.runtime import (
    TokenBudget,
    bind_model,
    restore_checkpoint,
    save_checkpoint,
    set_and_verify_ttl,
)
from duraseed.training.capability_dose import decide_dose
from duraseed.training.capability_dose_evidence import (
    CADENCE_UPDATES,
    DOSE_LEARNING_RATE,
    EPOCH_UPDATES,
    MAX_UPDATES,
    CapabilityDoseLiveEvidence,
    DoseEvaluationEvidence,
)


CHECKPOINT_TTL_SECONDS = 7 * 24 * 60 * 60


async def _sampler(
    inputs: Any, branch: Any, output: Path, journal: RemoteJournal, step: int
):
    journal.begin(
        "capability-dose-evaluation-sampler",
        {"method": "B-S", "step": step},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0, "fixed_usd": 0.05},
    )
    inputs.stage_a_ledger.reserve_call(TokenBudget(0, 0, 0), fixed_usd=0.05)
    try:
        sampler = (
            await branch.runtime.model.save_weights_and_get_sampling_client_async()
        )
    except Exception:
        inputs.stage_a_ledger.abort_call()
        raise
    inputs.stage_a_ledger.settle_call(TokenBudget(0, 0, 0))
    path = str(
        getattr(
            sampler,
            "model_id",
            f"ephemeral:{inputs.run_id}:capability-dose:B-S:step-{step}",
        )
    )
    append_jsonl(
        output / "branch-events.jsonl",
        {"event": "evaluation-sampler", "method": "B-S", "step": step, "path": path},
    )
    journal.complete({"operation": "capability-dose-evaluation-sampler", "path": path})
    return sampler, path


async def _evaluation(
    inputs: Any,
    origin: StageAOriginEvidence,
    branch: Any,
    sampler: Any,
    sampler_path: str,
    output: Path,
    journal: RemoteJournal,
    *,
    step: int,
    phase: str,
) -> DoseEvaluationEvidence:
    samples = 1 if phase == "cadence" else 2
    target = await evaluate_panel(
        inputs,
        sampler,
        output,
        role="targeted",
        samples_per_item=samples,
        sampler_path=sampler_path,
        training_step=step,
        label=f"dose-{phase}-step-{step}-target",
        origin_sampler_path=origin.boundary_sampler_path,
        journal=journal,
        method="B-S",
    )
    sentinel = None
    if phase == "cadence":
        sentinel = await evaluate_panel(
            inputs,
            sampler,
            output,
            role="sentinel",
            samples_per_item=1,
            sampler_path=sampler_path,
            training_step=step,
            label=f"dose-cadence-step-{step}-sentinel",
            origin_sampler_path=origin.boundary_sampler_path,
            journal=journal,
            method="B-S",
        )
    origin_target = observations(origin.target_generations, origin.target_rewards)
    origin_sentinel = observations(origin.sentinel_generations, origin.sentinel_rewards)
    return DoseEvaluationEvidence(
        step,
        phase,  # type: ignore[arg-type]
        panel_evidence(origin_target, target),
        panel_evidence(origin_sentinel, sentinel) if sentinel is not None else None,
        tuple(branch.metrics),
        True,
    )


async def _retain_and_restore(
    inputs: Any,
    branch: Any,
    output: Path,
    journal: RemoteJournal,
    step: int,
) -> tuple[str, str]:
    journal.begin(
        "save-capability-dose-checkpoint",
        {"method": "B-S", "step": step},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0, "fixed_usd": 0.10},
    )
    pair = await save_checkpoint(
        branch.runtime,
        name=f"{inputs.run_id}-B-S-dose-step-{step}",
        ttl_seconds=CHECKPOINT_TTL_SECONDS,
        ledger=inputs.stage_a_ledger,
        reserved_storage_usd=0.10,
    )
    append_jsonl(
        output / "checkpoints.jsonl",
        {
            "method": "B-S",
            "learning_rate": DOSE_LEARNING_RATE,
            "step": step,
            "sampler": pair.sampler_path,
            "state": pair.state_path,
            "ttl_seconds": pair.ttl_seconds,
            "selected_for_stage_b": True,
        },
    )
    journal.complete({"operation": "save-capability-dose-checkpoint", "step": step})
    journal.begin(
        "restore-capability-dose-weights-only",
        {"state_path": pair.state_path, "step": step},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    client = await restore_checkpoint(
        inputs.runtime,
        pair.state_path,
        full_state=False,
        ledger=inputs.stage_a_ledger,
        user_metadata={"gate": "capability-dose-restore", "method": "B-S"},
    )
    bind_model(inputs.runtime.sdk, inputs.runtime.service, client)
    journal.complete({"operation": "restore-capability-dose-weights-only"})
    journal.begin(
        "verify-capability-dose-checkpoint-ttl",
        {"step": step},
        {"prefill_tokens": 0, "sample_tokens": 0, "train_tokens": 0},
    )
    ttl = await set_and_verify_ttl(
        inputs.runtime,
        inputs.rest_client,
        (pair.sampler_path, pair.state_path),
        ttl_seconds=CHECKPOINT_TTL_SECONDS,
        ledger=inputs.stage_a_ledger,
    )
    atomic_write_bytes(
        output / "checkpoint-ttl-audit.json",
        canonical_json_bytes(
            {
                "schema_version": "duraseed-capability-dose-ttl-v1",
                "rows": tuple(
                    {
                        **asdict(row),
                        "expires_at": row.expires_at.isoformat(),
                        "checkpoint_type": str(row.checkpoint_type),
                    }
                    for row in ttl
                ),
            }
        ),
    )
    journal.complete({"operation": "verify-capability-dose-checkpoint-ttl"})
    return pair.sampler_path, pair.state_path


async def run_capability_dose_arm(
    inputs: Any,
    origin: StageAOriginEvidence,
    output: Path,
    journal: RemoteJournal,
) -> CapabilityDoseLiveEvidence:
    """Train until a frozen terminal branch, then retain exactly one state pair."""

    branch = await restore_branch(
        inputs,
        origin.boundary_state_path,
        "B-S",
        DOSE_LEARNING_RATE,
        output,
        journal,
    )
    pools = ordered_pools(inputs.prompt_pools)
    evaluations = []
    terminal = None
    for step in range(1, MAX_UPDATES + 1):
        await supervised_update(
            inputs,
            branch,
            step,
            pools,
            inputs.dose_sources,
            output,
            journal,
            schedule_step=((step - 1) % EPOCH_UPDATES) + 1,
        )
        if step not in CADENCE_UPDATES and step != MAX_UPDATES:
            continue
        sampler, path = await _sampler(inputs, branch, output, journal, step)
        phase = "cadence" if step in CADENCE_UPDATES else "epoch_cap"
        evaluations.append(
            await _evaluation(
                inputs,
                origin,
                branch,
                sampler,
                path,
                output,
                journal,
                step=step,
                phase=phase,
            )
        )
        decision = decide_dose(tuple(evaluations))
        if decision.action == "confirm":
            evaluations.append(
                await _evaluation(
                    inputs,
                    origin,
                    branch,
                    sampler,
                    path,
                    output,
                    journal,
                    step=step,
                    phase="confirmation",
                )
            )
            decision = decide_dose(tuple(evaluations))
        if decision.action != "continue":
            terminal = decision
            break
    if terminal is None:
        raise RunnerGateError(
            "capability-dose schedule ended without a terminal branch"
        )
    sampler_path, state_path = await _retain_and_restore(
        inputs, branch, output, journal, terminal.update
    )
    return CapabilityDoseLiveEvidence(
        inputs.prompt_pools.artifact.targeted_panel.value,
        inputs.prompt_pools.artifact.sentinel_panel.value,
        origin.boundary_sampler_path,
        origin.boundary_state_path,
        tuple(evaluations),
        tuple(branch.metrics),
        terminal,
        sampler_path,
        state_path,
        True,
    )


__all__ = ["run_capability_dose_arm"]
