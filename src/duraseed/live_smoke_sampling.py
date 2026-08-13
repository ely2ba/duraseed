"""Sampling evidence and acceptance reduction for the live smoke gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from duraseed.live_smoke_gate import PROTOCOL_MAX_TOKENS, SmokeRun
from duraseed.run_records import append_jsonl
from duraseed.runtime import (
    MODEL_ID,
    RuntimeBundle,
    SampleObservation,
    SamplingCoordinates,
    SamplingTask,
    TokenLedger,
    sample_seeded,
    sampling_stops,
)
from duraseed.training.reward import verify_task_completion


@dataclass(slots=True)
class SmokeSampler:
    """Collect durable raw rows and reduce only the smoke acceptance claims."""

    run_id: str
    seed: int
    run_directory: Path
    ledger: TokenLedger
    run: SmokeRun
    observations: list[SampleObservation] = field(default_factory=list)

    async def collect(
        self,
        runtime: RuntimeBundle,
        sampler: Any,
        task: SamplingTask,
        *,
        label: str,
        stage: Literal["base", "m0", "stage_a", "stage_b"],
        path: str,
        count: int,
        max_tokens: int,
    ) -> tuple[SampleObservation, ...]:
        purpose: Literal["training", "evaluation"] = (
            "evaluation" if "eval" in label or "probe" in label else "training"
        )
        coordinates = SamplingCoordinates(
            self.run_id,
            label,
            purpose,
            stage,
            0,
            path,
            path,
            self.seed,
            f"live-smoke-gate.{label}",
        )

        async def operation() -> tuple[SampleObservation, ...]:
            return await sample_seeded(
                runtime,
                sampler,
                task,
                coordinates,
                group_size=count,
                max_tokens=max_tokens,
                temperature=1.0,
                top_p=0.95,
                ledger=self.ledger,
            )

        def persist(rows: tuple[SampleObservation, ...]) -> dict[str, object]:
            for row in rows:
                append_jsonl(self.run_directory / "generations.jsonl", row.generation)
                append_jsonl(self.run_directory / "rewards.jsonl", row.reward)
            return {
                "sample_ids": [row.generation.sample_id for row in rows],
                "stop_reasons": [row.generation.stop_reason for row in rows],
            }

        rows = await self.run.paid(
            f"sample:{label}",
            self.ledger,
            operation,
            reservation={"group_size": count, "max_tokens": max_tokens},
            persist=persist,
        )
        self.observations.extend(rows)
        return rows

    async def probe_transport(
        self,
        runtime: RuntimeBundle,
        client: Any,
        tces: SamplingTask,
        maps: SamplingTask,
        group_size: int,
    ) -> tuple[SampleObservation, ...]:
        """Measure the frozen cap once; this smoke never re-selects it."""

        return (
            *await self.collect(
                runtime,
                client,
                tces,
                label="probe-tces-4096",
                stage="base",
                path=MODEL_ID,
                count=group_size,
                max_tokens=PROTOCOL_MAX_TOKENS,
            ),
            *await self.collect(
                runtime,
                client,
                maps,
                label="probe-maps-4096",
                stage="base",
                path=MODEL_ID,
                count=group_size,
                max_tokens=PROTOCOL_MAX_TOKENS,
            ),
        )

    def acceptance(
        self,
        runtime: RuntimeBundle,
        tasks: tuple[SamplingTask, ...],
        *,
        probe_rows: tuple[SampleObservation, ...],
    ) -> dict[str, Any]:
        exact_tasks = {task.task_id: task.exact_task for task in tasks}
        parity = all(
            (
                offline := verify_task_completion(
                    row.generation.completion_text,
                    exact_tasks[row.generation.task_id],
                )
            )
            == row.reward.exact_verification
            and row.generation.reward == row.reward.reward == offline.reward
            for row in self.observations
        )
        required_stops = {"</answer>", "\nUser:", "\n\nUser:"}
        configured_stops = {str(value) for value in sampling_stops(runtime.renderer)}
        classifications = [_transport_class(row) for row in probe_rows]
        reasons = [str(row.generation.stop_reason).lower() for row in probe_rows]
        terminations = [
            str(row.generation.renderer_termination).lower() for row in probe_rows
        ]
        counts = {
            name: classifications.count(name) for name in sorted(set(classifications))
        }
        reason_counts = {name: reasons.count(name) for name in sorted(set(reasons))}
        termination_counts = {
            name: terminations.count(name) for name in sorted(set(terminations))
        }
        stop_ok = (
            required_stops.issubset(configured_stops)
            and counts.get("configured_stop", 0) >= 1
            and counts.get("unclassified", 0) == 0
        )
        return {
            "online_offline_reward_parity": parity,
            "stop_contract_verified": stop_ok,
            "stop_contract": {
                "required_stops": sorted(required_stops),
                "configured_stops": sorted(configured_stops),
                "classification_counts": counts,
                "stop_reason_counts": reason_counts,
                "renderer_termination_counts": termination_counts,
                "wrapper_closed_count": sum(_wrapper_closed(row) for row in probe_rows),
                "minimum_clean_configured_stops": 1,
            },
            "max_tokens": {
                "protocol_value": PROTOCOL_MAX_TOKENS,
                "sample_count": len(probe_rows),
                "truncated_count": sum(_truncated(row) for row in probe_rows),
                "runtime_diagnostic_passed": stop_ok,
                "selection_performed": False,
                "frozen_evidence": {
                    "artifact": (
                        "frozen/v0/runs/tinker-calibration/tces-cap/"
                        "tces-cap-reference-20260809T204337Z/tces_cap_summary.json"
                    ),
                    "sha256": (
                        "sha256:d2505e16bec8dc4374e4a5917d80a71095018407ab5d95937c06dd7997e9ae76"
                    ),
                },
            },
        }


def _truncated(row: SampleObservation) -> bool:
    return str(row.generation.stop_reason).lower().endswith("length")


def _transport_class(row: SampleObservation) -> str:
    reason = str(row.generation.stop_reason).lower()
    termination = str(row.generation.renderer_termination).lower()
    if reason == "length":
        return "length"
    if reason == "stop" and termination.endswith("stop_sequence"):
        return "configured_stop"
    if reason == "stop" and termination.endswith("eos"):
        return "eos"
    if reason == "stop" and termination.endswith("malformed"):
        return "renderer_malformed_stop"
    return "unclassified"


def _wrapper_closed(row: SampleObservation) -> bool:
    completion = row.generation.completion_text.strip()
    return (
        completion.count("<answer>") == 1
        and completion.count("</answer>") == 1
        and completion.endswith("</answer>")
    )


__all__ = ["SmokeSampler"]
