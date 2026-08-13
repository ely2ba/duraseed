"""Authorization and durable run lifecycle for the live smoke gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence, TypeVar

from duraseed.config import PilotConfig
from duraseed.data.io import atomic_write_bytes
from duraseed.data.manifests import DatasetManifest, write_manifest
from duraseed.run_records import (
    RunRecord,
    RunStatus,
    append_jsonl,
    create_run_directory,
    write_run_record,
)
from duraseed.runners import (
    Action,
    LaunchAuthorization,
    RunPlan,
    authorize_launch,
)
from duraseed.runtime import MODEL_ID, RENDERER_NAME, TokenLedger


PHASE_LABEL = "live-smoke-gate"
TOTAL_CAP_USD = Decimal("25")
TTL_SECONDS = 7 * 24 * 60 * 60
PROTOCOL_MAX_TOKENS = 4096
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SmokeSettings:
    run_id: str
    config_path: Path = Path("duraseed_pilot_config.yaml")
    output_root: Path = Path("runs/live-smoke")
    seed: int = 5
    max_tokens: int = PROTOCOL_MAX_TOKENS
    group_size: int = 8

    def __post_init__(self) -> None:
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be one safe filename token")
        if self.seed < 0 or self.group_size < 2:
            raise ValueError("seed must be nonnegative and group_size at least two")
        if self.max_tokens != PROTOCOL_MAX_TOKENS:
            raise ValueError("the smoke must use the frozen 4096-token contract")


def build_plan() -> RunPlan:
    return RunPlan(
        PHASE_LABEL,
        (Action("real vertical slice and retained checkpoint proof", TOTAL_CAP_USD),),
        ("billing_reconciled", "human_launch_approval"),
        "duraseed live-smoke --dry-run",
        "pytest tests/unit/test_live_smoke.py",
        (
            "duraseed live-smoke --execute --run-id RUN_ID "
            "--project-id PROJECT_ID --authorized-cost-usd 25 "
            "--confirm-billing-reconciled --confirm-human-launch"
        ),
    )


def preflight_text() -> str:
    plan = build_plan()
    return "\n".join(
        (
            "Live smoke gate (no Tinker calls, no writes):",
            *(f"- {action.name}: ${action.cost_cap_usd}" for action in plan.actions),
            f"Remote cost cap: ${plan.remote_cost_cap_usd}",
            f"Dry-run: {plan.dry_run_command}",
            f"Mock: {plan.mock_command}",
            f"Execute after explicit authorization: {plan.authorization_command}",
            "Pass: real data, exact reward parity, stop contract, resume/branch.",
        )
    )


def authorize(
    *,
    execute: bool,
    authorized_cost_usd: str | None,
    billing_reconciled: bool,
    human_approval: bool,
) -> LaunchAuthorization:
    return authorize_launch(
        build_plan(),
        execute=execute,
        authorized_cost_usd=authorized_cost_usd,
        preconditions={
            "billing_reconciled": billing_reconciled,
            "human_launch_approval": human_approval,
        },
    )


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write_bytes(path, payload.encode())


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


@dataclass(slots=True)
class SmokeRun:
    directory: Path
    record: RunRecord

    @classmethod
    def start(
        cls,
        settings: SmokeSettings,
        config: PilotConfig,
        manifests: Sequence[DatasetManifest],
        *,
        evidence_origin: Literal["remote", "mock"],
        project_id: str,
    ) -> "SmokeRun":
        directory = create_run_directory(settings.output_root, settings.run_id)
        manifest_dir = directory / "manifests"
        manifest_dir.mkdir()
        for manifest in manifests:
            write_manifest(manifest_dir / f"{manifest.split}.json", manifest)
        started = datetime.now(UTC)
        protocol = config.model_dump(mode="json")["protocol"]["version"]
        record = RunRecord(
            protocol_version=str(protocol),
            git_commit=_git_commit(),
            resolved_config_hash=config.resolved_config_hash(),
            run_kind="engineering_smoke",
            method=None,
            seed=settings.seed,
            model_id=MODEL_ID,
            renderer=RENDERER_NAME,
            lora_rank=32,
            task_manifest_ids={m.split: m.manifest_id for m in manifests},
            status=RunStatus.RUNNING,
            started_at=started,
            updated_at=started,
            project_id=project_id,
            authorized_cost_usd=25.0,
            reserved_cost_usd=25.0,
        )
        write_run_record(directory, record)
        write_json(
            directory / "preflight.json",
            {
                "phase_label": PHASE_LABEL,
                "total_cap_usd": 25,
                "evidence_origin": evidence_origin,
            },
        )
        return cls(directory, record)

    async def paid(
        self,
        name: str,
        ledger: TokenLedger,
        operation: Callable[[], Awaitable[T]],
        *,
        reservation: Mapping[str, object],
        persist: Callable[[T], Mapping[str, object]],
    ) -> T:
        """Leave an unresolved marker unless durable completion evidence exists."""

        marker = self.directory / "pending_operation.json"
        if marker.exists():
            raise RuntimeError("an unresolved paid operation blocks this run")
        write_json(
            marker,
            {
                "phase_label": PHASE_LABEL,
                "operation": name,
                "reservation": dict(reservation),
                "authorized_cap_usd": 25,
                "committed_cost_before_usd": ledger.committed_cost_usd,
                "observed_cost_before_usd": ledger.observed_cost_usd,
            },
        )
        result = await operation()
        evidence = dict(persist(result))
        append_jsonl(
            self.directory / "operations.jsonl",
            {
                "phase_label": PHASE_LABEL,
                "operation": name,
                "status": "completed",
                "evidence": evidence,
                "committed_cost_after_usd": ledger.committed_cost_usd,
                "observed_cost_after_usd": ledger.observed_cost_usd,
            },
        )
        marker.unlink()
        return result

    def finish(
        self,
        status: RunStatus,
        ledger: TokenLedger,
        *,
        sampler_path: str | None = None,
        state_path: str | None = None,
        deviations: list[str] | None = None,
    ) -> None:
        finished = datetime.now(UTC)
        update: dict[str, Any] = {
            "status": status,
            "updated_at": finished,
            "finished_at": finished,
            "prompt_tokens": ledger.observed.prefill,
            "sampled_tokens": ledger.observed.sample,
            "train_tokens": ledger.observed.train,
            "cost_usd": ledger.observed_cost_usd,
            "deviations": deviations or [],
        }
        if sampler_path is not None:
            update["final_sampler_checkpoint_path"] = sampler_path
        if state_path is not None:
            update["final_state_checkpoint_path"] = state_path
        write_run_record(self.directory, self.record.model_copy(update=update))


__all__ = [
    "PHASE_LABEL",
    "PROTOCOL_MAX_TOKENS",
    "TOTAL_CAP_USD",
    "TTL_SECONDS",
    "SmokeRun",
    "SmokeSettings",
    "authorize",
    "build_plan",
    "preflight_text",
    "write_json",
]
