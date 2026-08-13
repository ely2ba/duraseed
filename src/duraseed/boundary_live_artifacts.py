"""Durable artifacts for the one boundary-extension live runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.data.manifests import DatasetManifest, write_manifest
from duraseed.run_records import (
    GenerationRecord,
    RewardRecord,
    RunRecord,
    RunStatus,
    append_jsonl,
    read_run_record,
    write_jsonl,
    write_run_record,
)
from duraseed.runners import RunnerGateError
from duraseed.runtime import PRICE_SNAPSHOT, TokenBudget, TokenLedger


def _json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value, sort_keys=True, indent=2, allow_nan=False, default=str
        ).encode()
        + b"\n"
    )
    atomic_write_bytes(path, payload)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid boundary artifact: {path.name}") from error


class BoundaryLiveArtifacts:
    """Canonical group journal plus restart-safe row projections."""

    def __init__(
        self,
        directory: Path,
        *,
        preflight: Mapping[str, Any],
        new_run: RunRecord,
    ) -> None:
        self.directory = directory
        self.journal = directory / "observation_groups.jsonl"
        self.pending = directory / "pending_group.json"
        expected = json.loads(json.dumps(preflight, sort_keys=True, default=str))
        if directory.exists():
            if _read_json(directory / "preflight.json") != expected:
                raise RunnerGateError("restart preflight differs from the original run")
            existing = read_run_record(directory)
            immutable = (
                "protocol_version",
                "git_commit",
                "resolved_config_hash",
                "model_id",
                "renderer",
                "lora_rank",
                "parent_tinker_checkpoint_path",
                "project_id",
                "authorized_cost_usd",
            )
            if any(
                getattr(existing, name) != getattr(new_run, name) for name in immutable
            ):
                raise RunnerGateError("restart run identity differs from run.json")
            if any(
                existing.task_manifest_ids.get(name) != manifest_id
                for name, manifest_id in new_run.task_manifest_ids.items()
            ):
                raise RunnerGateError("restart manifest identity differs from run.json")
            self.run = existing
            if self.pending.exists():
                value = _read_json(self.pending)
                raise RunnerGateError(
                    "ambiguous in-flight boundary group; reconcile before restart: "
                    f"{value.get('action')} / {value.get('task_id')}"
                )
        else:
            directory.mkdir(parents=True)
            _json(directory / "preflight.json", expected)
            self.run = new_run
            write_run_record(directory, self.run)
        self.groups = self._load_groups()
        self._sync_projections()

    def _load_groups(self) -> dict[tuple[str, str], tuple[Any, ...]]:
        result: dict[tuple[str, str], tuple[Any, ...]] = {}
        sample_ids: set[str] = set()
        if not self.journal.exists():
            return result
        try:
            lines = self.journal.read_text(encoding="utf-8").splitlines()
            for line in lines:
                value = json.loads(line)
                action, task_id = value["action"], value["task_id"]
                key = (action, task_id)
                if key in result:
                    raise RunnerGateError(
                        "group journal contains a duplicate task group"
                    )
                generations = tuple(
                    GenerationRecord.model_validate_json(json.dumps(row))
                    for row in value["generations"]
                )
                rewards = tuple(
                    RewardRecord.model_validate_json(json.dumps(row))
                    for row in value["rewards"]
                )
                if not generations or len(generations) != len(rewards):
                    raise RunnerGateError("group journal has an incomplete exact join")
                by_sample = {row.sample_id: row for row in rewards}
                if set(by_sample) != {row.sample_id for row in generations}:
                    raise RunnerGateError(
                        "group journal generation/reward join differs"
                    )
                if any(
                    row.task_id != task_id
                    or by_sample[row.sample_id].task_id != task_id
                    or row.reward != by_sample[row.sample_id].reward
                    for row in generations
                ):
                    raise RunnerGateError(
                        "group journal task or reward identity differs"
                    )
                ids = {row.sample_id for row in generations}
                if len(ids) != len(generations) or ids.intersection(sample_ids):
                    raise RunnerGateError("group journal contains duplicate samples")
                sample_ids.update(ids)
                result[key] = (*generations, *rewards)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RunnerGateError("invalid boundary group journal") from error
        return result

    def _rows(self) -> tuple[tuple[GenerationRecord, ...], tuple[RewardRecord, ...]]:
        generations, rewards = [], []
        for rows in self.groups.values():
            midpoint = len(rows) // 2
            generations.extend(rows[:midpoint])
            rewards.extend(rows[midpoint:])
        return tuple(generations), tuple(rewards)

    def _sync_one(
        self, path: Path, expected: tuple[Any, ...], model: type[Any]
    ) -> None:
        if path.exists():
            try:
                observed = tuple(
                    model.model_validate_json(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            except ValueError as error:
                raise RunnerGateError(f"invalid row projection: {path.name}") from error
            if observed != expected[: len(observed)]:
                raise RunnerGateError(
                    f"row projection diverges from journal: {path.name}"
                )
        write_jsonl(path, expected)

    def _sync_projections(self) -> None:
        generations, rewards = self._rows()
        self._sync_one(
            self.directory / "generations.jsonl", generations, GenerationRecord
        )
        self._sync_one(self.directory / "rewards.jsonl", rewards, RewardRecord)

    def completed_group(
        self,
        action: str,
        task_id: str,
        *,
        manifest_id: str,
        run_id: str,
        sample_indices: tuple[int, ...],
    ) -> tuple[tuple[GenerationRecord, ...], tuple[RewardRecord, ...]] | None:
        rows = self.groups.get((action, task_id))
        if rows is None:
            return None
        midpoint = len(rows) // 2
        generations = tuple(rows[:midpoint])
        rewards = tuple(rows[midpoint:])
        if (
            tuple(row.sample_index for row in generations) != sample_indices
            or {row.task_manifest_id for row in generations} != {manifest_id}
            or {row.run_id for row in generations} != {run_id}
        ):
            raise RunnerGateError("completed group differs from planned coordinates")
        return generations, rewards

    def append_group(self, action: str, task_id: str, rows: tuple[Any, ...]) -> None:
        if (action, task_id) in self.groups or not rows:
            raise RunnerGateError("attempted to append an empty or duplicate group")
        pending = _read_json(self.pending) if self.pending.exists() else {}
        if pending.get("action") != action or pending.get("task_id") != task_id:
            raise RunnerGateError("completed group lacks its durable pending marker")
        generations = tuple(row.generation for row in rows)
        rewards = tuple(row.reward for row in rows)
        append_jsonl(
            self.journal,
            {
                "action": action,
                "task_id": task_id,
                "generations": [row.model_dump(mode="json") for row in generations],
                "rewards": [row.model_dump(mode="json") for row in rewards],
            },
        )
        self.groups[(action, task_id)] = (*generations, *rewards)
        self.pending.unlink()
        self._sync_projections()

    def begin_group(
        self,
        action: str,
        task_id: str,
        *,
        manifest_id: str,
        run_id: str,
        sample_indices: tuple[int, ...],
        reservation: TokenBudget,
    ) -> None:
        """Persist ambiguity evidence before any paid request can begin."""

        if self.pending.exists() or (action, task_id) in self.groups:
            raise RunnerGateError("cannot begin an already pending or completed group")
        _json(
            self.pending,
            {
                "action": action,
                "task_id": task_id,
                "manifest_id": manifest_id,
                "run_id": run_id,
                "sample_indices": sample_indices,
                "reserved_tokens": asdict(reservation),
            },
        )

    def write_manifest(self, name: str, manifest: DatasetManifest) -> None:
        path = self.directory / name
        if (
            path.exists()
            and DatasetManifest.model_validate_json(path.read_bytes()) != manifest
        ):
            raise RunnerGateError(f"restart manifest differs: {name}")
        write_manifest(path, manifest)

    def add_manifest_identity(self, name: str, manifest_id: str) -> None:
        prior = self.run.task_manifest_ids.get(name)
        if prior not in (None, manifest_id):
            raise RunnerGateError("run manifest identity changed during execution")
        manifests = {**self.run.task_manifest_ids, name: manifest_id}
        self.run = self.run.model_copy(update={"task_manifest_ids": manifests})
        write_run_record(self.directory, self.run)

    def restore_ledger(
        self, action: str, limits: TokenBudget, authorized_usd: Decimal
    ) -> TokenLedger:
        ledger = TokenLedger(limits, float(authorized_usd))
        path = self.directory / "billing.json"
        action_row = (
            _read_json(path).get("actions", {}).get(action) if path.exists() else None
        )
        if action_row is not None:
            try:
                committed = TokenBudget(**action_row["committed_tokens"])
                observed = TokenBudget(**action_row["observed_tokens"])
                if any(
                    getattr(committed, name) > getattr(limits, name)
                    or getattr(observed, name) > getattr(committed, name)
                    for name in ("prefill", "sample", "train")
                ):
                    raise ValueError
                ledger.committed, ledger.observed = committed, observed
            except (KeyError, TypeError, ValueError) as error:
                raise RunnerGateError(
                    "billing ledger cannot be resumed safely"
                ) from error
        generations = tuple(
            row
            for (group_action, _), rows in self.groups.items()
            if group_action == action
            for row in rows[: len(rows) // 2]
        )
        committed_floor = TokenBudget(
            sum(row.prompt_tokens for row in generations),
            sum(int(row.sampling_max_tokens or 0) for row in generations),
            0,
        )
        observed_floor = TokenBudget(
            committed_floor.prefill,
            sum(row.sampled_tokens for row in generations),
            0,
        )
        ledger.committed = TokenBudget(
            max(ledger.committed.prefill, committed_floor.prefill),
            max(ledger.committed.sample, committed_floor.sample),
            0,
        )
        ledger.observed = TokenBudget(
            max(ledger.observed.prefill, observed_floor.prefill),
            max(ledger.observed.sample, observed_floor.sample),
            0,
        )
        if (
            ledger.committed.prefill > limits.prefill
            or ledger.committed.sample > limits.sample
            or ledger.committed_cost_usd > float(authorized_usd)
        ):
            raise RunnerGateError("resumed usage exceeds the action cap")
        return ledger

    def write_billing(self, ledgers: Mapping[str, TokenLedger]) -> None:
        prior = (
            _read_json(self.directory / "billing.json")
            if (self.directory / "billing.json").exists()
            else {}
        )
        actions = {
            **prior.get("actions", {}),
            **{
                name: {
                    "committed_tokens": asdict(ledger.committed),
                    "observed_tokens": asdict(ledger.observed),
                    "committed_cost_usd": ledger.committed_cost_usd,
                    "observed_cost_usd": ledger.observed_cost_usd,
                    "authorized_cost_usd": ledger.authorized_usd,
                }
                for name, ledger in ledgers.items()
            },
        }
        _json(
            self.directory / "billing.json",
            {
                "status": "local_usage_pending_console_reconciliation",
                "price_snapshot_id": PRICE_SNAPSHOT.snapshot_id,
                "actions": actions,
                "observed_cost_usd": sum(
                    row["observed_cost_usd"] for row in actions.values()
                ),
                "conservative_committed_cost_usd": sum(
                    row["committed_cost_usd"] for row in actions.values()
                ),
            },
        )

    def finish(self, status: RunStatus, ledgers: Mapping[str, TokenLedger]) -> None:
        now = datetime.now(UTC)
        observed = TokenBudget(
            sum(row.observed.prefill for row in ledgers.values()),
            sum(row.observed.sample for row in ledgers.values()),
            0,
        )
        self.run = self.run.model_copy(
            update={
                "status": status,
                "updated_at": now,
                "finished_at": now if status is not RunStatus.RUNNING else None,
                "prompt_tokens": observed.prefill,
                "sampled_tokens": observed.sample,
                "cost_usd": sum(row.observed_cost_usd for row in ledgers.values()),
            }
        )
        write_run_record(self.directory, self.run)
        self.write_billing(ledgers)

    def record_error(self, action: str, error: BaseException) -> None:
        append_jsonl(
            self.directory / "errors.jsonl",
            {
                "action": action,
                "error_type": type(error).__name__,
                "message": str(error),
                "time": datetime.now(UTC).isoformat(),
            },
        )

    def write_result(self, value: Any) -> None:
        _json(self.directory / "result.json", asdict(value))


__all__ = ["BoundaryLiveArtifacts"]
