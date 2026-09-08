"""Thin replay adapter over the existing Tinker primitives and durable journal."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import math
from types import SimpleNamespace

from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes
from duraseed.replay_inputs import read_json, verify_tokenizer
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import (
    RuntimeBundle,
    TokenBudget,
    TokenLedger,
    apply_update,
    bind_model,
    create_sampler,
    create_service,
    load_sdk,
    restore_checkpoint,
    save_checkpoint,
)


def write_json(path: Path, value: dict) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


class ReplayRemote:
    """One logical operation at a time, four concurrent item groups inside evals.

    A pending journal survives exceptions. Resume never retries it. A clean resume
    uses durable segment checkpoints; incomplete training segments need explicit
    evidence reconciliation and are not silently replayed.
    """

    def __init__(
        self, repo: Path, config: dict, root: Path, project_id: str, *, smoke=False
    ):
        self.repo, self.config, self.root = repo, config, root
        preflight = read_json(repo / config["preflight"]["path"])
        tokens = (
            {"prefill": 0, "sample": 0, "train": config["max_length"]}
            if smoke
            else preflight["token_budget"]
        )
        self.ledger = TokenLedger(
            TokenBudget(tokens["prefill"], tokens["sample"], tokens["train"]),
            20.0 if smoke else math.nextafter(float(preflight["main_usd"]), math.inf),
        )
        self.journal = RemoteJournal(root / "remote", reconciled_resume=True)
        billing = root / "billing.json"
        previous = {}
        if self.journal.sequence and not billing.exists():
            raise ValueError(
                "completed requests have no durable billing/session record"
            )
        if billing.exists():
            previous = read_json(billing)
            self.ledger.reserve_call(
                TokenBudget(**previous["committed"]),
                fixed_usd=previous["committed_fixed_usd"],
            )
            self.ledger.settle_call(TokenBudget(**previous["observed"]))
        floor = self.journal.floor
        if self.ledger.committed != TokenBudget(
            *(
                int(floor[k])
                for k in ("prefill_tokens", "sample_tokens", "train_tokens")
            )
        ) or not math.isclose(
            self.ledger.committed_fixed_usd,
            float(floor["fixed_usd"]),
            rel_tol=0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "durable billing and completed-call reservation floor disagree"
            )
        sdk = load_sdk()
        self.run_id = root.name
        self.session_ids = list(previous.get("session_ids", []))
        self.begin("session", {"run_id": self.run_id, "project_id": project_id})
        service = create_service(
            sdk,
            project_id=project_id,
            user_metadata={"run_id": root.name, "study": "replay-v1"},
        )
        session_id = str(service._get_session_holder().get_session_id())
        if not session_id.strip():
            raise ValueError("Tinker session identity was not returned")
        if session_id not in self.session_ids:
            self.session_ids.append(session_id)
        self.runtime = RuntimeBundle(sdk, service, None, None, None)
        self.eval_inputs = SimpleNamespace(
            config=SimpleNamespace(evaluation=config["evaluation"]),
            run_id=self.run_id,
            runtime=self.runtime,
            ledger=self.ledger,
        )
        self.complete({"phase": "session", "session_id": session_id})

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "session_ids": self.session_ids,
            "committed": asdict(self.ledger.committed),
            "observed": asdict(self.ledger.observed),
            "committed_fixed_usd": self.ledger.committed_fixed_usd,
            "observed_fixed_usd": self.ledger.observed_fixed_usd,
            "observed_token_cost_plus_storage_reservation_usd": self.ledger.observed_cost_usd,
            "storage_note": "fixed charges are reserved byte-time bounds, not settled invoice actuals",
        }

    def begin(
        self, operation: str, coordinate: dict, tokens=TokenBudget(0, 0, 0), fixed=0.0
    ):
        self.journal.begin(
            operation,
            coordinate,
            {
                "prefill_tokens": tokens.prefill,
                "sample_tokens": tokens.sample,
                "train_tokens": tokens.train,
                "fixed_usd": fixed,
            },
        )
        write_json(
            self.root / "progress.json",
            {"phase": operation, **coordinate, "pending": True},
        )

    def complete(self, evidence: dict):
        write_json(self.root / "billing.json", self.snapshot())
        self.journal.complete(evidence)
        write_json(self.root / "progress.json", {**evidence, "pending": False})

    async def restore(
        self, path: str, *, full_state: bool, coordinate: dict, sources=()
    ):
        self.begin("restore", {**coordinate, "full_state": full_state, "path": path})
        model = await restore_checkpoint(
            self.runtime,
            path,
            full_state=full_state,
            ledger=self.ledger,
            user_metadata={
                "study": "replay-v1",
                **{k: str(v) for k, v in coordinate.items()},
            },
        )
        self.runtime = bind_model(self.runtime.sdk, self.runtime.service, model)
        self.eval_inputs.runtime = self.runtime
        if sources:
            verify_tokenizer(self.runtime, self.repo, self.config, sources)
        self.complete(
            {"phase": "restore", **coordinate, "path": path, "full_state": full_state}
        )

    async def update(self, datums, lr: float, path: Path, coordinate: dict):
        if path.exists():
            raise ValueError(
                "incomplete segment contains committed updates; reconcile, never replay"
            )
        tokens = TokenBudget(0, 0, sum(int(d.model_input.length) for d in datums))
        self.begin("training", coordinate, tokens)
        metrics = await apply_update(
            self.runtime,
            datums,
            loss_fn="cross_entropy",
            learning_rate=lr,
            ledger=self.ledger,
        )
        value = {**coordinate, "train_tokens": tokens.train, "metrics": metrics}
        write_json(path, value)
        self.complete(
            {
                "phase": "training",
                **coordinate,
                "evidence": str(path.relative_to(self.root)),
            }
        )

    async def save(self, name: str, path: Path, coordinate: dict) -> dict:
        storage = self.config["storage_per_pair_usd"]
        self.begin("checkpoint", coordinate, fixed=storage)
        pair = await save_checkpoint(
            self.runtime,
            name=name,
            ttl_seconds=self.config["checkpoint_ttl_seconds"],
            ledger=self.ledger,
            reserved_storage_usd=storage,
        )
        rest = self.runtime.service.create_rest_client()
        for remote_path, bound in (
            (pair.state_path, 2_000_000_000),
            (pair.sampler_path, 500_000_000),
        ):
            parsed = (
                self.runtime.sdk.tinker.ParsedCheckpointTinkerPath.from_tinker_path(
                    remote_path
                )
            )
            listing = await rest.list_checkpoints_async(parsed.training_run_id)
            record = next(
                (r for r in listing.checkpoints if r.tinker_path == remote_path), None
            )
            if (
                record is None
                or not 0 < record.size_bytes <= bound
                or record.expires_at is None
            ):
                raise ValueError(
                    "checkpoint size/retention differs from reserved storage"
                )
            remaining = (record.expires_at - datetime.now(UTC)).total_seconds()
            if (
                not self.config["checkpoint_ttl_seconds"] - 300
                <= remaining
                <= self.config["checkpoint_ttl_seconds"]
            ):
                raise ValueError(
                    "checkpoint expiry differs from reserved 30-day retention"
                )
        value = {**coordinate, **asdict(pair)}
        write_json(path, value)
        self.complete(
            {
                "phase": "checkpoint",
                **coordinate,
                "evidence": str(path.relative_to(self.root)),
            }
        )
        return value

    async def sampler(self, path: str, coordinate: dict):
        self.begin("sampler", coordinate)
        value = await create_sampler(
            self.runtime, ledger=self.ledger, checkpoint_path=path
        )
        self.complete({"phase": "sampler", **coordinate, "path": path})
        return value
