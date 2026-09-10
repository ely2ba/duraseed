"""Endpoint-clone sampling and fixed-policy concurrent RL updates."""

from __future__ import annotations

import asyncio
from math import fsum
from types import SimpleNamespace

from duraseed.pilot0_contract import EPHEMERAL_SAMPLER_FIXED_USD
from duraseed.pilot0_data import scheduled_stage_a_records
from duraseed.provenance import derive_namespaced_seed
from duraseed.replay_remote import ReplayRemote, write_json
from duraseed.runners.pilot0_updates import _group_seeds
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import (
    SamplingCoordinates,
    SamplingTask,
    TokenBudget,
    TokenLedger,
    apply_update,
    rl_datums,
    sample_seeded,
)
from duraseed.tasks.tces import render_prompt
from duraseed.training.grpo import grouped_reward_diagnostics
from duraseed.training.stage_a_update_health import (
    StageAUpdateHealthFailure,
    StageAUpdateHealthFailureEvidence,
    write_stage_a_update_health_failure,
)
from duraseed.training_metric_errors import NonFiniteTrainingMetricError


def _raw(row):
    return {
        "prompt_token_ids": list(row.prompt.to_ints()),
        "generation": row.generation.model_dump(mode="json"),
        "reward": row.reward.model_dump(mode="json"),
    }


class EndpointRemote(ReplayRemote):
    """One worker ledger; other continuation workers have disjoint allocations.

    Group sampling reserves one aggregate bound, then uses bounded child ledgers.
    On failure, submitted groups finish and save their evidence; no new groups
    are dispatched, and the aggregate journal remains pending for inspection.
    """

    async def _groups(
        self,
        source,
        manifest,
        records,
        sampler,
        sampler_path,
        origin,
        output,
        *,
        step,
        purpose,
        concurrency,
    ):
        if output.exists():
            raise ValueError("existing sampling output: inspect it, never resubmit")
        records = tuple(records)
        prompts = [
            self.runtime.renderer.build_generation_prompt(
                [{"role": "user", "content": render_prompt(row.to_task())}],
                role="assistant",
            )
            for row in records
        ]
        cap, draws = self.config["tces_max_tokens"], 8
        reservations = [
            TokenBudget(int(p.length) * draws, cap * draws, 0) for p in prompts
        ]
        total = TokenBudget(0, 0, 0)
        for budget in reservations:
            total = total.plus(budget)
        coordinate = {
            "stage": "stage_a",
            "update": step,
            "purpose": purpose,
            "groups": len(records),
            "samples": len(records) * draws,
            "sampler_path": sampler_path,
        }
        self.begin("sampling", coordinate, total)
        self.ledger.reserve_call(total)
        work, results, children, failure = iter(enumerate(records)), {}, [], None

        async def worker():
            nonlocal failure
            while failure is None:
                item = next(work, None)
                if item is None:
                    return
                index, record = item
                child = TokenLedger(reservations[index], self.ledger.authorized_usd)
                children.append(child)
                directory = output / f"group-{index:04d}"
                journal = RemoteJournal(directory)
                group_coordinate = {
                    **coordinate,
                    "group": index,
                    "task_id": record.task_id,
                }
                journal.begin(
                    "endpoint-sample-group",
                    group_coordinate,
                    {
                        "prefill_tokens": reservations[index].prefill,
                        "sample_tokens": reservations[index].sample,
                        "train_tokens": 0,
                    },
                )
                seeds = (
                    _group_seeds(source.seed, step, index, record.task_id)
                    if purpose == "rl_rollout"
                    else tuple(
                        derive_namespaced_seed(
                            source.seed, "endpoint-clone.corpus", record.task_id, sample
                        )
                        for sample in range(draws)
                    )
                )
                requests = []

                async def submitted(**kwargs):
                    request = asyncio.create_task(sampler.sample_async(**kwargs))
                    requests.append(request)
                    return await request

                try:
                    rows = await sample_seeded(
                        self.runtime,
                        SimpleNamespace(sample_async=submitted),
                        SamplingTask(
                            manifest.manifest_id,
                            record.task_id,
                            "tces",
                            manifest.split,
                            render_prompt(record.to_task()),
                            record.to_task(),
                            record.item_index,
                            record.intended_family,
                            "training",
                        ),
                        SamplingCoordinates(
                            self.run_id,
                            f"{purpose}-u{step}-g{index}",
                            "training",
                            "stage_a",
                            step,
                            sampler_path,
                            origin,
                            source.seed,
                            "pilot0.stage_a.bg_rollout"
                            if purpose == "rl_rollout"
                            else "endpoint-clone.corpus",
                            "B-G",
                        ),
                        group_size=draws,
                        max_tokens=cap,
                        temperature=float(self.config["evaluation"]["temperature"]),
                        top_p=float(self.config["evaluation"]["top_p"]),
                        ledger=child,
                        explicit_seeds=seeds,
                    )
                    write_json(
                        directory / "samples.json",
                        {
                            **group_coordinate,
                            "records": [_raw(row) for row in rows],
                        },
                    )
                    journal.complete({"phase": "sampling", "samples": len(rows)})
                    results[index] = rows
                    write_json(
                        self.root / "progress.json",
                        {
                            "phase": "sampling",
                            **coordinate,
                            "pending": True,
                            "committed_groups": len(results),
                            "committed_samples": sum(len(v) for v in results.values()),
                        },
                    )
                except BaseException as error:
                    # sample_seeded may raise after one of its eight requests
                    # fails. Let the other submitted requests settle too.
                    await asyncio.gather(*requests, return_exceptions=True)
                    if failure is None:
                        failure = error

        try:
            await asyncio.gather(
                *(worker() for _ in range(min(concurrency, len(records))))
            )
            if failure is not None:
                raise failure
        except BaseException:
            self.ledger.abort_call()
            raise
        actual = TokenBudget(0, 0, 0)
        for child in children:
            actual = actual.plus(child.observed)
        self.ledger.settle_call(actual)
        self.complete(
            {
                "phase": "sampling",
                **coordinate,
                "committed_groups": len(results),
                "committed_samples": sum(len(v) for v in results.values()),
                "evidence": str(output.relative_to(self.root)),
            }
        )
        return [results[index] for index in range(len(records))]

    async def sample_corpus(self, source, manifest, checkpoint, output):
        """Keep all eight raw token sequences per frozen distillation prompt."""
        if manifest.record_count != 800:
            raise ValueError("endpoint corpus requires exactly 800 prompts")
        sampler = await self.sampler(checkpoint["sampler_path"], {"purpose": "corpus"})
        groups = await self._groups(
            source,
            manifest,
            manifest.records,
            sampler,
            checkpoint["sampler_path"],
            checkpoint["sampler_path"],
            output,
            step=checkpoint["update"],
            purpose="corpus",
            concurrency=self.config.get("corpus_group_concurrency", 8),
        )
        rows = tuple(_raw(row) for group in groups for row in group)
        write_json(
            output / "summary.json",
            {
                "manifest_id": manifest.manifest_id,
                "teacher": checkpoint,
                "samples": len(rows),
                "groups": len(groups),
                "mean_response_length": fsum(
                    len(r["generation"]["completion_token_ids"]) for r in rows
                )
                / len(rows),
                "filtering": "none; all responses and observed termination tokens retained",
            },
        )
        return rows

    async def rl_update(
        self,
        source,
        pools,
        *,
        step,
        learning_rate,
        boundary_sampler_path,
        output,
    ):
        """Same 16 groups, seeds, rewards and update as Pilot; only dispatch differs."""
        if output.exists():
            raise ValueError("existing RL update directory: never replay it")
        coordinate = {
            "seed": source.seed,
            "arm": "T",
            "stage": "stage_a",
            "update": step,
        }
        self.begin("ephemeral_sampler", coordinate, fixed=EPHEMERAL_SAMPLER_FIXED_USD)
        self.ledger.reserve_call(
            TokenBudget(0, 0, 0), fixed_usd=EPHEMERAL_SAMPLER_FIXED_USD
        )
        try:
            sampler = (
                await self.runtime.model.save_weights_and_get_sampling_client_async()
            )
        except BaseException:
            self.ledger.abort_call()
            raise
        self.ledger.settle_call(TokenBudget(0, 0, 0))
        sampler_path = f"ephemeral:{self.run_id}:teacher:u{step}"
        self.complete(
            {"phase": "ephemeral_sampler", **coordinate, "path": sampler_path}
        )
        records = scheduled_stage_a_records(
            pools, source.prompt_pools.artifact.bg_group_order, step
        )
        groups = await self._groups(
            source,
            source.prompt_pools.a_rl_train_manifest,
            records,
            sampler,
            sampler_path,
            boundary_sampler_path,
            output / "rollouts",
            step=step,
            purpose="rl_rollout",
            concurrency=self.config.get("rl_group_concurrency", 16),
        )
        mixed_rows, advantages, logprobs, grouped_advantages = [], [], [], []
        all_zero = all_one = mixed = 0
        for rows in groups:
            diagnostics = grouped_reward_diagnostics(
                [float(r.reward.reward) for r in rows], group_size=8
            )
            all_zero += diagnostics.all_zero_group_count
            all_one += diagnostics.all_one_group_count
            mixed += diagnostics.mixed_group_count
            logprobs.extend(v for row in rows for v in row.logprobs)
            group_adv = (
                diagnostics.centered_advantages[0]
                if diagnostics.mixed_group_count
                else (0.0,) * 8
            )
            grouped_advantages.append(list(group_adv))
            if diagnostics.mixed_group_count:
                mixed_rows.extend(rows)
                advantages.extend(group_adv)
        write_json(
            output / "group-advantages.json",
            {"update": step, "advantages": grouped_advantages},
        )

        def health_failure(reason, committed, metric=None):
            evidence = StageAUpdateHealthFailureEvidence(
                "B-G",
                learning_rate,
                step,
                "screen" if step <= 10 else "continuation",
                reason,
                step - 1,
                committed,
                len(records),
                len(records) * 8,
                mixed,
                all_zero,
                all_one,
                metric,
            )
            write_stage_a_update_health_failure(output, evidence)
            return StageAUpdateHealthFailure(evidence)

        if not mixed:
            raise health_failure("zero_mixed_group", False)
        datums = rl_datums(self.runtime, mixed_rows, advantages)
        tokens = TokenBudget(0, 0, sum(int(d.model_input.length) for d in datums))
        self.begin("training", coordinate, tokens)
        try:
            metrics = await apply_update(
                self.runtime,
                datums,
                loss_fn="importance_sampling",
                learning_rate=learning_rate,
                ledger=self.ledger,
            )
        except NonFiniteTrainingMetricError as error:
            failure = health_failure(
                "nonfinite_training_metric", True, error.metric_name
            )
            self.complete(
                {"phase": "training", **coordinate, "health_failure": str(failure)}
            )
            raise failure from error
        metrics.update(
            mixed_group_rate=mixed / len(records),
            mixed_group_count=float(mixed),
            all_zero_group_count=float(all_zero),
            all_one_group_count=float(all_one),
            mean_sampled_token_surprisal=-fsum(logprobs) / len(logprobs),
        )
        value = {
            **coordinate,
            "train_tokens": tokens.train,
            "metrics": metrics,
            "groups": len(groups),
            "samples": sum(len(g) for g in groups),
        }
        write_json(output / "update.json", value)
        self.complete(
            {
                "phase": "training",
                **coordinate,
                "evidence": str((output / "update.json").relative_to(self.root)),
                "committed_updates": step,
                "committed_groups": step * len(groups),
                "committed_samples": step * sum(len(g) for g in groups),
            }
        )
        return value
