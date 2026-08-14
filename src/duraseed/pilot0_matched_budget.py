"""Pessimistic cap for the post-Pilot matched-Stage-A follow-up."""

from __future__ import annotations

from dataclasses import dataclass

from duraseed.pilot0_contract import (
    STAGE_B_GRID,
    STAGE_B_MAX_TOKENS,
    Pilot0Inputs,
)
from duraseed.pilot0_data import stage_b_sources
from duraseed.runtime import PRICE_SNAPSHOT, TokenBudget, UsageQuantities, sft_datum
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.tasks.maps import render_prompt as render_maps_prompt
from duraseed.tasks.tces import render_prompt as render_tces_prompt


@dataclass(frozen=True, slots=True)
class MatchedPilotBudget:
    tokens: TokenBudget
    fixed_storage_usd: float
    upper_bound_usd: float
    candidate_checkpoint_count: int


AUTHORIZATION_SCHEMA = "duraseed-pilot0-matched-authorization-v1"


def validate_matched_authorization(
    inputs: Pilot0Inputs, authorization: dict, preflight_raw: bytes, preflight: dict
) -> str:
    """Bind one explicit dollar cap to the deterministic matched preflight."""

    try:
        authorized = float(authorization["authorized_usd"])
        valid = (
            set(authorization)
            == {
                "schema_version",
                "status",
                "preflight_sha256",
                "matched_run_id",
                "authorizer",
                "authorized_at_utc",
                "authorized_usd",
                "no_rerun_authorized",
            }
            and authorization["schema_version"] == AUTHORIZATION_SCHEMA
            and authorization["status"] == "accepted"
            and authorization["preflight_sha256"] == sha256_bytes(preflight_raw)
            and authorization["matched_run_id"] == inputs.run_id
            and str(authorization["authorizer"]).strip()
            and str(authorization["authorized_at_utc"]).strip()
            and authorization["no_rerun_authorized"] is True
            and authorized == inputs.ledger.authorized_usd
            and authorized >= float(preflight["required_upper_bound_usd"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerGateError("matched follow-up authorization is malformed") from error
    if not valid:
        raise RunnerGateError("matched follow-up lacks its exact bound authorization")
    return sha256_bytes(canonical_json_bytes(authorization))


def _prompt_length(inputs: Pilot0Inputs, record: object) -> int:
    prompt_text = (
        render_tces_prompt(record.to_task())  # type: ignore[attr-defined]
        if getattr(record, "task_family", None) == "tces"
        else render_maps_prompt(record.to_task())  # type: ignore[attr-defined]
    )
    prompt = inputs.runtime.renderer.build_generation_prompt(
        [{"role": "user", "content": prompt_text}], role="assistant"
    )
    return int(prompt.length)


def _evaluation(
    inputs: Pilot0Inputs,
    manifest: object,
    *,
    repetitions: int,
    samples: int,
    max_tokens: int,
) -> TokenBudget:
    records = manifest.records  # type: ignore[attr-defined]
    prompt_tokens = sum(_prompt_length(inputs, row) for row in records)
    return TokenBudget(
        prompt_tokens * samples * repetitions,
        len(records) * samples * max_tokens * repetitions,
        0,
    )


def _sum(values: list[TokenBudget]) -> TokenBudget:
    total = TokenBudget(0, 0, 0)
    for value in values:
        total = total.plus(value)
    return total


def calculate_matched_pilot_budget(
    inputs: Pilot0Inputs, candidate_plan: dict
) -> MatchedPilotBudget:
    """Reserve 32-draw selection plus four complete matched-origin B paths."""

    cells = candidate_plan["cells"]
    candidate_counts = {
        (int(cell["seed"]), str(cell["method"])): len(cell["candidates"])
        for cell in cells
    }
    budgets: list[TokenBudget] = []
    selection_samples = int(
        inputs.config.statistics.checkpoint_selection_max_samples_per_item
    )
    pilot_samples = int(inputs.config.evaluation["pilot_samples_per_item"])
    monitor_samples = int(inputs.config.stage_a.monitor_samples_per_item)
    stage_a_tokens = int(inputs.acquisition.selected_max_tokens)
    for source in inputs.seed_sources:
        candidate_count = sum(
            count
            for (seed, _), count in candidate_counts.items()
            if seed == source.seed
        )
        budgets.append(
            _evaluation(
                inputs,
                source.a_validation,
                repetitions=candidate_count,
                samples=selection_samples,
                max_tokens=stage_a_tokens,
            )
        )
        maps = [sft_datum(inputs.runtime, row) for row in stage_b_sources(source)]
        train = 0
        for step in range(1, STAGE_B_GRID[-1] + 1):
            train += sum(
                int(maps[((step - 1) * 32 + offset) % len(maps)].model_input.length)
                for offset in range(32)
            )
        budgets.append(TokenBudget(0, 0, train * 2))
        budgets.append(
            _evaluation(
                inputs,
                source.b_validation,
                repetitions=2 * len(STAGE_B_GRID),
                samples=pilot_samples,
                max_tokens=STAGE_B_MAX_TOKENS,
            )
        )
        budgets.append(
            _evaluation(
                inputs,
                source.prompt_pools.a_monitor_manifest,
                repetitions=2 * (len(STAGE_B_GRID) - 1),
                samples=monitor_samples,
                max_tokens=stage_a_tokens,
            )
        )
        budgets.append(
            _evaluation(
                inputs,
                source.a_validation,
                repetitions=2,
                samples=selection_samples,
                max_tokens=stage_a_tokens,
            )
        )
    tokens = _sum(budgets)
    cell_count = len(cells)
    fixed = cell_count * (9 * 0.1 + 1.0)
    upper = (
        PRICE_SNAPSHOT.cost(
            UsageQuantities(
                prefill_tokens=tokens.prefill,
                sample_tokens=tokens.sample,
                train_tokens=tokens.train,
            )
        )
        + fixed
    )
    return MatchedPilotBudget(tokens, fixed, upper, sum(candidate_counts.values()))


__all__ = [
    "MatchedPilotBudget",
    "calculate_matched_pilot_budget",
    "validate_matched_authorization",
]
