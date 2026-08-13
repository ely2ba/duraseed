"""Seeded grouped sampling with exact DuraSeed run-record projection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from typing import Any, Literal

from duraseed.provenance import derive_namespaced_seed
from duraseed.run_records import GenerationRecord, MethodCode, RewardRecord
from duraseed.runtime.client import RuntimeBundle
from duraseed.runtime.ledger import TokenBudget, TokenLedger
from duraseed.training.reward import verify_task_completion


ANSWER_TAG_STOP = "</answer>"
ROLE_COLON_USER_STOP = "\nUser:"


@dataclass(frozen=True, slots=True)
class SamplingTask:
    manifest_id: str
    task_id: str
    task_family: Literal["tces", "maps"]
    source_split: str
    prompt_text: str
    exact_task: Any
    item_index: int
    assigned_family_id: str | None = None
    panel_role: str | None = None


@dataclass(frozen=True, slots=True)
class SamplingCoordinates:
    run_id: str
    label: str
    purpose: Literal["training", "evaluation"]
    checkpoint_stage: Literal["base", "m0", "stage_a", "stage_b"]
    training_step: int
    sampler_checkpoint_path: str
    origin_sampler_checkpoint_path: str
    experiment_seed: int
    seed_namespace: str
    method: MethodCode | None = None


@dataclass(frozen=True, slots=True)
class SampleObservation:
    generation: GenerationRecord
    reward: RewardRecord
    prompt: Any
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]


def sampling_stops(renderer: Any) -> tuple[str, ...]:
    stops = list(renderer.get_stop_sequences())
    for value in (ANSWER_TAG_STOP, ROLE_COLON_USER_STOP):
        if value not in stops:
            stops.append(value)
    return tuple(stops)


def _completion(tokenizer: Any, tokens: tuple[int, ...]) -> str:
    content = list(tokens)
    eos = getattr(tokenizer, "eos_token_id", None)
    if content and eos is not None and content[-1] == eos:
        content.pop()
    text = str(tokenizer.decode(content))
    for delimiter in ("\n\nUser:", ROLE_COLON_USER_STOP):
        if text.endswith(delimiter):
            return text[: -len(delimiter)]
    return text


def _termination(
    runtime: RuntimeBundle,
    tokens: tuple[int, ...],
    stop_reason: Any,
) -> str:
    _, termination = runtime.renderer.parse_response(list(tokens))
    if str(stop_reason) == "stop" and str(runtime.tokenizer.decode(tokens)).endswith(
        (ANSWER_TAG_STOP, ROLE_COLON_USER_STOP)
    ):
        return "stop_sequence"
    if termination is None or not str(termination).strip():
        raise RuntimeError("renderer returned an empty termination")
    return str(termination)


async def sample_seeded(
    runtime: RuntimeBundle,
    sampler: Any,
    task: SamplingTask,
    coordinates: SamplingCoordinates,
    *,
    group_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    ledger: TokenLedger,
    sample_index_start: int = 0,
    explicit_seeds: tuple[int, ...] | None = None,
) -> tuple[SampleObservation, ...]:
    """Make independently seeded one-sample calls for one prompt group."""

    if group_size < 1 or max_tokens < 1 or sample_index_start < 0:
        raise ValueError("group_size and max_tokens must be positive")
    if (
        not math.isfinite(temperature)
        or temperature <= 0
        or not math.isfinite(top_p)
        or not 0 < top_p <= 1
    ):
        raise ValueError("temperature and top_p must be finite and in range")
    prompt = runtime.renderer.build_generation_prompt(
        [{"role": "user", "content": task.prompt_text}],
        role="assistant",
    )
    prompt_tokens = int(prompt.length)
    reserved = TokenBudget(prompt_tokens * group_size, max_tokens * group_size, 0)
    sample_indices = tuple(range(sample_index_start, sample_index_start + group_size))
    if explicit_seeds is not None:
        valid_shape = len(explicit_seeds) == group_size == len(set(explicit_seeds))
        valid_values = all(type(seed) is int and seed >= 0 for seed in explicit_seeds)
        if not valid_shape or not valid_values:
            raise ValueError("explicit sampling seeds must match the group")
        sample_seeds = explicit_seeds
    else:
        sample_seeds = tuple(
            derive_namespaced_seed(
                coordinates.experiment_seed,
                coordinates.seed_namespace,
                task.task_family,
                task.task_id,
                task.item_index,
                sample_index,
            )
            for sample_index in sample_indices
        )
    ledger.reserve_call(reserved)
    try:
        responses = await asyncio.gather(
            *(
                sampler.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=runtime.sdk.tinker.SamplingParams(
                        max_tokens=max_tokens,
                        seed=seed,
                        stop=list(sampling_stops(runtime.renderer)),
                        temperature=temperature,
                        top_k=-1,
                        top_p=top_p,
                    ),
                )
                for seed in sample_seeds
            )
        )
        sequences = []
        for response in responses:
            values = tuple(response.sequences)
            if len(values) != 1:
                raise RuntimeError("each seeded request must return exactly one sample")
            sequence = values[0]
            if getattr(sequence, "stop_reason", None) is None:
                raise RuntimeError("sample omitted its stop reason")
            tokens = tuple(int(value) for value in sequence.tokens)
            if sequence.logprobs is None:
                raise RuntimeError("sample omitted completion log-probabilities")
            logprobs = tuple(float(value) for value in sequence.logprobs)
            if len(tokens) != len(logprobs) or any(
                not math.isfinite(value) for value in logprobs
            ):
                raise RuntimeError(
                    "sample tokens and finite log-probabilities must align"
                )
            sequences.append((sequence, tokens, logprobs))
    except Exception:
        ledger.abort_call()
        raise
    ledger.settle_call(
        TokenBudget(
            prompt_tokens * group_size,
            sum(len(tokens) for _, tokens, _ in sequences),
            0,
        )
    )

    rows: list[SampleObservation] = []
    for offset, (sequence, tokens, logprobs) in enumerate(sequences):
        sample_index = sample_indices[offset]
        completion = _completion(runtime.tokenizer, tokens)
        verification = verify_task_completion(completion, task.exact_task)
        sample_id = (
            f"{coordinates.run_id}:{coordinates.label}:{task.task_family}:"
            f"{task.task_id}:{task.item_index}:sample-{sample_index}:cap-{max_tokens}"
        )
        generation = GenerationRecord(
            sample_id=sample_id,
            sample_index=sample_index,
            sampling_seed=sample_seeds[offset],
            purpose=coordinates.purpose,
            checkpoint_stage=coordinates.checkpoint_stage,
            training_step=coordinates.training_step,
            sampler_checkpoint_path=coordinates.sampler_checkpoint_path,
            task_manifest_id=task.manifest_id,
            task_id=task.task_id,
            task_family=task.task_family,
            source_split=task.source_split,
            prompt_text=task.prompt_text,
            completion_text=completion,
            prompt_tokens=prompt_tokens,
            sampled_tokens=len(tokens),
            sampling_temperature=temperature,
            sampling_top_p=top_p,
            sampling_max_tokens=max_tokens,
            run_id=coordinates.run_id,
            method=coordinates.method,
            seed=coordinates.experiment_seed,
            origin_sampler_checkpoint_path=coordinates.origin_sampler_checkpoint_path,
            item_index=task.item_index,
            assigned_family_id=task.assigned_family_id,
            family_id=verification.strategy_family_id,
            panel_role=task.panel_role,
            completion_token_ids=tokens,
            completion_logprobs=logprobs,
            stop_reason=str(sequence.stop_reason),
            renderer_termination=_termination(runtime, tokens, sequence.stop_reason),
            reward=verification.reward,
            advantage=None,
        )
        reward = RewardRecord(
            reward_id=f"reward:{sample_id}",
            sample_id=sample_id,
            task_id=task.task_id,
            reward=verification.reward,
            exact_verification=verification,
        )
        rows.append(SampleObservation(generation, reward, prompt, tokens, logprobs))
    return tuple(rows)


__all__ = [
    "SampleObservation",
    "SamplingCoordinates",
    "SamplingTask",
    "sample_seeded",
    "sampling_stops",
]
