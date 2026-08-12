"""Pinned role-colon conversion for supervised and grouped-RL data."""

from __future__ import annotations

import math
from typing import Any, Sequence

from duraseed.runtime.client import RuntimeBundle
from duraseed.training.sft import VerifiedSourceRecord
from duraseed.training.teacher_allocation import (
    TeacherTokenCounts,
    TeacherTokenMeasurer,
)


SUPERVISED_MAX_LENGTH = 1024


def _messages(source: VerifiedSourceRecord) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": source.prompt_text},
        {"role": "assistant", "content": source.verified_completion_text},
    ]


def sft_datum(
    runtime: RuntimeBundle,
    source: VerifiedSourceRecord,
    *,
    max_length: int = SUPERVISED_MAX_LENGTH,
) -> Any:
    """Build one mean-reduced LAST_ASSISTANT_MESSAGE supervised datum."""

    messages = _messages(source)
    full_input, _ = runtime.renderer.build_supervised_example(
        messages,
        train_on_what=runtime.sdk.train_on_what.LAST_ASSISTANT_MESSAGE,
    )
    if int(full_input.length) > max_length:
        raise ValueError(
            "supervised source would be truncated by the 1024-token contract"
        )
    datum = runtime.sdk.conversation_to_datum(
        messages,
        runtime.renderer,
        max_length=max_length,
        train_on_what=runtime.sdk.train_on_what.LAST_ASSISTANT_MESSAGE,
        reduction="mean",
    )
    if int(datum.model_input.length) != int(full_input.length) - 1:
        raise ValueError("supervised datum was truncated or shifted unexpectedly")
    return datum


def teacher_token_measurer(
    runtime: RuntimeBundle,
    *,
    max_length: int = SUPERVISED_MAX_LENGTH,
) -> TeacherTokenMeasurer:
    """Measure exact prompt/target counts from source text, never candidate metadata."""

    def measure(source: VerifiedSourceRecord) -> TeacherTokenCounts:
        messages = _messages(source)
        full_input, full_weights = runtime.renderer.build_supervised_example(
            messages,
            train_on_what=runtime.sdk.train_on_what.LAST_ASSISTANT_MESSAGE,
        )
        full_length = int(full_input.length)
        if full_length > max_length:
            raise ValueError(
                "teacher example would be truncated by the 1024-token contract"
            )
        if len(full_weights) != full_length:
            raise ValueError("role-colon full weights do not align with model input")
        datum = sft_datum(runtime, source, max_length=max_length)
        datum_length = int(datum.model_input.length)
        if datum_length != full_length - 1:
            raise ValueError("supervised datum was truncated or shifted unexpectedly")
        try:
            weights = tuple(
                float(value) for value in datum.loss_fn_inputs["weights"].data
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError("supervised datum omitted its token weights") from error
        if len(weights) != datum_length or any(
            not math.isfinite(weight) or weight < 0 for weight in weights
        ):
            raise ValueError("supervised datum returned invalid token weights")
        positive = tuple(index for index, weight in enumerate(weights) if weight > 0)
        if not positive or positive != tuple(range(positive[0], datum_length)):
            raise ValueError(
                "LAST_ASSISTANT_MESSAGE weights must be one nonempty suffix"
            )
        prompt = positive[0] + 1
        target = len(positive)
        if datum_length != prompt + target - 1:
            raise ValueError("prompt and target counts do not reconstruct the datum")
        generation = runtime.renderer.build_generation_prompt(
            [{"role": "user", "content": source.prompt_text}],
            role="assistant",
        )
        if int(generation.length) != prompt or tuple(datum.model_input.to_ints())[
            :prompt
        ] != tuple(generation.to_ints()):
            raise ValueError("supervised prefix differs from the generation prompt")
        return TeacherTokenCounts(prompt=prompt, target=target)

    return measure


def rl_datums(
    runtime: RuntimeBundle,
    observations: Sequence[Any],
    advantages: Sequence[float],
) -> list[Any]:
    """Convert sampled token/logprob observations into importance-sampling data."""

    if len(observations) != len(advantages):
        raise ValueError("observations and advantages must align")
    datums: list[Any] = []
    for row, advantage in zip(observations, advantages, strict=True):
        try:
            numeric_advantage = float(advantage)
            tokens = tuple(int(value) for value in row.tokens)
            logprobs = tuple(float(value) for value in row.logprobs)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "RL observations must contain numeric sample data"
            ) from error
        if not math.isfinite(numeric_advantage):
            raise ValueError("advantages must be finite")
        if (
            not tokens
            or len(tokens) != len(logprobs)
            or any(not math.isfinite(value) for value in logprobs)
        ):
            raise ValueError("RL tokens and finite log-probabilities must align")
        observation_length = int(row.prompt.length) - 1
        if observation_length < 0:
            raise ValueError("RL generation prompts must contain at least one token")
        model_input = row.prompt.append(
            runtime.sdk.tinker.EncodedTextChunk(tokens=list(tokens[:-1]))
        )
        generated_positions = int(model_input.length) - observation_length
        if generated_positions != len(tokens):
            raise ValueError("RL model input does not preserve the generation offset")
        datums.append(
            runtime.sdk.tinker.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": [0] * observation_length + list(tokens),
                    "logprobs": [0.0] * observation_length + list(logprobs),
                    "advantages": [0.0] * observation_length
                    + [numeric_advantage] * generated_positions,
                },
            )
        )
    return datums


__all__ = [
    "SUPERVISED_MAX_LENGTH",
    "rl_datums",
    "sft_datum",
    "teacher_token_measurer",
]
