"""One live teacher/student client across the endpoint-clone acquisition cadence."""

from __future__ import annotations

from math import fsum

from duraseed.pilot0_data import ordered_stage_a_pools
from duraseed.replay_evaluation import evaluate
from duraseed.replay_remote import write_json


def raw_datums(runtime, rows, *, max_length, mean_response_length=None):
    """Unfiltered sampled-token NLL with a single corpus-wide normalization."""
    rows = tuple(rows)
    if not rows:
        raise ValueError("the unfiltered corpus is empty")
    mean = fsum(len(row["generation"]["completion_token_ids"]) for row in rows) / len(
        rows
    )
    if mean_response_length is not None and mean_response_length != mean:
        raise ValueError(
            "corpus-wide normalization differs from sampled response lengths"
        )
    if mean <= 0:
        raise ValueError("the corpus contains no response tokens")
    datums = []
    for row in rows:
        prompt = list(row["prompt_token_ids"])
        tokens = list(row["generation"]["completion_token_ids"])
        if not prompt or len(prompt) + len(tokens) - 1 > max_length:
            raise ValueError("raw sampled datum is empty-prefix or would be truncated")
        # Predict precisely the observed suffix. No rendering, trimming, EOS
        # insertion, or special treatment of invalid/capped/empty responses.
        prefix = len(prompt) - 1
        inputs = (prompt + tokens)[:-1]
        datums.append(
            runtime.sdk.tinker.Datum(
                model_input=runtime.sdk.tinker.ModelInput.from_ints(inputs),
                loss_fn_inputs={
                    "target_tokens": [0] * prefix + tokens,
                    "weights": [0.0] * prefix + [1.0 / mean] * len(tokens),
                },
            )
        )
    return datums


def _fresh(remote, arm):
    directory = remote.root / arm / "stage_a"
    if directory.exists():
        raise ValueError(
            "acquisition directory already exists; inspect, never restart automatically"
        )
    return directory


async def _cadence(remote, source, checkpoint, arm, directory):
    result = await evaluate(
        remote,
        source,
        checkpoint,
        source.a_cadence,
        stage="stage_a",
        update=checkpoint["update"],
        arm=arm,
        purpose="cadence",
        draws=1,
        cap=4096,
        output=directory / "cadence",
    )
    return {"update": checkpoint["update"], **result}


async def train_teacher(remote, source, m0, pools=None):
    directory = _fresh(remote, "T")
    await remote.restore(
        m0["state_path"],
        full_state=False,
        coordinate={"arm": "T", "stage": "stage_a", "update": 0},
    )
    pools = ordered_stage_a_pools(source) if pools is None else pools
    cadence, checkpoints = [], []
    for step in range(1, remote.config.get("teacher_updates", 30) + 1):
        await remote.rl_update(
            source,
            pools,
            step=step,
            learning_rate=remote.config.get("teacher_lr", 1e-5),
            boundary_sampler_path=m0["sampler_path"],
            output=directory / f"u{step}",
        )
        if step % 10 == 0:
            checkpoint = await remote.save(
                f"{remote.run_id}-teacher-u{step}",
                directory / f"u{step}" / "checkpoint.json",
                {
                    "arm": "T",
                    "stage": "stage_a",
                    "update": step,
                    "seed": source.seed,
                    "origin_state_path": m0["state_path"],
                },
            )
            checkpoints.append(checkpoint)
            cadence.append(
                await _cadence(remote, source, checkpoint, "T", directory / f"u{step}")
            )
    result = {
        "checkpoint": checkpoints[-1],
        "checkpoints": checkpoints,
        "cadence": cadence,
    }
    write_json(directory / "result.json", result)
    return result


async def train_student(remote, source, m0, corpus_rows, order):
    directory = _fresh(remote, "S")
    rows, order = tuple(corpus_rows), tuple(order)
    if len(rows) != 6400 or sorted(order) != list(range(len(rows))):
        raise ValueError(
            "student needs all 6,400 samples and one complete frozen permutation"
        )
    await remote.restore(
        m0["state_path"],
        full_state=False,
        coordinate={"arm": "S", "stage": "stage_a", "update": 0},
    )
    datums = raw_datums(remote.runtime, rows, max_length=remote.config["max_length"])
    cadence, checkpoints = [], []
    stop = remote.config.get("student_updates", 294)
    for step in range(1, stop + 1):
        indices = [order[((step - 1) * 32 + j) % len(order)] for j in range(32)]
        await remote.update(
            [datums[i] for i in indices],
            remote.config.get("stage_a_lr", 1e-4),
            directory / f"u{step}" / "update.json",
            {
                "arm": "S",
                "stage": "stage_a",
                "update": step,
                "examples": step * 32,
                "seed": source.seed,
            },
        )
        if step % 10 == 0 or step == stop:
            checkpoint = await remote.save(
                f"{remote.run_id}-student-u{step}",
                directory / f"u{step}" / "checkpoint.json",
                {
                    "arm": "S",
                    "stage": "stage_a",
                    "update": step,
                    "seed": source.seed,
                    "origin_state_path": m0["state_path"],
                },
            )
            checkpoints.append(checkpoint)
            cadence.append(
                await _cadence(remote, source, checkpoint, "S", directory / f"u{step}")
            )
    result = {
        "checkpoint": checkpoints[-1],
        "checkpoints": checkpoints,
        "cadence": cadence,
    }
    write_json(directory / "result.json", result)
    return result
