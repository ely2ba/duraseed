"""Fixed endpoint-clone MAPS continuations with a live optimizer per repetition."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from duraseed.pilot0_data import stage_b_sources
from duraseed.replay_evaluation import evaluate
from duraseed.replay_inputs import read_json
from duraseed.replay_remote import write_json
from duraseed.runners.pilot0_sampling import _panel_role
from duraseed.runtime import sft_datum

RUN_STOPS = {"T1": 480, "T2": 20, "S1": 480, "S2": 20}
DENSE_GRID = tuple(range(21))
LATE_GRID = (40, 80, 160, 320, 480)
MAPS_GRID = (0, 1, 2, 5, 10, 20, *LATE_GRID)
PANELS = {
    "a_monitor": (384, 4, 4096),
    "b_validation": (512, 16, 128),
    "a_validation": (512, 16, 4096),
}


def grids(stop):
    if stop not in (20, 480):
        raise ValueError("only the approved 20/480-update continuations are allowed")
    return {
        "a_monitor": tuple(u for u in (*DENSE_GRID, *LATE_GRID) if u <= stop),
        "b_validation": tuple(u for u in MAPS_GRID if u <= stop),
        "a_validation": (480,) if stop == 480 else (),
    }


async def run_continuation(remote, source, origin: dict, label: str, stop: int):
    """One weights-only entry, no restore between updates, independent evaluations.

    The caller supplies a separate remote/journal and preallocated budget for each
    repetition. Partial executions require explicit reconciliation, never replay.
    """
    if RUN_STOPS.get(label) != stop:
        raise ValueError("continuation label/duration differs from the fixed matrix")
    manifest_by_panel = {
        "a_monitor": source.prompt_pools.a_monitor_manifest,
        "b_validation": source.b_validation,
        "a_validation": source.a_validation,
    }
    for panel, manifest in manifest_by_panel.items():
        if manifest.record_count != PANELS[panel][0]:
            raise ValueError(f"{panel} item count differs from the frozen schedule")
        if panel.startswith("a_"):
            roles = Counter(_panel_role(source, row) for row in manifest.records)
            expected = manifest.record_count // 2
            if roles != {"targeted": expected, "sentinel": expected}:
                raise ValueError(f"{panel} requires equal targeted/sentinel roles")
    record = {
        "schema": "endpoint-clone-continuation-v1",
        "label": label,
        "seed": source.seed,
        "stop": stop,
        "origin": origin,
        "grids": {k: list(v) for k, v in grids(stop).items()},
        "evaluation_namespace": f"endpoint_clone.{label}",
        "fresh_optimizer": True,
    }
    plan = remote.root / "continuation.json"
    if plan.exists() and read_json(plan) != record:
        raise ValueError("retained continuation identity changed")
    terminal = remote.root / "result.json"
    if terminal.exists():
        result = read_json(terminal)
        if not plan.exists() or result.get("status") != "COMPLETED":
            raise ValueError("continuation terminal record is inconsistent")
        return result
    stage_root = remote.root / "stage_b"
    if stage_root.exists():
        raise ValueError("partial continuation requires reconciliation; never replay")
    write_json(plan, record)
    records = stage_b_sources(source)
    if len(records) != 4096 or remote.config["stage_b_lr"] != 3e-4:
        raise ValueError("Stage-B data count or learning rate differs from the recipe")
    coordinate = {"seed": source.seed, "replay_arm": label, "stage": "stage_b"}
    await remote.restore(
        origin["state_path"],
        full_state=False,
        coordinate={**coordinate, "update": 0},
        sources=records,
    )
    # The model and optimizer stay resident for the entire continuation. Every
    # immutable sampler checkpoint can be evaluated without restoring training.
    datums = [
        sft_datum(remote.runtime, row, max_length=remote.config["max_length"])
        for row in records
    ]
    checkpoint = {
        **origin,
        **coordinate,
        "update": 0,
        "origin_sampler_path": origin["sampler_path"],
    }
    write_json(stage_root / "u0" / "checkpoint.json", checkpoint)
    schedule, evaluated, samples, tokens = grids(stop), [], 0, 0
    for update in range(stop + 1):
        point = stage_root / f"u{update}"
        if update:
            batch = [datums[((update - 1) * 32 + j) % len(datums)] for j in range(32)]
            await remote.update(
                batch,
                3e-4,
                point / f"update-{update}.json",
                {**coordinate, "update": update},
            )
            tokens += sum(int(d.model_input.length) for d in batch)
            if update in schedule["a_monitor"]:
                checkpoint = await remote.save(
                    f"{remote.run_id}-{label}-stage-b-u{update}",
                    point / "checkpoint.json",
                    {**coordinate, "update": update},
                )
                checkpoint["origin_sampler_path"] = origin["sampler_path"]
        for panel, updates in schedule.items():
            if update not in updates:
                continue
            _, draws, cap = PANELS[panel]
            result = await evaluate(
                remote,
                source,
                checkpoint,
                manifest_by_panel[panel],
                stage="stage_b",
                update=update,
                arm=label,
                purpose=panel,
                draws=draws,
                cap=cap,
                output=point / panel,
                seed_namespace=f"endpoint_clone.{label}.{panel}.{update}",
            )
            samples += result["row_count"]
            evaluated.append(
                {"update": update, "panel": panel, "completions": result["row_count"]}
            )
    result = {
        "status": "COMPLETED",
        "label": label,
        "updates": stop,
        "training_examples": stop * 32,
        "training_tokens": tokens,
        "evaluation_completions": samples,
        "evaluations": evaluated,
        "checkpoint_pairs": len(schedule["a_monitor"]) - 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "billing": remote.snapshot(),
    }
    write_json(terminal, result)
    return result
