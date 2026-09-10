"""Local endpoint-clone populations and one finite, worker-partitioned cost bound."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

from duraseed.config import load_pilot_config
from duraseed.data.manifests import build_manifest, read_manifest, write_manifest
from duraseed.data.sealing import ExecutionContext
from duraseed.data.splits import tces_numeric_key
from duraseed.data.stage_a_prompt_pools import PromptPoolStratum
from duraseed.endpoint_confirmation import build_confirmation, validate_confirmation
from duraseed.pilot0_data import (
    ordered_stage_a_pools,
    scheduled_stage_a_records,
    stage_b_sources,
)
from duraseed.provenance import sha256_bytes
from duraseed.replay_data import batch_indices, order_key
from duraseed.replay_data_archive import read_source
from duraseed.replay_data_tokens import local_runtime
from duraseed.replay_inputs import read_json
from duraseed.replay_remote import write_json
from duraseed.runners.pilot0_sampling import _prompt
from duraseed.runtime import TokenBudget, sft_datum
from duraseed.runtime.billing import PRICE_SNAPSHOT, UsageQuantities

SOURCE_CONFIG = Path("runs/replay-v1/preparation/config.json")
SEED = 11
TEACHER_UPDATES = 30
STUDENT_GRID = (*range(10, 291, 10), 294)
LONG_MONITOR_GRID = (*range(21), 40, 80, 160, 320, 480)
LONG_MAPS_GRID = (0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480)
CLONE_COUNTS = {
    PromptPoolStratum.BOUNDARY: 400,
    PromptPoolStratum.INTERMEDIATE: 200,
    PromptPoolStratum.BROAD_RANDOM: 200,
}


def teacher_records(source):
    pools = ordered_stage_a_pools(source)
    return tuple(
        row
        for step in range(1, TEACHER_UPDATES + 1)
        for row in scheduled_stage_a_records(
            pools, source.prompt_pools.artifact.bg_group_order, step
        )
    )


def clone_manifest(source):
    teacher_ids = {row.task_id for row in teacher_records(source)}
    pools = ordered_stage_a_pools(source)
    rows = []
    for stratum, count in CLONE_COUNTS.items():
        available = [row for row in pools[stratum] if row.task_id not in teacher_ids]
        if len(available) < count:
            raise ValueError(f"insufficient unseen teacher prompts in {stratum}")
        rows.extend(available[:count])
    original = source.prompt_pools.a_rl_train_manifest
    manifest = build_manifest(
        name="endpoint-clone-prompts",
        split=original.split,
        generator_version=original.generator_version,
        root_seed=original.root_seed,
        records=rows,
        parent_manifest_id=original.manifest_id,
        metadata={
            "selection": "first unused rows of each fixed interleaved stratum",
            "counts": {key.value: value for key, value in CLONE_COUNTS.items()},
            "teacher_updates": TEACHER_UPDATES,
            "draws_per_prompt": 8,
        },
    )
    validate_clone(source, manifest)
    return manifest


def validate_clone(source, manifest):
    teacher = teacher_records(source)
    forbidden = (
        *teacher,
        *source.a_validation.records,
        *source.prompt_pools.a_monitor_manifest.records,
    )
    numeric = {tces_numeric_key(row) for row in forbidden}
    content = {row.content_hash for row in forbidden}
    artifact = source.prompt_pools.artifact
    sentinel = set(artifact.sentinel_family_ids)
    families = (
        artifact.boundary_family_ids,
        artifact.intermediate_family_ids,
        artifact.broad_random_family_ids,
    )
    counts = [
        sum(row.intended_family in group for row in manifest.records)
        for group in families
    ]
    if (
        len(teacher) != 480
        or len({row.task_id for row in teacher}) != 480
        or manifest.record_count != 800
        or counts != [400, 200, 200]
        or len({tces_numeric_key(row) for row in manifest.records}) != 800
        or any(
            tces_numeric_key(row) in numeric
            or row.content_hash in content
            or sentinel.intersection(row.valid_family_ids)
            for row in manifest.records
        )
    ):
        raise ValueError(
            "clone prompts violate unused-prompt, stratum, or sentinel separation"
        )


def corpus_coordinates(manifest):
    rows = [(row.task_id, draw) for row in manifest.records for draw in range(8)]
    return tuple(
        sorted(rows, key=lambda pair: (order_key(SEED, f"{pair[0]}:{pair[1]}"), pair))
    )


def _base_inputs(repo):
    original = read_json(repo / SOURCE_CONFIG)
    block = original["blocks"][str(SEED)]
    source, old_preflight = read_source(repo / block["source_run"], SEED)
    if old_preflight["lineage"]["m0_state_path"] != block["m0_state_path"]:
        raise ValueError("source M0 identity differs from original run")
    config = {
        key: deepcopy(original[key])
        for key in (
            "batch_size",
            "evaluation",
            "checkpoint_ttl_seconds",
            "storage_per_pair_usd",
            "tokenizer_path",
            "project_id",
            "stage_a_lr",
            "stage_b_lr",
        )
    }
    config.update(
        {
            "namespace": "endpoint-clone",
            "tces_max_tokens": 4096,
            "seed": SEED,
            "source_block": SEED,
            "source_run": block["source_run"],
            "m0_state_path": block["m0_state_path"],
            "m0_sampler_path": block["m0_sampler_path"],
            "teacher_updates": TEACHER_UPDATES,
            "teacher_lr": 1e-5,
            "stage_a_grid": list(STUDENT_GRID),
            "stage_b_grid": list(LONG_MAPS_GRID),
            "monitor_grid": list(LONG_MONITOR_GRID),
            "corpus_order_seed": SEED,
            "confirmation_seed": 20260910,
            "runs": {
                name: {"stop": stop, "source": origin}
                for name, stop, origin in (
                    ("T1", 480, "teacher"),
                    ("S1", 480, "student"),
                    ("T2", 20, "teacher"),
                    ("S2", 20, "student"),
                )
            },
        }
    )
    return config, source


def build_preflight(config, source, clone, confirmation, runtime):
    prompt_cache = {}

    def lengths(manifest_or_rows):
        rows = getattr(manifest_or_rows, "records", manifest_or_rows)
        for row in rows:
            if row.task_id not in prompt_cache:
                prompt_cache[row.task_id] = int(
                    runtime.renderer.build_generation_prompt(
                        [{"role": "user", "content": _prompt(row)}], role="assistant"
                    ).length
                )
        return [prompt_cache[row.task_id] for row in rows]

    panels = {
        "cadence": source.a_cadence,
        "selection": source.a_validation,
        "confirmation": confirmation,
        "monitor": source.prompt_pools.a_monitor_manifest,
        "maps": source.b_validation,
    }
    panel_lengths = {name: lengths(panel) for name, panel in panels.items()}
    teacher_lengths = lengths(teacher_records(source))
    clone_lengths = lengths(clone)
    by_id = dict(
        zip((row.task_id for row in clone.records), clone_lengths, strict=True)
    )
    corpus_lengths = [by_id[task] + 4096 - 1 for task, _ in corpus_coordinates(clone)]
    config["max_length"] = max(max(corpus_lengths) + 1, 4217)
    student_updates = [
        sum(corpus_lengths[i] for i in batch_indices(6400, step))
        for step in range(1, 295)
    ]
    maps_lengths = [
        int(sft_datum(runtime, row, max_length=config["max_length"]).model_input.length)
        for row in stage_b_sources(source)
    ]
    maps_updates = [
        sum(maps_lengths[((step - 1) * 32 + j) % len(maps_lengths)] for j in range(32))
        for step in range(1, 481)
    ]
    workers = {}

    def worker(name, pairs, ephemeral=0):
        workers[name] = {
            "components": [],
            "token_budget": TokenBudget(0, 0, 0),
            "storage": {
                "pairs": pairs,
                "usd_per_pair": 0.25,
                "reserved_usd": pairs * 0.25,
                "ttl_seconds": config["checkpoint_ttl_seconds"],
            },
            "ephemeral_sampler_reserve_usd": ephemeral * 0.05,
        }

    def append(name, component, budget, **counts):
        entry = workers[name]
        entry["token_budget"] = entry["token_budget"].plus(budget)
        entry["components"].append({"component": component, **counts, **asdict(budget)})

    def sample(name, component, lens, draws, points, cap):
        n = len(lens) * draws * points
        counts = dict(
            items=len(lens), draws=draws, points=points, cap=cap, completions=n
        )
        budget = TokenBudget(sum(lens) * draws * points, n * cap, 0)
        append(name, component, budget, **counts)

    worker("acquisition", 33, 30)
    sample("acquisition", "teacher_rollouts", teacher_lengths, 8, 1, 4096)
    append(
        "acquisition",
        "teacher_train",
        TokenBudget(0, 0, sum(n + 4096 - 1 for n in teacher_lengths) * 8),
        updates=30,
        groups=480,
        draws=8,
    )
    sample("acquisition", "teacher_cadence", panel_lengths["cadence"], 1, 3, 4096)
    sample("acquisition", "clone_corpus", clone_lengths, 8, 1, 4096)
    append(
        "acquisition",
        "student_train",
        TokenBudget(0, 0, sum(student_updates)),
        updates=294,
        batch=32,
    )
    sample("acquisition", "student_cadence", panel_lengths["cadence"], 1, 30, 4096)
    sample("acquisition", "selection_profiles", panel_lengths["selection"], 16, 4, 4096)
    sample(
        "acquisition",
        "confirmation_profiles",
        panel_lengths["confirmation"],
        16,
        3,
        4096,
    )
    for name, settings in config["runs"].items():
        stop = settings["stop"]
        monitor_grid = [u for u in LONG_MONITOR_GRID if u <= stop]
        maps_grid = [u for u in LONG_MAPS_GRID if u <= stop]
        worker(name, len(monitor_grid) - 1)
        append(
            name,
            "stage_b_train",
            TokenBudget(0, 0, sum(maps_updates[:stop])),
            updates=stop,
            batch=32,
        )
        sample(
            name,
            "stage_b_monitor",
            panel_lengths["monitor"],
            4,
            len(monitor_grid),
            4096,
        )
        sample(name, "stage_b_maps", panel_lengths["maps"], 16, len(maps_grid), 128)
        if stop == 480:
            sample(name, "final_arithmetic", panel_lengths["selection"], 16, 1, 4096)
    total = TokenBudget(0, 0, 0)
    for value in workers.values():
        tokens = value["token_budget"]
        raw = PRICE_SNAPSHOT.cost(
            UsageQuantities(
                prefill_tokens=tokens.prefill,
                sample_tokens=tokens.sample,
                train_tokens=tokens.train,
            )
        )
        raw += value["storage"]["reserved_usd"] + value["ephemeral_sampler_reserve_usd"]
        value["main_usd"] = str(
            Decimal(str(raw)).quantize(Decimal(".01"), rounding=ROUND_CEILING)
        )
        value["token_budget"] = asdict(tokens)
        total = total.plus(tokens)
    ceiling = sum(
        (Decimal(value["main_usd"]) for value in workers.values()), Decimal(0)
    )
    return {
        "workers": workers,
        "main_usd": str(ceiling),
        "token_budget": asdict(total),
        "price_snapshot": PRICE_SNAPSHOT.snapshot_id,
        "storage": {
            "pairs": sum(v["storage"]["pairs"] for v in workers.values()),
            "reserved_usd": sum(v["storage"]["reserved_usd"] for v in workers.values()),
        },
        "measurements": {
            "teacher_prompt_tokens": teacher_lengths,
            "clone_prompt_tokens": by_id,
            "panel_prompt_tokens": panel_lengths,
            "student_maximum_train_tokens_per_update": student_updates,
            "stage_b_train_tokens_per_update": maps_updates,
        },
        "budget_scope": "All five worker caps sum to the single approved package; no independent extra allocations.",
    }


def prepare(repo: Path, root: Path):
    if (root / "config.json").exists() or (
        root / "confirmation_manifest.json"
    ).exists():
        raise FileExistsError(f"endpoint inputs already exist: {root}")
    config, source = _base_inputs(repo)
    clone = clone_manifest(source)
    generator = load_pilot_config(repo / "duraseed_pilot_config.yaml").tasks.tces
    from duraseed.tasks.tces import TCESGeneratorConfig

    confirmation = build_confirmation(
        source, TCESGeneratorConfig(**generator.generator_kwargs())
    )
    runtime = local_runtime(Path(config["tokenizer_path"]))
    preflight = build_preflight(config, source, clone, confirmation, runtime)
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root / "clone_manifest.json", clone)
    write_manifest(root / "confirmation_manifest.json", confirmation)
    write_json(root / "preflight.json", preflight)

    def ref(path):
        return {
            "path": str(path.relative_to(repo)),
            "sha256": sha256_bytes(path.read_bytes()),
        }

    config.update(
        {
            "run_id": root.name,
            "source_config": ref(repo / SOURCE_CONFIG),
            "preflight": ref(root / "preflight.json"),
            "clone_manifest": ref(root / "clone_manifest.json"),
            "confirmation_manifest": ref(root / "confirmation_manifest.json"),
            "authorized_package_usd": preflight["main_usd"],
        }
    )
    write_json(root / "config.json", config)
    load_inputs(repo, root)
    return preflight


def load_inputs(repo: Path, root: Path):
    config = read_json(root / "config.json")
    source, _ = read_source(repo / config["source_run"], SEED)
    manifests = []
    for name in ("clone_manifest", "confirmation_manifest"):
        reference = config[name]
        path = repo / reference["path"]
        if sha256_bytes(path.read_bytes()) != reference["sha256"]:
            raise ValueError(f"prepared {name} changed")
        manifests.append(read_manifest(path, context=ExecutionContext.SELECTION))
    clone, confirmation = manifests
    validate_clone(source, clone)
    validate_confirmation(source, confirmation)
    preflight = read_json(repo / config["preflight"]["path"])
    if Decimal(preflight["main_usd"]) > Decimal(
        config["authorized_package_usd"]
    ) or sum(Decimal(v["main_usd"]) for v in preflight["workers"].values()) != Decimal(
        preflight["main_usd"]
    ):
        raise ValueError(
            "endpoint preflight exceeds authorization or partitions disagree"
        )
    return config, source, clone, confirmation
