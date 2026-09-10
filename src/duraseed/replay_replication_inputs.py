"""Offline inputs for one order-seed replication on frozen source block 11."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from duraseed.pilot0_data import stage_b_sources
from duraseed.provenance import sha256_bytes
from duraseed.replay_data import batch_indices, order_key
from duraseed.replay_data_tokens import local_runtime
from duraseed.replay_inputs import block_inputs, candidate_manifest, load_config
from duraseed.replay_matching import ARMS, STAGE_A_GRID, STAGE_B_GRID
from duraseed.replay_remote import write_json
from duraseed.runners.pilot0_sampling import _prompt
from duraseed.runtime import TokenBudget, sft_datum
from duraseed.runtime.billing import PRICE_SNAPSHOT, UsageQuantities

SEED = 47
SOURCE_BLOCK = 11
SOURCE_CONFIG = Path("runs/replay-v1/preparation/config.json")
EVALUATIONS = (
    ("stage_a_cadence", "a_cadence", 192, 1, 30, 4096),
    ("candidate_assessments", "targeted_a_validation", 256, 16, 3, 4096),
    ("pre_b_and_final_validation", "a_validation", 512, 16, 2, 4096),
    ("stage_b_monitor_including_pre_b", "a_monitor", 384, 4, 11, 4096),
    ("stage_b_maps_including_pre_b", "b_validation", 512, 16, 11, 128),
)


def ordered_records(records):
    """Reorder paired rows only: every original prompt and completion is retained."""
    maps = {arm: {row.task_id: row for row in records[arm]} for arm in ARMS}
    ids = set(maps[ARMS[0]])
    if any(
        len(maps[arm]) != len(records[arm]) or set(maps[arm]) != ids for arm in ARMS
    ):
        raise ValueError("replication corpus must contain paired unique task IDs")
    if any(
        maps[ARMS[0]][key].prompt_text != maps[ARMS[1]][key].prompt_text for key in ids
    ):
        raise ValueError("replication arms must share exact prompts")
    order = sorted(ids, key=lambda task_id: (order_key(SEED, task_id), task_id))
    return {arm: tuple(maps[arm][key] for key in order) for arm in ARMS}


def load_inputs(repo: Path):
    """Seed 47 changes order/sampling, not the frozen block-11 data population."""
    original = load_config(repo, repo / SOURCE_CONFIG)
    source, records = block_inputs(repo, original, SOURCE_BLOCK)
    if len(records["R-S"]) != 579:
        raise ValueError("source block 11 must retain its 579 frozen shared prompts")
    config = deepcopy(original)
    config["blocks"] = {str(SEED): deepcopy(original["blocks"][str(SOURCE_BLOCK)])}
    config["seeds"] = [SEED]
    config["source_block"] = SOURCE_BLOCK
    config["order_seed"] = SEED
    config["replication_scope"] = (
        "One paired acquisition-order/sampling seed on the unchanged block-11 "
        "corpus and evaluation populations; not a new corpus or task sample."
    )
    for key in (
        "targeted_matching",
        "owner_launch_direction",
        "launch_status",
        "implementation_commit",
        "implementation_files",
    ):
        config.pop(key, None)
    return config, replace(source, seed=SEED), ordered_records(records)


def build_preflight(config, source, records, runtime):
    """Render locally and price exactly the fixed two-arm maximum workload."""
    total = TokenBudget(0, 0, 0)
    components = []
    measurements = {}

    def append(name, tokens, **counts):
        nonlocal total
        total = total.plus(tokens)
        components.append({"component": name, **counts, **asdict(tokens)})

    def datum_lengths(rows):
        return [
            int(
                sft_datum(
                    runtime, row, max_length=config["max_length"]
                ).model_input.length
            )
            for row in rows
        ]

    for arm in ARMS:
        lengths = datum_lengths(records[arm])
        updates = [
            sum(lengths[index] for index in batch_indices(len(lengths), update))
            for update in range(1, 295)
        ]
        measurements[f"stage_a_{arm}_per_update_train_tokens"] = updates
        append(
            f"stage_a_{arm}_train",
            TokenBudget(0, 0, sum(updates)),
            arms=1,
            updates=294,
            batch=32,
            presentations=294 * 32,
        )
    maps = stage_b_sources(source)
    lengths = datum_lengths(maps)
    updates = [
        sum(
            lengths[((update - 1) * 32 + offset) % len(lengths)] for offset in range(32)
        )
        for update in range(1, 481)
    ]
    measurements["stage_b_per_update_train_tokens"] = updates
    append(
        "stage_b_train",
        TokenBudget(0, 0, 2 * sum(updates)),
        arms=2,
        updates=480,
        batch=32,
        presentations=2 * 480 * 32,
    )
    panels = {
        "a_cadence": source.a_cadence,
        "targeted_a_validation": candidate_manifest(source),
        "a_validation": source.a_validation,
        "a_monitor": source.prompt_pools.a_monitor_manifest,
        "b_validation": source.b_validation,
    }
    for name, panel, items, draws, points, cap in EVALUATIONS:
        manifest = panels[panel]
        if manifest.record_count != items:
            raise ValueError(f"frozen {panel} count differs from {items}")
        prefill = (
            sum(
                int(
                    runtime.renderer.build_generation_prompt(
                        [{"role": "user", "content": _prompt(row)}], role="assistant"
                    ).length
                )
                for row in manifest.records
            )
            * draws
            * 2
            * points
        )
        append(
            name,
            TokenBudget(prefill, items * draws * 2 * points * cap, 0),
            items=items,
            draws=draws,
            arms=2,
            points=points,
            cap=cap,
            completions=items * draws * 2 * points,
        )
    pairs = 2 * (len(STAGE_A_GRID) + len(STAGE_B_GRID) - 1)
    storage = pairs * config["storage_per_pair_usd"]
    token_cost = PRICE_SNAPSHOT.cost(
        UsageQuantities(
            prefill_tokens=total.prefill,
            sample_tokens=total.sample,
            train_tokens=total.train,
        )
    )
    return {
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "source_block": SOURCE_BLOCK,
        "components": components,
        "token_budget": asdict(total),
        "main_usd": str(token_cost + storage),
        "price_snapshot": PRICE_SNAPSHOT.snapshot_id,
        "storage": {
            "pairs": pairs,
            "usd_per_pair": config["storage_per_pair_usd"],
            "reserved_usd": storage,
            "ttl_seconds": config["checkpoint_ttl_seconds"],
        },
        "training_token_measurements": measurements,
        "scope": "Finite full-cap workload bound for the existing ledger; no new approval step or extra sampling.",
    }


def prepare(repo: Path, root: Path, protocol_path: Path):
    """Write a new launch envelope without changing or duplicating frozen corpora."""
    if root.exists():
        raise FileExistsError(f"refusing to overwrite a prepared run: {root}")
    config, source, records = load_inputs(repo)
    runtime = local_runtime(Path(config["tokenizer_path"]))
    preflight = build_preflight(config, source, records, runtime)

    def ref(path):
        return {
            "path": str(path.relative_to(repo)),
            "sha256": sha256_bytes(path.read_bytes()),
        }

    protocol_path = (
        repo / protocol_path if not protocol_path.is_absolute() else protocol_path
    )
    config["protocol"] = ref(protocol_path)
    config["source_config"] = ref(repo / SOURCE_CONFIG)
    write_json(root / "preflight.json", preflight)
    config["preflight"] = ref(root / "preflight.json")
    write_json(root / "config.json", config)
    write_json(
        root / "corpus_order.json",
        {
            "seed": SEED,
            "source_block": SOURCE_BLOCK,
            "source_corpus": config["blocks"][str(SEED)]["corpus"],
            "source_audit": config["blocks"][str(SEED)]["audit"],
            "task_ids": [row.task_id for row in records["R-S"]],
            "rule": "sort by (order_key(47, task_id), task_id), shared by both arms; wrap fixed order at 32 examples/update",
            "unchanged": "All source prompts, arm-specific traces, manifests, family roles and M0; MAPS data order is unchanged.",
        },
    )
    return preflight
