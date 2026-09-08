"""Export completed replay evidence locally; no sampling or analysis recomputation."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean, median

ARMS = ("R-S", "R-P")
PRIVATE_KEYS = set(
    "project_id session_id session_ids tokenizer_path current_balance_evidence "
    "current_balance_usd other_committed_usd owner_launch_direction "
    "owner_direction user_metadata".split()
)
FORBIDDEN = re.compile(
    r"/Users/|/home/|tinker://|ephemeral:|"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|"
    r'"(?:project_id|session_ids?|api_key|password|secret)"',
    re.I,
)


def read(path):
    return json.loads(path.read_bytes())


def digest(value):
    return "sha256:" + sha256(value).hexdigest()


def subset(value, names):
    return {key: value[key] for key in names.split()}


def clean(value):
    if isinstance(value, dict):
        return {
            key: clean(item) for key, item in value.items() if key not in PRIVATE_KEYS
        }
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str) and value.startswith(("tinker://", "ephemeral:", "/")):
        return "opaque-" + digest(value.encode())
    return value


def privacy_check(text):
    if FORBIDDEN.search(text):
        raise ValueError("public export contains a private identifier or machine path")


def dump(output, name, value, written):
    text = (
        value
        if isinstance(value, str)
        else json.dumps(clean(value), indent=2, sort_keys=True) + "\n"
    )
    privacy_check(text)
    target = output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    written.append(name)


def lengths(values):
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": mean(values),
        "median": median(values),
        "maximum": max(values),
    }


def training(root, seed, arm, stage, cap, selected=None):
    files = sorted((root / f"seed-{seed}" / arm / stage).glob("u*/update-*.json"))
    rows = [read(path) for path in files]
    if sorted(row["update"] for row in rows) != list(range(1, cap + 1)):
        raise ValueError(f"missing or duplicate {stage} updates for {seed}/{arm}")
    if any(
        row["seed"] != seed or row["replay_arm"] != arm or row["stage"] != stage
        for row in rows
    ):
        raise ValueError("training identity changed")
    return {
        "completed_updates": cap,
        "full_presentations": cap * 32,
        "full_train_tokens": sum(row["train_tokens"] for row in rows),
        "selected_update": selected,
        "selected_presentations": selected * 32 if selected is not None else None,
        "selected_train_tokens": sum(
            row["train_tokens"] for row in rows if row["update"] <= selected
        )
        if selected is not None
        else None,
    }


def corpus_package(preparation, source, matching):
    lineages, stats = [], {}
    for seed in (11, 29):
        directory = preparation / f"block-{seed}"
        corpus = [
            json.loads(line)
            for line in (directory / "corpus.jsonl").read_text().splitlines()
        ]
        audit = read(directory / "audit.json")
        arms = {}
        for arm in ARMS:
            selected = matching[str(seed)]["selected"]
            counts = training(
                source, seed, arm, "stage_a", 294, selected[arm] if selected else None
            )
            if counts["full_train_tokens"] != audit["per_arm_total_train_tokens"][arm]:
                raise ValueError(
                    "actual Stage-A training tokens differ from frozen corpus schedule"
                )
            arms[arm] = {
                **counts,
                "unique_trace_lengths": {
                    key: lengths([row["tokens"][arm][key] for row in corpus])
                    for key in ("target_tokens", "full_rendered_tokens", "train_tokens")
                },
            }
        stats[str(seed)] = {
            "unique_prompts": len(corpus),
            "arms": arms,
            "stratum_counts": dict(Counter(row["stratum"] for row in corpus)),
            "assigned_family_count": len({row["assigned_family_id"] for row in corpus}),
            "original_policy_sampled_tokens": lengths(
                [row["source"]["sampled_tokens"] for row in corpus]
            ),
            "audit": {
                key: audit[key]
                for key in (
                    "sample_count",
                    "eligible_sample_count",
                    "sampled_unique_prompt_count",
                    "family_attrition",
                    "stratum_attrition",
                    "leakage",
                    "source_manifest_id",
                    "corpus_sha256",
                    "exclusions_sha256",
                )
            },
        }
        for order, row in enumerate(corpus):
            lineages.append(
                {
                    "seed": seed,
                    "order_index": order,
                    **{
                        key: row[key]
                        for key in (
                            "task_id",
                            "assigned_family_id",
                            "stratum",
                            "order_sha256",
                            "selection_sha256",
                            "tokens",
                        )
                    },
                    "source": clean(row["source"]),
                    "text_sha256": {
                        arm: {
                            "prompt": digest(row[arm]["prompt_text"].encode()),
                            "completion": digest(
                                row[arm]["verified_completion_text"].encode()
                            ),
                        }
                        for arm in ARMS
                    },
                }
            )
    return lineages, stats


def selection_package(source, continuation):
    candidates, cadence = [], []
    for seed in (11, 29):
        for arm in ARMS:
            for purpose, target in (("candidate", candidates), ("cadence", cadence)):
                for path in sorted(
                    (source / f"seed-{seed}" / arm / "stage_a").glob(
                        f"u*/{purpose}/result.json"
                    )
                ):
                    result = read(path)
                    target.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "update": int(path.parents[1].name[1:]),
                            **{
                                key: result[key]
                                for key in (
                                    "manifest_id",
                                    "item_count",
                                    "row_count",
                                    "samples_per_item",
                                    "item_counts",
                                )
                            },
                            "original_result_sha256": digest(path.read_bytes()),
                        }
                    )
    if len(candidates) != 12 or len(cadence) != 120:
        raise ValueError(
            "selection export must retain 12 candidates and 120 cadence evaluations"
        )
    return {
        "original_matching": read(source / "matching.json"),
        "continuation_matching": read(continuation / "matching.json"),
        "nominations": read(source / "nominations.json"),
        "cadence": cadence,
        "candidates": candidates,
        "continuation": clean(read(continuation / "continuation.json")),
    }


def configuration(repo, preparation, continuation):
    path = preparation / "config.json"
    config, preflight = read(path), read(preparation / "preflight.json")
    public = subset(
        config,
        "namespace arms seeds batch_size stage_a_lr stage_b_lr "
        "stage_a_grid stage_b_grid evaluation max_length checkpoint_ttl_seconds "
        "storage_per_pair_usd targeted_matching primary key_f2 "
        "implementation_commit implementation_files",
    )
    protocol = repo / config["protocol"]["path"]
    if digest(protocol.read_bytes()) != config["protocol"]["sha256"]:
        raise ValueError("referenced frozen protocol changed")
    public.update(
        {
            "projection_only_not_a_launch_config": True,
            "original_config_sha256": digest(path.read_bytes()),
            "protocol": config["protocol"],
            "original_preflight_sha256": digest(
                (preparation / "preflight.json").read_bytes()
            ),
            "continuation_override": clean(read(continuation / "continuation.json")),
            "blocks": {
                seed: {
                    "manifest_ids": b["manifest_ids"],
                    "m0_state_alias": clean(b["m0_state_path"]),
                    "m0_sampler_alias": clean(b["m0_sampler_path"]),
                    "original_corpus_sha256": b["corpus"]["sha256"],
                    "original_audit_sha256": b["audit"]["sha256"],
                }
                for seed, b in config["blocks"].items()
            },
            "runtime": {
                key: preflight[key]
                for key in (
                    "optimizer",
                    "sdk_version",
                    "cookbook_version",
                    "current_official_model",
                )
            },
        }
    )
    return public


def inventory(source, continuation, statistics):
    source_result, continuation_result = (
        read(source / "result.json"),
        read(continuation / "result.json"),
    )
    if (
        source_result["status"] != "NO_MATCH"
        or continuation_result["status"] != "COMPLETED"
    ):
        raise ValueError(
            "export requires the completed original and continuation outcomes"
        )
    ledgers = {
        "original_acquisition_and_selection": read(source / "billing.json"),
        "engineering_only": read(source / "engineering-only/billing.json"),
        "continuation": read(continuation / "billing.json"),
    }
    rates = read(continuation / "preflight.json")["prices_per_million_usd"]
    token_charges = sum(
        Decimal(str(rates[k])) * b["observed"][k] / 1_000_000
        for b in ledgers.values()
        for k in ("prefill", "sample", "train")
    )
    storage = sum(
        (Decimal(str(b["observed_fixed_usd"])) for b in ledgers.values()), Decimal(0)
    )
    result = {
        "source_run_id": source.name,
        "continuation_run_id": continuation.name,
        "original_block_status": {
            s: b["status"] for s, b in source_result["matching"].items()
        },
        "continuation_block_status": {},
        "stage_a": statistics,
        "stage_b": {},
        "billing": {
            "ledgers": clean(ledgers),
            "observed_tokens_plus_storage_allowances_usd": str(token_charges + storage),
            "storage_allowances_included_usd": str(storage),
            "observed_token_charges_usd": str(token_charges),
            "prices_per_million_usd": rates,
            "settled_invoice_actuals": "not available; these are local token-ledger figures, not provider invoice settlement",
            "unused_recovery_reservation_usd": read(continuation / "preflight.json")[
                "recovery_reservation_usd"
            ],
            "unused_reserve_authorizes_additional_experiments": False,
        },
    }
    for seed in (11, 29):
        match = continuation_result["matching"][str(seed)]
        result["continuation_block_status"][str(seed)] = (
            "COMPLETED" if match["selected"] else "NO_MATCH"
        )
        result["stage_b"][str(seed)] = {}
        for arm in ARMS:
            if match["selected"]:
                result["stage_b"][str(seed)][arm] = training(
                    continuation, seed, arm, "stage_b", 480
                )
            elif (continuation / f"seed-{seed}" / arm / "stage_b").exists():
                raise ValueError("NO_MATCH block unexpectedly has Stage-B files")
    return result


def export(source, continuation, preparation, output, repo):
    output.mkdir(parents=True, exist_ok=True)
    written, readout = [], read(continuation / "analysis/readout.json")
    if set(readout["blocks"]) != {"11", "29"}:
        raise ValueError("both source blocks must remain visible")
    document = (continuation / "analysis/readout.md").read_text()
    for seed, block in readout["blocks"].items():
        for arm, values in block["arms"].items():
            original = values["F3_profile"]
            public = f"profiles/seed-{seed}-{arm}.json"
            dump(output, public, read(continuation / original), written)
            values["F3_profile"] = public
            document = document.replace(
                f"[{original}](../{original})", f"[{public}]({public})"
            )
    dump(output, "readout.json", readout, written)
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    dump(output, "readout.md", document, written)
    selected, stats = corpus_package(
        preparation, source, read(continuation / "matching.json")
    )
    lineage = "".join(json.dumps(clean(row), sort_keys=True) + "\n" for row in selected)
    dump(output, "corpus-lineage.jsonl", lineage, written)
    dump(output, "selection.json", selection_package(source, continuation), written)
    dump(output, "inventory.json", inventory(source, continuation, stats), written)
    dump(
        output,
        "config.public.json",
        configuration(repo, preparation, continuation),
        written,
    )
    overview = (repo / "artifacts/replay-v1/followup/README.md").read_text()
    dump(output, "README.md", overview, written)
    return written


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("source-root", "continuation-root", "preparation-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    roots = [
        args.source_root.resolve(),
        args.continuation_root.resolve(),
        args.preparation_root.resolve(),
    ]
    output = args.output.resolve()
    if any(output == root or output.is_relative_to(root) for root in roots):
        raise ValueError("public export must not write inside original evidence")
    files = export(*roots, output, args.repo.resolve())
    print(
        json.dumps(
            {"files": files, "privacy_check": "passed", "bootstrap_recomputed": False},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
