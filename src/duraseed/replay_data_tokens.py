"""Offline full-trace tokenization and private replay corpus preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.replay_data import batch_indices, build_corpus
from duraseed.replay_data_archive import authenticated_rollouts, read_source
from duraseed.runtime import RuntimeBundle, load_sdk, sft_datum
from duraseed.training.sft import VerifiedSourceRecord


def local_runtime(tokenizer_path: Path) -> RuntimeBundle:
    """No ServiceClient, keys, network tokenization, or replacement model loading."""
    from transformers import AutoTokenizer

    sdk = load_sdk()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    renderer = sdk.get_renderer(
        "role_colon", tokenizer, model_name="Qwen/Qwen3.5-9B-Base"
    )
    return RuntimeBundle(sdk, None, None, renderer, tokenizer)


def measure_source(
    runtime: RuntimeBundle, source: VerifiedSourceRecord
) -> dict[str, int]:
    """Assert full shifted targets and completion-only unit-sum weights, no clipping."""
    messages = [
        {"role": "user", "content": source.prompt_text},
        {"role": "assistant", "content": source.verified_completion_text},
    ]
    full, mask = runtime.renderer.build_supervised_example(
        messages, train_on_what=runtime.sdk.train_on_what.LAST_ASSISTANT_MESSAGE
    )
    ids = tuple(full.to_ints())
    datum = sft_datum(runtime, source, max_length=len(ids))
    if (
        tuple(datum.model_input.to_ints()) != ids[:-1]
        or tuple(datum.loss_fn_inputs["target_tokens"].data) != ids[1:]
    ):
        raise ValueError("full supervised shifted inputs/targets changed")
    weights = tuple(float(value) for value in datum.loss_fn_inputs["weights"].data)
    expected = tuple(float(value) for value in mask[1:])
    denominator = math.fsum(expected)
    if (
        denominator <= 0
        or len(weights) != len(expected)
        or any(
            not math.isclose(a, b / denominator, rel_tol=1e-6, abs_tol=1e-9)
            for a, b in zip(weights, expected, strict=True)
        )
    ):
        raise ValueError("completion-only per-example mean loss mask changed")
    positive = [i for i, weight in enumerate(weights) if weight > 0]
    if positive != list(range(positive[0], len(weights))):
        raise ValueError("supervised loss mask is not one completion suffix")
    prompt = runtime.renderer.build_generation_prompt(messages[:1], role="assistant")
    if tuple(prompt.to_ints()) != ids[: int(prompt.length)] or positive[0] + 1 != int(
        prompt.length
    ):
        raise ValueError("supervised and generation prompt prefixes differ")
    return {
        "full_rendered_tokens": len(ids),
        "train_tokens": len(ids) - 1,
        "prompt_tokens": int(prompt.length),
        "target_tokens": len(positive),
    }


def tokenize_corpus(runtime: RuntimeBundle, corpus: list[dict]) -> dict:
    for row in corpus:
        row["tokens"] = {
            arm: measure_source(
                runtime, VerifiedSourceRecord.model_validate_json(json.dumps(row[arm]))
            )
            for arm in ("R-S", "R-P")
        }
        if (
            row["tokens"]["R-S"]["prompt_tokens"]
            != row["tokens"]["R-P"]["prompt_tokens"]
        ):
            raise ValueError("replay arms have unequal prompt tokenization")
    maximum = max(
        (
            values["full_rendered_tokens"]
            for row in corpus
            for values in row["tokens"].values()
        ),
        default=0,
    )
    if len(corpus) < 32:
        return {
            "max_length": maximum,
            "per_arm_per_update_train_tokens": {},
            "per_arm_total_train_tokens": {},
            "presentations_per_arm": 0,
        }
    lengths = {
        arm: [
            sum(
                corpus[index]["tokens"][arm]["train_tokens"]
                for index in batch_indices(len(corpus), update)
            )
            for update in range(1, 295)
        ]
        for arm in ("R-S", "R-P")
    }
    return {
        "max_length": maximum,
        "per_arm_per_update_train_tokens": lengths,
        "per_arm_total_train_tokens": {
            arm: sum(values) for arm, values in lengths.items()
        },
        "presentations_per_arm": 9408,
        "epochs_at_cap": 9408 / len(corpus),
        "conversion": "follow-up-only max_length; complete text, shifted targets and completion-only per-example mean mask verified",
    }


def prepare_block(
    run_root: Path, seed: int, tokenizer_path: Path, output: Path
) -> dict:
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen corpus preparation: {output}"
        )
    runtime = local_runtime(tokenizer_path)
    source, preflight = read_source(run_root, seed)
    corpus, decisions, audit = build_corpus(
        source,
        authenticated_rollouts(run_root, seed, runtime.tokenizer, source, preflight),
    )
    audit.update(tokenize_corpus(runtime, corpus))
    audit["source_preflight_sha256"] = sha256_bytes(
        (run_root / "preflight.json").read_bytes()
    )
    audit["m0_state_path"] = preflight["lineage"]["m0_state_path"]
    audit["m0_sampler_path"] = preflight["lineage"]["m0_sampler_path"]
    audit["tokenizer"] = {
        "model_id": "Qwen/Qwen3.5-9B-Base",
        "renderer": "role_colon",
        "sdk_version": runtime.sdk.sdk_version,
        "cookbook_version": runtime.sdk.cookbook_version,
        "snapshot_directory_name": tokenizer_path.name,
        "files_sha256": {
            name: hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest()
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        "identity_scope": "local pinned tokenizer; all archived rollout token arrays reproduce historical normalized text; compatibility is not provider weight/tokenizer identity",
        "remote_binding_requirement": "before billable training, provider tokenizer must reproduce frozen datum IDs/counts; mismatch stops rather than retokenizing or truncating silently",
    }
    if audit["max_length"] > 65536:
        audit["status"] = "DATA_BLOCKED"
        audit["blocker"] = "full rendered source exceeds documented 65536 context"
    output.mkdir(parents=True)
    for name, values in (
        ("corpus.jsonl", corpus),
        ("exclusions.jsonl", sorted(decisions, key=lambda row: row["sample_id"])),
    ):
        raw = b"".join(canonical_json_bytes(row) + b"\n" for row in values)
        (output / name).write_bytes(raw)
        audit[name.removesuffix(".jsonl") + "_sha256"] = sha256_bytes(raw)
    (output / "audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(11, 29), required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = prepare_block(args.run_root, args.seed, args.tokenizer_path, args.output)
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "seed",
                    "status",
                    "unique_prompts",
                    "eligible_sample_count",
                    "max_length",
                    "per_arm_total_train_tokens",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
