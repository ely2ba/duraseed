"""Compute REPLAY-V1 costs locally; no Tinker client, key, or paid operation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import inspect
import json
from pathlib import Path

from duraseed.pilot0_data import stage_b_sources
from duraseed.replay_budget import RATES, build_preflight
from duraseed.replay_data_archive import read_source
from duraseed.replay_data_tokens import local_runtime, measure_source
from duraseed.tasks.maps import render_prompt as maps_prompt
from duraseed.tasks.tces import render_prompt as tces_prompt


def sdk_operations(sdk):
    """Inspect installed classes and local Adam data only; never instantiate a client."""
    from tinker.lib.public_interfaces.rest_client import RestClient

    methods = {
        "ServiceClient": (
            "create_training_client_from_state_async",
            "create_training_client_from_state_with_optimizer_async",
        ),
        "TrainingClient": (
            "forward_backward_async",
            "optim_step_async",
            "save_state_async",
            "save_weights_for_sampler_async",
            "get_tokenizer",
        ),
        "SamplingClient": ("sample_async",),
        "RestClient": (
            "list_checkpoints_async",
            "get_checkpoint_archive_url_async",
            "set_checkpoint_ttl_from_tinker_path_async",
        ),
    }
    signatures = {
        f"{owner}.{name}": str(
            inspect.signature(
                getattr(
                    RestClient if owner == "RestClient" else getattr(sdk.tinker, owner),
                    name,
                )
            )
        )
        for owner, names in methods.items()
        for name in names
    }
    adam = sdk.tinker.types.AdamParams(
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-12,
        weight_decay=0.0,
        grad_clip_norm=0.0,
    )
    return {
        "method_signatures": signatures,
        "adam_fields": adam.model_dump(),
        "verification_scope": "installed SDK signatures/local parameter validation only; no service call",
    }


def measure_panels(runtime, source):
    def lengths(records):
        values = []
        for record in records:
            text = (tces_prompt if record.task_family == "tces" else maps_prompt)(
                record.to_task()
            )
            values.append(
                int(
                    runtime.renderer.build_generation_prompt(
                        [{"role": "user", "content": text}], role="assistant"
                    ).length
                )
            )
        return values

    targeted = set(source.prompt_pools.artifact.boundary_family_ids)
    prompt_lengths = {
        "a_cadence": lengths(source.a_cadence.records),
        "targeted_a_validation": lengths(
            record
            for record in source.a_validation.records
            if record.intended_family in targeted
        ),
        "a_validation": lengths(source.a_validation.records),
        "a_monitor": lengths(source.prompt_pools.a_monitor_manifest.records),
        "b_validation": lengths(source.b_validation.records),
    }
    datum_lengths = [
        measure_source(runtime, row)["train_tokens"] for row in stage_b_sources(source)
    ]
    return {
        "prompt_lengths": prompt_lengths,
        "stage_b_datum_lengths_in_manifest_order": datum_lengths,
        "stage_b_per_update_train_tokens": [
            sum(
                datum_lengths[((step - 1) * 32 + offset) % len(datum_lengths)]
                for offset in range(32)
            )
            for step in range(1, 481)
        ],
    }


def prepare(repo: Path, preparation: Path, tokenizer: Path, runs: dict[int, Path]):
    runtime = local_runtime(tokenizer)
    blocks, evidence = {}, {}
    for seed, run in sorted(runs.items()):
        audit_path = preparation / f"block-{seed}" / "audit.json"
        audit = json.loads(audit_path.read_bytes())
        if audit["status"] == "DATA_BLOCKED":
            raise ValueError(f"DATA_BLOCKED: source block {seed}")
        source, _ = read_source(run, seed)
        measured = measure_panels(runtime, source)
        blocks[seed] = {**audit, **measured}
        evidence[str(seed)] = {
            "source_run": str(run.relative_to(repo)),
            "source_audit": str(audit_path.relative_to(repo)),
            "tokenizer": audit["tokenizer"],
            "max_length": audit["max_length"],
            "per_arm_per_update_train_tokens": audit["per_arm_per_update_train_tokens"],
            **measured,
        }
    # Existing same-rank observed state=1,135,130,180; adapter file=378,407,600 bytes.
    # These are conservative operational reservations, not provider-guaranteed maxima.
    storage = dict(
        pairs=160,
        state_bytes=2_000_000_000,
        sampler_bytes=500_000_000,
        ttl_seconds=30 * 86400,
        backup_seconds=0,
        size_bound_verified=False,
        backup_policy_verified=False,
    )
    report = build_preflight(blocks, storage=storage)
    url = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
    # Public official models.json and storage-pricing docs inspected 2026-09-05.
    # This offline command deliberately does not refresh pricing or open a session.
    model = {
        "tinker_id": "Qwen/Qwen3.5-9B-Base",
        "context": "64K",
        **{key: str(RATES[key]) for key in ("prefill", "sample", "train")},
    }
    report.update(
        prepared_at_utc=datetime.now(UTC).isoformat(),
        input_measurements=evidence,
        current_official_model=model,
        pricing_source=url,
        pricing_verified_date="2026-09-05",
        sdk_version=runtime.sdk.sdk_version,
        cookbook_version=runtime.sdk.cookbook_version,
        installed_operations=sdk_operations(runtime.sdk),
        optimizer={
            "stage_a_learning_rate": "0.0001",
            "stage_b_learning_rate": "0.0003",
            "beta1": "0.9",
            "beta2": "0.95",
            "eps": "1e-12",
            "weight_decay": "0",
            "grad_clip_norm": "0",
            "schedule": "constant",
        },
        storage_sources={
            "observed_state_bytes": 1135130180,
            "observed_sampler_file_bytes": 378407600,
            "state_record": "artifacts/pilot0-pair2-lora-geometry/archive-manifest.json",
            "sampler_file": "artifacts/pilot0-pair2-lora-geometry/seed-29-B-S-step-20/adapter_model.safetensors",
            "pricing": "https://tinker-docs.thinkingmachines.ai/tinker/models/",
            "backup_policy": "https://tinker-docs.thinkingmachines.ai/tinker/data-model/#checkpoint-deletion-and-backups",
        },
        maximum_liability_status="UNRESOLVED: backup duration and provider size bound are not verified; reported ceiling is conditional and does not authorize launch",
        scientific_identity_status="Local tokenization compatibility only; provider M0 availability and exact datum/tokenizer binding require confirmation before billable training",
        all_in_maximum_liability_usd=None,
    )
    (preparation / "preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pair1-run", type=Path, required=True)
    parser.add_argument("--pair2-run", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.repo,
        args.preparation,
        args.tokenizer,
        {11: args.pair1_run, 29: args.pair2_run},
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "token_budget",
                    "main_usd",
                    "approval_ceiling_usd",
                    "blockers",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
