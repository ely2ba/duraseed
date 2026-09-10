"""Replay arm identity around unchanged Pilot sampling and profile reducers."""

from __future__ import annotations

from pathlib import Path

from duraseed.pilot0_evidence import read_evaluation
from duraseed.pilot0_profiles import pre_b_capability_profile
from duraseed.replay_inputs import read_json
from duraseed.replay_remote import write_json
from duraseed.runners.pilot0_sampling import _prompt, evaluate_manifest
from duraseed.runtime import TokenBudget


def raw_counts(result: dict, role: str | None = None) -> dict:
    rows = [
        row
        for row in result["item_counts"]
        if role is None or row["panel_role"] == role
    ]
    if not rows:
        raise ValueError("required evaluation role has no observations")
    return {
        "successes": sum(row["successes"] for row in rows),
        "trials": sum(row["trials"] for row in rows),
    }


async def evaluate(
    remote,
    source,
    checkpoint: dict,
    manifest,
    *,
    stage: str,
    update: int,
    arm: str,
    purpose: str,
    draws: int,
    cap: int,
    output: Path,
    seed_namespace: str | None = None,
):
    seed_namespace = seed_namespace or f"replay-v1.{stage}.{purpose}.{update}"
    coordinate = {
        "replay_arm": arm,
        "seed": source.seed,
        "stage": stage,
        "update": update,
        "purpose": purpose,
        "sampler_path": checkpoint["sampler_path"],
        "origin_sampler_path": checkpoint.get(
            "origin_sampler_path", checkpoint["sampler_path"]
        ),
        "manifest_id": manifest.manifest_id,
        "draws": draws,
        "cap": cap,
        "seed_namespace": seed_namespace,
    }
    identity_path = output / "replay-identity.json"
    if identity_path.exists() and read_json(identity_path) != coordinate:
        raise ValueError("cached replay evaluation has a different arm/checkpoint")
    completed = read_evaluation(output)
    if completed is not None:
        if not identity_path.exists():
            raise ValueError("cached replay evaluation lacks arm identity")
    write_json(identity_path, coordinate)
    sampler = None
    if completed is None:
        sampler = await remote.sampler(checkpoint["sampler_path"], coordinate)
        prefill = (
            sum(
                int(
                    remote.runtime.renderer.build_generation_prompt(
                        [{"role": "user", "content": _prompt(record)}], role="assistant"
                    ).length
                )
                for record in manifest.records
            )
            * draws
        )
        remote.begin(
            "evaluation",
            coordinate,
            TokenBudget(prefill, manifest.record_count * draws * cap, 0),
        )
    result = await evaluate_manifest(
        remote.eval_inputs,
        source,
        manifest=manifest,
        sampler=sampler,
        sampler_path=checkpoint["sampler_path"],
        origin_sampler_path=coordinate["origin_sampler_path"],
        # Legacy schema's method enum is intentionally not expanded or mislabeled.
        # The replay identity above and arm-bearing label are authoritative.
        method=None,
        checkpoint_stage=stage,
        training_step=update,
        label=f"replay-v1-seed{source.seed}-{arm}-{stage}-{update}-{purpose}",
        samples_per_item=draws,
        max_tokens=cap,
        seed_namespace=seed_namespace,
        output=output,
    )
    if completed is None:
        remote.complete(
            {"phase": "evaluation", **coordinate, "rows": result["row_count"]}
        )
    return result


def write_profile(source, arm: str, checkpoint: dict, validation: Path, output: Path):
    value = pre_b_capability_profile(
        origin_kind="replay-v1-selected",
        seed=source.seed,
        method=None,
        manifest=source.a_validation,
        evaluation_directories=(validation,),
        cover_thresholds=(0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75),
        expected_sampler_path=checkpoint["sampler_path"],
    )
    value["replay_arm"] = arm
    value["stage_a_update"] = checkpoint["update"]
    value["note"] = "Descriptive only; required draws only; never used for rematching."
    maps = read_evaluation(validation.parent / "b_validation")
    if maps is None:
        raise ValueError("pre-B profile omitted required MAPS baseline")
    value["stage_b_baseline"] = raw_counts(maps)
    write_json(output, value)
