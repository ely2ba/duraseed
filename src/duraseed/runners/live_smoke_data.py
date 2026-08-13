"""Deterministic verifier-backed inputs for the live engineering smoke."""

from __future__ import annotations

from dataclasses import dataclass

from duraseed.data.manifests import (
    DatasetManifest,
    TCESTaskManifestRecord,
    build_manifest,
    build_maps_record,
    build_tces_record,
)
from duraseed.data.splits import TCESSplitBuilder
from duraseed.provenance import derive_namespaced_seed
from duraseed.runtime import SamplingTask
from duraseed.tasks.maps import (
    MAPSGenerator,
    MAPSGeneratorConfig,
    render_prompt as render_maps_prompt,
    render_teacher_answer,
)
from duraseed.tasks.tces import (
    TCESGeneratorConfig,
    render_prompt as render_tces_prompt,
    render_teacher_completion,
)
from duraseed.training.sft import (
    VerifiedSourceRecord,
    build_solver_teacher_record,
    build_stage_b_maps_record,
)


@dataclass(frozen=True, slots=True)
class SmokeInputs:
    manifests: tuple[DatasetManifest, ...]
    tces_source: VerifiedSourceRecord
    maps_source: VerifiedSourceRecord
    tces_task: SamplingTask
    rl_tasks: tuple[SamplingTask, ...]
    maps_task: SamplingTask


def build_inputs(seed: int) -> SmokeInputs:
    """Build tiny real tasks; these select no scientific family or parameter."""

    tces_config = TCESGeneratorConfig(
        n_operands=2,
        operand_min=2,
        operand_max=12,
        target_min=-144,
        target_max=144,
        max_tree_depth=2,
        max_ast_nodes=3,
        max_answer_length=128,
        exclude_target_in_operands=False,
        max_attempts=256,
    )
    split_instances = TCESSplitBuilder(seed, tces_config).build_splits(
        {"a_seed_train": 1, "a_rl_train": 4}
    )
    manifests: list[DatasetManifest] = []
    records: dict[str, tuple[TCESTaskManifestRecord, ...]] = {}
    completions: dict[str, tuple[str, ...]] = {}
    for split, instances in split_instances.items():
        split_records = tuple(build_tces_record(instance) for instance in instances)
        records[split] = split_records
        completions[split] = tuple(
            render_teacher_completion(instance) for instance in instances
        )
        manifests.append(
            build_manifest(
                name=f"live-smoke-{split}",
                split=split,
                generator_version="1.0.0",
                root_seed=seed,
                records=split_records,
                metadata={"phase_label": "live-smoke-gate", "scientific": False},
            )
        )

    maps_config = MAPSGeneratorConfig(
        latent_program_min_length=2,
        latent_program_max_length=5,
        min_shortest_length=2,
        max_shortest_length=2,
    )
    maps_instances = {}
    maps_records = {}
    used_maps_ids: set[str] = set()
    for split in ("b_train", "b_validation"):
        generator = MAPSGenerator(
            derive_namespaced_seed(seed, "dataset.maps.split", split), maps_config
        )
        for index in range(32):
            instance = generator.generate(index)
            record = build_maps_record(instance, split=split)
            if record.task_id not in used_maps_ids:
                break
        else:  # pragma: no cover - the deterministic stream has ample space.
            raise RuntimeError("could not construct disjoint smoke MAPS tasks")
        used_maps_ids.add(record.task_id)
        maps_instances[split] = instance
        maps_records[split] = record
        manifests.append(
            build_manifest(
                name=f"live-smoke-shortest2-cap2-{split}",
                split=split,
                generator_version="1.0.0",
                root_seed=seed,
                records=[record],
                metadata={"phase_label": "live-smoke-gate", "scientific": False},
            )
        )
    by_split = {manifest.split: manifest for manifest in manifests}
    rl_records = records["a_rl_train"]

    def sampling_task(record: TCESTaskManifestRecord) -> SamplingTask:
        return SamplingTask(
            by_split["a_rl_train"].manifest_id,
            record.task_id,
            "tces",
            "a_rl_train",
            render_tces_prompt(record.to_task()),
            record.to_task(),
            record.item_index,
            assigned_family_id=record.intended_family,
            panel_role="engineering-smoke",
        )

    rl_tasks = tuple(sampling_task(record) for record in rl_records)
    return SmokeInputs(
        manifests=tuple(manifests),
        tces_source=build_solver_teacher_record(
            source_manifest=by_split["a_seed_train"],
            source_record=records["a_seed_train"][0],
            completion=completions["a_seed_train"][0],
        ),
        maps_source=build_stage_b_maps_record(
            source_manifest=by_split["b_train"],
            source_record=maps_records["b_train"],
            completion=render_teacher_answer(maps_instances["b_train"]),
        ),
        tces_task=rl_tasks[0],
        rl_tasks=rl_tasks,
        maps_task=SamplingTask(
            by_split["b_validation"].manifest_id,
            maps_records["b_validation"].task_id,
            "maps",
            "b_validation",
            render_maps_prompt(maps_records["b_validation"].to_task()),
            maps_records["b_validation"].to_task(),
            maps_records["b_validation"].item_index,
            panel_role="engineering-smoke",
        ),
    )


__all__ = ["SmokeInputs", "build_inputs"]
