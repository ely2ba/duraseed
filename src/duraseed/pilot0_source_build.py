"""Deterministic local construction of the two frozen Pilot source bundles."""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor
import os
from pathlib import Path
import shutil
import tempfile

from duraseed.config import PilotConfig
from duraseed.data.manifests import (
    DatasetManifest,
    TCESTaskManifestRecord,
    build_manifest,
    build_maps_record,
    read_manifest,
    write_manifest,
)
from duraseed.data.family_generation import (
    IndexedSplitGenerationJob,
    generate_split_candidate,
)
from duraseed.data.maps_splits import MAPSSplitBuilder
from duraseed.data.panel_split_manifest import build_panel_split_manifest
from duraseed.data.panels import FamilyPanelArtifact
from duraseed.data.sealing import ExecutionContext
from duraseed.data.splits import tces_numeric_key
from duraseed.data.stage_a_prompt_pools import (
    StageAPromptPoolBundle,
    read_stage_a_prompt_pool_bundle,
    write_stage_a_prompt_pool_bundle,
)
from duraseed.data import stage_a_prompt_pools as source_pools
from duraseed.pilot0_contract import PILOT_SEEDS, PilotSeedSources, STAGE_B_PROFILE
from duraseed.pilot0_prompt_pools import build_pilot_prompt_pools
from duraseed.tasks.maps import MAPSGeneratorConfig
from duraseed.tasks.tces import GeneratedTCESInstance, TCESGeneratorConfig


def _regenerate_templates(
    broad: DatasetManifest,
    config: TCESGeneratorConfig,
    executor: Executor,
) -> dict[str, GeneratedTCESInstance]:
    """Reproduce the accepted template stream with indexed work in parallel."""

    metadata = broad.metadata.get("templates")
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Pilot broad manifest omitted template provenance")
    by_family = {str(row["family_id"]): row for row in metadata}
    largest_index = max(int(row["template_item_index"]) for row in metadata)
    required = largest_index + 1
    accepted: list[GeneratedTCESInstance] = []
    seen_numeric: set[object] = set()
    seen_content: set[str] = set()
    next_index = 0
    scan_stop = max(1_024, required * 128)
    while len(accepted) < required:
        count = min(max(required - len(accepted), 64), scan_stop - next_index)
        if count < 1:
            raise RuntimeError("Pilot template stream exhausted its scan budget")
        candidates = executor.map(
            generate_split_candidate,
            (
                IndexedSplitGenerationJob(
                    broad.root_seed, config, "a_candidate", item_index
                )
                for item_index in range(next_index, next_index + count)
            ),
        )
        next_index += count
        for candidate in candidates:
            numeric = tces_numeric_key(candidate)
            if numeric in seen_numeric or candidate.content_hash in seen_content:
                continue
            seen_numeric.add(numeric)
            seen_content.add(candidate.content_hash)
            accepted.append(candidate)
    templates = {}
    for family_id, row in by_family.items():
        template = accepted[int(row["template_item_index"])]
        if template.intended_family != family_id or template.content_hash != row.get(
            "template_content_hash"
        ):
            raise ValueError("Pilot regenerated template differs from provenance")
        templates[family_id] = template
    return templates


def _balanced_validation(
    config: PilotConfig,
    boundary: Path,
    panel: FamilyPanelArtifact,
    pools: tuple[StageAPromptPoolBundle, ...],
    executor: Executor | None = None,
    templates: dict[str, GeneratedTCESInstance] | None = None,
) -> DatasetManifest:
    broad = read_manifest(
        boundary / "a_candidate_manifest.json", context=ExecutionContext.SELECTION
    )
    confirmation = read_manifest(
        boundary / "a_candidate_confirmation_manifest.json",
        context=ExecutionContext.SELECTION,
    )
    full = build_panel_split_manifest(
        config,
        artifact=panel,
        broad_manifest=broad,
        confirmation_manifest=confirmation,
        split="a_validation",
        items_per_family=22,
        executor=executor,
        templates=templates,
        forbidden_records=tuple(
            row
            for manifest in (
                broad,
                confirmation,
                *(value.a_rl_train_manifest for value in pools),
                *(value.a_monitor_manifest for value in pools),
            )
            for row in manifest.records
            if isinstance(row, TCESTaskManifestRecord)
        ),
    )
    retained: list[TCESTaskManifestRecord] = []
    for panel_families in (panel.panel_a_family_ids, panel.panel_b_family_ids):
        for family_index, family in enumerate(sorted(panel_families)):
            count = 22 if family_index < 4 else 21
            rows = sorted(
                (
                    row
                    for row in full.records
                    if isinstance(row, TCESTaskManifestRecord)
                    and row.intended_family == family
                ),
                key=lambda row: (row.item_index, row.task_id),
            )[:count]
            if len(rows) != count:
                raise RuntimeError("frozen panel lacks its Pilot validation rows")
            retained.extend(rows)
    return build_manifest(
        name="pilot0-a-validation",
        split="a_validation",
        generator_version="1.0.0",
        root_seed=config.seed,
        records=retained,
        metadata={
            "scope": "pilot0",
            "panel_balance": "256_per_panel",
            "family_allocation": "canonical_first_4_x22_remaining_8_x21",
            "source_full_manifest_id": full.manifest_id,
        },
    )


def _maps_manifests(config: PilotConfig) -> tuple[DatasetManifest, DatasetManifest]:
    values = config.tasks.maps.generator_kwargs()
    values["max_shortest_length"] = 2
    instances = MAPSSplitBuilder(
        config.seed, MAPSGeneratorConfig(**values)
    ).build_splits({"b_train": 4096, "b_validation": 512})
    manifests = []
    for split in ("b_train", "b_validation"):
        manifests.append(
            build_manifest(
                name=f"pilot0-{STAGE_B_PROFILE}-{split}",
                split=split,
                generator_version="1.0.0",
                root_seed=config.seed,
                records=[
                    build_maps_record(row, split=split) for row in instances[split]
                ],
                metadata={"scope": "pilot0", "profile": STAGE_B_PROFILE},
            )
        )
    return manifests[0], manifests[1]


def _cadence_manifest(
    config: PilotConfig, pool: StageAPromptPoolBundle
) -> DatasetManifest:
    records = []
    for family in sorted(
        pool.artifact.boundary_family_ids + pool.artifact.sentinel_family_ids
    ):
        rows = sorted(
            (
                row
                for row in pool.a_monitor_manifest.records
                if isinstance(row, TCESTaskManifestRecord)
                and row.intended_family == family
            ),
            key=lambda row: (row.item_index, row.task_id),
        )[:8]
        if len(rows) != 8:
            raise RuntimeError("Pilot cadence family is incomplete")
        records.extend(rows)
    return build_manifest(
        name=f"pilot0-seed-{pool.artifact.calibration_seed}-a-cadence",
        split="a_monitor",
        generator_version="1.0.0",
        root_seed=config.seed,
        records=records,
        parent_manifest_id=pool.a_monitor_manifest.manifest_id,
        metadata={"scope": "pilot0_cadence", "items_per_family": 8},
    )


def build_pilot_seed_sources(
    config: PilotConfig, boundary_directory: str | Path
) -> tuple[PilotSeedSources, PilotSeedSources]:
    """Build both pair bundles together so their validation population is common."""

    boundary = Path(boundary_directory)
    panel = FamilyPanelArtifact.model_validate_json(
        (boundary / "target_sentinel_panels.json").read_bytes()
    )
    _, broad, _, _, _ = source_pools._validated_source(  # noqa: SLF001
        boundary, config, 17
    )
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        generator_config = TCESGeneratorConfig(**config.tasks.tces.generator_kwargs())
        templates = _regenerate_templates(broad, generator_config, executor)
        pools = tuple(
            build_pilot_prompt_pools(
                boundary,
                config=config,
                pilot_seed=seed,
                executor=executor,
                templates=templates,
            )
            for seed in PILOT_SEEDS
        )
        validation = _balanced_validation(
            config,
            boundary,
            panel,
            pools,
            executor=executor,
            templates=templates,
        )
    b_train, b_validation = _maps_manifests(config)
    return tuple(
        PilotSeedSources(
            seed,
            pool,
            _cadence_manifest(config, pool),
            validation,
            b_train,
            b_validation,
        )
        for seed, pool in zip(PILOT_SEEDS, pools, strict=True)
    )  # type: ignore[return-value]


def write_pilot_seed_sources(directory: Path, source: PilotSeedSources) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_stage_a_prompt_pool_bundle(directory, source.prompt_pools)
    write_manifest(directory / "a_cadence_manifest.json", source.a_cadence)
    write_manifest(directory / "a_validation_manifest.json", source.a_validation)
    write_manifest(directory / "b_train_manifest.json", source.b_train)
    write_manifest(directory / "b_validation_manifest.json", source.b_validation)


def _read_pilot_seed_source(directory: Path, seed: int) -> PilotSeedSources:
    return PilotSeedSources(
        seed=seed,
        prompt_pools=read_stage_a_prompt_pool_bundle(directory),
        a_cadence=read_manifest(
            directory / "a_cadence_manifest.json",
            context=ExecutionContext.SELECTION,
        ),
        a_validation=read_manifest(
            directory / "a_validation_manifest.json",
            context=ExecutionContext.SELECTION,
        ),
        b_train=read_manifest(
            directory / "b_train_manifest.json",
            context=ExecutionContext.TRAINING,
        ),
        b_validation=read_manifest(
            directory / "b_validation_manifest.json",
            context=ExecutionContext.SELECTION,
        ),
    )


def _validate_source_set(
    sources: tuple[PilotSeedSources, PilotSeedSources],
) -> tuple[PilotSeedSources, PilotSeedSources]:
    if tuple(source.seed for source in sources) != PILOT_SEEDS:
        raise ValueError("Pilot source set must contain frozen seeds 11 and 29")
    for source in sources:
        pools = source.prompt_pools
        if (
            pools.artifact.calibration_seed != source.seed
            or source.a_cadence.parent_manifest_id
            != pools.a_monitor_manifest.manifest_id
            or source.a_cadence.split != "a_monitor"
            or source.a_cadence.record_count != 192
            or source.a_validation.split != "a_validation"
            or source.a_validation.record_count != 512
            or source.b_train.split != "b_train"
            or source.b_train.record_count != 4096
            or source.b_validation.split != "b_validation"
            or source.b_validation.record_count != 512
        ):
            raise ValueError("Pilot source set differs from its frozen populations")
    first, second = sources
    if (
        first.a_validation != second.a_validation
        or first.b_train != second.b_train
        or first.b_validation != second.b_validation
    ):
        raise ValueError("Pilot pairs must share validation and Stage-B populations")
    return sources


def read_pilot_seed_sources(
    directory: str | Path,
) -> tuple[PilotSeedSources, PilotSeedSources]:
    """Load a completed reusable source set; never generate inside a live launch."""

    root = Path(directory)
    return _validate_source_set(
        tuple(
            _read_pilot_seed_source(root / f"seed-{seed}", seed) for seed in PILOT_SEEDS
        )  # type: ignore[arg-type]
    )


def write_pilot_seed_source_set(
    directory: str | Path,
    sources: tuple[PilotSeedSources, PilotSeedSources],
) -> Path:
    """Atomically persist both pair sources after the complete local build."""

    values = _validate_source_set(sources)
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"Pilot source destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for source in values:
            write_pilot_seed_sources(staging / f"seed-{source.seed}", source)
        read_pilot_seed_sources(staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


__all__ = [
    "build_pilot_seed_sources",
    "read_pilot_seed_sources",
    "write_pilot_seed_source_set",
    "write_pilot_seed_sources",
]
