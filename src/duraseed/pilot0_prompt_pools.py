"""Pilot-seed prompt pools built from the authenticated frozen boundary source."""

from __future__ import annotations

from pathlib import Path
import random

from duraseed.config import PilotConfig
from duraseed.data.manifests import GENERATOR_VERSION, build_manifest
from duraseed.data.panel_capacity import (
    PANEL_SPLIT_MINIMUMS,
    audit_family_split_capacity,
)
from duraseed.data.splits import tces_numeric_key
from duraseed.data import stage_a_prompt_pools as pools
from duraseed.pilot0_contract import PILOT_SEEDS
from duraseed.provenance import (
    SeedNamespace,
    canonical_json_hash,
    derive_namespaced_seed,
)
from duraseed.tasks.tces import TCESGeneratorConfig


def build_pilot_prompt_pools(
    boundary_confirmation_directory: str | Path,
    *,
    config: PilotConfig,
    pilot_seed: int,
) -> pools.StageAPromptPoolBundle:
    """Build the frozen Stage-A pools for one crossed Pilot seed.

    The historical builder remains immutable and seed-17-specific.  This wrapper
    authenticates that same source through its original validator, then applies
    its unchanged generation rules to the predeclared Pilot orientation/seed.
    """

    if pilot_seed not in PILOT_SEEDS:
        raise pools.StageAPromptPoolError("Pilot prompt seed must be 11 or 29")
    boundary = Path(boundary_confirmation_directory).resolve()
    panel, broad, confirmation, candidate_rows, capacity_eligible = (
        pools._validated_source(boundary, config, 17)  # noqa: SLF001
    )
    targeted_panel, sentinel_panel, boundary_ids, sentinel_ids = (
        pools._panel_orientation(panel, pilot_seed)  # noqa: SLF001
    )
    ranking = pools._candidate_ranking(  # noqa: SLF001
        candidate_rows, panel.allocation_seed
    )
    selected_panel_ids = set(boundary_ids).union(sentinel_ids)
    if set(ranking[:24]) != selected_panel_ids or not set(ranking).issubset(
        capacity_eligible
    ):
        raise pools.StageAPromptPoolError("Pilot panels differ from frozen ranking")
    intermediate_ids = tuple(
        sorted(ranking[24 : 24 + pools.STAGE_A_FAMILIES_PER_STRATUM])
    )
    if len(intermediate_ids) != pools.STAGE_A_FAMILIES_PER_STRATUM:
        raise pools.StageAPromptPoolError("Pilot intermediate pool is incomplete")

    generator_config = TCESGeneratorConfig(**config.tasks.tces.generator_kwargs())
    templates = pools._regenerate_templates(broad, generator_config)  # noqa: SLF001
    required_templates = selected_panel_ids.union(intermediate_ids)
    if not required_templates.issubset(templates):
        raise pools.StageAPromptPoolError("Pilot family lacks template provenance")
    source_records = (*broad.records, *confirmation.records)
    used_numeric = {tces_numeric_key(record) for record in source_records}
    used_content = {record.content_hash for record in source_records}

    protected_panels = frozenset(selected_panel_ids)
    sentinel_families = frozenset(sentinel_ids)
    monitor_records = []
    for family_id in sorted(selected_panel_ids):
        generated = pools._generate_family_records(  # noqa: SLF001
            template=templates[family_id],
            generator_config=generator_config,
            root_seed=config.seed,
            split="a_monitor",
            count=pools.STAGE_A_MONITOR_ITEMS_PER_PANEL_FAMILY,
            used_numeric=used_numeric,
            used_content=used_content,
            forbidden_valid_families=protected_panels.difference({family_id}),
        )
        if generated is None:
            raise pools.StageAPromptPoolError("Pilot monitor capacity is incomplete")
        monitor_records.extend(generated)

    rl_records = []
    boundary_families = frozenset(boundary_ids)
    for family_id in sorted(set(boundary_ids).union(intermediate_ids)):
        generated = pools._generate_family_records(  # noqa: SLF001
            template=templates[family_id],
            generator_config=generator_config,
            root_seed=config.seed,
            split="a_rl_train",
            count=pools.STAGE_A_RL_ITEMS_PER_FAMILY,
            used_numeric=used_numeric,
            used_content=used_content,
            forbidden_valid_families=(
                sentinel_families
                if family_id in boundary_families
                else protected_panels
            ),
        )
        if generated is None:
            raise pools.StageAPromptPoolError("Pilot RL capacity is incomplete")
        rl_records.extend(generated)

    panel_id = canonical_json_hash(panel)
    random_seed = derive_namespaced_seed(
        config.seed,
        SeedNamespace.RANDOM_TEACHER_ALLOCATION,
        "stage_a.broad_random_prompt_pool",
        pilot_seed,
        panel_id,
    )
    broad_candidates = sorted(
        set(templates).difference(selected_panel_ids).difference(intermediate_ids)
    )
    random.Random(random_seed).shuffle(broad_candidates)
    broad_random_ids = []
    for family_id in broad_candidates:
        audit = audit_family_split_capacity(
            templates[family_id],
            generator_config,
            root_seed=config.seed,
            requirements=PANEL_SPLIT_MINIMUMS,
            forbidden_records=source_records,
        )
        if not audit.passed:
            continue
        generated = pools._generate_family_records(  # noqa: SLF001
            template=templates[family_id],
            generator_config=generator_config,
            root_seed=config.seed,
            split="a_rl_train",
            count=pools.STAGE_A_RL_ITEMS_PER_FAMILY,
            used_numeric=used_numeric,
            used_content=used_content,
            forbidden_valid_families=protected_panels,
        )
        if generated is None:
            continue
        broad_random_ids.append(family_id)
        rl_records.extend(generated)
        if len(broad_random_ids) == pools.STAGE_A_FAMILIES_PER_STRATUM:
            break
    if len(broad_random_ids) != pools.STAGE_A_FAMILIES_PER_STRATUM:
        raise pools.StageAPromptPoolError("Pilot broad-random pool is incomplete")
    broad_random_ids = tuple(sorted(broad_random_ids))

    metadata = {
        "scope": "stage_a_frozen_prompt_pool",
        "calibration_seed": pilot_seed,
        "family_panel_artifact_id": panel_id,
        "source_confirmation_manifest_id": confirmation.manifest_id,
    }
    rl_manifest = build_manifest(
        name=f"pilot0-seed-{pilot_seed}-a-rl-train",
        split="a_rl_train",
        generator_version=GENERATOR_VERSION,
        root_seed=config.seed,
        records=rl_records,
        parent_manifest_id=broad.manifest_id,
        metadata={
            **metadata,
            "items_per_family": pools.STAGE_A_RL_ITEMS_PER_FAMILY,
            "stratum_family_ids": {
                pools.PromptPoolStratum.BOUNDARY.value: list(boundary_ids),
                pools.PromptPoolStratum.INTERMEDIATE.value: list(intermediate_ids),
                pools.PromptPoolStratum.BROAD_RANDOM.value: list(broad_random_ids),
            },
        },
    )
    monitor_manifest = build_manifest(
        name=f"pilot0-seed-{pilot_seed}-a-monitor",
        split="a_monitor",
        generator_version=GENERATOR_VERSION,
        root_seed=config.seed,
        records=monitor_records,
        parent_manifest_id=broad.manifest_id,
        metadata={
            **metadata,
            "items_per_family": pools.STAGE_A_MONITOR_ITEMS_PER_PANEL_FAMILY,
            "targeted_family_ids": list(boundary_ids),
            "sentinel_family_ids": list(sentinel_ids),
        },
    )
    schedule_seed = derive_namespaced_seed(
        config.seed,
        SeedNamespace.DATA_ORDER,
        "stage_a.prompt_pool_schedule",
        pilot_seed,
        panel_id,
    )
    artifact = pools._build_artifact(  # noqa: SLF001
        calibration_seed=pilot_seed,
        family_panel_artifact_id=panel_id,
        candidate_table_id=panel.candidate_family_table_manifest_id,
        panel_matching_report_id=panel.panel_matching_report_id,
        source_broad_manifest_id=broad.manifest_id,
        source_confirmation_manifest_id=confirmation.manifest_id,
        rl_train_manifest_id=rl_manifest.manifest_id,
        monitor_manifest_id=monitor_manifest.manifest_id,
        targeted_panel=targeted_panel,
        sentinel_panel=sentinel_panel,
        boundary_family_ids=boundary_ids,
        sentinel_family_ids=sentinel_ids,
        intermediate_family_ids=intermediate_ids,
        broad_random_family_ids=broad_random_ids,
        allocations=(
            pools.PromptPoolAllocation(
                stratum=pools.PromptPoolStratum.BOUNDARY,
                weight_numerator=2,
                bs_slots=16,
                bg_groups=8,
            ),
            pools.PromptPoolAllocation(
                stratum=pools.PromptPoolStratum.INTERMEDIATE,
                weight_numerator=1,
                bs_slots=8,
                bg_groups=4,
            ),
            pools.PromptPoolAllocation(
                stratum=pools.PromptPoolStratum.BROAD_RANDOM,
                weight_numerator=1,
                bs_slots=8,
                bg_groups=4,
            ),
        ),
        random_allocation_seed=random_seed,
        schedule_seed=schedule_seed,
        bs_slot_order=pools._ordered_schedule(  # noqa: SLF001
            schedule_seed=schedule_seed,
            namespace="data_order.stage_a.bs_slots",
            boundary=16,
            intermediate=8,
            broad=8,
        ),
        bg_group_order=pools._ordered_schedule(  # noqa: SLF001
            schedule_seed=schedule_seed,
            namespace="data_order.stage_a.bg_groups",
            boundary=8,
            intermediate=4,
            broad=4,
        ),
        record_order="family_round_robin_then_task_id",
    )
    return pools.StageAPromptPoolBundle(
        artifact=artifact,
        a_rl_train_manifest=rl_manifest,
        a_monitor_manifest=monitor_manifest,
    )


__all__ = ["build_pilot_prompt_pools"]
