"""Frozen, manifest-backed Stage-A prompt pools.

The builder consumes a completed boundary-confirmation directory and constructs
the one prompt population shared by Stage-A methods.  It is deliberately local:
no training client, remote API, or run orchestration is available here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from enum import StrEnum
import json
from pathlib import Path
import random
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from duraseed.config import PilotConfig
from duraseed.data.io import atomic_write_bytes
from duraseed.data.leakage import audit_leakage
from duraseed.data.manifests import (
    DatasetManifest,
    GENERATOR_VERSION,
    TCESTaskManifestRecord,
    build_manifest,
    build_tces_record,
    read_manifest,
    write_manifest,
)
from duraseed.data.panel_capacity import (
    PANEL_CAPACITY_PROBE_MULTIPLIER,
    PANEL_FILTERED_SPLIT_SCAN_MULTIPLIER,
    PANEL_SELECTED_TEST_SINGLE_MINIMUM,
    PANEL_SPLIT_MINIMUMS,
    audit_family_split_capacity,
)
from duraseed.data.panels import FamilyPanelArtifact, PanelLabel
from duraseed.data.sealing import ExecutionContext
from duraseed.data.splits import (
    TCESSplitBuilder,
    derive_tces_split_seed,
    tces_numeric_key,
)
from duraseed.provenance import (
    MAX_ROOT_SEED,
    SeedNamespace,
    canonical_json_bytes,
    canonical_json_hash,
    derive_namespaced_seed,
    validate_sha256_id,
)
from duraseed.run_records import RunStatus, read_run_record
from duraseed.schemas import StrictModel
from duraseed.tasks.tces import (
    GeneratedTCESInstance,
    TCESFamilyGenerator,
    TCESGenerationError,
    TCESGeneratorConfig,
)


STAGE_A_PROMPT_POOL_SCHEMA_VERSION = "duraseed-stage-a-prompt-pools-v1"
STAGE_A_RL_ITEMS_PER_FAMILY = 64
STAGE_A_MONITOR_ITEMS_PER_PANEL_FAMILY = 16
STAGE_A_FAMILIES_PER_STRATUM = 12
STAGE_A_BS_SLOTS = 32
STAGE_A_BG_GROUPS = 16
STAGE_A_PROMPT_POOL_FILENAMES = {
    "artifact": "stage_a_prompt_pools.json",
    "rl_train": "a_rl_train_manifest.json",
    "monitor": "a_monitor_manifest.json",
}
_THREE_COHORT_SOURCE_KIND = "three_cohort_boundary_freeze_v1"


class StageAPromptPoolError(ValueError):
    """The frozen boundary evidence cannot produce the declared prompt pool."""


class PromptPoolStratum(StrEnum):
    """The three predeclared components of the Stage-A prompt distribution."""

    BOUNDARY = "boundary"
    INTERMEDIATE = "intermediate"
    BROAD_RANDOM = "broad_random"


class _PromptPoolModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PromptPoolAllocation(_PromptPoolModel):
    """Exact allocation for one stratum in both Stage-A objectives."""

    stratum: PromptPoolStratum
    family_count: Literal[12] = STAGE_A_FAMILIES_PER_STRATUM
    weight_numerator: Literal[1, 2]
    weight_denominator: Literal[4] = 4
    bs_slots: Literal[8, 16]
    bg_groups: Literal[4, 8]

    @model_validator(mode="after")
    def allocation_matches_stratum(self) -> Self:
        expected = {
            PromptPoolStratum.BOUNDARY: (2, 16, 8),
            PromptPoolStratum.INTERMEDIATE: (1, 8, 4),
            PromptPoolStratum.BROAD_RANDOM: (1, 8, 4),
        }[self.stratum]
        if (self.weight_numerator, self.bs_slots, self.bg_groups) != expected:
            raise ValueError("prompt-pool allocation differs from 0.50/0.25/0.25")
        return self


def _artifact_identity_payload(artifact: "StageAPromptPoolArtifact") -> dict[str, Any]:
    return {
        field_name: getattr(artifact, field_name)
        for field_name in type(artifact).model_fields
        if field_name != "artifact_id"
    }


class StageAPromptPoolArtifact(_PromptPoolModel):
    """Content-addressed family membership, manifests, and fixed schedules."""

    schema_version: Literal["duraseed-stage-a-prompt-pools-v1"] = (
        STAGE_A_PROMPT_POOL_SCHEMA_VERSION
    )
    artifact_id: str
    calibration_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    family_panel_artifact_id: str
    candidate_table_id: str
    panel_matching_report_id: str
    source_broad_manifest_id: str
    source_confirmation_manifest_id: str
    rl_train_manifest_id: str
    monitor_manifest_id: str
    targeted_panel: PanelLabel
    sentinel_panel: PanelLabel
    boundary_family_ids: tuple[str, ...]
    sentinel_family_ids: tuple[str, ...]
    intermediate_family_ids: tuple[str, ...]
    broad_random_family_ids: tuple[str, ...]
    allocations: tuple[PromptPoolAllocation, ...]
    random_allocation_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    schedule_seed: int = Field(ge=0, le=MAX_ROOT_SEED)
    bs_slot_order: tuple[PromptPoolStratum, ...]
    bg_group_order: tuple[PromptPoolStratum, ...]
    record_order: Literal["family_round_robin_then_task_id"] = (
        "family_round_robin_then_task_id"
    )

    @field_validator(
        "artifact_id",
        "family_panel_artifact_id",
        "candidate_table_id",
        "panel_matching_report_id",
        "source_broad_manifest_id",
        "source_confirmation_manifest_id",
        "rl_train_manifest_id",
        "monitor_manifest_id",
    )
    @classmethod
    def ids_are_canonical(cls, value: str) -> str:
        return validate_sha256_id(value)

    @field_validator(
        "boundary_family_ids",
        "sentinel_family_ids",
        "intermediate_family_ids",
        "broad_random_family_ids",
    )
    @classmethod
    def families_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            len(value) != STAGE_A_FAMILIES_PER_STRATUM
            or value != tuple(sorted(set(value)))
            or any(not family_id for family_id in value)
        ):
            raise ValueError("each prompt-pool family set must be 12 unique sorted IDs")
        return value

    @model_validator(mode="after")
    def pool_contract_is_exact(self) -> Self:
        family_sets = (
            set(self.boundary_family_ids),
            set(self.sentinel_family_ids),
            set(self.intermediate_family_ids),
            set(self.broad_random_family_ids),
        )
        if any(
            left.intersection(right)
            for index, left in enumerate(family_sets)
            for right in family_sets[index + 1 :]
        ):
            raise ValueError(
                "Stage-A family pools and the sentinel panel must be disjoint"
            )
        expected_strata = tuple(PromptPoolStratum)
        if tuple(row.stratum for row in self.allocations) != expected_strata:
            raise ValueError(
                "allocations must use canonical boundary/intermediate/random order"
            )
        if len(self.bs_slot_order) != STAGE_A_BS_SLOTS or Counter(
            self.bs_slot_order
        ) != Counter(
            {
                PromptPoolStratum.BOUNDARY: 16,
                PromptPoolStratum.INTERMEDIATE: 8,
                PromptPoolStratum.BROAD_RANDOM: 8,
            }
        ):
            raise ValueError("B-S slot order must contain exactly 16/8/8 strata")
        if len(self.bg_group_order) != STAGE_A_BG_GROUPS or Counter(
            self.bg_group_order
        ) != Counter(
            {
                PromptPoolStratum.BOUNDARY: 8,
                PromptPoolStratum.INTERMEDIATE: 4,
                PromptPoolStratum.BROAD_RANDOM: 4,
            }
        ):
            raise ValueError("B-G group order must contain exactly 8/4/4 strata")
        if self.artifact_id != canonical_json_hash(_artifact_identity_payload(self)):
            raise ValueError("artifact_id does not match prompt-pool content")
        return self


class StageAPromptPoolBundle(_PromptPoolModel):
    """The artifact and the two authenticated manifests it names."""

    artifact: StageAPromptPoolArtifact
    a_rl_train_manifest: DatasetManifest
    a_monitor_manifest: DatasetManifest

    @model_validator(mode="after")
    def manifests_match_artifact(self) -> Self:
        artifact = self.artifact
        rl = self.a_rl_train_manifest
        monitor = self.a_monitor_manifest
        if (
            rl.task_family != "tces"
            or rl.split != "a_rl_train"
            or rl.manifest_id != artifact.rl_train_manifest_id
            or rl.record_count
            != 3 * STAGE_A_FAMILIES_PER_STRATUM * STAGE_A_RL_ITEMS_PER_FAMILY
        ):
            raise ValueError(
                "a_rl_train manifest differs from the prompt-pool artifact"
            )
        if (
            monitor.task_family != "tces"
            or monitor.split != "a_monitor"
            or monitor.manifest_id != artifact.monitor_manifest_id
            or monitor.record_count
            != 2 * STAGE_A_FAMILIES_PER_STRATUM * STAGE_A_MONITOR_ITEMS_PER_PANEL_FAMILY
        ):
            raise ValueError("a_monitor manifest differs from the prompt-pool artifact")
        rl_counts = Counter(record.intended_family for record in rl.records)
        expected_rl_families = (
            artifact.boundary_family_ids
            + artifact.intermediate_family_ids
            + artifact.broad_random_family_ids
        )
        if set(rl_counts) != set(expected_rl_families) or set(rl_counts.values()) != {
            STAGE_A_RL_ITEMS_PER_FAMILY
        }:
            raise ValueError(
                "a_rl_train does not contain exactly 64 items per pool family"
            )
        monitor_counts = Counter(record.intended_family for record in monitor.records)
        panel_families = artifact.boundary_family_ids + artifact.sentinel_family_ids
        if set(monitor_counts) != set(panel_families) or set(
            monitor_counts.values()
        ) != {STAGE_A_MONITOR_ITEMS_PER_PANEL_FAMILY}:
            raise ValueError(
                "a_monitor does not contain exactly 16 items per panel family"
            )
        boundary = set(artifact.boundary_family_ids)
        sentinel = set(artifact.sentinel_family_ids)
        protected_panels = set(panel_families)
        for record in rl.records:
            forbidden = (
                sentinel if record.intended_family in boundary else protected_panels
            )
            if set(record.valid_family_ids).intersection(forbidden):
                raise ValueError(
                    "Stage-A training items expose a forbidden panel family"
                )
        if any(
            set(record.valid_family_ids).intersection(protected_panels)
            != {record.intended_family}
            for record in monitor.records
        ):
            raise ValueError(
                "Stage-A monitor items must isolate their intended panel family"
            )
        audit_leakage(
            {
                "a_rl_train": rl,
                "a_monitor": monitor,
            }
        ).assert_clean()
        return self


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageAPromptPoolError(f"invalid source artifact: {path.name}") from error
    if not isinstance(value, Mapping):
        raise StageAPromptPoolError(f"source artifact is not an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        rows = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise StageAPromptPoolError(f"invalid source evidence: {path.name}") from error
    if any(not isinstance(row, Mapping) for row in rows):
        raise StageAPromptPoolError(f"source evidence is not row objects: {path.name}")
    return rows


def _capacity_eligible_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    requirements: Mapping[str, int],
) -> frozenset[str]:
    expected = dict(requirements)
    family_ids = tuple(str(row.get("family_id", "")) for row in rows)
    if any(not family_id for family_id in family_ids) or len(family_ids) != len(
        set(family_ids)
    ):
        raise StageAPromptPoolError("capacity audits require unique family IDs")
    eligible: set[str] = set()
    for family_id, row in zip(family_ids, rows, strict=True):
        capacities = row.get("split_capacities")
        if not isinstance(capacities, list) or any(
            not isinstance(capacity, Mapping) for capacity in capacities
        ):
            raise StageAPromptPoolError("capacity audit rows are malformed")
        by_split = {str(capacity.get("split", "")): capacity for capacity in capacities}
        if not set(expected).issubset(by_split):
            raise StageAPromptPoolError("capacity audit omits a protected split")
        passed = all(
            by_split[split].get("required_instances") == required
            and by_split[split].get("passed") is True
            and type(by_split[split].get("available_disjoint_instances")) is int
            and int(by_split[split]["available_disjoint_instances"]) >= required
            for split, required in expected.items()
        )
        if row.get("passed") is True:
            if not passed:
                raise StageAPromptPoolError(
                    "capacity audit aggregate pass contradicts its split rows"
                )
            eligible.add(family_id)
    return frozenset(eligible)


def _source_directories(confirmation_directory: Path) -> tuple[Path, Path]:
    confirmation_plan = _read_json(confirmation_directory / "preflight.json")
    source = confirmation_plan.get("source")
    if not isinstance(source, Mapping):
        raise StageAPromptPoolError("confirmation preflight omitted its source")
    refinement_directory = Path(
        str(source.get("refinement_run_directory", ""))
    ).resolve()
    refinement_plan = _read_json(refinement_directory / "preflight.json")
    refinement_source = refinement_plan.get("source")
    if not isinstance(refinement_source, Mapping):
        raise StageAPromptPoolError("refinement preflight omitted its source")
    broad_directory = Path(
        str(refinement_source.get("broad_run_directory", ""))
    ).resolve()
    return refinement_directory, broad_directory


def _direct_composite_manifests(
    confirmation_directory: Path,
    run: Any,
) -> tuple[DatasetManifest, DatasetManifest] | None:
    plan = _read_json(confirmation_directory / "preflight.json")
    if plan.get("source_kind") != _THREE_COHORT_SOURCE_KIND:
        return None
    source = plan.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("source_kind") != _THREE_COHORT_SOURCE_KIND
    ):
        raise StageAPromptPoolError("composite boundary source is malformed")
    directories = source.get("confirmation_run_directories")
    if (
        not isinstance(directories, list)
        or len(directories) != 3
        or len({str(value) for value in directories}) != 3
    ):
        raise StageAPromptPoolError("composite boundary source requires three cohorts")
    for value in directories:
        source_run = read_run_record(Path(str(value)).resolve())
        if (
            source_run.status != RunStatus.COMPLETED
            or source_run.run_kind != "m0_calibration"
        ):
            raise StageAPromptPoolError("a composite boundary cohort is incomplete")
    broad = read_manifest(
        confirmation_directory / "a_candidate_manifest.json",
        context=ExecutionContext.SELECTION,
    )
    confirmation = read_manifest(
        confirmation_directory / "a_candidate_confirmation_manifest.json",
        context=ExecutionContext.SELECTION,
    )
    if (
        run.task_manifest_ids.get("boundary_composite_a_candidate") != broad.manifest_id
        or run.task_manifest_ids.get("boundary_confirmation_a_candidate")
        != confirmation.manifest_id
        or source.get("combined_broad_manifest_id") != broad.manifest_id
        or source.get("combined_confirmation_manifest_id") != confirmation.manifest_id
        or confirmation.metadata.get("source_broad_manifest_id") != broad.manifest_id
        or broad.root_seed != run.seed
        or confirmation.root_seed != run.seed
        or broad.metadata.get("cohort_ids") != ["initial", "extension_1", "extension_2"]
        or confirmation.metadata.get("cohort_ids")
        != ["initial", "extension_1", "extension_2"]
    ):
        raise StageAPromptPoolError("composite manifests differ from their source")
    return broad, confirmation


def _validated_source(
    confirmation_directory: Path,
    config: PilotConfig,
    calibration_seed: int,
) -> tuple[
    FamilyPanelArtifact,
    DatasetManifest,
    DatasetManifest,
    tuple[Mapping[str, Any], ...],
    frozenset[str],
]:
    if (
        calibration_seed != 17
        or calibration_seed not in config.statistics.calibration_seeds
    ):
        raise StageAPromptPoolError(
            "the first Stage-A calibration block is frozen at seed 17"
        )
    run = read_run_record(confirmation_directory)
    if run.status != RunStatus.COMPLETED or run.run_kind != "m0_calibration":
        raise StageAPromptPoolError("boundary confirmation is not a completed M0 run")
    summary = _read_json(confirmation_directory / "confirmation_summary.json")
    if summary.get("status") != "confirmed_panels_frozen":
        raise StageAPromptPoolError("boundary confirmation did not freeze 12/12 panels")
    panel = FamilyPanelArtifact.model_validate_json(
        (confirmation_directory / "target_sentinel_panels.json").read_text(
            encoding="utf-8"
        )
    )
    if len(panel.panel_a_family_ids) != 12 or len(panel.panel_b_family_ids) != 12:
        raise StageAPromptPoolError("Stage-A requires exact 12/12 frozen panels")
    candidate_payload = _read_json(
        confirmation_directory / "panel_candidate_table.json"
    )
    matching_payload = _read_json(confirmation_directory / "panel_matching_report.json")
    if (
        canonical_json_hash(candidate_payload)
        != panel.candidate_family_table_manifest_id
        or canonical_json_hash(matching_payload) != panel.panel_matching_report_id
        or summary.get("candidate_table_id") != panel.candidate_family_table_manifest_id
        or summary.get("panel_matching_report_id") != panel.panel_matching_report_id
    ):
        raise StageAPromptPoolError(
            "panel source IDs do not authenticate their content"
        )
    candidates = candidate_payload.get("candidates")
    if not isinstance(candidates, list) or any(
        not isinstance(row, Mapping) for row in candidates
    ):
        raise StageAPromptPoolError("panel candidate table is malformed")
    capacity_rows = _read_jsonl(confirmation_directory / "split_capacity_audits.jsonl")
    eligible = _capacity_eligible_ids(
        capacity_rows,
        requirements=dict(PANEL_SPLIT_MINIMUMS),
    )
    summary_eligible = summary.get("split_capacity_eligible_family_ids")
    if (
        summary.get("split_capacity_requirements") != dict(PANEL_SPLIT_MINIMUMS)
        or not isinstance(summary_eligible, list)
        or eligible != frozenset(str(value) for value in summary_eligible)
    ):
        raise StageAPromptPoolError("confirmation summary differs from capacity audits")
    selected_capacity_rows = _read_jsonl(
        confirmation_directory / "selected_panel_capacity_audits.jsonl"
    )
    selected_capacity = _capacity_eligible_ids(
        selected_capacity_rows,
        requirements={
            **dict(PANEL_SPLIT_MINIMUMS),
            "a_test_single": PANEL_SELECTED_TEST_SINGLE_MINIMUM,
        },
    )
    if selected_capacity != frozenset(
        (*panel.panel_a_family_ids, *panel.panel_b_family_ids)
    ):
        raise StageAPromptPoolError("selected panels differ from their capacity audits")
    direct = _direct_composite_manifests(confirmation_directory, run)
    if direct is not None:
        broad, confirmation = direct
        plan = _read_json(confirmation_directory / "preflight.json")
        source = plan.get("source")
        assert isinstance(source, Mapping)
        if (
            source.get("sampler_checkpoint_path") != panel.m0_checkpoint_path
            or source.get("state_checkpoint_path") != run.parent_tinker_checkpoint_path
            or isinstance(source.get("training_step"), bool)
            or not isinstance(source.get("training_step"), int)
            or int(source["training_step"]) < 0
        ):
            raise StageAPromptPoolError("composite M0 lineage differs from its panels")
        return panel, broad, confirmation, tuple(candidates), eligible
    refinement_directory, broad_directory = _source_directories(confirmation_directory)
    refinement_run = read_run_record(refinement_directory)
    broad_run = read_run_record(broad_directory)
    if (
        refinement_run.status != RunStatus.COMPLETED
        or broad_run.status != RunStatus.COMPLETED
    ):
        raise StageAPromptPoolError("boundary source lineage is incomplete")
    broad = read_manifest(
        broad_directory / "a_candidate_manifest.json",
        context=ExecutionContext.SELECTION,
    )
    refinement = read_manifest(
        refinement_directory / "a_candidate_manifest.json",
        context=ExecutionContext.SELECTION,
    )
    confirmation = read_manifest(
        confirmation_directory / "a_candidate_confirmation_manifest.json",
        context=ExecutionContext.SELECTION,
    )
    if broad != refinement:
        raise StageAPromptPoolError(
            "refinement changed the authenticated broad manifest"
        )
    if (
        confirmation.metadata.get("source_broad_manifest_id") != broad.manifest_id
        or run.task_manifest_ids.get("boundary_confirmation_a_candidate")
        != confirmation.manifest_id
    ):
        raise StageAPromptPoolError(
            "confirmation manifest differs from its run lineage"
        )
    return panel, broad, confirmation, tuple(candidates), eligible


def _candidate_ranking(
    candidates: Sequence[Mapping[str, Any]], allocation_seed: int
) -> tuple[str, ...]:
    supplied_ids = tuple(str(row.get("family_id", "")) for row in candidates)
    if any(not family_id for family_id in supplied_ids) or len(supplied_ids) != len(
        set(supplied_ids)
    ):
        raise StageAPromptPoolError(
            "panel candidate family IDs must be nonempty and unique"
        )
    tie_seed = derive_namespaced_seed(
        allocation_seed,
        "family_selection.panel_candidate_ties",
    )
    family_ids = tuple(sorted(supplied_ids))
    shuffled = list(family_ids)
    random.Random(tie_seed).shuffle(shuffled)
    tie_rank = {family_id: rank for rank, family_id in enumerate(shuffled)}
    by_family = {str(candidate["family_id"]): candidate for candidate in candidates}
    try:
        return tuple(
            sorted(
                family_ids,
                key=lambda family_id: (
                    -float(by_family[family_id]["informative_group_probability_i8"]),
                    -int(by_family[family_id]["available_disjoint_instances"]),
                    tie_rank[family_id],
                    family_id,
                ),
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise StageAPromptPoolError(
            "panel candidate ranking fields are malformed"
        ) from error


def _panel_orientation(
    panel: FamilyPanelArtifact, calibration_seed: int
) -> tuple[PanelLabel, PanelLabel, tuple[str, ...], tuple[str, ...]]:
    assignment = next(
        (
            row
            for row in panel.seed_block_assignments
            if row.training_seed == calibration_seed
        ),
        None,
    )
    if assignment is None:
        raise StageAPromptPoolError("panel artifact omits calibration seed 17")
    by_label = {
        PanelLabel.A: panel.panel_a_family_ids,
        PanelLabel.B: panel.panel_b_family_ids,
    }
    return (
        assignment.targeted_panel,
        assignment.sentinel_panel,
        by_label[assignment.targeted_panel],
        by_label[assignment.sentinel_panel],
    )


def _regenerate_templates(
    broad_manifest: DatasetManifest,
    generator_config: TCESGeneratorConfig,
) -> dict[str, GeneratedTCESInstance]:
    metadata = broad_manifest.metadata.get("templates")
    if not isinstance(metadata, list) or not metadata:
        raise StageAPromptPoolError("broad manifest omitted template provenance")
    by_family: dict[str, Mapping[str, Any]] = {}
    for row in metadata:
        if not isinstance(row, Mapping) or not isinstance(row.get("family_id"), str):
            raise StageAPromptPoolError("broad template provenance is malformed")
        by_family[str(row["family_id"])] = row
    largest_index = max(int(row["template_item_index"]) for row in by_family.values())
    candidates = TCESSplitBuilder(
        broad_manifest.root_seed, generator_config
    ).lazy_split("a_candidate", size=largest_index + 1)
    templates: dict[str, GeneratedTCESInstance] = {}
    for family_id, row in by_family.items():
        template = candidates[int(row["template_item_index"])]
        if template.intended_family != family_id or template.content_hash != row.get(
            "template_content_hash"
        ):
            raise StageAPromptPoolError("regenerated template differs from provenance")
        templates[family_id] = template
    return templates


def _generate_family_records(
    *,
    template: GeneratedTCESInstance,
    generator_config: TCESGeneratorConfig,
    root_seed: int,
    split: Literal["a_rl_train", "a_monitor"],
    count: int,
    used_numeric: set[object],
    used_content: set[str],
    forbidden_valid_families: frozenset[str] = frozenset(),
) -> tuple[TCESTaskManifestRecord, ...] | None:
    split_seed = derive_tces_split_seed(root_seed, split)
    config = replace(
        generator_config,
        split=split,
        min_valid_families=1,
        max_valid_families=None,
    )
    generator = TCESFamilyGenerator(split_seed, template, config)
    local_numeric: set[object] = set()
    local_content: set[str] = set()
    selected: list[TCESTaskManifestRecord] = []
    scan_multiplier = (
        PANEL_FILTERED_SPLIT_SCAN_MULTIPLIER
        if forbidden_valid_families
        else PANEL_CAPACITY_PROBE_MULTIPLIER
    )
    for item_index in range(count * scan_multiplier):
        try:
            instance = generator.generate(item_index)
        except TCESGenerationError:
            continue
        if forbidden_valid_families.intersection(instance.valid_family_ids):
            continue
        numeric = tces_numeric_key(instance)
        if (
            numeric in used_numeric
            or numeric in local_numeric
            or instance.content_hash in used_content
            or instance.content_hash in local_content
        ):
            continue
        selected.append(build_tces_record(instance))
        local_numeric.add(numeric)
        local_content.add(instance.content_hash)
        if len(selected) == count:
            used_numeric.update(local_numeric)
            used_content.update(local_content)
            return tuple(selected)
    return None


def _flatten_additional_forbidden(
    values: Iterable[DatasetManifest | TCESTaskManifestRecord],
) -> tuple[TCESTaskManifestRecord, ...]:
    """Normalize optional TCES records/manifests into one forbidden record set."""

    records: list[TCESTaskManifestRecord] = []
    for value in values:
        if isinstance(value, DatasetManifest):
            if value.task_family != "tces":
                raise TypeError(
                    "additional forbidden inputs must be TCES records or TCES manifests"
                )
            candidates = value.records
        else:
            candidates = (value,)
        if any(not isinstance(record, TCESTaskManifestRecord) for record in candidates):
            raise TypeError(
                "additional forbidden inputs must be TCES records or TCES manifests"
            )
        records.extend(candidates)
    return tuple(records)


def _ordered_schedule(
    *, schedule_seed: int, namespace: str, boundary: int, intermediate: int, broad: int
) -> tuple[PromptPoolStratum, ...]:
    values = (
        [PromptPoolStratum.BOUNDARY] * boundary
        + [PromptPoolStratum.INTERMEDIATE] * intermediate
        + [PromptPoolStratum.BROAD_RANDOM] * broad
    )
    seed = derive_namespaced_seed(schedule_seed, namespace)
    random.Random(seed).shuffle(values)
    return tuple(values)


def _build_artifact(
    **values: Any,
) -> StageAPromptPoolArtifact:
    payload = {
        "schema_version": STAGE_A_PROMPT_POOL_SCHEMA_VERSION,
        "artifact_id": "sha256:" + "0" * 64,
        **values,
    }
    provisional = StageAPromptPoolArtifact.model_construct(**payload)
    payload["artifact_id"] = canonical_json_hash(
        _artifact_identity_payload(provisional)
    )
    return StageAPromptPoolArtifact(**payload)


def build_stage_a_prompt_pools(
    boundary_confirmation_directory: str | Path,
    *,
    config: PilotConfig,
    calibration_seed: int = 17,
    additional_forbidden: Iterable[DatasetManifest | TCESTaskManifestRecord] = (),
) -> StageAPromptPoolBundle:
    """Build seed-17 pools locally from completed, authenticated boundary evidence.

    ``additional_forbidden`` may contain TCES records or whole TCES manifests.
    Their numeric identities and content hashes are excluded from both generated
    manifests, allowing already-frozen datasets to remain globally disjoint.
    """

    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    confirmation_directory = Path(boundary_confirmation_directory).resolve()
    panel, broad, confirmation, candidate_rows, capacity_eligible = _validated_source(
        confirmation_directory,
        config,
        calibration_seed,
    )
    targeted_panel, sentinel_panel, boundary_ids, sentinel_ids = _panel_orientation(
        panel, calibration_seed
    )
    ranking = _candidate_ranking(candidate_rows, panel.allocation_seed)
    selected_panel_ids = set(boundary_ids).union(sentinel_ids)
    if set(ranking[:24]) != selected_panel_ids:
        raise StageAPromptPoolError(
            "frozen panels differ from the confirmation ranking"
        )
    if not set(ranking).issubset(capacity_eligible):
        raise StageAPromptPoolError(
            "a ranked confirmation finalist failed its capacity audit"
        )
    intermediate_ids = ranking[24 : 24 + STAGE_A_FAMILIES_PER_STRATUM]
    if len(intermediate_ids) != STAGE_A_FAMILIES_PER_STRATUM:
        raise StageAPromptPoolError("fewer than 12 ranked intermediate families remain")
    intermediate_ids = tuple(sorted(intermediate_ids))

    generator_config = TCESGeneratorConfig(**config.tasks.tces.generator_kwargs())
    templates = _regenerate_templates(broad, generator_config)
    required_templates = selected_panel_ids.union(intermediate_ids)
    if not required_templates.issubset(templates):
        raise StageAPromptPoolError(
            "a frozen panel/intermediate family lacks provenance"
        )
    extra_forbidden = _flatten_additional_forbidden(additional_forbidden)
    source_records = (*broad.records, *confirmation.records, *extra_forbidden)
    used_numeric = {tces_numeric_key(record) for record in source_records}
    used_content = {record.content_hash for record in source_records}

    protected_panels = frozenset(selected_panel_ids)
    sentinel_family_set = frozenset(sentinel_ids)
    monitor_records: list[TCESTaskManifestRecord] = []
    for family_id in tuple(sorted(selected_panel_ids)):
        generated = _generate_family_records(
            template=templates[family_id],
            generator_config=generator_config,
            root_seed=config.seed,
            split="a_monitor",
            count=STAGE_A_MONITOR_ITEMS_PER_PANEL_FAMILY,
            used_numeric=used_numeric,
            used_content=used_content,
            forbidden_valid_families=protected_panels.difference({family_id}),
        )
        if generated is None:
            raise StageAPromptPoolError(
                f"panel family lacks global monitor capacity: {family_id}"
            )
        monitor_records.extend(generated)

    rl_records: list[TCESTaskManifestRecord] = []
    boundary_family_set = frozenset(boundary_ids)
    for family_id in tuple(sorted(set(boundary_ids).union(intermediate_ids))):
        generated = _generate_family_records(
            template=templates[family_id],
            generator_config=generator_config,
            root_seed=config.seed,
            split="a_rl_train",
            count=STAGE_A_RL_ITEMS_PER_FAMILY,
            used_numeric=used_numeric,
            used_content=used_content,
            forbidden_valid_families=(
                sentinel_family_set
                if family_id in boundary_family_set
                else protected_panels
            ),
        )
        if generated is None:
            raise StageAPromptPoolError(
                f"family lacks global RL-pool capacity: {family_id}"
            )
        rl_records.extend(generated)

    random_allocation_seed = derive_namespaced_seed(
        config.seed,
        SeedNamespace.RANDOM_TEACHER_ALLOCATION,
        "stage_a.broad_random_prompt_pool",
        calibration_seed,
        canonical_json_hash(panel),
    )
    broad_candidates = sorted(
        set(templates).difference(selected_panel_ids).difference(intermediate_ids)
    )
    random.Random(random_allocation_seed).shuffle(broad_candidates)
    broad_random_ids: list[str] = []
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
        generated = _generate_family_records(
            template=templates[family_id],
            generator_config=generator_config,
            root_seed=config.seed,
            split="a_rl_train",
            count=STAGE_A_RL_ITEMS_PER_FAMILY,
            used_numeric=used_numeric,
            used_content=used_content,
            forbidden_valid_families=protected_panels,
        )
        if generated is None:
            continue
        broad_random_ids.append(family_id)
        rl_records.extend(generated)
        if len(broad_random_ids) == STAGE_A_FAMILIES_PER_STRATUM:
            break
    if len(broad_random_ids) != STAGE_A_FAMILIES_PER_STRATUM:
        raise StageAPromptPoolError(
            "fewer than 12 broad-random families satisfy the frozen gate"
        )
    broad_random_family_ids = tuple(sorted(broad_random_ids))

    panel_artifact_id = canonical_json_hash(panel)
    common_metadata = {
        "scope": "stage_a_frozen_prompt_pool",
        "calibration_seed": calibration_seed,
        "family_panel_artifact_id": panel_artifact_id,
        "source_confirmation_manifest_id": confirmation.manifest_id,
    }
    rl_manifest = build_manifest(
        name="stage-a-frozen-a-rl-train",
        split="a_rl_train",
        generator_version=GENERATOR_VERSION,
        root_seed=config.seed,
        records=rl_records,
        parent_manifest_id=broad.manifest_id,
        metadata={
            **common_metadata,
            "items_per_family": STAGE_A_RL_ITEMS_PER_FAMILY,
            "stratum_family_ids": {
                PromptPoolStratum.BOUNDARY.value: list(boundary_ids),
                PromptPoolStratum.INTERMEDIATE.value: list(intermediate_ids),
                PromptPoolStratum.BROAD_RANDOM.value: list(broad_random_family_ids),
            },
        },
    )
    monitor_manifest = build_manifest(
        name="stage-a-panel-a-monitor",
        split="a_monitor",
        generator_version=GENERATOR_VERSION,
        root_seed=config.seed,
        records=monitor_records,
        parent_manifest_id=broad.manifest_id,
        metadata={
            **common_metadata,
            "items_per_family": STAGE_A_MONITOR_ITEMS_PER_PANEL_FAMILY,
            "targeted_family_ids": list(boundary_ids),
            "sentinel_family_ids": list(sentinel_ids),
        },
    )
    schedule_seed = derive_namespaced_seed(
        config.seed,
        SeedNamespace.DATA_ORDER,
        "stage_a.prompt_pool_schedule",
        calibration_seed,
        panel_artifact_id,
    )
    allocations = (
        PromptPoolAllocation(
            stratum=PromptPoolStratum.BOUNDARY,
            weight_numerator=2,
            bs_slots=16,
            bg_groups=8,
        ),
        PromptPoolAllocation(
            stratum=PromptPoolStratum.INTERMEDIATE,
            weight_numerator=1,
            bs_slots=8,
            bg_groups=4,
        ),
        PromptPoolAllocation(
            stratum=PromptPoolStratum.BROAD_RANDOM,
            weight_numerator=1,
            bs_slots=8,
            bg_groups=4,
        ),
    )
    artifact = _build_artifact(
        calibration_seed=calibration_seed,
        family_panel_artifact_id=panel_artifact_id,
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
        broad_random_family_ids=broad_random_family_ids,
        allocations=allocations,
        random_allocation_seed=random_allocation_seed,
        schedule_seed=schedule_seed,
        bs_slot_order=_ordered_schedule(
            schedule_seed=schedule_seed,
            namespace="data_order.stage_a.bs_slots",
            boundary=16,
            intermediate=8,
            broad=8,
        ),
        bg_group_order=_ordered_schedule(
            schedule_seed=schedule_seed,
            namespace="data_order.stage_a.bg_groups",
            boundary=8,
            intermediate=4,
            broad=4,
        ),
        record_order="family_round_robin_then_task_id",
    )
    return StageAPromptPoolBundle(
        artifact=artifact,
        a_rl_train_manifest=rl_manifest,
        a_monitor_manifest=monitor_manifest,
    )


def write_stage_a_prompt_pool_bundle(
    output_directory: str | Path, bundle: StageAPromptPoolBundle
) -> Path:
    """Write the canonical local artifact and its two manifests."""

    if not isinstance(bundle, StageAPromptPoolBundle):
        raise TypeError("bundle must be a StageAPromptPoolBundle")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_manifest(
        directory / STAGE_A_PROMPT_POOL_FILENAMES["rl_train"],
        bundle.a_rl_train_manifest,
    )
    write_manifest(
        directory / STAGE_A_PROMPT_POOL_FILENAMES["monitor"],
        bundle.a_monitor_manifest,
    )
    atomic_write_bytes(
        directory / STAGE_A_PROMPT_POOL_FILENAMES["artifact"],
        canonical_json_bytes(bundle.artifact) + b"\n",
    )
    return directory


def read_stage_a_prompt_pool_bundle(
    directory: str | Path,
) -> StageAPromptPoolBundle:
    """Read and re-authenticate a previously written local prompt-pool bundle."""

    source = Path(directory)
    artifact_path = source / STAGE_A_PROMPT_POOL_FILENAMES["artifact"]
    try:
        artifact_bytes = artifact_path.read_bytes()
        artifact = StageAPromptPoolArtifact.model_validate_json(artifact_bytes)
    except (OSError, ValueError) as error:
        raise StageAPromptPoolError("invalid Stage-A prompt-pool artifact") from error
    if artifact_bytes != canonical_json_bytes(artifact) + b"\n":
        raise StageAPromptPoolError("Stage-A prompt-pool artifact is not canonical")
    return StageAPromptPoolBundle(
        artifact=artifact,
        a_rl_train_manifest=read_manifest(
            source / STAGE_A_PROMPT_POOL_FILENAMES["rl_train"],
            context=ExecutionContext.TRAINING,
        ),
        a_monitor_manifest=read_manifest(
            source / STAGE_A_PROMPT_POOL_FILENAMES["monitor"],
            context=ExecutionContext.SELECTION,
        ),
    )


__all__ = [
    "PromptPoolAllocation",
    "PromptPoolStratum",
    "STAGE_A_BG_GROUPS",
    "STAGE_A_BS_SLOTS",
    "STAGE_A_FAMILIES_PER_STRATUM",
    "STAGE_A_MONITOR_ITEMS_PER_PANEL_FAMILY",
    "STAGE_A_PROMPT_POOL_FILENAMES",
    "STAGE_A_PROMPT_POOL_SCHEMA_VERSION",
    "STAGE_A_RL_ITEMS_PER_FAMILY",
    "StageAPromptPoolArtifact",
    "StageAPromptPoolBundle",
    "StageAPromptPoolError",
    "build_stage_a_prompt_pools",
    "read_stage_a_prompt_pool_bundle",
    "write_stage_a_prompt_pool_bundle",
]
