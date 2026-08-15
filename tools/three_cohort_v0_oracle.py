#!/usr/bin/env python3
"""Run the archived v0 three-cohort reducer on normalized production inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from duraseed.config import load_pilot_config
from duraseed.data.boundary import BoundaryFamilySummary, BoundaryItemSummary
from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
from duraseed.data.panel_capacity import (
    PANEL_SELECTED_TEST_SINGLE_MINIMUM,
    PANEL_SPLIT_MINIMUMS,
)
from duraseed.data.panel_matching import build_family_panel_candidate
from duraseed.data.stage_a_prompt_pools import _candidate_ranking
from duraseed.provenance import canonical_json_bytes
from duraseed.tasks.tces import TCESGeneratorConfig
from duraseed import tinker_boundary_confirmation as confirmation
from duraseed import tinker_boundary_panel_freeze as freeze


_EXPECTED_SOURCES = {
    "src/duraseed/tinker_boundary_panel_freeze.py": (
        "55820094e4d99143fa37ec926ec15afdfd402fabb14021093f676919aa4e1432"
    ),
    "src/duraseed/tinker_boundary_confirmation.py": (
        "b1fa035143f2d035304b95e477f917132eb4bf276f9604aadad8bab199e43a24"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(value: object) -> DatasetManifest:
    return DatasetManifest.model_validate_json(
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    )


def _summary(value: dict) -> BoundaryFamilySummary:
    fields = dict(value)
    fields["items"] = tuple(BoundaryItemSummary(**row) for row in fields["items"])
    fields["observed_strategy_family_ids"] = tuple(
        fields["observed_strategy_family_ids"]
    )
    return BoundaryFamilySummary(**fields)


def _cohort(value: dict) -> SimpleNamespace:
    return SimpleNamespace(
        cohort_id=value["cohort_id"],
        broad_manifest=_manifest(value["broad_manifest"]),
        confirmation_manifest=_manifest(value["confirmation_manifest"]),
        finalist_summaries=tuple(_summary(row) for row in value["finalist_summaries"]),
        sampler_checkpoint_path=value["sampler_checkpoint_path"],
        locked_eligible_family_ids=tuple(value["locked_eligible_family_ids"]),
    )


def _settings_projection(config) -> dict:
    return {
        "allocation_seed": config.family_panels.frozen_allocation_seed,
        "capacity_root_seed": config.seed,
        "generator_kwargs": config.tasks.tces.generator_kwargs(),
        "matching_covariates": [
            value.value for value in config.family_panels.matching_covariates
        ],
        "panel_size": config.family_panels.size_profiles.confirmatory.targeted,
        "training_seeds": [
            *config.statistics.calibration_seeds,
            *config.statistics.pilot_seeds,
            *config.statistics.confirmatory_seeds,
        ],
    }


def _reduce(source: dict, config) -> dict:
    cohorts = tuple(_cohort(value) for value in source["cohorts"])
    combined_broad = freeze._combine_broad_manifests(cohorts)
    combined_confirmation = freeze._combine_confirmation_manifests(
        cohorts, broad_manifest_id=combined_broad.manifest_id
    )
    source_records = (*combined_broad.records, *combined_confirmation.records)
    summaries = tuple(
        sorted(
            (row for cohort in cohorts for row in cohort.finalist_summaries),
            key=lambda row: row.intended_family_id,
        )
    )
    finalist_ids = tuple(row.intended_family_id for row in summaries)
    if len(finalist_ids) != len(set(finalist_ids)):
        raise RuntimeError("v0 cohort finalist families overlap")
    generator = TCESGeneratorConfig(**config.tasks.tces.generator_kwargs())
    templates = confirmation._regenerate_family_templates(
        generator, combined_broad, finalist_ids
    )
    audits = confirmation._audit_family_capacities(
        templates,
        finalist_ids,
        generator,
        root_seed=config.seed,
        forbidden_records=source_records,
        protected_family_ids=finalist_ids,
    )
    if tuple(row.family_id for row in audits) != finalist_ids:
        raise RuntimeError("v0 combined capacity order changed")
    audit_by_family = {row.family_id: row for row in audits}
    summary_by_family = {row.intended_family_id: row for row in summaries}
    observation = tuple(
        row.intended_family_id
        for row in summaries
        if freeze.assess_confirmation_observation_gate(row).eligible
    )
    locked = cohorts[0].locked_eligible_family_ids
    if (
        len(locked) != 15
        or not set(locked).issubset(observation)
        or any(not audit_by_family[family_id].passed for family_id in locked)
    ):
        raise RuntimeError("v0 combined capacity reclassified a locked passer")
    eligible = tuple(
        family_id for family_id in observation if audit_by_family[family_id].passed
    )
    records_by_family: dict[str, list[TCESTaskManifestRecord]] = {
        family_id: [] for family_id in eligible
    }
    for record in source_records:
        if (
            isinstance(record, TCESTaskManifestRecord)
            and record.intended_family in records_by_family
        ):
            records_by_family[record.intended_family].append(record)
    renderer, train_on_what = freeze._load_renderer(
        SimpleNamespace(model_id=source["model_id"], renderer=source["renderer"])
    )
    token_counts = {
        family_id: confirmation._teacher_trace_token_counts(
            summary_by_family[family_id],
            tuple(records_by_family[family_id]),
            renderer,
            train_on_what,
        )
        for family_id in eligible
    }
    candidates = tuple(
        build_family_panel_candidate(
            summary_by_family[family_id],
            records_by_family[family_id],
            teacher_trace_token_counts=token_counts[family_id],
            available_disjoint_instances=(
                audit_by_family[family_id].available_disjoint_instances
            ),
        )
        for family_id in eligible
    )
    ranked_ids = _candidate_ranking(
        tuple(asdict(row) for row in candidates),
        config.family_panels.frozen_allocation_seed,
    )
    resolved = None
    selected_audits = ()
    if len(candidates) >= freeze.BOUNDARY_PANEL_FREEZE_MINIMUM_CANDIDATES:
        provisional = confirmation._resolve_panel_selection(
            candidates,
            config,
            m0_checkpoint_path=cohorts[0].sampler_checkpoint_path,
        )
        selected = provisional.match.selected_family_ids
        selected_audits = confirmation._audit_family_capacities(
            confirmation._regenerate_family_templates(
                generator, combined_broad, selected
            ),
            selected,
            generator,
            root_seed=config.seed,
            requirements={
                **dict(PANEL_SPLIT_MINIMUMS),
                "a_test_single": PANEL_SELECTED_TEST_SINGLE_MINIMUM,
            },
            forbidden_records=source_records,
            protected_family_ids=selected,
        )
        if tuple(row.family_id for row in selected_audits) != selected:
            raise RuntimeError("v0 selected capacity order changed")
        if selected_audits and all(row.passed for row in selected_audits):
            resolved = provisional
    artifact = resolved.artifact if resolved else None
    summary = {
        "status": (
            "confirmed_panels_frozen"
            if artifact is not None
            else "confirmation_complete_panel_selection_unresolved"
        ),
        "source_kind": freeze.BOUNDARY_PANEL_FREEZE_SOURCE_KIND,
        "selection_performed": artifact is not None,
        "panel_assignment_performed": artifact is not None,
        "pilot_started": False,
        "cohort_ids": [row.cohort_id for row in cohorts],
        "finalist_family_count": len(summaries),
        "observation_eligible_family_count": len(observation),
        "observation_eligible_family_ids": list(observation),
        "split_capacity_requirements": dict(PANEL_SPLIT_MINIMUMS),
        "split_capacity_eligible_family_count": sum(row.passed for row in audits),
        "split_capacity_eligible_family_ids": [
            row.family_id for row in audits if row.passed
        ],
        "panel_candidate_family_count": len(candidates),
        "panel_candidate_family_ids": [row.family_id for row in candidates],
        "minimum_required_for_panel_construction": (
            freeze.BOUNDARY_PANEL_FREEZE_MINIMUM_CANDIDATES
        ),
        "selected_panel_split_capacity_passed": bool(selected_audits)
        and all(row.passed for row in selected_audits),
        "panel_a_family_ids": list(artifact.panel_a_family_ids) if artifact else [],
        "panel_b_family_ids": list(artifact.panel_b_family_ids) if artifact else [],
        "candidate_table_id": resolved.candidate_table_id if resolved else None,
        "panel_matching_report_id": (resolved.matching_report_id if resolved else None),
        "next_step": (
            "continue remaining pre-pilot calibration"
            if artifact
            else "stop Phase 3 unresolved; do not extend or weaken the design"
        ),
    }
    return {
        "combined_broad_manifest": combined_broad,
        "combined_confirmation_manifest": combined_confirmation,
        "finalist_summaries": summaries,
        "confirmation_family_table": tuple(
            freeze._family_row(row) for row in summaries
        ),
        "capacity_audits": audits,
        "observation_eligible_family_ids": observation,
        "eligible_family_ids": eligible,
        "teacher_trace_token_counts": token_counts,
        "candidates": candidates,
        "ranked_family_ids": ranked_ids,
        "candidate_payload": resolved.candidate_payload if resolved else None,
        "candidate_table_id": resolved.candidate_table_id if resolved else None,
        "match": resolved.match if resolved else None,
        "matching_report_payload": (
            resolved.matching_report_payload if resolved else None
        ),
        "matching_report_id": resolved.matching_report_id if resolved else None,
        "intermediate_family_ids": (
            tuple(sorted(ranked_ids[24:36])) if artifact is not None else ()
        ),
        "selected_capacity_audits": selected_audits,
        "panel_artifact": artifact,
        "confirmation_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for relative, expected in _EXPECTED_SOURCES.items():
        if _sha256(args.source_root / relative) != expected:
            raise RuntimeError(f"archived v0 source changed: {relative}")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    config = load_pilot_config(args.source_root / "duraseed_pilot_config.yaml")
    if _settings_projection(config) != source["settings"]:
        raise RuntimeError("v0/v1 freeze-setting projections differ")
    args.output.write_bytes(canonical_json_bytes(_reduce(source, config)) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
