from __future__ import annotations

import json
from pathlib import Path

import pytest

from duraseed.config import load_pilot_config
from duraseed.data.boundary_sources import (
    BoundarySourceContract,
    BoundarySourceError,
    validate_completed_confirmation_contract,
    validate_confirmation_source_contract,
)
from duraseed.data.manifests import DatasetManifest
from duraseed.run_records import GenerationRecord, RunRecord


ROOT = Path(__file__).resolve().parents[2]
BOUNDARY = ROOT / "frozen/v0/runs/tinker-calibration/boundary"


def _frozen_initial_confirmation():
    broad = DatasetManifest.model_validate_json(
        (
            BOUNDARY / "boundary-broad-20260809T214400Z/a_candidate_manifest.json"
        ).read_text()
    )
    directory = BOUNDARY / "boundary-confirm-20260810T144517Z"
    confirmed = DatasetManifest.model_validate_json(
        (directory / "a_candidate_confirmation_manifest.json").read_text()
    )
    run = RunRecord.model_validate_json((directory / "run.json").read_text())
    generations = tuple(
        GenerationRecord.model_validate_json(line)
        for line in (directory / "generations.jsonl").read_text().splitlines()
    )
    first = generations[0]
    source = BoundarySourceContract(
        cohort_id="initial",
        cohort_ordinal_start=0,
        prior_run_ids=(
            "boundary-broad-20260809T214400Z",
            "boundary-refine-20260810T005730Z",
        ),
        sampler_checkpoint_path=first.sampler_checkpoint_path,
        training_step=first.training_step,
        model_id=run.model_id,
        renderer=run.renderer,
        lora_rank=run.lora_rank,
        state_checkpoint_path=run.parent_tinker_checkpoint_path,
        project_id=str(run.project_id),
        protocol_version="legacy-prepilot-v1",
        resolved_config_hash="legacy-source-config-hash",
        broad_manifest_id=broad.manifest_id,
    )
    plan = {
        "run_id": "boundary-confirm-20260810T144517Z",
        "project_id": run.project_id,
        "source": {
            "broad_manifest_id": broad.manifest_id,
            "prior_run_ids": list(source.prior_run_ids),
            "sampler_checkpoint_path": source.sampler_checkpoint_path,
            "training_step": source.training_step,
            "source_protocol_version": source.protocol_version,
            "source_resolved_config_hash": source.resolved_config_hash,
            "current_protocol_version": run.protocol_version,
            "current_resolved_config_hash": run.resolved_config_hash,
            "compatible_pre_pilot_amendment": "docs/rfc-prepilot-estimands-and-selection.md",
        },
        "measurement": {
            "manifest_id": confirmed.manifest_id,
            "temperature": first.sampling_temperature,
            "top_p": first.sampling_top_p,
            "max_tokens": first.sampling_max_tokens,
        },
    }
    return broad, confirmed, run, generations, source, plan


def test_completed_initial_confirmation_replays_frozen_evidence() -> None:
    broad, confirmed, run, generations, source, plan = _frozen_initial_confirmation()

    run_id, sample_ids = validate_completed_confirmation_contract(
        source_contract=source,
        confirmation_run=run,
        confirmation_plan=plan,
        broad_manifest=broad,
        confirmation_manifest=confirmed,
        confirmation_generations=generations,
        config=load_pilot_config(ROOT / "duraseed_pilot_config.yaml"),
    )

    assert run_id == "boundary-confirm-20260810T144517Z"
    assert len(sample_ids) == len(set(sample_ids)) == 2432


def test_extension1_source_contract_replays_frozen_handoff() -> None:
    broad_dir = BOUNDARY / "boundary-broad-extension1-20260810T223000Z"
    refine_dir = BOUNDARY / "boundary-refine-extension1-20260811T104725Z"
    initial = DatasetManifest.model_validate_json(
        (
            BOUNDARY / "boundary-broad-20260809T214400Z/a_candidate_manifest.json"
        ).read_text()
    )
    manifest = DatasetManifest.model_validate_json(
        (broad_dir / "a_candidate_manifest.json").read_text()
    )

    def generations(directory: Path) -> tuple[GenerationRecord, ...]:
        return tuple(
            GenerationRecord.model_validate_json(line)
            for line in (directory / "generations.jsonl").read_text().splitlines()
        )

    contract = validate_confirmation_source_contract(
        config=load_pilot_config(ROOT / "duraseed_pilot_config.yaml"),
        broad_run=RunRecord.model_validate_json((broad_dir / "run.json").read_text()),
        refinement_run=RunRecord.model_validate_json(
            (refine_dir / "run.json").read_text()
        ),
        broad_plan=json.loads((broad_dir / "preflight.public.json").read_text()),
        refinement_plan=json.loads((refine_dir / "preflight.public.json").read_text()),
        broad_manifest=manifest,
        refinement_manifest=DatasetManifest.model_validate_json(
            (refine_dir / "a_candidate_manifest.json").read_text()
        ),
        broad_generations=generations(broad_dir),
        refinement_generations=generations(refine_dir),
        expected_parent_manifest_id=initial.manifest_id,
    )

    assert contract.cohort_id == "extension_1"
    assert contract.cohort_ordinal_start == 64
    assert contract.prior_run_ids == (
        "boundary-broad-extension1-20260810T223000Z",
        "boundary-refine-extension1-20260811T104725Z",
    )


def test_completed_confirmation_rejects_identity_and_sampling_tampering() -> None:
    broad, confirmed, run, generations, source, plan = _frozen_initial_confirmation()
    config = load_pilot_config(ROOT / "duraseed_pilot_config.yaml")
    with pytest.raises(BoundarySourceError, match="selected-M0 identity"):
        validate_completed_confirmation_contract(
            source_contract=source,
            confirmation_run=run.model_copy(update={"model_id": "wrong"}),
            confirmation_plan=plan,
            broad_manifest=broad,
            confirmation_manifest=confirmed,
            confirmation_generations=generations,
            config=config,
        )
    tampered = generations[0].model_copy(update={"sampling_max_tokens": 1})
    with pytest.raises(BoundarySourceError, match="observation contract"):
        validate_completed_confirmation_contract(
            source_contract=source,
            confirmation_run=run,
            confirmation_plan=plan,
            broad_manifest=broad,
            confirmation_manifest=confirmed,
            confirmation_generations=(tampered, *generations[1:]),
            config=config,
        )
