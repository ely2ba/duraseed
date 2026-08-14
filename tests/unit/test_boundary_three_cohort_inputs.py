from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from duraseed.boundary_consolidated_journal import load_consolidated_journal
from duraseed.boundary_three_cohort_inputs import load_three_cohort_inputs
from duraseed.data.manifests import build_manifest, build_tces_record
from duraseed.provenance import derive_namespaced_seed
from duraseed.run_records import GenerationRecord, RewardRecord, write_jsonl
from duraseed.runners import RunnerGateError
from duraseed.schemas import VerificationFailure, VerificationResult
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig


def _evidence():
    root_seed = 313
    split = "a_candidate"
    task = TCESGenerator(
        derive_namespaced_seed(root_seed, "dataset.tces.split", split),
        TCESGeneratorConfig(
            n_operands=3,
            max_tree_depth=3,
            max_ast_nodes=5,
            max_valid_expressions=1_000,
            split=split,
        ),
    ).generate(0)
    record = build_tces_record(task)
    manifest = build_manifest(
        name="synthetic-boundary-journal",
        task_family="tces",
        split=split,
        generator_version="1.0.0",
        root_seed=root_seed,
        records=(record,),
    )
    verification = VerificationResult(
        valid_answer_tag=True,
        valid_lexing=True,
        valid_syntax=True,
        valid_operand_multiset=True,
        valid_operations=True,
        valid_intermediates=True,
        correct_target=False,
        reward=0.0,
        failure_code=VerificationFailure.WRONG_TARGET,
    )
    generations = tuple(
        GenerationRecord(
            sample_id=f"sample-{index}",
            sample_index=index,
            sampling_seed=10_000 + index,
            purpose="evaluation",
            checkpoint_stage="m0",
            training_step=2,
            sampler_checkpoint_path="tinker://m0/sampler",
            task_manifest_id=manifest.manifest_id,
            task_id=record.task_id,
            task_family="tces",
            source_split="a_candidate",
            prompt_text="prompt",
            completion_text="<answer>0</answer>",
            prompt_tokens=1,
            sampled_tokens=1,
            sampling_temperature=1.0,
            sampling_top_p=0.95,
            sampling_max_tokens=4096,
            run_id="synthetic-run:extension2-broad",
            seed=5,
            origin_sampler_checkpoint_path="tinker://m0/sampler",
            item_index=record.item_index,
            assigned_family_id=record.intended_family,
            stop_reason="stop",
            reward=0.0,
        )
        for index in range(4)
    )
    rewards = tuple(
        RewardRecord(
            reward_id=f"reward-{index}",
            sample_id=generation.sample_id,
            task_id=record.task_id,
            reward=0.0,
            exact_verification=verification,
        )
        for index, generation in enumerate(generations)
    )
    return manifest, generations, rewards


def _write_bundle(
    root: Path,
    manifest,
    generations,
    rewards,
    *,
    projected_generations=None,
) -> None:
    write_jsonl(
        root / "observation_groups.jsonl",
        (
            {
                "action": "extension2-broad",
                "task_id": manifest.records[0].task_id,
                "generations": [row.model_dump(mode="json") for row in generations],
                "rewards": [row.model_dump(mode="json") for row in rewards],
            },
        ),
    )
    write_jsonl(root / "generations.jsonl", projected_generations or generations)
    write_jsonl(root / "rewards.jsonl", rewards)


def test_consolidated_journal_loads_exact_raw_projections(tmp_path: Path) -> None:
    manifest, generations, rewards = _evidence()
    _write_bundle(tmp_path, manifest, generations, rewards)

    loaded_g, loaded_r, tasks = load_consolidated_journal(
        tmp_path,
        run_id="synthetic-run",
        manifests={"extension2-broad": manifest},
    )

    assert loaded_g["extension2-broad"] == generations
    assert loaded_r["extension2-broad"] == rewards
    assert tasks["extension2-broad"] == {manifest.records[0].task_id}


def test_three_cohort_loader_stays_closed_before_confirmation(tmp_path: Path) -> None:
    with pytest.raises(RunnerGateError, match="requires completed Extension-2"):
        load_three_cohort_inputs(None, tmp_path / "not-read", tmp_path)  # type: ignore[arg-type]


def test_consolidated_journal_rejects_projection_tamper(tmp_path: Path) -> None:
    manifest, generations, rewards = _evidence()
    changed = generations[0].model_copy(update={"completion_text": "changed"})
    _write_bundle(
        tmp_path,
        manifest,
        generations,
        rewards,
        projected_generations=(changed, *generations[1:]),
    )

    with pytest.raises(RunnerGateError, match="projections differ"):
        load_consolidated_journal(
            tmp_path,
            run_id="synthetic-run",
            manifests={"extension2-broad": manifest},
        )


def test_consolidated_journal_rejects_coordinate_tamper(tmp_path: Path) -> None:
    manifest, generations, rewards = _evidence()
    changed = generations[0].model_copy(update={"sample_index": 9})
    tampered = (changed, *generations[1:])
    _write_bundle(tmp_path, manifest, tampered, rewards)

    with pytest.raises(RunnerGateError, match="coordinates changed"):
        load_consolidated_journal(
            tmp_path,
            run_id="synthetic-run",
            manifests={"extension2-broad": manifest},
        )


def test_preserved_source_rejects_sha_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duraseed.boundary_three_cohort_inputs as loader

    relative = "source.json"
    (tmp_path / relative).write_text("changed", encoding="utf-8")
    monkeypatch.setattr(loader, "_PRESERVED_FILES", (relative,))
    monkeypatch.setattr(loader, "_FROZEN_PREFIX", Path("frozen"))
    monkeypatch.setattr(
        loader,
        "_provenance_hashes",
        lambda: {"frozen/source.json": hashlib.sha256(b"original").hexdigest()},
    )

    with pytest.raises(RunnerGateError, match="SHA-256 changed"):
        loader._authenticate_preserved_source(tmp_path)
