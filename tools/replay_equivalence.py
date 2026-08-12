#!/usr/bin/env python3
"""Replay frozen v0 evidence through the immutable v1 instrument."""

from __future__ import annotations

import csv
from dataclasses import asdict
from functools import cache
import hashlib
import json
from math import fsum
from pathlib import Path
from typing import Any

from duraseed.data.boundary import (
    assess_confirmation_observation_gate,
    summarize_m0_boundary,
)
from duraseed.data.boundary_confirmation import (
    choose_refinement_family_ids,
    reduce_confirmation_evidence,
)
from duraseed.data.manifests import (
    MAPSTaskManifestRecord,
    TCESTaskManifestRecord,
    read_manifest,
)
from duraseed.data.sealing import ExecutionContext
from duraseed.data.panel_capacity import FamilyCapacityAudit, FamilySplitCapacity
from duraseed.evaluation.analysis import (
    BinomialObservation,
    base_stage_a_stage_b_transition_counts,
    equal_item_posterior_mean,
    normalized_stage_b_aucs,
    posterior_soft_cover,
    unbiased_pass_at_k,
)
from duraseed.run_records import GenerationRecord, RewardRecord
from duraseed.tasks.maps.verifier import verify_maps
from duraseed.tasks.tces.verifier import verify_completion

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "frozen/v0"
BOUNDARY = FROZEN / "runs/tinker-calibration/boundary"
BROAD = BOUNDARY / "boundary-broad-20260809T214400Z"
REFINE = BOUNDARY / "boundary-refine-20260810T005730Z"
CONFIRM = BOUNDARY / "boundary-confirm-20260810T144517Z"
MAPS = FROZEN / "runs/tinker-calibration/maps-probe/maps-shortest2-20260810T144606Z"
SMOKE = FROZEN / "artifacts/smoke"
HASHES = {
    path: digest
    for digest, path in (
        line.split("  ", 1)
        for line in (ROOT / "provenance/v0-artifacts.sha256").read_text().splitlines()
    )
}
TAUS = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75)
COVER_SHA256 = "eda23463171ace06ce33e7b8d3ba51518e35cc87f5ed4e5bfc68c7e0e4193b88"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@cache
def checked(path: Path) -> Path:
    relative = path.relative_to(ROOT).as_posix()
    require(relative in HASHES and path.is_file(), f"missing input: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == HASHES[relative], f"SHA-256 mismatch: {relative}")
    return path


def data(path: Path) -> Any:
    return json.loads(checked(path).read_text())


def rows(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(line) for line in checked(path).read_text().splitlines())


def run_rows(directory: Path) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    def load(name: str, model: Any) -> tuple[Any, ...]:
        return tuple(
            model.model_validate_json(line)
            for line in checked(directory / name).read_text().splitlines()
        )

    return load("generations.jsonl", GenerationRecord), load(
        "rewards.jsonl", RewardRecord
    )


def manifest(path: Path) -> Any:
    return read_manifest(checked(path), context=ExecutionContext.SELECTION)


def compare_summaries(
    summaries: tuple[Any, ...], recorded: tuple[Any, ...], label: str
) -> None:
    by_family = {item.intended_family_id: item for item in summaries}
    require(len(by_family) == len(summaries), f"{label}: duplicate family")
    for row in recorded:
        actual = json.loads(json.dumps(asdict(by_family[row["intended_family_id"]])))
        require(
            actual == {key: row[key] for key in actual}, f"{label}: aggregate differs"
        )


def verifier_replay(
    task_manifest: Any,
    generations: tuple[GenerationRecord, ...],
    rewards: tuple[RewardRecord, ...],
    family: str,
) -> dict[str, Any]:
    tasks = {row.task_id: row for row in task_manifest.records}
    linked = {row.sample_id: row for row in rewards}
    require(len(linked) == len(rewards), f"{family}: duplicate linkage")
    require(
        {row.sample_id for row in generations} == set(linked),
        f"{family}: linkage differs",
    )
    for generation in generations:
        record = tasks[generation.task_id]
        if family == "tces":
            require(isinstance(record, TCESTaskManifestRecord), "TCES record differs")
            actual = verify_completion(generation.completion_text, record.to_task())
        else:
            require(isinstance(record, MAPSTaskManifestRecord), "MAPS record differs")
            actual = verify_maps(
                generation.completion_text, record.to_task()
            ).to_shared_result()
        expected = linked[generation.sample_id]
        require(
            actual == expected.exact_verification,
            f"{family}: structured result differs",
        )
        require(generation.reward == expected.reward, f"{family}: reward differs")
    return {
        "rows": len(generations),
        "structured_differences": 0,
    }


def replay_boundary() -> tuple[dict[str, Any], dict[str, Any]]:
    broad_manifest = manifest(BROAD / "a_candidate_manifest.json")
    confirm_manifest = manifest(CONFIRM / "a_candidate_confirmation_manifest.json")
    broad_g, broad_r = run_rows(BROAD)
    refine_g, refine_r = run_rows(REFINE)
    confirm_g, confirm_r = run_rows(CONFIRM)
    tces = verifier_replay(confirm_manifest, confirm_g, confirm_r, "tces")
    families = {row.assigned_family_id for row in confirm_g}
    require(None not in families and len(families) == 38, "TCES coverage differs")
    require(len(confirm_g) - 38 >= 200, "TCES audit too small")
    tces.update(covered_families=38, additional_rows=len(confirm_g) - 38)

    stage1 = summarize_m0_boundary(broad_manifest, broad_g, broad_r, group_size=8)
    rows1 = rows(BROAD / "candidate_family_table.jsonl")
    compare_summaries(stage1, rows1, "Stage 1")
    successes = {row["intended_family_id"]: row["total_successes"] for row in rows1}
    positive, audit = choose_refinement_family_ids(successes)
    zero = tuple(name for name, count in successes.items() if count == 0)
    require(
        all(
            row["broad_screen_observed_success"] == (row["total_successes"] > 0)
            for row in rows1
        ),
        "Stage 1 decisions differ",
    )
    stage2 = summarize_m0_boundary(
        broad_manifest,
        (*broad_g, *refine_g),
        (*broad_r, *refine_r),
        group_size=8,
        expected_run_ids=(BROAD.name, REFINE.name),
    )
    rows2 = rows(REFINE / "refined_family_table.jsonl")
    compare_summaries(stage2, rows2, "Stage 2")

    def role(name: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                row["intended_family_id"]
                for row in rows2
                if row["refinement_role"] == name
            )
        )

    require(role("success_positive_candidate") == positive, "Stage 1 candidates differ")
    require(role("zero_success_audit") == tuple(sorted(audit)), "Stage 1 audit differs")
    require(
        set(role("broad_only")) == set(zero).difference(audit), "Stage 1 rejects differ"
    )
    refined = set(positive).union(audit)
    rows3 = rows(CONFIRM / "confirmation_family_table.jsonl")
    capacity_rows = rows(CONFIRM / "split_capacity_audits.jsonl")
    capacities = tuple(
        FamilyCapacityAudit(
            family_id=row["family_id"],
            split_capacities=tuple(
                FamilySplitCapacity(**item) for item in row["split_capacities"]
            ),
            available_disjoint_instances=row["available_disjoint_instances"],
            passed=row["passed"],
        )
        for row in capacity_rows
    )

    stage3 = summarize_m0_boundary(
        broad_manifest,
        (*broad_g, *refine_g, *confirm_g),
        (*broad_r, *refine_r, *confirm_r),
        group_size=8,
        expected_run_ids=(BROAD.name, REFINE.name, CONFIRM.name),
        additional_manifests=(confirm_manifest,),
    )
    compare_summaries(stage3, rows3, "Stage 3")
    evidence = reduce_confirmation_evidence(stage2, stage3, tuple(refined), capacities)
    finalists = evidence.stage2_finalist_family_ids
    require(
        finalists == tuple(sorted(row["intended_family_id"] for row in rows3)),
        "Stage 2 differs",
    )
    require(
        evidence.capacity_cleared_family_ids == finalists, "Stage 2 capacity differs"
    )
    by_family = {item.intended_family_id: item for item in stage3}
    decisions = []
    for row in rows3:
        result = assess_confirmation_observation_gate(
            by_family[row["intended_family_id"]]
        )
        require(
            (result.eligible, list(result.exclusion_reasons))
            == (
                row["observation_gate_eligible"],
                row["observation_gate_exclusion_reasons"],
            ),
            "Stage 3 decision differs",
        )
        decisions.append(
            (result.intended_family_id, result.eligible, result.exclusion_reasons)
        )
    passed = tuple(sorted(item[0] for item in decisions if item[1]))
    failed = tuple(sorted(item[0] for item in decisions if not item[1]))
    require((len(passed), len(failed)) == (15, 23), "Stage 3 counts differ")
    require(passed == evidence.final_eligible_family_ids, "Stage 3 reducer differs")
    require(
        passed
        == tuple(
            sorted(
                data(CONFIRM / "confirmation_summary.json")[
                    "observation_eligible_family_ids"
                ]
            )
        ),
        "Stage 3 IDs differ",
    )
    digest = hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()
    return tces, {
        "stage1": {
            "families": 64,
            "success_positive": len(positive),
            "audit": len(audit),
        },
        "stage2": {"families": 64, "finalists": len(finalists)},
        "stage3": {"finalists": 38, "passed": len(passed), "failed": len(failed)},
        "decision_sha256": digest,
        "differences": 0,
    }


def replay_maps() -> dict[str, Any]:
    generations, rewards = run_rows(MAPS)
    report = verifier_replay(
        manifest(MAPS / "manifests/shortest2_cap2_b_validation.json"),
        generations,
        rewards,
        "maps",
    )
    linked = {row.sample_id: row.reward for row in rewards}
    steps = tuple(sorted({row.training_step for row in generations}))
    scores = tuple(
        fsum(linked[row.sample_id] for row in generations if row.training_step == step)
        / sum(row.training_step == step for row in generations)
        for step in steps
    )
    auc = normalized_stage_b_aucs(steps, {"MAPS": scores})["MAPS"]
    values = (auc.absolute_auc, auc.raw_gain_auc, auc.headroom_normalized_gain_auc)
    require(
        values == (0.30164642333984376, 0.25281829833984376, 0.2657966504103982),
        "MAPS calibration AUC differs",
    )
    report["scientific_curve"] = {
        "steps": steps,
        "scores": scores,
        "auc_variants": values,
    }
    return report


def replay_analysis() -> dict[str, Any]:
    expected = data(SMOKE / "analysis.json")
    require(
        expected["fixture_kind"] == "synthetic_analysis_only"
        and not expected["scientific_claim"],
        "analysis fixture scope differs",
    )
    with checked(SMOKE / "item_transitions.csv").open(newline="") as handle:
        records = tuple(csv.DictReader(handle))
    fields = {
        "M0": "base_successes",
        "post-A": "stage_a_successes",
        "post-B": "stage_b_successes",
    }
    observations = {
        stage: tuple(
            BinomialObservation(int(row[field]), int(row["samples"])) for row in records
        )
        for stage, field in fields.items()
    }
    means = {
        stage: equal_item_posterior_mean(items) for stage, items in observations.items()
    }
    require(
        means
        == {"M0": 0.08823529411764706, "post-A": 0.5, "post-B": 0.38235294117647056},
        "posterior means differ",
    )
    cover = {
        tau: tuple(posterior_soft_cover(observations[stage], tau) for stage in fields)
        for tau in TAUS
    }
    cover_bytes = json.dumps(cover, sort_keys=True, separators=(",", ":")).encode()
    require(
        hashlib.sha256(cover_bytes).hexdigest() == COVER_SHA256, "Cover curve differs"
    )
    require(
        {stage: cover[0.25][index] for index, stage in enumerate(fields)}
        == expected["posterior_soft_cover"],
        "primary Cover differs",
    )
    hard = base_stage_a_stage_b_transition_counts(
        zip(observations["M0"], observations["post-A"], observations["post-B"]),
        expected["tau"],
        reliable_probability=expected["hard_reliable_probability"],
        unreliable_probability=expected["hard_unreliable_probability"],
    )
    hard_rows = {
        " -> ".join(state.value for state in key): count
        for key, count in sorted(
            hard.items(), key=lambda item: tuple(map(str, item[0]))
        )
    }
    require(hard_rows == expected["hard_transition_counts"], "transitions differ")
    k = expected["pass_at_k"]["k"]
    pass_k = {
        stage: fsum(
            unbiased_pass_at_k(item.successes, item.trials, k) for item in items
        )
        / len(items)
        for stage, items in observations.items()
    }
    require(pass_k == expected["pass_at_k"]["mean_by_stage"], "pass@k differs")
    return {"items": len(records), "cover_tau_count": len(cover), "differences": 0}


tces, boundary = replay_boundary()
print(
    json.dumps(
        {
            "status": "passed",
            "tces": tces,
            "boundary": boundary,
            "maps": replay_maps(),
            "analysis": replay_analysis(),
            "total_differences": 0,
        },
        indent=2,
        sort_keys=True,
    )
)
