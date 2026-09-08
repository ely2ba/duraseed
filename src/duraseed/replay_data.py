"""Pure, success-conditioned archived-policy/solver corpus selection for replay v1."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any, Iterable

from duraseed.data.leakage import audit_leakage
from duraseed.pilot0_contract import PilotSeedSources
from duraseed.pilot0_data import _tces_completion
from duraseed.provenance import canonical_json_hash
from duraseed.training.reward import verify_task_completion
from duraseed.training.sft import (
    build_current_policy_verified_record,
    build_solver_teacher_record,
)


def selection_key(seed: int, task_id: str, sample_id: str) -> tuple[str, str]:
    text = f"duraseed-replay-v1|{seed}|{task_id}|{sample_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), sample_id


def order_key(seed: int, task_id: str) -> str:
    text = f"duraseed-replay-order-v1|{seed}|{task_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def batch_indices(size: int, update: int) -> tuple[int, ...]:
    if size < 32 or not 1 <= update <= 294:
        raise ValueError("replay requires >=32 unique prompts and updates 1..294")
    return tuple(((update - 1) * 32 + j) % size for j in range(32))


def build_corpus(
    source: PilotSeedSources, records: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep full historical text; selection never uses length except the cap rule."""

    manifest = source.prompt_pools.a_rl_train_manifest
    tasks = {row.task_id: row for row in manifest.records}
    pools = source.prompt_pools.artifact
    strata = {
        family: label
        for label, families in (
            ("boundary", pools.boundary_family_ids),
            ("intermediate", pools.intermediate_family_ids),
            ("broad_random", pools.broad_random_family_ids),
        )
        for family in families
    }
    sentinel = set(pools.sentinel_family_ids)
    leakage = audit_leakage(
        {
            "a_rl_train": manifest,
            "a_monitor": source.prompt_pools.a_monitor_manifest,
            "a_validation": source.a_validation,
        }
    ).assert_clean()
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions, seen = [], set()
    family_counts: dict[str, Counter] = defaultdict(Counter)
    observed_tasks: dict[str, set[str]] = defaultdict(set)
    for item in records:
        row, reward = item["generation"], item["reward"]
        sample_id, task_id = row["sample_id"], row["task_id"]
        if sample_id in seen:
            raise ValueError("duplicate authenticated sample identifier")
        seen.add(sample_id)
        task = tasks.get(task_id)
        if task is None or row["task_manifest_id"] != manifest.manifest_id:
            raise ValueError("rollout task is outside the authorized manifest")
        family = task.intended_family
        if family not in strata or family in sentinel:
            raise ValueError("replay prompt violates stratum/sentinel separation")
        if (
            row["seed"] != source.seed
            or row["method"] != "B-G"
            or row["purpose"] != "training"
            or row["source_split"] != "a_rl_train"
            or row["assigned_family_id"] != family
            or row["item_index"] != task.item_index
            or row["task_family"] != "tces"
        ):
            raise ValueError("rollout source coordinates differ from its manifest")
        verification = verify_task_completion(row["completion_text"], task.to_task())
        if (
            reward["sample_id"] != sample_id
            or reward["task_id"] != task_id
            or verification.model_dump(mode="json") != reward["exact_verification"]
            or verification.reward != reward["reward"]
            or verification.reward != row["reward"]
        ):
            raise ValueError(f"historical verifier disagreement: {sample_id}")
        reasons = []
        if not 1 <= row["training_step"] <= 50:
            reasons.append("outside_fixed_update_cutoff")
        if verification.reward != 1.0:
            reasons.append("not_verifier_correct")
        if (
            row["stop_reason"] == "length"
            or row["sampled_tokens"] >= row["sampling_max_tokens"]
        ):
            reasons.append("capped")
        decision = {
            "sample_id": sample_id,
            "task_id": task_id,
            "assigned_family_id": family,
            "stratum": strata[family],
            "update": row["training_step"],
            "reasons": reasons,
            "selected": False,
        }
        decisions.append(decision)
        observed_tasks[family].add(task_id)
        family_counts[family]["sampled"] += 1
        for reason in reasons:
            family_counts[family][reason] += 1
        if not reasons:
            family_counts[family]["eligible_samples"] += 1
            eligible[task_id].append({**item, "decision": decision})
    selected = []
    for task_id in sorted(
        eligible, key=lambda value: (order_key(source.seed, value), value)
    ):
        candidates = eligible[task_id]
        chosen = min(
            candidates,
            key=lambda item: selection_key(
                source.seed, task_id, item["generation"]["sample_id"]
            ),
        )
        for item in candidates:
            item["decision"]["reasons"] = [
                "selected" if item is chosen else "eligible_not_selected"
            ]
            item["decision"]["selected"] = item is chosen
        row, task = chosen["generation"], tasks[task_id]
        solver = build_solver_teacher_record(
            source_manifest=manifest,
            source_record=task,
            completion=_tces_completion(task),
        )
        policy = build_current_policy_verified_record(
            source_manifest=manifest,
            source_record=task,
            completion=row["completion_text"],
        )
        if (
            solver.prompt_text != row["prompt_text"]
            or policy.prompt_text != solver.prompt_text
        ):
            raise ValueError("stored prompt text differs from the frozen task renderer")
        family_counts[task.intended_family]["selected_prompts"] += 1
        selected.append(
            {
                "task_id": task_id,
                "assigned_family_id": task.intended_family,
                "stratum": strata[task.intended_family],
                "order_sha256": order_key(source.seed, task_id),
                "selection_sha256": selection_key(
                    source.seed, task_id, row["sample_id"]
                )[0],
                "source": {
                    **chosen["source"],
                    "sample_id": row["sample_id"],
                    "sample_index": row["sample_index"],
                    "sampling_seed": row["sampling_seed"],
                    "rollout_for_update": row["training_step"],
                    "generating_completed_update": row["training_step"] - 1,
                    "sampler_checkpoint_path": row["sampler_checkpoint_path"],
                    "sampled_tokens": row["sampled_tokens"],
                    "stop_reason": row["stop_reason"],
                    "completion_sha256": hashlib.sha256(
                        row["completion_text"].encode()
                    ).hexdigest(),
                    "record_sha256": canonical_json_hash(row),
                },
                "R-S": solver.model_dump(mode="json"),
                "R-P": policy.model_dump(mode="json"),
            }
        )
    stratum_counts: dict[str, Counter] = defaultdict(Counter)
    for family in strata:
        counts = family_counts[family]
        counts["manifest_prompts"] = sum(
            row.intended_family == family for row in manifest.records
        )
        counts["sampled_unique_prompts"] = len(observed_tasks[family])
        counts["unsampled_manifest_prompts"] = (
            counts["manifest_prompts"] - counts["sampled_unique_prompts"]
        )
        counts["no_eligible_trace_prompts"] = (
            counts["sampled_unique_prompts"] - counts["selected_prompts"]
        )
    for family, counts in family_counts.items():
        stratum_counts[strata[family]].update(counts)
    return (
        selected,
        decisions,
        {
            "schema": "duraseed-replay-corpus-v1",
            "seed": source.seed,
            "status": "READY" if len(selected) >= 32 else "DATA_BLOCKED",
            "unique_prompts": len(selected),
            "sample_count": len(decisions),
            "manifest_prompt_count": len(tasks),
            "sampled_unique_prompt_count": sum(
                len(values) for values in observed_tasks.values()
            ),
            "eligible_sample_count": sum(len(values) for values in eligible.values()),
            "source_manifest_id": manifest.manifest_id,
            "leakage": leakage.to_dict(),
            "family_attrition": {
                family: dict(family_counts[family]) for family in sorted(strata)
            },
            "stratum_attrition": {
                key: dict(value) for key, value in sorted(stratum_counts.items())
            },
            "trace_source": "archived B-G acquisition updates 1..50; success-conditioned; not on-policy for replay",
            "verification_scope": "final answer only, not every rationale step",
            "minimum_scope": "32 unique prompts is operational only, not scientific adequacy",
            "mixture": "observed success-conditioned mixture; no rebalancing",
        },
    )
