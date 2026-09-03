"""Local, descriptive pair-2 MAPS failure/length and baseline-item summaries."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/pilot0/pilot0-pair2-seed29-20260830T204211Z"
OUT = Path(__file__).resolve().parent
GRID = (0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480)
METHODS = ("B-S", "B-G")


def relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def evaluation_path(method: str, index: int) -> Path:
    segment = "step-0" if index == 0 else f"steps-{GRID[index - 1]}-{GRID[index]}"
    return RUN / "seed-29" / method / "stage-b" / segment / "b-validation"


def rows(path: Path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def summarize(method: str, index: int, manifest: dict) -> dict:
    directory = evaluation_path(method, index)
    rewards = {}
    for reward in rows(directory / "rewards.jsonl"):
        key = reward["sample_id"]
        assert key not in rewards, f"Duplicate reward: {key}"
        rewards[key] = reward
    counts, failures, stops, cross = Counter(), Counter(), Counter(), Counter()
    lengths, character_lengths, caps, manifest_ids = [], [], set(), set()
    item_successes, item_draws, draw_keys, seen = Counter(), Counter(), set(), set()
    for generation in rows(directory / "generations.jsonl"):
        key, task = generation["sample_id"], generation["task_id"]
        assert key not in seen and key in rewards
        seen.add(key)
        reward = rewards[key]
        assert reward["task_id"] == task and task in manifest["by_id"]
        assert generation["item_index"] == manifest["by_id"][task]["item_index"]
        assert generation["method"] == method
        assert generation["training_step"] == GRID[index]
        assert generation["source_split"] == "b_validation"
        draw = (task, generation["sample_index"])
        assert draw not in draw_keys and 0 <= draw[1] < 16
        draw_keys.add(draw)
        check = reward["exact_verification"]
        assert reward["reward"] == generation["reward"] == check["reward"]
        success = int(reward["reward"] == 1.0)
        assert reward["reward"] in (0.0, 1.0)
        code = check["failure_code"]
        assert (code is None) == bool(success)
        cap, length = generation["sampling_max_tokens"], generation["sampled_tokens"]
        cap_hit = length >= cap
        tag_bad, syntax_bad = not check["valid_answer_tag"], not check["valid_syntax"]
        format_bad = tag_bad or syntax_bad
        cap_or_format = cap_hit or format_bad
        counts.update({
            "completions": 1, "successes": success, "failures": 1 - success,
            "length_stop_count": cap_hit, "missing_tag_count": code == "missing_answer_tag",
            "invalid_tag_count": tag_bad, "syntactically_invalid_count": syntax_bad,
            "format_invalid_count": format_bad,
            "cap_or_format_failure_count": cap_or_format and not success,
            "cap_or_format_success_count": cap_or_format and bool(success),
            "wrong_target_count": code == "wrong_target",
        })
        if code is not None:
            failures[code] += 1
        stops[generation["stop_reason"]] += 1
        cross[(bool(success), cap_hit, tag_bad, syntax_bad)] += 1
        lengths.append(length)
        character_lengths.append(len(generation["completion_text"]))
        caps.add(cap)
        manifest_ids.add(generation["task_manifest_id"])
        item_successes[task] += success
        item_draws[task] += 1
    assert seen == set(rewards), "Generation/reward sample sets differ"
    assert set(item_draws) == set(manifest["by_id"]), "Manifest item set differs"
    assert set(item_draws.values()) == {16} and len(item_draws) == 512
    assert manifest_ids == {manifest["manifest_id"]} and caps == {128}
    n = counts["completions"]
    rates = {key.removesuffix("_count") + "_rate": value / n
             for key, value in counts.items() if key.endswith("_count")}
    rates.update({"pass_at_1": counts["successes"] / n,
                  "all_failure_rate": counts["failures"] / n,
                  "cap_or_format_share_of_failures": counts["cap_or_format_failure_count"]
                  / counts["failures"] if counts["failures"] else 0.0})
    return {
        "method": method, "update": GRID[index], "items": 512, "draws_per_item": 16,
        "cap_tokens": 128, **dict(counts), **rates,
        "completion_tokens_mean": mean(lengths), "completion_tokens_median": median(lengths),
        "completion_characters_mean": mean(character_lengths),
        "completion_characters_median": median(character_lengths),
        "failure_code_counts": dict(sorted(failures.items())),
        "recorded_stop_reason_counts": dict(sorted(stops.items())),
        "success_cap_tag_syntax_cross_tab": [
            {"success": key[0], "at_token_cap": key[1], "invalid_tag": key[2],
             "invalid_syntax": key[3], "count": value}
            for key, value in sorted(cross.items())],
        "item_successes": dict(sorted(item_successes.items())),
        "sources": [relative(directory / name)
                    for name in ("generations.jsonl", "rewards.jsonl")],
    }


def baseline_overlap(series: dict, manifest: dict) -> dict:
    baseline = series["B-G"][0]["item_successes"]
    baseline_set = {task for task, success in baseline.items() if success > 0}
    comparisons = []
    for method in METHODS:
        for update in (80, 480):
            successes = series[method][GRID.index(update)]["item_successes"]
            solved = {task for task, success in successes.items() if success > 0}
            overlap, union = baseline_set & solved, baseline_set | solved
            on_baseline = sum(successes[task] for task in baseline_set)
            comparisons.append({
                "method": method, "update": update, "solved_item_count": len(solved),
                "overlap_item_count": len(overlap), "union_item_count": len(union),
                "baseline_item_recall": len(overlap) / len(baseline_set),
                "jaccard": len(overlap) / len(union),
                "baseline_fraction_of_later_solved_items": len(overlap) / len(solved),
                "successful_completions_total": sum(successes.values()),
                "successful_completions_on_baseline_items": on_baseline,
                "fraction_later_successes_on_baseline_items": on_baseline / sum(successes.values()),
                "later_solved_ids": sorted(solved), "overlap_ids": sorted(overlap),
                "baseline_not_later_solved_ids": sorted(baseline_set - solved),
                "later_solved_not_baseline_ids": sorted(solved - baseline_set),
            })
    return {
        "definition": "Solved item = at least one exact success among its 16 stored draws.",
        "baseline_method": "B-G", "baseline_update": 0,
        "baseline_solved_item_count": len(baseline_set),
        "baseline_successful_completions": sum(baseline.values()),
        "baseline_successes_per_solved_item_histogram": dict(sorted(Counter(
            baseline[task] for task in baseline_set).items())),
        "baseline_solved_items": [
            {"task_id": task, "item_index": manifest["by_id"][task]["item_index"],
             "successes_out_of_16": baseline[task],
             "shortest_family_ids": manifest["by_id"][task]["shortest_family_ids"]}
            for task in sorted(baseline_set)],
        "comparisons": comparisons,
    }


def table(headers: list, values: list) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in values])


def markdown(data: dict) -> str:
    text = ["# Pair-2 F2 failure, length, and baseline concentration diagnostics",
            "Local-only descriptive analysis; no new sampling, no gate or design authority.",
            "## Definitions and provenance", data["definitions"],
            f"Manifest: `{data['manifest_path']}`. Every checkpoint has 512 manifest items × "
            "16 draws = 8,192 completions, with a 128-token cap. Sample-id/task-id joins, "
            "draw uniqueness, item sets, and manifest IDs are checked. Generation token IDs "
            "and log-probabilities are not retained by this analysis.",
            "All exact source paths, item IDs, counts, rates, and cross-tabs are in "
            "[`f2_diagnostics.json`](f2_diagnostics.json). Rates below are percentages of "
            "all 8,192 completions unless stated otherwise."]
    for method in METHODS:
        text += [f"## {method}: completion trajectories", table(
            ["Update", "Correct", "Pass@1 %", "Cap %", "Missing tag %",
             "Invalid tag %", "Invalid syntax %", "All failure %", "Tokens mean", "Median"],
            [[row["update"], row["successes"], f"{100 * row['pass_at_1']:.4f}",
              f"{100 * row['length_stop_rate']:.4f}", f"{100 * row['missing_tag_rate']:.4f}",
              f"{100 * row['invalid_tag_rate']:.4f}",
              f"{100 * row['syntactically_invalid_rate']:.4f}",
              f"{100 * row['all_failure_rate']:.4f}",
              f"{row['completion_tokens_mean']:.4f}", row["completion_tokens_median"]]
             for row in data["series"][method]])]
        codes = sorted({key for row in data["series"][method]
                        for key in row["failure_code_counts"]})
        text += [f"### {method}: failure-code counts", table(
            ["Update"] + codes, [[row["update"]]
            + [row["failure_code_counts"].get(code, 0) for code in codes]
            for row in data["series"][method]])]
    text += ["## Question on record", data["question"],
             "Cap/format failure union = completion fails exact verification and is at "
             "the token cap, has an invalid answer tag, or has invalid syntax. Categories "
             "overlap and this union is not a causal classification.", table(
                 ["B-G update", "Failures", "Cap/format union", "Share of failures %",
                  "Wrong target", "Cap count", "Invalid tag", "Invalid syntax"],
                 [[row["update"], row["failures"], row["cap_or_format_failure_count"],
                   f"{100 * row['cap_or_format_share_of_failures']:.4f}",
                   row["wrong_target_count"], row["length_stop_count"],
                   row["invalid_tag_count"], row["syntactically_invalid_count"]]
                  for row in data["series"]["B-G"] if 1 <= row["update"] <= 80])]
    early = [row for row in data["series"]["B-G"] if 1 <= row["update"] <= 40]
    at40, at80 = [data["series"]["B-G"][GRID.index(step)] for step in (40, 80)]
    text += [f"Across updates 1–40, cap/format failures account for "
             f"{100 * min(row['cap_or_format_share_of_failures'] for row in early):.4f}%–"
             f"{100 * max(row['cap_or_format_share_of_failures'] for row in early):.4f}% "
             "of failures. At updates 40 → 80, exact successes are "
             f"{at40['successes']} → {at80['successes']}; cap counts "
             f"{at40['length_stop_count']} → {at80['length_stop_count']}; invalid-tag counts "
             f"{at40['invalid_tag_count']} → {at80['invalid_tag_count']}; invalid-syntax counts "
             f"{at40['syntactically_invalid_count']} → {at80['syntactically_invalid_count']}; "
             f"wrong-target counts {at40['wrong_target_count']} → {at80['wrong_target_count']}. "
             "These are aggregate trajectories; they do not identify a causal mechanism."]
    overlap = data["baseline_concentration"]
    text += ["## B-G update-0 solved-item concentration", overlap["definition"],
             f"B-G update 0: {overlap['baseline_solved_item_count']}/512 items with ≥1 success; "
             f"{overlap['baseline_successful_completions']}/8,192 successful completions.", table(
                 ["Arm", "Update", "Solved items", "Overlap", "Baseline recall %",
                  "Jaccard", "Successes on baseline / total", "Success share %"],
                 [[row["method"], row["update"], row["solved_item_count"],
                   row["overlap_item_count"], f"{100 * row['baseline_item_recall']:.4f}",
                   f"{row['jaccard']:.6f}",
                   f"{row['successful_completions_on_baseline_items']} / "
                   f"{row['successful_completions_total']}",
                   f"{100 * row['fraction_later_successes_on_baseline_items']:.4f}"]
                  for row in overlap["comparisons"]]),
             "Baseline membership is defined by observed successes in only 16 draws, "
             "not a latent ability label. The item sets below refer to these same stored "
             "512 items, without resampling or additional evaluation.",
             "### Complete B-G update-0 solved-item list", table(
                 ["Manifest item index", "Task ID", "Successes / 16"],
                 [[row["item_index"], f"`{row['task_id']}`", row["successes_out_of_16"]]
                  for row in overlap["baseline_solved_items"]])]
    return "\n\n".join(text) + "\n"


def build() -> dict:
    path = RUN / "pilot-inputs/b_validation_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["by_id"] = {row["task_id"]: row for row in manifest["records"]}
    assert len(manifest["by_id"]) == manifest["record_count"] == 512
    series = {method: [summarize(method, index, manifest) for index in range(len(GRID))]
              for method in METHODS}
    return {
        "analysis_role": "post_hoc_descriptive_no_gate_authority", "run_id": RUN.name,
        "manifest_path": relative(path), "stage_b_updates": GRID,
        "definitions": "Pass@1 is exact-success count / completion count. Length-stop is "
        "`sampled_tokens >= sampling_max_tokens`, matching the archived profile reducer; "
        "the recorded stop_reason counts are also retained. Missing-tag is the recorded "
        "`missing_answer_tag` failure code; invalid-tag is `!valid_answer_tag`; syntactic "
        "invalidity is `!valid_syntax` (MAPS valid_program). These are distinct from all "
        "verification failures, which also include legal-but-wrong-target programs. "
        "Program-too-long concerns MAPS instruction count, not the 128-token cap. "
        "Length uses stored sampled_tokens, not retokenization. Definitions follow "
        "`src/duraseed/tasks/maps/verifier.py` and "
        "`src/duraseed/pilot0_profiles.py` without changing either.",
        "question": "Is B-G's flat phase (updates 1–40) dominated by cap/format failures "
        "that then resolve at the 40–80 takeoff?",
        "series": series, "baseline_concentration": baseline_overlap(series, manifest),
        "checks": {"evaluations": 22, "completion_reward_joins": 22 * 512 * 16,
                   "unique_item_draws_per_evaluation": 8192, "shared_manifest_items": 512,
                   "manifest_ids_verified": True, "generation_reward_values_verified": True},
    }


def main() -> None:
    data = build()
    (OUT / "f2_diagnostics.json").write_text(json.dumps(data, indent=2) + "\n")
    document = markdown(data)
    (OUT / "f2_diagnostics.md").write_text(document)
    print(document.split("### Complete B-G update-0 solved-item list")[0])


if __name__ == "__main__":
    main()
