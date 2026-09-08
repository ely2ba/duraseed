"""Prospectively fixed, local-only estimands for replay and the Pilot audit."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256

GRID = (0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480)
THRESHOLDS = (0.06, 0.07, 0.10, 0.20)
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_NAMESPACE = "duraseed-replay-v1|pilot-audit|"


def bootstrap_seed(contrast: str, *, namespace=BOOTSTRAP_NAMESPACE) -> int:
    """Frozen before calculation: first eight SHA256 bytes, unsigned big endian."""
    return int.from_bytes(sha256((namespace + contrast).encode()).digest()[:8], "big")


def auc(values, grid=GRID):
    import numpy as np

    v, t = np.asarray(values), np.asarray(grid)
    if v.shape[-1] != len(t) or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("AUC requires a complete increasing grid")
    return np.sum(np.diff(t) * (v[..., :-1] + v[..., 1:]) / 2, axis=-1) / (t[-1] - t[0])


def half_life(values, grid=GRID):
    if values[0] <= 0:
        return {"status": "undefined_zero_baseline", "update": None}
    target = values[0] / 2
    for i in range(1, len(grid)):
        if values[i] <= target < values[i - 1]:
            f = (values[i - 1] - target) / (values[i - 1] - values[i])
            return {
                "status": "crossed",
                "update": float(grid[i - 1] + f * (grid[i] - grid[i - 1])),
                "bracket": list(grid[i - 1 : i + 1]),
            }
    return {"status": "not_reached", "update": None, "right_censored_after": grid[-1]}


def attainment(maps, retention, threshold, grid=GRID):
    """First upward crossing; baseline attainment is not matched learning."""
    if len(maps) != len(grid) or len(retention) != len(grid):
        raise ValueError("attainment requires aligned complete curves")
    if maps[0] >= threshold:
        return {
            "status": "baseline_exceeded",
            "update": 0,
            "retention": float(retention[0]),
            "bracket": [0, 0],
        }
    for i in range(1, len(grid)):
        if maps[i - 1] < threshold <= maps[i]:
            f = (threshold - maps[i - 1]) / (maps[i] - maps[i - 1])
            return {
                "status": "crossed",
                "update": float(grid[i - 1] + f * (grid[i] - grid[i - 1])),
                "retention": float(
                    retention[i - 1] + f * (retention[i] - retention[i - 1])
                ),
                "bracket": list(grid[i - 1 : i + 1]),
            }
    return {"status": "not_reached", "update": None, "retention": None, "bracket": None}


def paired_interval(
    values, contrast, *, replicates=BOOTSTRAP_REPLICATES, namespace=BOOTSTRAP_NAMESPACE
):
    """Values are item-level paired trajectory functionals, never single draws."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all():
        raise ValueError("paired bootstrap requires finite item functionals")
    seed = bootstrap_seed(contrast, namespace=namespace)
    rng = np.random.default_rng(seed)
    boots = np.empty(replicates)
    for begin in range(0, replicates, 250):
        end = min(begin + 250, replicates)
        indices = rng.integers(len(v), size=(end - begin, len(v)))
        boots[begin:end] = v[indices].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975], method="linear")
    return {
        "contrast": contrast,
        "estimate": float(v.mean()),
        "ci95": [float(lo), float(hi)],
        "paired_items": len(v),
        "seed": seed,
        "replicates": replicates,
        "undefined_replicates": 0,
        "method": "NumPy PCG64 paired item-cluster trajectory percentile; pointwise descriptive",
    }


def failure_summary(records):
    """Disjoint authoritative outcomes and separately overlapping indicators.

    Each compact record contains the original verifier result and recorded cap.
    It is never repaired, reparsed, or silently reclassified as correct.
    """
    codes, overlap, grouped = Counter(), Counter(), Counter()
    lengths = []
    for row in records:
        check = row["verification"]
        correct = check["reward"] == 1.0
        code = check["failure_code"]
        if check["reward"] not in (0, 1) or (code is None) != correct:
            raise ValueError("authoritative correctness and failure code disagree")
        codes["correct" if correct else code] += 1
        cap = row["sampled_tokens"] >= row["sampling_max_tokens"]
        bad_tag, bad_syntax = not check["valid_answer_tag"], not check["valid_syntax"]
        parse_failure = bad_tag or bad_syntax or not check["valid_lexing"]
        group = (
            "correct"
            if correct
            else "output_contract_or_parse"
            if parse_failure
            else "executable_wrong_target"
            if code == "wrong_target"
            else "other_well_formed_task_failure"
        )
        grouped[group] += 1
        overlap.update(
            {
                "cap_reached": cap,
                "invalid_tag": bad_tag,
                "invalid_syntax": bad_syntax,
                "cap_or_parse_failure": not correct and (cap or parse_failure),
                "output_valid": not parse_failure,
                "output_valid_correct": correct and not parse_failure,
            }
        )
        lengths.append(row["sampled_tokens"])
    n = sum(codes.values())
    if not n:
        raise ValueError("no authoritative completion records")
    from statistics import mean, median

    valid = overlap["output_valid"]
    return {
        "completions": n,
        "unconditional_success": codes["correct"] / n,
        "disjoint_authoritative_codes": dict(sorted(codes.items())),
        "disjoint_groups": dict(sorted(grouped.items())),
        "overlapping_indicators": dict(sorted(overlap.items())),
        "conditional_output_valid": {
            "successes": overlap["output_valid_correct"],
            "denominator": valid,
            "accuracy": overlap["output_valid_correct"] / valid if valid else None,
        },
        "tokens_mean": mean(lengths),
        "tokens_median": median(lengths),
    }
