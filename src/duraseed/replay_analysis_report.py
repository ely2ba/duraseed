"""Small descriptive Pilot-audit package; original evidence is read-only."""

from __future__ import annotations

import json


def fmt(value):
    return "undefined / unavailable" if value is None else f"{value:.8f}"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def markdown(data):
    lines = [
        "# Pilot-0 existing-data audit",
        "",
        "Descriptive, post-hoc analysis of the two completed pairs. No new sampling, training, verifier relaxation, or gate changes. Original Pilot reports and uncertainty settings remain intact.",
        "",
        "## Reproduce before extending",
        "",
        table(
            [
                "Pair",
                "Stored F1/F2 cells",
                "Original intervals",
                "Maximum interval discrepancy",
            ],
            [
                [
                    r["pair"],
                    "exact",
                    r["old_intervals_reproduced"],
                    r["max_absolute_interval_discrepancy"],
                ]
                for r in data["reproduction"]
            ],
        ),
        "",
        "All new contrasts below are **B-G minus B-S**. The reproduced historical intervals in `summary.json` retain their original **B-S minus B-G** direction. Raw Pass@1 is unconditional exact successes divided by generated completions; all failures stay in the denominator. Posterior scores are item-averaged Jeffreys means and are kept separately for provenance.",
        "",
        "New pointwise 95% intervals use 50,000 paired item-trajectory resamples. NumPy PCG64 receives the unsigned big-endian integer from the first eight SHA256 digest bytes of `duraseed-replay-v1|pilot-audit|<contrast>`. Contrast names and resulting seeds are retained in JSON. Each item retains both arms, its whole trajectory, and all realized within-item draws. The conditional intervals do not include training-run, checkpoint-selection, or fresh-completion uncertainty; no pooling across the two source blocks is performed. No optional family-block sensitivity was added.",
        "",
        "AUC integrates the linear checkpoint grid and divides by the window width. The first-attainment thresholds 0.06, 0.07, 0.10, 0.20 are post-hoc sensitivities for Pilot 0, fixed before this calculation. Crossing and half-life summaries below are descriptive point estimates, not fitted decay constants or observed intermediate checkpoints; no bootstrap non-crossings were discarded to form an interval.",
        "",
        "![Absolute and own-baseline-relative learning](learning.svg)",
        "",
        "![Retention and chronological downstream-performance paths](retention.svg)",
        "",
        "![Disjoint early verifier outcomes](failures.svg)",
        "",
    ]
    for pair in data["pairs"]:
        lines += [
            f"## Pair {pair['pair']}",
            "",
            "### Absolute performance, gain, and endpoint",
            "",
            table(
                [
                    "Arm",
                    "Baseline",
                    "Absolute AUC 0–40",
                    "Gain AUC 0–40",
                    "Absolute AUC 0–480",
                    "Gain AUC 0–480",
                    "Endpoint 480",
                ],
                [
                    [
                        m,
                        fmt(s["maps_baseline"]),
                        fmt(s["maps_absolute_auc"]["40"]),
                        fmt(s["maps_gain_auc"]["40"]),
                        fmt(s["maps_absolute_auc"]["480"]),
                        fmt(s["maps_gain_auc"]["480"]),
                        fmt(s["maps_endpoint480"]),
                    ]
                    for m, s in pair["summaries"].items()
                ],
            ),
            "",
            "Gain contrast = absolute-AUC contrast − baseline contrast:",
            "",
            table(
                [
                    "Window",
                    "Absolute difference",
                    "Baseline difference",
                    "Gain difference",
                    "Numerical residual",
                ],
                [
                    [
                        str(d["window"]),
                        fmt(d["absolute_difference"]),
                        fmt(d["baseline_difference"]),
                        fmt(d["gain_difference"]),
                        f"{d['identity_residual']:.2g}",
                    ]
                    for d in pair["decomposition"]
                ],
            ),
            "",
            "### Unconditional retention",
            "",
        ]
        retention = []
        for method, summary in pair["summaries"].items():
            for role, r in summary["retention"].items():
                half = r["half_life"]
                retention.append(
                    [
                        method,
                        role,
                        fmt(r["baseline"]),
                        fmt(r["absolute_auc_0_20"]),
                        fmt(r["relative_auc_0_20"]),
                        fmt(half["update"]),
                        str(half.get("bracket", half["status"])),
                    ]
                )
        lines += [
            table(
                [
                    "Arm",
                    "Role",
                    "Raw baseline",
                    "Absolute AUC 0–20",
                    "Own-baseline-relative AUC",
                    "Half-life",
                    "Bracket/status",
                ],
                retention,
            ),
            "",
            "These scores are total observed pre-B performance, not survival restricted to newly acquired items.",
            "",
            "### Paired item uncertainty",
            "",
            table(
                ["Contrast", "Estimate", "95% lower", "95% upper", "Items"],
                [
                    [
                        r["contrast"],
                        fmt(r["estimate"]),
                        fmt(r["ci95"][0]),
                        fmt(r["ci95"][1]),
                        r["paired_items"],
                    ]
                    for r in pair["contrasts"]
                ],
            ),
            "",
            "### First attainment at comparable absolute MAPS performance",
            "",
        ]
        crossing = []
        for row in pair["attainment"]:
            for method, c in row["arms"].items():
                crossing.append(
                    [
                        row["threshold"],
                        method,
                        c["status"],
                        fmt(c["update"]),
                        str(c["bracket"]),
                        fmt(c["retention"]),
                        fmt(row["retention_difference_B_G_minus_B_S"])
                        if method == "B-G"
                        else "—",
                    ]
                )
        lines += [
            table(
                [
                    "Threshold",
                    "Arm",
                    "Status",
                    "Interpolated update",
                    "Observed bracket",
                    "Targeted TCES",
                    "Eligible B-G−B-S",
                ],
                crossing,
            ),
            "",
            "`baseline_exceeded` is not equivalent learning progress. A two-arm post-training difference is printed only when both arms have a first crossing from below. Interpolation uses the same adjacent interval for update and retention and never extrapolates or erases a reversal.",
            "",
            "### Full raw and posterior trajectories",
            "",
        ]
        trajectory = []
        for role, arms in pair["curves"].items():
            for method, curve in arms.items():
                for step, raw, posterior in zip(
                    pair["grid"], curve["raw"], curve["posterior"]
                ):
                    trajectory.append(
                        [
                            role,
                            method,
                            step,
                            fmt(raw),
                            fmt(posterior),
                            fmt(raw - curve["raw"][0]) if role == "maps" else "—",
                        ]
                    )
        lines += [
            table(
                [
                    "Panel/role",
                    "Arm",
                    "Update",
                    "Raw Pass@1",
                    "Jeffreys posterior",
                    "MAPS raw gain",
                ],
                trajectory,
            ),
            "",
            "### Failure decomposition",
            "",
            "Authoritative codes and grouped outcomes are mutually exclusive. Overlapping cap/tag/syntax indicators are printed separately and must not be added as disjoint failures. `output_contract_or_parse` uses the original invalid-tag/lexing/syntax flags; `executable_wrong_target` is the original `wrong_target` code; other well-formed task failures include operand/operation/arithmetic constraints. Conditional output-valid accuracy is shown with its denominator and unconditional score. It does not establish that an invalid output would have been correct.",
            "",
        ]
        failure_rows = []
        code_rows = []
        for r in pair["failures"]:
            valid, overlap = r["conditional_output_valid"], r["overlapping_indicators"]
            failure_rows.append(
                [
                    r["role"],
                    r["method"],
                    r["update"],
                    r["completions"],
                    fmt(r["unconditional_success"]),
                    f"{valid['successes']}/{valid['denominator']}",
                    fmt(valid["accuracy"]),
                    overlap["cap_reached"],
                    overlap["invalid_tag"],
                    overlap["invalid_syntax"],
                    overlap["cap_or_parse_failure"],
                    fmt(r["tokens_mean"]),
                    fmt(r["tokens_median"]),
                ]
            )
            code_rows.append(
                [
                    r["role"],
                    r["method"],
                    r["update"],
                    json.dumps(r["disjoint_authoritative_codes"], sort_keys=True),
                ]
            )
        lines += [
            table(
                [
                    "Role",
                    "Arm",
                    "Update",
                    "N",
                    "Unconditional",
                    "Valid successes/N",
                    "Conditional",
                    "Cap",
                    "Bad tag",
                    "Bad syntax",
                    "Failure union",
                    "Mean tokens",
                    "Median",
                ],
                failure_rows,
            ),
            "",
            table(
                [
                    "Role",
                    "Arm",
                    "Update",
                    "Disjoint authoritative codes (including correct)",
                ],
                code_rows,
            ),
            "",
        ]
    lines += [
        "## Audit conclusion",
        "",
        audit_conclusion(data),
        "",
        "## Reproduction and exclusions",
        "",
        "`per-item-counts.jsonl` retains all item trajectories and their realized draw rewards in sample-index order. `figure-data.json` is the exact data used for the figures; `summary.json` retains numerical definitions, old-interval reproduction, new seeds, and all failure denominators. The private source index is saved separately under ignored `runs/replay-v1/`; raw generations and verifier decisions remain in the immutable original run directories. No sealed tests, new completions, or alternative verifier were used.",
        "",
        "Rebuild into a new destination from the repository root:",
        "",
        "```sh",
        "PYTHONPATH=src uv run python -m duraseed.replay_analysis_pilot --output artifacts/replay-v1/pilot-audit-reproduction",
        "```",
        "",
        "Data unavailable: "
        + (
            "none for the requested Pilot item counts and raw early-task failure decomposition."
            if not data["unavailable"]
            else str(data["unavailable"])
        ),
        "",
    ]
    return "\n".join(lines)


def audit_conclusion(data):
    parts = []
    for pair in data["pairs"]:
        s, g = pair["summaries"]["B-S"], pair["summaries"]["B-G"]
        retention = (
            g["retention"]["targeted"]["absolute_auc_0_20"]
            - s["retention"]["targeted"]["absolute_auc_0_20"]
        )
        d40, d480 = pair["decomposition"]
        f = {
            r["method"]: r
            for r in pair["failures"]
            if r["role"] == "targeted" and r["update"] == 2
        }
        parts.append(
            f"Pair {pair['pair']}: B-G−B-S targeted raw retention AUC 0–20 is {retention:.8f}; MAPS absolute AUC 0–480 difference is {d480['absolute_difference']:.8f}, whereas its gain-AUC difference is {d480['gain_difference']:.8f}. In the early 0–40 window, absolute and gain differences are {d40['absolute_difference']:.8f} and {d40['gain_difference']:.8f}, with baseline difference {d40['baseline_difference']:.8f}. At update 2, output-valid targeted success is B-S {f['B-S']['conditional_output_valid']['successes']}/{f['B-S']['conditional_output_valid']['denominator']} and B-G {f['B-G']['conditional_output_valid']['successes']}/{f['B-G']['conditional_output_valid']['denominator']}; invalid outputs remain failures in the primary raw scores."
        )
        start = next(
            r["conditional_output_valid"]
            for r in pair["failures"]
            if r["method"] == "B-S" and r["role"] == "targeted" and r["update"] == 0
        )
        now = f["B-S"]["conditional_output_valid"]
        parts.append(
            f"Pair {pair['pair']}: B-S output-valid targeted completions increase from {start['denominator']}/768 at baseline to {now['denominator']}/768 at update 2, while verified successes decrease from {start['successes']} to {now['successes']}. Thus this early unconditional decline is not solely an increase in output-format failures. Later failure composition differs by arm, as retained in the full code table; this does not imply preservation or loss of an unobserved latent capability."
        )
    parts.append(
        "What survives: B-G has the larger unconditional targeted retention AUC over 0–20 and the longer targeted half-life in both pairs. What weakens: the early B-S MAPS gain advantage is much smaller on absolute performance in Pair 1, and reverses direction on absolute performance in Pair 2. The 0–480 absolute-AUC contrast has opposite signs across pairs. At first attainment of 0.10 or 0.20 absolute MAPS success, targeted retention is already near zero in both pairs; all these rows remain in the table. Larger gain from a lower starting point is not an absolute-performance advantage, and these observations do not establish a uniformly better retention–learning tradeoff."
    )
    parts.append(
        "Task-validity conditioning changes the denominator and cannot identify hidden arithmetic capability or repair invalid responses. The planned controlled replay can compare the bundled trace-source intervention under a shared supervised acquisition recipe and shared prompts; it cannot isolate an RL objective effect, separate trace content from length/format/strategy, or establish general plasticity from two reused source blocks."
    )
    return "\n\n".join(parts)
