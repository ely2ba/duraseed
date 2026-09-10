"""Prospective endpoint-clone matching and descriptive repeat-distance analysis."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations


ROLES = ("targeted", "sentinel")
STUDENT_GRID = (*range(10, 291, 10), 294)
RUNS = ("T1", "T2", "S1", "S2")
PASS_TOLERANCES = {"targeted": Fraction(3, 100), "sentinel": Fraction(5, 100)}


def _items(value: dict, *, count: int, draws: int) -> dict:
    rows = value.get("items", value.get("item_counts", []))
    grouped = {role: {} for role in ROLES}
    for row in rows:
        role, task = row["panel_role"], row["task_id"]
        successes, trials = row["successes"], row["trials"]
        if (
            role not in grouped
            or task in grouped[role]
            or type(successes) is not int
            or type(trials) is not int
            or trials != draws
            or not 0 <= successes <= trials
        ):
            raise ValueError("endpoint comparison requires complete raw item counts")
        grouped[role][task] = Fraction(successes, trials)
    if any(len(rows) != count for rows in grouped.values()):
        raise ValueError("endpoint comparison has an incomplete role panel")
    if set(grouped["targeted"]) & set(grouped["sentinel"]):
        raise ValueError("one endpoint item cannot have two roles")
    return grouped


def _mean(values) -> Fraction:
    values = tuple(values)
    return sum(values, Fraction()) / len(values)


def _distance(left: dict, right: dict) -> dict:
    if any(set(left[role]) != set(right[role]) for role in ROLES):
        raise ValueError("paired comparison changed its evaluation items")
    return {
        role: _mean(abs(value - right[role][task]) for task, value in rows.items())
        for role, rows in left.items()
    }


def nominate(cadence: list[dict], teacher: dict) -> dict:
    """Rank three of all 30 completed cadences by both-role raw accuracy."""
    if sorted(row["update"] for row in cadence) != list(STUDENT_GRID):
        raise ValueError("nomination requires all 30 frozen student cadences")
    reference = _items(teacher, count=96, draws=1)
    target = {role: _mean(rows.values()) for role, rows in reference.items()}
    ranked = []
    for row in cadence:
        items = _items(row, count=96, draws=1)
        _distance(items, reference)
        scores = {role: _mean(items[role].values()) for role in ROLES}
        distance = sum(
            abs(scores[role] - target[role]) / PASS_TOLERANCES[role] for role in ROLES
        )
        ranked.append(
            {
                "update": row["update"],
                "nomination_distance": str(distance),
                "raw_pass_at_1": {role: float(scores[role]) for role in ROLES},
            }
        )
    ranked.sort(key=lambda row: (Fraction(row["nomination_distance"]), row["update"]))
    return {
        "teacher_update": 30,
        "teacher_raw_pass_at_1": {role: float(target[role]) for role in ROLES},
        "nominees": ranked[:3],
        "cadence": ranked,
    }


def _profile_metrics(profile: dict, items: dict, role: str) -> dict:
    panel = profile["panels"][role]
    if panel["item_count"] != 256 or panel["completion_count"] != 4096:
        raise ValueError("clone profiles require 256 items and 16 draws per role")
    # The existing profile exposes an invalid rate; recover its integer count.
    invalid = panel["syntactically_invalid_rate"] * panel["completion_count"]
    if abs(invalid - round(invalid)) > 1e-8 or not 0 <= invalid <= 4096:
        raise ValueError("profile validity does not represent observed counts")
    return {
        "raw_pass_at_1": _mean(items[role].values()),
        "coverage_at_16": Fraction(
            sum(value > 0 for value in items[role].values()), 256
        ),
        "median_length": Fraction(str(panel["completion_token_length"]["median"])),
        "validity": 1 - Fraction(round(invalid), 4096),
        "strategy_families": Fraction(panel["unique_verified_strategy_family_count"]),
    }


def profile_gates(student: dict, teacher: dict) -> dict:
    """Arithmetic gates only; recompute Pass@1 and coverage from raw counts."""
    si = _items(student, count=256, draws=16)
    ti = _items(teacher, count=256, draws=16)
    distances = _distance(si, ti)
    gates = []
    for role in ROLES:
        sm = _profile_metrics(student, si, role)
        tm = _profile_metrics(teacher, ti, role)
        for name in sm:
            s, t = sm[name], tm[name]
            ratio_metric = name in ("median_length", "strategy_families")
            if ratio_metric:
                passed = s == 0 if t == 0 else Fraction(2, 3) <= s / t <= Fraction(3, 2)
                limit = {
                    "ratio_lower": "2/3",
                    "ratio_upper": "3/2",
                    "zero_reference_requires_zero": True,
                }
            else:
                tolerance = (
                    PASS_TOLERANCES[role]
                    if name == "raw_pass_at_1"
                    else Fraction(1, 10)
                )
                passed = abs(s - t) <= tolerance
                limit = {"absolute_tolerance": str(tolerance)}
            gates.append(
                {
                    "role": role,
                    "metric": name,
                    "student": float(s),
                    "teacher": float(t),
                    "passed": passed,
                    **limit,
                }
            )
    return {
        "passed": all(row["passed"] for row in gates),
        "gates": gates,
        "per_role_item_distance": {
            role: float(value) for role, value in distances.items()
        },
        "equal_role_item_distance": str(_mean(distances.values())),
    }


def select(nomination: dict, teacher: dict, candidates: dict[int, dict]) -> dict:
    """Select one passing nominee; selection itself never authorizes Stage B."""
    nominees = nomination["nominees"]
    if (
        nomination["teacher_update"] != 30
        or len(nominees) != 3
        or len({row["update"] for row in nominees}) != 3
        or set(candidates) != {row["update"] for row in nominees}
    ):
        raise ValueError("selection must assess exactly the three frozen nominees")
    assessments = []
    for row in nominees:
        assessment = profile_gates(candidates[row["update"]], teacher)
        assessments.append({**row, **assessment})
    passing = [row for row in assessments if row["passed"]]
    passing.sort(
        key=lambda row: (
            Fraction(row["equal_role_item_distance"]),
            Fraction(row["nomination_distance"]),
            row["update"],
        )
    )
    return {
        "schema_version": "duraseed-endpoint-clone-matching-v1",
        "status": "SELECTED" if passing else "NO_MATCH",
        "teacher_update": 30,
        "assessments": assessments,
        "selection_item_ids": {
            role: sorted(rows)
            for role, rows in _items(teacher, count=256, draws=16).items()
        },
        "selected_update": passing[0]["update"] if passing else None,
        "stage_b_allowed": False,
    }


def confirm(
    selection: dict, teacher_first: dict, teacher_repeat: dict, student: dict
) -> dict:
    """One confirmation of the irrevocable selection; failure has no fallback."""
    if selection["status"] != "SELECTED" or selection["selected_update"] is None:
        raise ValueError("confirmation requires one irrevocably selected candidate")
    if (
        student.get("stage_a_update", selection["selected_update"])
        != selection["selected_update"]
    ):
        raise ValueError("confirmation replaced the irrevocably selected checkpoint")
    gates = profile_gates(student, teacher_first)
    t1 = _items(teacher_first, count=256, draws=16)
    t2 = _items(teacher_repeat, count=256, draws=16)
    si = _items(student, count=256, draws=16)
    selection_ids = {
        task for rows in selection["selection_item_ids"].values() for task in rows
    }
    if selection_ids.intersection(task for rows in t1.values() for task in rows):
        raise ValueError("confirmation must be item-disjoint from clone selection")
    student_distance = _distance(si, t1)
    repeat_distance = _distance(t2, t1)
    agreement = []
    for role in ROLES:
        excess = student_distance[role] - repeat_distance[role]
        agreement.append(
            {
                "role": role,
                "student_teacher_distance": float(student_distance[role]),
                "teacher_repeat_distance": float(repeat_distance[role]),
                "excess_item_distance": float(excess),
                "absolute_tolerance": "3/100",
                "passed": excess <= Fraction(3, 100),
            }
        )
    passed = gates["passed"] and all(row["passed"] for row in agreement)
    return {
        **selection,
        "status": "MATCHED" if passed else "CONFIRMATION_FAILED",
        "confirmation": {**gates, "item_agreement": agreement, "passed": passed},
        "stage_b_allowed": passed,
    }


def trajectory_analysis(runs: dict[str, list[dict]]) -> dict:
    """Both roles' six dense absolute distances and repeat-adjusted separation."""
    if set(runs) != set(RUNS):
        raise ValueError("trajectory analysis requires T1, T2, S1, S2")
    curves = {role: {} for role in ROLES}
    reference = None
    for run in RUNS:
        rows = sorted(runs[run], key=lambda row: row["update"])
        if [row["update"] for row in rows] != list(range(21)):
            raise ValueError("primary trajectory requires every update from 0 to 20")
        values = []
        for row in rows:
            items = _items(row, count=192, draws=4)
            if reference is None:
                reference = items
            _distance(items, reference)
            values.append(items)
        for role in ROLES:
            curves[role][run] = [_mean(items[role].values()) for items in values]
    result = {}
    for role in ROLES:
        distances = {}
        for left, right in combinations(RUNS, 2):
            gaps = [
                abs(x - y)
                for x, y in zip(curves[role][left], curves[role][right], strict=True)
            ]
            distances[f"{left}:{right}"] = (
                sum((gaps[i] + gaps[i + 1]) / 2 for i in range(20)) / 20
            )
        cross = _mean(distances[f"{t}:{s}"] for t in ("T1", "T2") for s in ("S1", "S2"))
        repeat = (distances["T1:T2"] + distances["S1:S2"]) / 2
        result[role] = {
            "curves": {
                run: [float(value) for value in values]
                for run, values in curves[role].items()
            },
            "distances": {pair: float(value) for pair, value in distances.items()},
            "mean_cross_distance": float(cross),
            "mean_repeat_distance": float(repeat),
            "excess_separation": float(cross - repeat),
        }
    return {
        "schema_version": "duraseed-endpoint-trajectory-distance-v1",
        "grid": list(range(21)),
        "metric": "trapezoid_of_absolute_raw_pass_at_1_differences_divided_by_20",
        "analysis_role": "descriptive_repeatability_comparison_no_equivalence_verdict",
        "panels": result,
    }
