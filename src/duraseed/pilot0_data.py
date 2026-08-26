"""Authenticated teacher and fixed data schedules for Pilot 0."""

from __future__ import annotations

from collections import Counter, defaultdict

from duraseed.data.manifests import MAPSTaskManifestRecord, TCESTaskManifestRecord
from duraseed.data.stage_a_prompt_pools import PromptPoolStratum
from duraseed.pilot0_contract import PilotSeedSources
from duraseed.runners import RunnerGateError
from duraseed.tasks.tces import generate_teacher_trace, strategy_family_id
from duraseed.tasks.tces.ast import (
    BinaryExpression,
    BinaryOperator,
    Expression,
    IntegerLiteral,
)
from duraseed.training.capability_dose_evidence import EPOCH_UPDATES
from duraseed.training.sft import (
    VerifiedSourceRecord,
    build_solver_teacher_record,
    build_stage_b_maps_record,
)


_FAMILY_OPERATORS = {
    "ADD": BinaryOperator.ADD,
    "SUB": BinaryOperator.SUB,
    "MUL": BinaryOperator.MUL,
    "DIV": BinaryOperator.DIV,
}


def _family_expression(record: TCESTaskManifestRecord) -> Expression:
    """Rebuild the manifest's frozen representative without exhaustive search."""

    structure = record.intended_family.split("|", maxsplit=1)[0]
    ranked_operands = tuple(
        value
        for _, value in sorted(
            enumerate(record.operands), key=lambda item: (item[1], item[0])
        )
    )

    def parse(position: int) -> tuple[Expression, int]:
        if position < len(structure) and structure[position] == "r":
            end = position + 1
            while end < len(structure) and structure[end].isdigit():
                end += 1
            try:
                rank = int(structure[position + 1 : end])
                if rank < 1:
                    raise ValueError("rank must be positive")
                return IntegerLiteral(ranked_operands[rank - 1]), end
            except (IndexError, ValueError) as error:
                raise RunnerGateError(
                    "Pilot-0 solver family has an invalid rank"
                ) from error
        operator_name = next(
            (
                name
                for name in _FAMILY_OPERATORS
                if structure.startswith(f"{name}(", position)
            ),
            None,
        )
        if operator_name is None:
            raise RunnerGateError("Pilot-0 solver family has an invalid node")
        left, cursor = parse(position + len(operator_name) + 1)
        if cursor >= len(structure) or structure[cursor] != ",":
            raise RunnerGateError("Pilot-0 solver family lacks a child separator")
        right, cursor = parse(cursor + 1)
        if cursor >= len(structure) or structure[cursor] != ")":
            raise RunnerGateError("Pilot-0 solver family lacks a closing parenthesis")
        return BinaryExpression(
            _FAMILY_OPERATORS[operator_name], left, right
        ), cursor + 1

    expression, end = parse(0)
    if end != len(structure) or strategy_family_id(expression, record.operands) != (
        record.intended_family
    ):
        raise RunnerGateError("Pilot-0 solver family reconstruction changed identity")
    return expression


def _tces_completion(record: TCESTaskManifestRecord) -> str:
    expression = _family_expression(record)
    return generate_teacher_trace(expression)


_SOLVER_CACHE: dict[str, dict[str, VerifiedSourceRecord]] = {}


def stage_a_solver_sources(source: PilotSeedSources) -> dict[str, VerifiedSourceRecord]:
    cached = _SOLVER_CACHE.get(source.prompt_pools.a_rl_train_manifest.manifest_id)
    if cached is not None:
        return cached
    pools = ordered_stage_a_pools(source)
    selected_ids = {
        row.task_id
        for step in range(1, EPOCH_UPDATES + 1)
        for row in scheduled_stage_a_records(
            pools, source.prompt_pools.artifact.bs_slot_order, step
        )
    }
    if len(selected_ids) != 1_552:
        raise RunnerGateError("Pilot-0 B-S epoch is not the frozen 1,552 traces")
    selected_records = []
    for record in source.prompt_pools.a_rl_train_manifest.records:
        if not isinstance(record, TCESTaskManifestRecord):
            raise RunnerGateError("Pilot-0 Stage-A training manifest is not TCES")
        if record.task_id in selected_ids:
            selected_records.append(record)
    result = {
        record.task_id: build_solver_teacher_record(
            source_manifest=source.prompt_pools.a_rl_train_manifest,
            source_record=record,
            completion=_tces_completion(record),
        )
        for record in selected_records
    }
    if set(result) != selected_ids:
        raise RunnerGateError("Pilot-0 B-S schedule omits a solver source")
    _SOLVER_CACHE[source.prompt_pools.a_rl_train_manifest.manifest_id] = result
    return result


def ordered_stage_a_pools(
    source: PilotSeedSources,
) -> dict[PromptPoolStratum, tuple[TCESTaskManifestRecord, ...]]:
    artifact = source.prompt_pools.artifact
    by_stratum = {
        PromptPoolStratum.BOUNDARY: artifact.boundary_family_ids,
        PromptPoolStratum.INTERMEDIATE: artifact.intermediate_family_ids,
        PromptPoolStratum.BROAD_RANDOM: artifact.broad_random_family_ids,
    }
    result = {}
    for stratum, families in by_stratum.items():
        by_family = {
            family: tuple(
                sorted(
                    (
                        row
                        for row in source.prompt_pools.a_rl_train_manifest.records
                        if isinstance(row, TCESTaskManifestRecord)
                        and row.intended_family == family
                    ),
                    key=lambda row: row.task_id,
                )
            )
            for family in families
        }
        if {len(rows) for rows in by_family.values()} != {64}:
            raise RunnerGateError("Pilot-0 Stage-A pool changed its 64-item strata")
        result[stratum] = tuple(
            by_family[family][item] for item in range(64) for family in families
        )
    return result


def scheduled_stage_a_records(
    pools: dict[PromptPoolStratum, tuple[TCESTaskManifestRecord, ...]],
    order: tuple[PromptPoolStratum, ...],
    step: int,
) -> tuple[TCESTaskManifestRecord, ...]:
    counts: dict[PromptPoolStratum, int] = defaultdict(int)
    per_step = Counter(order)
    selected = []
    for stratum in order:
        offset = (step - 1) * per_step[stratum] + counts[stratum]
        selected.append(pools[stratum][offset % len(pools[stratum])])
        counts[stratum] += 1
    return tuple(selected)


def stage_b_sources(source: PilotSeedSources) -> tuple[VerifiedSourceRecord, ...]:
    values = []
    for record in source.b_train.records:
        if not isinstance(record, MAPSTaskManifestRecord):
            raise RunnerGateError("Pilot-0 Stage-B training manifest is not MAPS")
        values.append(
            build_stage_b_maps_record(
                source_manifest=source.b_train,
                source_record=record,
                completion=f"<answer>{record.shortest_programs[0]}</answer>",
            )
        )
    return tuple(values)


__all__ = [
    "ordered_stage_a_pools",
    "scheduled_stage_a_records",
    "stage_a_solver_sources",
    "stage_b_sources",
]
