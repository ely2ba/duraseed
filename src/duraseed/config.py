"""Small configuration surface for the carried instrument and next two gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class TCESConfig(_Section):
    n_operands: int
    operand_min: int
    operand_max: int
    require_distinct_operands: bool
    target_min: int
    target_max: int
    require_integer_target: bool
    allowed_ops: tuple[str, ...]
    allow_fractional_intermediates: bool
    max_abs_intermediate: int
    max_denominator: int
    max_tree_depth: int
    max_ast_nodes: int
    max_answer_length: int
    exclude_target_in_operands: bool
    exclude_trivial_identity_steps: bool
    require_positive_intermediates: bool
    min_valid_families: int
    max_valid_families: int | None
    min_valid_expressions: int
    max_valid_expressions: int | None
    ood_n_operands: int
    max_attempts: int

    def generator_kwargs(self, *, ood: bool = False) -> dict[str, Any]:
        values = self.model_dump(exclude={"ood_n_operands"})
        if ood:
            values["n_operands"] = self.ood_n_operands
        return values


class MAPSConfig(_Section):
    moduli: tuple[int, ...]
    add_constants: tuple[int, ...]
    mul_constants: tuple[int, ...]
    include_neg: bool
    allowed_instruction_count: int
    latent_program_min_length: int
    latent_program_max_length: int
    min_shortest_length: int
    max_shortest_length: int
    max_shortest_programs: int
    max_shortest_families: int
    min_latent_distractors: int
    require_inverse_pair: bool
    blocked_shortest_family_ids: tuple[str, ...]
    max_attempts: int

    def generator_kwargs(self) -> dict[str, Any]:
        return self.model_dump()


class TaskConfig(_Section):
    tces: TCESConfig
    maps: MAPSConfig


class StatisticsConfig(_Section):
    calibration_seeds: tuple[int, ...]


class BoundaryScanConfig(_Section):
    broad_samples_per_item: int
    refinement_samples_per_item_min: int
    refinement_samples_per_item_max: int


class TeacherDoseConfig(_Section):
    demonstrations_per_family: tuple[int, ...]
    selected_demonstrations_per_family: int | None
    calibration_batch_size: int
    calibration_updates: int


class StageAConfig(_Section):
    provisional_checkpoint_updates: tuple[int, ...]
    selected_max_updates: int | None
    provisional_max_tokens: int
    selected_max_tokens: int | None
    selected_rl_configuration: dict[str, Any] | None
    entropy_collapse_gate_passed: bool


class StageBConfig(_Section):
    selected_profile: str
    selected_max_updates: int


class LearningRateChoice(_Section):
    grid: tuple[float, ...]
    selected: float | None


class LearningRatesConfig(_Section):
    format_warmup: LearningRateChoice
    teacher_seed_sft: LearningRateChoice
    static_sft: LearningRateChoice
    on_policy_sft: LearningRateChoice
    group_relative_rl: LearningRateChoice
    stage_b_sft: LearningRateChoice


class SpendCapsConfig(_Section):
    grant_usd: float
    confirmatory_max_uncommitted_fraction: float
    per_run_usd: float | None
    total_pilot_usd: float | None


class TinkerConfig(_Section):
    model_id: str
    renderer_name: str
    lora_rank: int
    project_id_env: str
    max_sampled_tokens: int
    group_size: int
    groups_per_batch: int
    learning_rates: LearningRatesConfig
    spend_caps: SpendCapsConfig


class PilotConfig(_Section):
    seed: int
    tasks: TaskConfig
    statistics: StatisticsConfig
    boundary_scan: BoundaryScanConfig
    teacher_dose: TeacherDoseConfig
    stage_a: StageAConfig
    stage_b: StageBConfig
    tinker: TinkerConfig

    def unresolved_values(self) -> tuple[str, ...]:
        unresolved: list[str] = []
        if self.teacher_dose.selected_demonstrations_per_family is None:
            unresolved.append("teacher_dose.selected_demonstrations_per_family")
        if self.stage_a.selected_max_updates is None:
            unresolved.append("stage_a.selected_max_updates")
        if self.stage_a.selected_max_tokens is None:
            unresolved.append("stage_a.selected_max_tokens")
        if self.stage_a.selected_rl_configuration is None:
            unresolved.append("stage_a.selected_rl_configuration")
        if not self.stage_a.entropy_collapse_gate_passed:
            unresolved.append("stage_a.entropy_collapse_gate_passed")
        for name in (
            "teacher_seed_sft",
            "static_sft",
            "on_policy_sft",
            "group_relative_rl",
        ):
            if getattr(self.tinker.learning_rates, name).selected is None:
                unresolved.append(f"tinker.learning_rates.{name}.selected")
        return tuple(unresolved)

    def canonical_resolved_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()

    def resolved_config_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_resolved_bytes()).hexdigest()


class PilotConfigError(ValueError):
    pass


class UnresolvedPilotSelectionError(PilotConfigError):
    pass


def load_pilot_config(
    path: str | Path, *, require_launch_ready: bool = False
) -> PilotConfig:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = PilotConfig.model_validate(payload)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        raise PilotConfigError(f"invalid pilot configuration: {path}") from error
    unresolved = config.unresolved_values()
    if require_launch_ready and unresolved:
        raise UnresolvedPilotSelectionError(
            "unresolved pilot selections: " + ", ".join(unresolved)
        )
    return config


__all__ = [
    "PilotConfig",
    "PilotConfigError",
    "UnresolvedPilotSelectionError",
    "load_pilot_config",
]
