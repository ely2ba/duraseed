"""Deterministic split-capacity audit for confirmed TCES families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from duraseed.data.splits import (
    TCES_SPLIT_SPECS,
    derive_tces_split_seed,
    tces_numeric_key,
)
from duraseed.tasks.tces import (
    GeneratedTCESInstance,
    TCESFamilyGenerator,
    TCESGenerationError,
    TCESGeneratorConfig,
)


# These are the smallest per-family prefixes already required by the frozen
# calibration/evaluation design.  The Stage-A prompt-pool demand is checked
# again after its duration is selected; tasks may be sampled repeatedly.
PANEL_SPLIT_MINIMUMS: tuple[tuple[str, int], ...] = (
    ("a_monitor", 16),
    ("a_rl_train", 64),
    ("a_seed_gate", 8),
    ("a_seed_train", 16),
    ("a_validation", 22),
)
PANEL_SELECTED_TEST_SINGLE_MINIMUM = 22
PANEL_CAPACITY_PROBE_MULTIPLIER = 2
PANEL_FILTERED_SPLIT_SCAN_MULTIPLIER = 8


class PanelCapacityError(ValueError):
    """A family-capacity audit was malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class FamilySplitCapacity:
    """Bounded deterministic capacity for one family in one protected split."""

    split: str
    required_instances: int
    probe_indices: int
    available_disjoint_instances: int
    first_generation_failure_index: int | None
    passed: bool


@dataclass(frozen=True, slots=True)
class FamilyCapacityAudit:
    """Cross-split disjoint-instance evidence for one exact TCES family."""

    family_id: str
    split_capacities: tuple[FamilySplitCapacity, ...]
    available_disjoint_instances: int
    passed: bool

    @property
    def available_by_split(self) -> dict[str, int]:
        return {
            row.split: row.available_disjoint_instances for row in self.split_capacities
        }


def _validated_requirements(
    requirements: Mapping[str, int] | tuple[tuple[str, int], ...],
) -> dict[str, int]:
    values = dict(requirements)
    known = {spec.name for spec in TCES_SPLIT_SPECS}
    forbidden = {"a_candidate", "a_ood"}
    if (
        not values
        or set(values).difference(known)
        or set(values).intersection(forbidden)
    ):
        raise PanelCapacityError(
            "capacity requirements must use protected in-distribution TCES splits"
        )
    if any(
        not isinstance(name, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        for name, count in values.items()
    ):
        raise PanelCapacityError("capacity requirements must be positive integers")
    return values


def _family_config_for_split(
    base: TCESGeneratorConfig,
    split: str,
) -> TCESGeneratorConfig:
    # Panel membership is filtered from the completed enumeration below.
    # Doing it inside the latent generator would repeat that same expensive
    # enumeration up to max_attempts times for each independent item index.
    return replace(
        base,
        split=split,
        min_valid_families=1,
        max_valid_families=None,
    )


def _passes_split_filter(
    instance: GeneratedTCESInstance,
    split: str,
    protected_family_ids: frozenset[str],
) -> bool:
    if split == "a_test_single":
        return frozenset(instance.valid_family_ids).intersection(
            protected_family_ids
        ) == {instance.intended_family}
    return True


def audit_family_split_capacity(
    template: GeneratedTCESInstance,
    base_config: TCESGeneratorConfig,
    *,
    root_seed: int,
    requirements: Mapping[str, int] | tuple[tuple[str, int], ...] = (
        PANEL_SPLIT_MINIMUMS
    ),
    probe_multiplier: int = PANEL_CAPACITY_PROBE_MULTIPLIER,
    forbidden_records: Sequence[object] = (),
    protected_family_ids: Sequence[str] = (),
) -> FamilyCapacityAudit:
    """Probe fixed split prefixes while enforcing global numeric disjointness.

    The bounded probe is a pre-treatment matching covariate, not a claim that a
    family has infinitely many instances.  A generation failure before the
    required prefix is complete fails that split closed.
    """

    if not isinstance(template, GeneratedTCESInstance):
        raise TypeError("template must be a GeneratedTCESInstance")
    if not isinstance(base_config, TCESGeneratorConfig):
        raise TypeError("base_config must be a TCESGeneratorConfig")
    if (
        isinstance(probe_multiplier, bool)
        or not isinstance(probe_multiplier, int)
        or probe_multiplier < 1
    ):
        raise ValueError("probe_multiplier must be a positive integer")
    minimums = _validated_requirements(requirements)
    protected = frozenset(protected_family_ids)
    if "a_test_single" in minimums and (
        template.intended_family not in protected
        or any(
            not isinstance(family_id, str) or not family_id for family_id in protected
        )
    ):
        raise PanelCapacityError(
            "a_test_single capacity requires the complete protected family set"
        )
    forbidden = tuple(forbidden_records)
    used_numeric: set[object] = {tces_numeric_key(record) for record in forbidden}
    used_content: set[str] = set()
    for record in forbidden:
        content_hash = getattr(record, "content_hash", None)
        if not isinstance(content_hash, str) or not content_hash:
            raise TypeError("forbidden records must expose a content_hash")
        used_content.add(content_hash)
    rows: list[FamilySplitCapacity] = []
    for spec in TCES_SPLIT_SPECS:
        required = minimums.get(spec.name)
        if required is None:
            continue
        probe_indices = required * (
            PANEL_FILTERED_SPLIT_SCAN_MULTIPLIER
            if spec.name == "a_test_single"
            else probe_multiplier
        )
        generator = TCESFamilyGenerator(
            derive_tces_split_seed(root_seed, spec.name),
            template,
            _family_config_for_split(base_config, spec.name),
        )
        available = 0
        first_failure: int | None = None
        for item_index in range(probe_indices):
            try:
                instance = generator.generate(item_index)
            except TCESGenerationError:
                if first_failure is None:
                    first_failure = item_index
                continue
            if instance.intended_family != template.intended_family:
                raise PanelCapacityError("family generator changed the exact family")
            if not _passes_split_filter(instance, spec.name, protected):
                continue
            numeric_key = tces_numeric_key(instance)
            if numeric_key in used_numeric or instance.content_hash in used_content:
                continue
            used_numeric.add(numeric_key)
            used_content.add(instance.content_hash)
            available += 1
            if spec.name == "a_test_single" and available == required:
                break
        rows.append(
            FamilySplitCapacity(
                split=spec.name,
                required_instances=required,
                probe_indices=probe_indices,
                available_disjoint_instances=available,
                first_generation_failure_index=first_failure,
                passed=available >= required,
            )
        )
    return FamilyCapacityAudit(
        family_id=template.intended_family,
        split_capacities=tuple(rows),
        available_disjoint_instances=sum(
            row.available_disjoint_instances for row in rows
        ),
        passed=all(row.passed for row in rows),
    )


__all__ = [
    "FamilyCapacityAudit",
    "FamilySplitCapacity",
    "PANEL_CAPACITY_PROBE_MULTIPLIER",
    "PANEL_FILTERED_SPLIT_SCAN_MULTIPLIER",
    "PANEL_SELECTED_TEST_SINGLE_MINIMUM",
    "PANEL_SPLIT_MINIMUMS",
    "PanelCapacityError",
    "audit_family_split_capacity",
]
