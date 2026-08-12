"""Deterministic, disjoint construction of the declared MAPS splits."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from duraseed.provenance import derive_namespaced_seed
from duraseed.tasks.maps import (
    GeneratedMAPSInstance,
    MAPSGenerationError,
    MAPSGenerator,
    MAPSGeneratorConfig,
)


MAPS_SPLIT_NAMES: tuple[str, ...] = (
    "b_train",
    "b_validation",
    "b_monitor",
    "b_test",
)
MAPS_SPLIT_SIZES: Mapping[str, int] = MappingProxyType(
    {
        "b_train": 4_096,
        "b_validation": 512,
        "b_monitor": 256,
        "b_test": 512,
    }
)


class MAPSSplitConstructionError(RuntimeError):
    """A deterministic MAPS split bundle could not satisfy its contract."""


def derive_maps_split_seed(root_seed: int, split: str) -> int:
    """Derive the stable seed for one declared MAPS split."""

    if split not in MAPS_SPLIT_SIZES:
        raise ValueError(f"unknown MAPS split: {split!r}")
    return derive_namespaced_seed(root_seed, "dataset.maps.split", split)


class MAPSSplitBuilder:
    """Build deterministic MAPS prefixes without within/cross-split duplicates."""

    def __init__(
        self,
        root_seed: int,
        config: MAPSGeneratorConfig | None = None,
    ) -> None:
        derive_namespaced_seed(root_seed, "dataset.maps.split", "validation")
        self.root_seed = root_seed
        self.config = config or MAPSGeneratorConfig()

    def build_splits(
        self,
        requested_sizes: Mapping[str, int],
    ) -> dict[str, tuple[GeneratedMAPSInstance, ...]]:
        """Build requested splits in frozen order, skipping semantic duplicates."""

        unknown = set(requested_sizes).difference(MAPS_SPLIT_NAMES)
        if unknown:
            raise ValueError(f"unknown MAPS splits: {sorted(unknown)!r}")
        normalized: dict[str, int] = {}
        for split, size in requested_sizes.items():
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"size for {split!r} must be a non-negative integer")
            ceiling = MAPS_SPLIT_SIZES[split]
            if size > ceiling:
                raise ValueError(
                    f"requested {size} items exceeds {split!r} ceiling {ceiling}"
                )
            normalized[split] = size

        used_task_ids: set[str] = set()
        built: dict[str, tuple[GeneratedMAPSInstance, ...]] = {}
        for split in MAPS_SPLIT_NAMES:
            if split not in normalized:
                continue
            size = normalized[split]
            generator = MAPSGenerator(
                derive_maps_split_seed(self.root_seed, split), self.config
            )
            accepted: list[GeneratedMAPSInstance] = []
            candidate_index = 0
            scan_stop = max(1_024, size * 8)
            while len(accepted) < size:
                if candidate_index >= scan_stop:
                    raise MAPSSplitConstructionError(
                        f"split {split!r} exhausted its deterministic scan budget"
                    )
                try:
                    candidate = generator.generate(candidate_index)
                except MAPSGenerationError as error:
                    raise MAPSSplitConstructionError(
                        f"MAPS generation failed for split {split!r}, candidate "
                        f"index {candidate_index}"
                    ) from error
                candidate_index += 1
                task_id = candidate.task.task_id
                if not isinstance(task_id, str) or not task_id:
                    raise MAPSSplitConstructionError(
                        "MAPS generator returned an item without a task identity"
                    )
                if task_id in used_task_ids:
                    continue
                used_task_ids.add(task_id)
                accepted.append(candidate)
            built[split] = tuple(accepted)
        return built


__all__ = [
    "MAPSSplitBuilder",
    "MAPSSplitConstructionError",
    "MAPS_SPLIT_NAMES",
    "MAPS_SPLIT_SIZES",
    "derive_maps_split_seed",
]
