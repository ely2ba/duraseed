"""Deterministic lazy construction of the declared TCES data splits.

The specification treats split sizes as ceilings.  Merely creating a
``LazyTCESSplit`` therefore performs no generation: items are materialized only
when an index is requested.  Each split receives a namespaced seed derived from
the dataset root seed, so access order and rejection in another split cannot
perturb its candidate stream.

For a collection of splits, use :meth:`TCESSplitBuilder.build_splits`.  It walks
the frozen declaration order and rejects repeated numeric tasks and content
hashes, which gives teacher and evaluation splits disjoint numeric instances.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, overload

from duraseed.provenance import MAX_ROOT_SEED, derive_namespaced_seed
from duraseed.tasks.tces.generator import (
    GeneratedTCESInstance,
    TCESGenerator,
    TCESGeneratorConfig,
)


FamilyFilterMode = Literal["any", "single", "multi"]
TCESNumericKey = tuple[tuple[int, ...], int, int]


@dataclass(frozen=True, slots=True)
class TCESSplitSpec:
    """One frozen TCES split declaration from the project specification."""

    name: str
    purpose: str
    default_size: int
    family_filter: FamilyFilterMode = "any"


@dataclass(frozen=True, slots=True)
class TCESOODProfile:
    """Explicit frozen profile for the currently supported operand-count OOD.

    Tree-motif holdouts require a motif-aware generator filter that is not part
    of Vertical Slice 2.  Until that exists, ``a_ood`` is fail-closed and only
    accepts a declared operand-count profile distinct from the ID config.
    ``metadata_against`` is suitable for manifest ``difficulty_features`` or
    aggregate manifest metadata.
    """

    name: str
    config: TCESGeneratorConfig
    held_out_dimension: Literal["operand_count"] = "operand_count"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("OOD profile name must be a non-empty string")
        if not isinstance(self.config, TCESGeneratorConfig):
            raise TypeError("OOD profile config must be a TCESGeneratorConfig")
        if self.held_out_dimension != "operand_count":
            raise ValueError("Vertical Slice 2 supports operand-count OOD only")

    def metadata_against(
        self, in_distribution: TCESGeneratorConfig
    ) -> dict[str, object]:
        if self.config.n_operands == in_distribution.n_operands:
            raise ValueError("OOD operand count must differ from the ID operand count")
        return {
            "ood_profile": self.name,
            "held_out_dimension": self.held_out_dimension,
            "in_distribution_operand_count": in_distribution.n_operands,
            "ood_operand_count": self.config.n_operands,
        }


TCES_SPLIT_SPECS: tuple[TCESSplitSpec, ...] = (
    TCESSplitSpec("a_candidate", "base boundary map and family selection", 8_192),
    TCESSplitSpec(
        "a_seed_train", "teacher examples for selected/control families", 4_096
    ),
    TCESSplitSpec("a_seed_gate", "unseen seed-dose reachability decision", 768),
    TCESSplitSpec("a_rl_train", "frozen consolidation prompt pool", 16_384),
    TCESSplitSpec("a_monitor", "low-cost frequent checkpoint evaluation", 384),
    TCESSplitSpec("a_validation", "hyperparameter and checkpoint matching", 512),
    TCESSplitSpec(
        "a_test_single",
        "confirmatory single-family items",
        512,
        family_filter="single",
    ),
    TCESSplitSpec(
        "a_test_multi",
        "confirmatory multi-family items",
        512,
        family_filter="multi",
    ),
    TCESSplitSpec("a_ood", "held-out operand count or tree motifs", 512),
)

TCES_SPLIT_NAMES: tuple[str, ...] = tuple(spec.name for spec in TCES_SPLIT_SPECS)
TCES_SPLIT_SIZES: Mapping[str, int] = MappingProxyType(
    {spec.name: spec.default_size for spec in TCES_SPLIT_SPECS}
)
TEACHER_SPLITS: frozenset[str] = frozenset({"a_seed_train", "b_train"})
EVALUATION_SPLITS: frozenset[str] = frozenset(
    {
        "a_candidate",
        "a_seed_gate",
        "a_monitor",
        "a_validation",
        "a_test_single",
        "a_test_multi",
        "a_ood",
        "b_monitor",
        "b_validation",
        "b_test",
    }
)
FINAL_TEST_SPLITS: frozenset[str] = frozenset(
    {"a_test_single", "a_test_multi", "b_test"}
)

_SPECS_BY_NAME: Mapping[str, TCESSplitSpec] = MappingProxyType(
    {spec.name: spec for spec in TCES_SPLIT_SPECS}
)


class SplitConstructionError(RuntimeError):
    """A deterministic split stream could not satisfy its declared contract."""


def get_tces_split_spec(name: str) -> TCESSplitSpec:
    """Return a declared split specification, rejecting unknown split names."""

    try:
        return _SPECS_BY_NAME[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown TCES split: {name!r}") from error


def derive_tces_split_seed(root_seed: int, split: str) -> int:
    """Derive the shared provenance contract's stable 63-bit split seed."""

    get_tces_split_spec(split)
    return derive_namespaced_seed(root_seed, "dataset.tces.split", split)


def exact_valid_family_count(item: object) -> int | None:
    """Read an exact family count from a generated instance or manifest record.

    A generated enumeration explicitly marked incomplete is never accepted as
    evidence for a family-count filter.  Manifest records are expected to have
    been created only from complete primary enumerations.
    """

    enumeration = getattr(item, "enumeration", None)
    if enumeration is not None:
        complete = getattr(enumeration, "complete", None)
        if complete is not True:
            return None

    value = getattr(item, "valid_family_count", None)
    if value is None and enumeration is not None:
        families = getattr(enumeration, "families", None)
        if families is not None:
            value = len(families)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def is_single_family(item: object) -> bool:
    """Return whether the complete valid family set has exactly one member."""

    return exact_valid_family_count(item) == 1


def is_multi_family(
    item: object,
    *,
    minimum: int = 3,
    maximum: int = 8,
) -> bool:
    """Return whether the exact family count lies in the declared 3..8 band."""

    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or minimum < 1
        or maximum < minimum
    ):
        raise ValueError("family-count bounds must be positive ordered integers")
    count = exact_valid_family_count(item)
    return count is not None and minimum <= count <= maximum


def tces_numeric_key(item: object) -> TCESNumericKey:
    """Return the operand-multiset and normalized-target identity of an item."""

    task = getattr(item, "task", item)
    operands = getattr(task, "operands", None)
    target = getattr(task, "target", None)
    if (
        not isinstance(operands, (tuple, list))
        or not operands
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in operands
        )
    ):
        raise TypeError("TCES item must expose integer operands")
    numerator = getattr(target, "numerator", None)
    denominator = getattr(target, "denominator", None)
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise TypeError("TCES item must expose a normalized exact target")
    return (tuple(sorted(operands)), numerator, denominator)


def _content_hash(item: object) -> str:
    value = getattr(item, "content_hash", None)
    if not isinstance(value, str) or not value:
        task = getattr(item, "task", None)
        value = getattr(task, "content_hash", None)
    if not isinstance(value, str) or not value:
        raise TypeError("TCES item must expose a non-empty content_hash")
    return value


def _config_for_split(
    base: TCESGeneratorConfig,
    spec: TCESSplitSpec,
    ood_profile: TCESOODProfile | None = None,
) -> TCESGeneratorConfig:
    source = base
    if spec.name == "a_ood":
        if ood_profile is None:
            raise ValueError(
                "a_ood requires an explicit TCESOODProfile; refusing to generate "
                "in-distribution data under an OOD label"
            )
        # Validation also produces the metadata contract used by manifests.
        ood_profile.metadata_against(base)
        source = ood_profile.config
    updates: dict[str, object] = {"split": spec.name}
    if spec.family_filter == "single":
        updates.update(min_valid_families=1, max_valid_families=1)
    elif spec.family_filter == "multi":
        updates.update(min_valid_families=3, max_valid_families=8)
    return replace(source, **updates)


class LazyTCESSplit(Sequence[GeneratedTCESInstance]):
    """A bounded, cached, deterministic view over one TCES split stream."""

    def __init__(
        self,
        *,
        split_spec: TCESSplitSpec,
        generator_seed: int,
        generator_config: TCESGeneratorConfig,
        size: int,
        generation_metadata: Mapping[str, object] | None = None,
        forbidden_numeric_keys: Iterable[TCESNumericKey] = (),
        forbidden_content_hashes: Iterable[str] = (),
    ) -> None:
        if (
            isinstance(generator_seed, bool)
            or not isinstance(generator_seed, int)
            or not 0 <= generator_seed <= MAX_ROOT_SEED
        ):
            raise ValueError(
                f"generator_seed must be an integer in [0, {MAX_ROOT_SEED}]"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("split size must be a non-negative integer")
        if size > split_spec.default_size:
            raise ValueError(
                f"requested {size} items exceeds {split_spec.name!r} ceiling "
                f"{split_spec.default_size}"
            )
        self.split_spec = split_spec
        self.generator_seed = generator_seed
        self.generator_config = generator_config
        self.generation_metadata: Mapping[str, object] = MappingProxyType(
            dict(generation_metadata or {})
        )
        self._generator = TCESGenerator(generator_seed, generator_config)
        self._size = size
        self._next_candidate_index = 0
        self._scan_stop = max(1_024, size * 128)
        self._forbidden_numeric = frozenset(forbidden_numeric_keys)
        self._forbidden_content = frozenset(forbidden_content_hashes)
        self._seen_numeric: set[TCESNumericKey] = set()
        self._seen_content: set[str] = set()
        self._cache: list[GeneratedTCESInstance] = []

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, index: int) -> GeneratedTCESInstance: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[GeneratedTCESInstance, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> GeneratedTCESInstance | tuple[GeneratedTCESInstance, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(self._size))
            )
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("split indices must be integers or slices")
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError("TCES split index out of range")
        self._ensure(index)
        return self._cache[index]

    @property
    def generated_count(self) -> int:
        """Number of accepted items materialized so far."""

        return len(self._cache)

    @property
    def scanned_candidate_count(self) -> int:
        """Number of deterministic generator indices examined so far."""

        return self._next_candidate_index

    def take(self, count: int) -> tuple[GeneratedTCESInstance, ...]:
        """Materialize and return the first ``count`` accepted items."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if count > self._size:
            raise ValueError("count exceeds the lazy split size")
        return tuple(self[index] for index in range(count))

    def _ensure(self, accepted_index: int) -> None:
        while len(self._cache) <= accepted_index:
            if self._next_candidate_index >= self._scan_stop:
                raise SplitConstructionError(
                    f"split {self.split_spec.name!r} exhausted its deterministic "
                    "candidate scan budget"
                )
            candidate_index = self._next_candidate_index
            self._next_candidate_index += 1
            try:
                instance = self._generator.generate(candidate_index)
            except Exception as error:
                raise SplitConstructionError(
                    f"TCES generation failed for split {self.split_spec.name!r}, "
                    f"candidate index {candidate_index}"
                ) from error

            if getattr(instance.task, "split", None) != self.split_spec.name:
                raise SplitConstructionError("generator returned the wrong split label")
            if self.split_spec.family_filter == "single" and not is_single_family(
                instance
            ):
                raise SplitConstructionError(
                    "single-family generator returned a non-single-family item"
                )
            if self.split_spec.family_filter == "multi" and not is_multi_family(
                instance
            ):
                raise SplitConstructionError(
                    "multi-family generator returned an out-of-band item"
                )

            numeric_key = tces_numeric_key(instance)
            content_hash = _content_hash(instance)
            if (
                numeric_key in self._forbidden_numeric
                or numeric_key in self._seen_numeric
                or content_hash in self._forbidden_content
                or content_hash in self._seen_content
            ):
                continue
            self._seen_numeric.add(numeric_key)
            self._seen_content.add(content_hash)
            self._cache.append(instance)


class TCESSplitBuilder:
    """Construct reproducible split streams from one dataset root seed."""

    def __init__(
        self,
        root_seed: int,
        config: TCESGeneratorConfig | None = None,
        *,
        ood_profile: TCESOODProfile | None = None,
    ) -> None:
        # Validate against the single repository-wide root-seed contract before
        # retaining the value.  The derived result is recomputed per split.
        derive_namespaced_seed(root_seed, "dataset.tces.split", "validation")
        self.root_seed = root_seed
        self.config = config or TCESGeneratorConfig()
        if ood_profile is not None:
            ood_profile.metadata_against(self.config)
        self.ood_profile = ood_profile

    def split_generation_metadata(self, split: str) -> Mapping[str, object]:
        """Return immutable metadata to copy into a split/dataset manifest.

        In particular, callers that use :meth:`build_splits` can retain the OOD
        profile declaration without reconstructing the corresponding lazy
        stream.
        """

        spec = get_tces_split_spec(split)
        if spec.name != "a_ood":
            return MappingProxyType({})
        if self.ood_profile is None:
            raise ValueError(
                "a_ood requires an explicit TCESOODProfile; no manifest "
                "metadata is available"
            )
        return MappingProxyType(self.ood_profile.metadata_against(self.config))

    def lazy_split(
        self,
        split: str,
        *,
        size: int | None = None,
        forbidden_numeric_keys: Iterable[TCESNumericKey] = (),
        forbidden_content_hashes: Iterable[str] = (),
    ) -> LazyTCESSplit:
        """Create a lazy split without generating any item eagerly."""

        spec = get_tces_split_spec(split)
        requested_size = spec.default_size if size is None else size
        split_config = _config_for_split(self.config, spec, self.ood_profile)
        split_seed = derive_tces_split_seed(self.root_seed, split)
        generation_metadata = self.split_generation_metadata(split)
        return LazyTCESSplit(
            split_spec=spec,
            generator_seed=split_seed,
            generator_config=split_config,
            size=requested_size,
            generation_metadata=generation_metadata,
            forbidden_numeric_keys=forbidden_numeric_keys,
            forbidden_content_hashes=forbidden_content_hashes,
        )

    def build_splits(
        self,
        requested_sizes: Mapping[str, int],
    ) -> dict[str, tuple[GeneratedTCESInstance, ...]]:
        """Build requested split prefixes with deterministic global disjointness.

        Input mapping order is ignored.  The frozen specification order decides
        which stream owns a colliding numeric instance, making reruns and
        differently ordered configuration files byte-for-byte equivalent.
        """

        unknown = set(requested_sizes).difference(TCES_SPLIT_NAMES)
        if unknown:
            raise ValueError(f"unknown TCES splits: {sorted(unknown)!r}")
        normalized: dict[str, int] = {}
        for name, size in requested_sizes.items():
            spec = get_tces_split_spec(name)
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"size for {name!r} must be a non-negative integer")
            if size > spec.default_size:
                raise ValueError(
                    f"requested {size} items exceeds {name!r} ceiling "
                    f"{spec.default_size}"
                )
            normalized[name] = size

        used_numeric: set[TCESNumericKey] = set()
        used_content: set[str] = set()
        built: dict[str, tuple[GeneratedTCESInstance, ...]] = {}
        for spec in TCES_SPLIT_SPECS:
            if spec.name not in normalized:
                continue
            size = normalized[spec.name]
            stream = self.lazy_split(
                spec.name,
                size=size,
                forbidden_numeric_keys=used_numeric,
                forbidden_content_hashes=used_content,
            )
            instances = stream.take(size)
            built[spec.name] = instances
            used_numeric.update(tces_numeric_key(instance) for instance in instances)
            used_content.update(_content_hash(instance) for instance in instances)
        return built


__all__ = [
    "EVALUATION_SPLITS",
    "FINAL_TEST_SPLITS",
    "FamilyFilterMode",
    "LazyTCESSplit",
    "SplitConstructionError",
    "TCESNumericKey",
    "TCESOODProfile",
    "TCESSplitBuilder",
    "TCESSplitSpec",
    "TCES_SPLIT_NAMES",
    "TCES_SPLIT_SIZES",
    "TCES_SPLIT_SPECS",
    "TEACHER_SPLITS",
    "derive_tces_split_seed",
    "exact_valid_family_count",
    "get_tces_split_spec",
    "is_multi_family",
    "is_single_family",
    "tces_numeric_key",
]
