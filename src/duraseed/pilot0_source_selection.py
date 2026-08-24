"""Deterministic fast-path with exact sequential fallback for Pilot sources."""

from __future__ import annotations

from duraseed.data import stage_a_prompt_pools as pools
from duraseed.data.family_generation import select_family_candidates
from duraseed.data.manifests import TCESTaskManifestRecord
from duraseed.tasks.tces import GeneratedTCESInstance, TCESGeneratorConfig


def select_or_generate(
    candidates: tuple[TCESTaskManifestRecord, ...] | None,
    *,
    template: GeneratedTCESInstance,
    generator_config: TCESGeneratorConfig,
    root_seed: int,
    split: str,
    count: int,
    used_numeric: set[object],
    used_content: set[str],
    forbidden: frozenset[str],
) -> tuple[TCESTaskManifestRecord, ...] | None:
    """Use an early prefix when sufficient, else reproduce sequential selection."""

    selected = None
    if candidates is not None:
        selected = select_family_candidates(
            candidates,
            count=count,
            forbidden_valid_families=forbidden,
            used_numeric=used_numeric,
            used_content=used_content,
        )
    if selected is not None:
        return selected
    return pools._generate_family_records(  # noqa: SLF001
        template=template,
        generator_config=generator_config,
        root_seed=root_seed,
        split=split,
        count=count,
        used_numeric=used_numeric,
        used_content=used_content,
        forbidden_valid_families=forbidden,
    )


__all__ = ["select_or_generate"]
