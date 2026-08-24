from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from duraseed.data import stage_a_prompt_pools as pools
from duraseed.data.family_generation import (
    FamilyGenerationJob,
    FilteredFamilyGenerationJob,
    generate_filtered_family,
)
from duraseed.data.splits import tces_numeric_key
from duraseed.data.manifests import build_manifest, build_tces_record
from duraseed.data.splits import TCESSplitBuilder
from duraseed.pilot0_source_build import _regenerate_templates
from duraseed.pilot0_source_selection import select_or_generate
from duraseed.tasks.tces import (
    GeneratedTCESInstance,
    TCESGenerator,
    TCESGeneratorConfig,
)


def _fixture() -> tuple[GeneratedTCESInstance, TCESGeneratorConfig]:
    config = TCESGeneratorConfig(
        n_operands=3,
        operand_min=2,
        operand_max=20,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_attempts=256,
    )
    template = TCESGenerator(
        17,
        replace(config, split="a_candidate"),
    ).generate(0)
    return template, config


def _early_candidates(template: GeneratedTCESInstance, config: TCESGeneratorConfig):
    return generate_filtered_family(
        FilteredFamilyGenerationJob(
            FamilyGenerationJob(template, config, 11, "a_rl_train", 32),
            4,
            frozenset(),
            frozenset(),
            frozenset(),
        )
    )


def test_early_family_generation_matches_sequential_reference() -> None:
    template, config = _fixture()
    candidates = _early_candidates(template, config)
    fast = select_or_generate(
        candidates,
        template=template,
        generator_config=config,
        root_seed=11,
        split="a_rl_train",
        count=4,
        used_numeric=set(),
        used_content=set(),
        forbidden=frozenset(),
    )
    reference = pools._generate_family_records(  # noqa: SLF001
        template=template,
        generator_config=config,
        root_seed=11,
        split="a_rl_train",
        count=4,
        used_numeric=set(),
        used_content=set(),
    )
    assert fast == reference


def test_cross_family_collision_uses_exact_sequential_fallback() -> None:
    template, config = _fixture()
    candidates = _early_candidates(template, config)
    assert candidates is not None
    blocked = {tces_numeric_key(candidates[0])}
    fast = select_or_generate(
        candidates,
        template=template,
        generator_config=config,
        root_seed=11,
        split="a_rl_train",
        count=4,
        used_numeric=set(blocked),
        used_content=set(),
        forbidden=frozenset(),
    )
    reference = pools._generate_family_records(  # noqa: SLF001
        template=template,
        generator_config=config,
        root_seed=11,
        split="a_rl_train",
        count=4,
        used_numeric=set(blocked),
        used_content=set(),
    )
    assert fast == reference


def test_parallel_template_replay_matches_accepted_split_indices() -> None:
    _, config = _fixture()
    accepted = TCESSplitBuilder(11, config).lazy_split("a_candidate", size=12).take(12)
    templates = {}
    for index, instance in enumerate(accepted):
        templates.setdefault(instance.intended_family, (index, instance))
        if len(templates) == 4:
            break
    manifest = build_manifest(
        name="template-replay-test",
        split="a_candidate",
        generator_version="1.0.0",
        root_seed=11,
        records=[build_tces_record(instance) for instance in accepted],
        metadata={
            "templates": [
                {
                    "family_id": family,
                    "template_item_index": index,
                    "template_content_hash": instance.content_hash,
                }
                for family, (index, instance) in templates.items()
            ]
        },
    )
    with ProcessPoolExecutor(max_workers=2) as executor:
        replay = _regenerate_templates(manifest, config, executor)
    assert replay == {family: instance for family, (_, instance) in templates.items()}
