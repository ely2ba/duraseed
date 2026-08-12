from __future__ import annotations

from types import SimpleNamespace

import pytest

from duraseed.config import load_pilot_config
from duraseed.data.matching import FamilyBlockMatchStatus
from duraseed.training.teacher_allocation_freeze import (
    build_teacher_allocation_freeze,
)
from duraseed.training.teacher_allocation_sources import (
    TeacherAllocationSources,
    validate_teacher_allocation_sources,
)
from tests.unit.teacher_allocation_fixtures import fixed_token_measurer, freezer_sources


@pytest.fixture(scope="module")
def sources() -> TeacherAllocationSources:
    source = freezer_sources(load_pilot_config("duraseed_pilot_config.yaml"))
    return validate_teacher_allocation_sources(source)


def _patch_loop(
    monkeypatch: pytest.MonkeyPatch,
    *,
    statuses: list[FamilyBlockMatchStatus],
    stream_size: int = 257,
) -> list[int]:
    import duraseed.training.teacher_allocation_freeze as freezer

    generated: list[int] = []
    target_a = (SimpleNamespace(structure_key=("eligible",)),)
    target_b = (SimpleNamespace(structure_key=("eligible",)),)

    class _StreamBuilder:
        def __init__(self, root_seed, generator_config) -> None:
            assert root_seed == 5
            del generator_config

        def lazy_split(self, split: str, *, size: int):
            assert split == "a_candidate"
            return tuple(
                SimpleNamespace(intended_family=f"random-{index:03d}")
                for index in range(size)
            )

    def candidates(manifest, panel_ids, token_measurer):  # type: ignore[no-untyped-def]
        del panel_ids, token_measurer
        if manifest is sources:
            return ()
        return tuple(SimpleNamespace(family_block_record=object()) for _ in range(16))

    def generate(source, template, config, used_numeric, used_content, multiplier):  # type: ignore[no-untyped-def]
        del source, template, config, used_numeric, used_content, multiplier
        generated.append(1)
        return (object(),) * 16

    def match(target, random, source):  # type: ignore[no-untyped-def]
        del target, random, source
        status = statuses.pop(0)
        return SimpleNamespace(
            status=status,
            passed=status is FamilyBlockMatchStatus.SELECTED,
        )

    monkeypatch.setattr(freezer, "TCESSplitBuilder", _StreamBuilder)
    monkeypatch.setattr(freezer, "candidate_rows", candidates)
    monkeypatch.setattr(
        freezer,
        "target_blocks",
        lambda source, candidates, panel: target_a if panel.value == "A" else target_b,
    )
    monkeypatch.setattr(freezer, "family_structure_key", lambda *args: ("eligible",))
    monkeypatch.setattr(freezer, "generate_random_family", generate)
    monkeypatch.setattr(freezer, "match_orientation", match)
    monkeypatch.setattr(
        freezer,
        "build_manifest",
        lambda **kwargs: SimpleNamespace(records=tuple(kwargs["records"])),
    )
    monkeypatch.setattr(
        freezer,
        "_complete_result",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    return generated


def test_direct_loop_selects_first_joint_prefix(
    sources: TeacherAllocationSources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = [
        FamilyBlockMatchStatus.INFEASIBLE,
        FamilyBlockMatchStatus.INFEASIBLE,
        FamilyBlockMatchStatus.SELECTED,
        FamilyBlockMatchStatus.SELECTED,
    ]
    generated = _patch_loop(monkeypatch, statuses=statuses)

    result = build_teacher_allocation_freeze(
        sources=sources,
        token_measurer=fixed_token_measurer,
        template_ceiling=3,
    )

    assert result.status is FamilyBlockMatchStatus.SELECTED
    assert result.prefix_length == 2
    assert result.failure is None
    assert len(generated) == 2


@pytest.mark.parametrize(
    ("statuses", "failure"),
    [
        ([FamilyBlockMatchStatus.SEARCH_EXHAUSTED], "panel_a_match_search_exhausted"),
        (
            [
                FamilyBlockMatchStatus.INFEASIBLE,
                FamilyBlockMatchStatus.SEARCH_EXHAUSTED,
            ],
            "panel_b_match_search_exhausted",
        ),
    ],
)
def test_direct_loop_stops_on_match_search_exhaustion(
    sources: TeacherAllocationSources,
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[FamilyBlockMatchStatus],
    failure: str,
) -> None:
    generated = _patch_loop(monkeypatch, statuses=list(statuses))
    result = build_teacher_allocation_freeze(
        sources=sources,
        token_measurer=fixed_token_measurer,
        template_ceiling=3,
    )

    assert result.status is FamilyBlockMatchStatus.SEARCH_EXHAUSTED
    assert result.prefix_length == 1
    assert result.failure == failure
    assert len(generated) == 1


def test_later_panel_a_exhaustion_clears_stale_panel_b_report(
    sources: TeacherAllocationSources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = [
        FamilyBlockMatchStatus.INFEASIBLE,
        FamilyBlockMatchStatus.INFEASIBLE,
        FamilyBlockMatchStatus.SEARCH_EXHAUSTED,
    ]
    _patch_loop(monkeypatch, statuses=statuses)
    result = build_teacher_allocation_freeze(
        sources=sources,
        token_measurer=fixed_token_measurer,
        template_ceiling=3,
    )

    assert result.status is FamilyBlockMatchStatus.SEARCH_EXHAUSTED
    assert result.prefix_length == 2
    assert result.report_b is None


def test_direct_loop_enforces_frozen_seed_train_cap(
    sources: TeacherAllocationSources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = [FamilyBlockMatchStatus.INFEASIBLE] * (2 * 256)
    generated = _patch_loop(monkeypatch, statuses=statuses)
    result = build_teacher_allocation_freeze(
        sources=sources,
        token_measurer=fixed_token_measurer,
        template_ceiling=257,
    )

    assert result.status is FamilyBlockMatchStatus.INFEASIBLE
    assert result.prefix_length == 257
    assert result.failure == "a_seed_train_ceiling_reached_without_joint_match"
    assert len(generated) == 256


def test_direct_loop_records_full_ceiling_with_only_trailing_exclusions(
    sources: TeacherAllocationSources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duraseed.training.teacher_allocation_freeze as freezer

    _patch_loop(monkeypatch, statuses=[])
    monkeypatch.setattr(freezer, "family_structure_key", lambda *args: ("excluded",))
    result = build_teacher_allocation_freeze(
        sources=sources,
        token_measurer=fixed_token_measurer,
        template_ceiling=3,
    )

    assert result.status is FamilyBlockMatchStatus.INFEASIBLE
    assert result.prefix_length == 3
    assert result.failure == "a_candidate_ceiling_reached_without_joint_match"
    assert result.exclusion_counts == {"not_structurally_eligible": 3}
    assert result.random_records == []
