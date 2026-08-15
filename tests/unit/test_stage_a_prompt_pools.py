from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
from duraseed.data.panels import PanelLabel
from duraseed.data.stage_a_prompt_pools import (
    PromptPoolAllocation,
    PromptPoolStratum,
    StageAPromptPoolBundle,
    StageAPromptPoolError,
    _build_artifact,
    _capacity_eligible_ids,
    _candidate_ranking,
    _direct_composite_manifests,
    _flatten_additional_forbidden,
    _generate_family_records,
    _ordered_schedule,
    read_stage_a_prompt_pool_bundle,
    write_stage_a_prompt_pool_bundle,
)
from duraseed.run_records import RunStatus
from duraseed.tasks.tces import TCESGeneratorConfig


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _allocations() -> tuple[PromptPoolAllocation, ...]:
    return (
        PromptPoolAllocation(
            stratum=PromptPoolStratum.BOUNDARY,
            weight_numerator=2,
            bs_slots=16,
            bg_groups=8,
        ),
        PromptPoolAllocation(
            stratum=PromptPoolStratum.INTERMEDIATE,
            weight_numerator=1,
            bs_slots=8,
            bg_groups=4,
        ),
        PromptPoolAllocation(
            stratum=PromptPoolStratum.BROAD_RANDOM,
            weight_numerator=1,
            bs_slots=8,
            bg_groups=4,
        ),
    )


def _artifact():
    schedule_seed = 919
    return _build_artifact(
        calibration_seed=17,
        family_panel_artifact_id=_sha("1"),
        candidate_table_id=_sha("2"),
        panel_matching_report_id=_sha("3"),
        source_broad_manifest_id=_sha("4"),
        source_confirmation_manifest_id=_sha("5"),
        rl_train_manifest_id=_sha("6"),
        monitor_manifest_id=_sha("7"),
        targeted_panel=PanelLabel.A,
        sentinel_panel=PanelLabel.B,
        boundary_family_ids=tuple(f"boundary-{index:02d}" for index in range(12)),
        sentinel_family_ids=tuple(f"sentinel-{index:02d}" for index in range(12)),
        intermediate_family_ids=tuple(
            f"intermediate-{index:02d}" for index in range(12)
        ),
        broad_random_family_ids=tuple(f"random-{index:02d}" for index in range(12)),
        allocations=_allocations(),
        random_allocation_seed=818,
        schedule_seed=schedule_seed,
        bs_slot_order=_ordered_schedule(
            schedule_seed=schedule_seed,
            namespace="data_order.stage_a.bs_slots",
            boundary=16,
            intermediate=8,
            broad=8,
        ),
        bg_group_order=_ordered_schedule(
            schedule_seed=schedule_seed,
            namespace="data_order.stage_a.bg_groups",
            boundary=8,
            intermediate=4,
            broad=4,
        ),
        record_order="family_round_robin_then_task_id",
    )


def test_prompt_pool_artifact_freezes_exact_mixture_and_authenticates_content() -> None:
    artifact = _artifact()

    assert Counter(artifact.bs_slot_order) == {
        PromptPoolStratum.BOUNDARY: 16,
        PromptPoolStratum.INTERMEDIATE: 8,
        PromptPoolStratum.BROAD_RANDOM: 8,
    }
    assert Counter(artifact.bg_group_order) == {
        PromptPoolStratum.BOUNDARY: 8,
        PromptPoolStratum.INTERMEDIATE: 4,
        PromptPoolStratum.BROAD_RANDOM: 4,
    }
    assert artifact.bs_slot_order == _artifact().bs_slot_order

    changed = artifact.model_dump(mode="python")
    changed["schedule_seed"] += 1
    with pytest.raises(ValidationError, match="artifact_id"):
        type(artifact).model_validate(changed)


def test_prompt_pool_artifact_rejects_family_overlap() -> None:
    artifact = _artifact()
    changed = artifact.model_dump(mode="python")
    changed["broad_random_family_ids"] = changed["boundary_family_ids"]

    with pytest.raises(ValidationError, match="must be disjoint"):
        type(artifact).model_validate(changed)


@pytest.mark.parametrize("contaminated_split", ["a_rl_train", "a_monitor"])
def test_prompt_pool_bundle_rejects_complete_valid_family_panel_contamination(
    contaminated_split: str,
) -> None:
    artifact = _artifact()
    boundary = artifact.boundary_family_ids[0]
    sentinel = artifact.sentinel_family_ids[0]
    rl_records = [
        SimpleNamespace(intended_family=family_id, valid_family_ids=(family_id,))
        for family_id in (
            artifact.boundary_family_ids
            + artifact.intermediate_family_ids
            + artifact.broad_random_family_ids
        )
        for _ in range(64)
    ]
    monitor_records = [
        SimpleNamespace(intended_family=family_id, valid_family_ids=(family_id,))
        for family_id in artifact.boundary_family_ids + artifact.sentinel_family_ids
        for _ in range(16)
    ]
    records = rl_records if contaminated_split == "a_rl_train" else monitor_records
    records[0] = SimpleNamespace(
        intended_family=boundary,
        valid_family_ids=(boundary, sentinel),
    )
    bundle = SimpleNamespace(
        artifact=artifact,
        a_rl_train_manifest=SimpleNamespace(
            task_family="tces",
            split="a_rl_train",
            manifest_id=artifact.rl_train_manifest_id,
            record_count=len(rl_records),
            records=tuple(rl_records),
        ),
        a_monitor_manifest=SimpleNamespace(
            task_family="tces",
            split="a_monitor",
            manifest_id=artifact.monitor_manifest_id,
            record_count=len(monitor_records),
            records=tuple(monitor_records),
        ),
    )

    with pytest.raises(ValueError, match="forbidden panel|isolate"):
        StageAPromptPoolBundle.manifests_match_artifact(bundle)  # type: ignore[arg-type]


def test_confirmation_ranking_is_order_invariant_and_uses_capacity_second() -> None:
    rows = [
        {
            "family_id": "low",
            "informative_group_probability_i8": 0.40,
            "available_disjoint_instances": 999,
        },
        {
            "family_id": "high-small",
            "informative_group_probability_i8": 0.60,
            "available_disjoint_instances": 100,
        },
        {
            "family_id": "high-large",
            "informative_group_probability_i8": 0.60,
            "available_disjoint_instances": 101,
        },
    ]

    expected = ("high-large", "high-small", "low")
    assert _candidate_ranking(rows, 101) == expected
    assert _candidate_ranking(tuple(reversed(rows)), 101) == expected

    ties = [
        {
            "family_id": family_id,
            "informative_group_probability_i8": 0.5,
            "available_disjoint_instances": 100,
        }
        for family_id in ("tie-a", "tie-b", "tie-c")
    ]
    assert _candidate_ranking(ties, 101) == _candidate_ranking(
        tuple(reversed(ties)), 101
    )

    with pytest.raises(StageAPromptPoolError, match="unique"):
        _candidate_ranking(rows + [rows[0]], 101)


def test_capacity_evidence_requires_exact_passing_split_rows() -> None:
    rows = (
        {
            "family_id": "family-a",
            "passed": True,
            "split_capacities": [
                {
                    "split": "a_rl_train",
                    "required_instances": 64,
                    "available_disjoint_instances": 64,
                    "passed": True,
                }
            ],
        },
    )
    assert _capacity_eligible_ids(rows, requirements={"a_rl_train": 64}) == frozenset(
        {"family-a"}
    )

    contradictory = (
        {
            **rows[0],
            "split_capacities": [{**rows[0]["split_capacities"][0], "passed": False}],
        },
    )
    with pytest.raises(StageAPromptPoolError, match="contradicts"):
        _capacity_eligible_ids(contradictory, requirements={"a_rl_train": 64})


def test_composite_source_allows_two_cohorts_from_one_completed_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broad_id = _sha("a")
    confirmation_id = _sha("b")
    cohort_ids = ["initial", "extension_1", "extension_2"]
    initial = tmp_path / "initial-confirmation"
    consolidated = tmp_path / "consolidated-confirmation"
    plan = {
        "source_kind": "three_cohort_boundary_freeze_v1",
        "source": {
            "source_kind": "three_cohort_boundary_freeze_v1",
            "confirmation_run_directories": [
                str(initial),
                str(consolidated),
                str(consolidated),
            ],
            "combined_broad_manifest_id": broad_id,
            "combined_confirmation_manifest_id": confirmation_id,
        },
    }
    broad = SimpleNamespace(
        manifest_id=broad_id,
        root_seed=17,
        metadata={"cohort_ids": cohort_ids},
    )
    confirmation = SimpleNamespace(
        manifest_id=confirmation_id,
        root_seed=17,
        metadata={
            "cohort_ids": cohort_ids,
            "source_broad_manifest_id": broad_id,
        },
    )
    run = SimpleNamespace(
        seed=17,
        task_manifest_ids={
            "boundary_composite_a_candidate": broad_id,
            "boundary_confirmation_a_candidate": confirmation_id,
        },
    )
    checked_directories = []

    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools._read_json", lambda _path: plan
    )

    def fake_read_run_record(path):
        checked_directories.append(path)
        return SimpleNamespace(
            status=RunStatus.COMPLETED,
            run_kind="m0_calibration",
        )

    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.read_run_record",
        fake_read_run_record,
    )
    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.read_manifest",
        lambda path, *, context: (
            broad if path.name == "a_candidate_manifest.json" else confirmation
        ),
    )

    assert _direct_composite_manifests(tmp_path, run) == (broad, confirmation)
    assert checked_directories == [
        initial.resolve(),
        consolidated.resolve(),
        consolidated.resolve(),
    ]


def test_family_record_scan_excludes_panel_overlap_without_committing_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeInstance:
        def __init__(self, item_index: int) -> None:
            self.item_index = item_index
            self.content_hash = f"content-{item_index}"
            self.valid_family_ids = (
                ("broad", "panel") if item_index == 0 else ("broad",)
            )

    class FakeGenerator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def generate(self, item_index: int) -> FakeInstance:
            return FakeInstance(item_index)

    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.TCESFamilyGenerator", FakeGenerator
    )
    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.tces_numeric_key",
        lambda instance: (instance.item_index,),
    )
    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.build_tces_record",
        lambda instance: instance.content_hash,
    )
    used_numeric: set[object] = set()
    used_content: set[str] = set()

    records = _generate_family_records(
        template=object(),  # type: ignore[arg-type]
        generator_config=TCESGeneratorConfig(),
        root_seed=11,
        split="a_rl_train",
        count=2,
        used_numeric=used_numeric,
        used_content=used_content,
        forbidden_valid_families=frozenset({"panel"}),
    )

    assert records == ("content-1", "content-2")
    assert used_numeric == {(1,), (2,)}
    assert used_content == {"content-1", "content-2"}


def test_family_record_scan_can_isolate_one_intended_panel_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeInstance:
        def __init__(self, item_index: int) -> None:
            self.item_index = item_index
            self.content_hash = f"content-{item_index}"
            self.valid_family_ids = {
                0: ("target-a", "sentinel-b"),
                1: ("target-a", "target-c"),
                2: ("target-a",),
            }.get(item_index, ("target-a",))

    class FakeGenerator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def generate(self, item_index: int) -> FakeInstance:
            return FakeInstance(item_index)

    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.TCESFamilyGenerator", FakeGenerator
    )
    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.tces_numeric_key",
        lambda instance: (instance.item_index,),
    )
    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.build_tces_record",
        lambda instance: instance.content_hash,
    )

    records = _generate_family_records(
        template=object(),  # type: ignore[arg-type]
        generator_config=TCESGeneratorConfig(),
        root_seed=11,
        split="a_monitor",
        count=1,
        used_numeric=set(),
        used_content=set(),
        forbidden_valid_families=frozenset({"sentinel-b", "target-c"}),
    )

    assert records == ("content-2",)


def test_family_record_scan_does_not_partially_commit_on_capacity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeInstance:
        content_hash = "only"
        valid_family_ids = ("broad",)

    class FakeGenerator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def generate(self, _item_index: int) -> FakeInstance:
            return FakeInstance()

    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.TCESFamilyGenerator", FakeGenerator
    )
    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.tces_numeric_key", lambda _instance: (1,)
    )
    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.build_tces_record", lambda _instance: "only"
    )
    used_numeric: set[object] = {("source",)}
    used_content = {"source"}

    assert (
        _generate_family_records(
            template=object(),  # type: ignore[arg-type]
            generator_config=TCESGeneratorConfig(),
            root_seed=11,
            split="a_rl_train",
            count=2,
            used_numeric=used_numeric,
            used_content=used_content,
        )
        is None
    )
    assert used_numeric == {("source",)}
    assert used_content == {"source"}


def test_additional_forbidden_accepts_records_and_tces_manifests() -> None:
    first = TCESTaskManifestRecord.model_construct()
    second = TCESTaskManifestRecord.model_construct()
    manifest = DatasetManifest.model_construct(task_family="tces", records=(second,))

    assert _flatten_additional_forbidden((first, manifest)) == (first, second)


@pytest.mark.parametrize(
    "value",
    (
        SimpleNamespace(task_family="maps"),
        DatasetManifest.model_construct(task_family="maps", records=()),
    ),
)
def test_additional_forbidden_rejects_non_tces_records(value: object) -> None:
    with pytest.raises(TypeError, match="TCES records or TCES manifests"):
        _flatten_additional_forbidden((value,))  # type: ignore[arg-type]


def test_prompt_pool_bundle_write_and_authenticated_read(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact()
    rl = DatasetManifest.model_construct()
    monitor = DatasetManifest.model_construct()
    bundle = StageAPromptPoolBundle.model_construct(
        artifact=artifact,
        a_rl_train_manifest=rl,
        a_monitor_manifest=monitor,
    )
    written: dict[str, DatasetManifest] = {}

    def fake_write_manifest(path, manifest) -> None:
        written[path.name] = manifest

    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.write_manifest", fake_write_manifest
    )
    write_stage_a_prompt_pool_bundle(tmp_path, bundle)

    assert set(written) == {"a_rl_train_manifest.json", "a_monitor_manifest.json"}
    assert (tmp_path / "stage_a_prompt_pools.json").read_bytes().endswith(b"\n")

    def fake_read_manifest(path, *, context):
        del context
        return written[path.name]

    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.read_manifest", fake_read_manifest
    )
    monkeypatch.setattr(
        "duraseed.data.stage_a_prompt_pools.StageAPromptPoolBundle",
        lambda **values: SimpleNamespace(**values),
    )

    loaded = read_stage_a_prompt_pool_bundle(tmp_path)
    assert loaded.artifact == artifact
    assert loaded.a_rl_train_manifest is rl
    assert loaded.a_monitor_manifest is monitor

    artifact_path = tmp_path / "stage_a_prompt_pools.json"
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    with pytest.raises(StageAPromptPoolError, match="not canonical"):
        read_stage_a_prompt_pool_bundle(tmp_path)
