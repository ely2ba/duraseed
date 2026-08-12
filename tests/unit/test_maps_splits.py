from __future__ import annotations

import pytest

from duraseed.data.maps_splits import (
    MAPSSplitBuilder,
    MAPS_SPLIT_NAMES,
    MAPS_SPLIT_SIZES,
    derive_maps_split_seed,
)
from duraseed.provenance import derive_namespaced_seed


def test_declared_maps_bundle_is_constructible_and_globally_disjoint() -> None:
    bundle = MAPSSplitBuilder(11).build_splits(MAPS_SPLIT_SIZES)

    assert tuple(bundle) == MAPS_SPLIT_NAMES
    assert {split: len(rows) for split, rows in bundle.items()} == dict(
        MAPS_SPLIT_SIZES
    )
    task_ids = [row.task.task_id for rows in bundle.values() for row in rows]
    assert len(task_ids) == len(set(task_ids))


def test_maps_split_construction_ignores_request_mapping_order() -> None:
    forward = MAPSSplitBuilder(41).build_splits({"b_train": 32, "b_validation": 32})
    reverse = MAPSSplitBuilder(41).build_splits({"b_validation": 32, "b_train": 32})

    assert {
        split: tuple(row.task.task_id for row in rows)
        for split, rows in forward.items()
    } == {
        split: tuple(row.task.task_id for row in rows)
        for split, rows in reverse.items()
    }


def test_maps_split_seed_uses_the_shared_namespace_contract() -> None:
    assert derive_maps_split_seed(11, "b_validation") == derive_namespaced_seed(
        11, "dataset.maps.split", "b_validation"
    )
    assert derive_maps_split_seed(11, "b_train") != derive_maps_split_seed(
        11, "b_validation"
    )


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        ({"unknown": 1}, "unknown MAPS splits"),
        ({"b_monitor": True}, "non-negative integer"),
        ({"b_monitor": 257}, "ceiling 256"),
    ],
)
def test_maps_split_contract_rejects_invalid_requests(
    requested: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MAPSSplitBuilder(11).build_splits(requested)
