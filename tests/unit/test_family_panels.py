from __future__ import annotations

import pytest
from pydantic import ValidationError

from duraseed.data import FamilyPanelArtifact as ExportedFamilyPanelArtifact
from duraseed.data.panels import (
    FamilyPanelArtifact,
    PanelLabel,
    SeedBlockPanelAssignment,
)
from duraseed.provenance import MAX_ROOT_SEED


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _assignment(
    seed: int,
    targeted: PanelLabel,
) -> SeedBlockPanelAssignment:
    sentinel = PanelLabel.B if targeted is PanelLabel.A else PanelLabel.A
    return SeedBlockPanelAssignment(
        training_seed=seed,
        targeted_panel=targeted,
        sentinel_panel=sentinel,
    )


def _payload() -> dict[str, object]:
    return {
        "m0_checkpoint_path": "tinker://checkpoint/m0",
        "candidate_family_table_manifest_id": SHA_A,
        "panel_matching_report_id": SHA_B,
        "allocation_seed": 101,
        "panel_a_family_ids": ("test-family-a-1", "test-family-a-2"),
        "panel_b_family_ids": ("test-family-b-1", "test-family-b-2"),
        "seed_block_assignments": (
            _assignment(11, PanelLabel.A),
            _assignment(29, PanelLabel.B),
            _assignment(47, PanelLabel.A),
            _assignment(71, PanelLabel.B),
        ),
    }


def test_resolved_panel_artifact_round_trips_without_selecting_membership() -> None:
    artifact = FamilyPanelArtifact.model_validate(_payload())

    assert ExportedFamilyPanelArtifact is FamilyPanelArtifact
    assert artifact.schema_version == "duraseed-family-panels-v1"
    assert artifact.panel_a_family_ids == (
        "test-family-a-1",
        "test-family-a-2",
    )
    assert [
        assignment.targeted_panel for assignment in artifact.seed_block_assignments
    ] == [
        PanelLabel.A,
        PanelLabel.B,
        PanelLabel.A,
        PanelLabel.B,
    ]
    assert (
        FamilyPanelArtifact.model_validate_json(artifact.model_dump_json()) == artifact
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("candidate_family_table_manifest_id", "not-a-hash"),
        ("panel_matching_report_id", "sha256:" + "A" * 64),
    ),
)
def test_panel_provenance_ids_must_be_canonical_sha256(
    field: str,
    value: str,
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValidationError, match="canonical sha256"):
        FamilyPanelArtifact.model_validate(payload)


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"panel_a_family_ids": ()}, "at least 1 item"),
        (
            {"panel_b_family_ids": ("test-family-b-1",)},
            "equal family counts",
        ),
        (
            {
                "panel_b_family_ids": (
                    "test-family-a-2",
                    "test-family-b-1",
                )
            },
            "must be disjoint",
        ),
        (
            {
                "panel_a_family_ids": (
                    "test-family-a-1",
                    "test-family-a-1",
                )
            },
            "unique and canonically sorted",
        ),
        (
            {
                "panel_a_family_ids": (
                    "test-family-a-2",
                    "test-family-a-1",
                )
            },
            "unique and canonically sorted",
        ),
    ),
)
def test_panel_family_sets_are_nonempty_equal_disjoint_and_unique(
    changes: dict[str, object],
    match: str,
) -> None:
    payload = _payload()
    payload.update(changes)

    with pytest.raises(ValidationError, match=match):
        FamilyPanelArtifact.model_validate(payload)


def test_each_seed_block_assigns_distinct_targeted_and_sentinel_panels() -> None:
    with pytest.raises(ValidationError, match="must be distinct"):
        SeedBlockPanelAssignment(
            training_seed=11,
            targeted_panel=PanelLabel.A,
            sentinel_panel=PanelLabel.A,
        )

    payload = _payload()
    assignments = list(payload["seed_block_assignments"])
    assignments[0] = {
        "training_seed": 11,
        "targeted_panel": "C",
        "sentinel_panel": "B",
    }
    payload["seed_block_assignments"] = tuple(assignments)
    with pytest.raises(ValidationError, match="targeted_panel"):
        FamilyPanelArtifact.model_validate(payload)


def test_seed_block_assignments_are_unique_canonical_and_balanced() -> None:
    duplicate = _payload()
    duplicate["seed_block_assignments"] = (
        _assignment(11, PanelLabel.A),
        _assignment(11, PanelLabel.B),
    )
    with pytest.raises(ValidationError, match="unique training seeds"):
        FamilyPanelArtifact.model_validate(duplicate)

    unordered = _payload()
    unordered["seed_block_assignments"] = (
        _assignment(29, PanelLabel.B),
        _assignment(11, PanelLabel.A),
    )
    with pytest.raises(ValidationError, match="sorted by training seed"):
        FamilyPanelArtifact.model_validate(unordered)

    unbalanced = _payload()
    unbalanced["seed_block_assignments"] = (
        _assignment(11, PanelLabel.A),
        _assignment(29, PanelLabel.A),
    )
    with pytest.raises(ValidationError, match="balance targeted roles"):
        FamilyPanelArtifact.model_validate(unbalanced)


@pytest.mark.parametrize("allocation_seed", (-1, True, "101", MAX_ROOT_SEED + 1))
def test_allocation_seed_uses_the_strict_repository_seed_contract(
    allocation_seed: object,
) -> None:
    payload = _payload()
    payload["allocation_seed"] = allocation_seed

    with pytest.raises(ValidationError, match="allocation_seed"):
        FamilyPanelArtifact.model_validate(payload)


def test_unknown_artifact_fields_fail_closed() -> None:
    payload = _payload()
    payload["selected_before_m0"] = True

    with pytest.raises(ValidationError, match="selected_before_m0"):
        FamilyPanelArtifact.model_validate(payload)
