from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from duraseed.data.boundary_confirmation import ConfirmationEvidence
from duraseed.data.panel_capacity import FamilyCapacityAudit
from duraseed.runners import RunnerGateError
from duraseed.runners.boundary_confirm_resume import _result_payload
from duraseed.runners.boundary_extension import BoundaryBlockResult
from tests.unit.test_boundary_scan import _shared_family_manifest


def test_extension2_result_payload_serializes_nested_manifest() -> None:
    manifest = _shared_family_manifest(1)
    family_id = manifest.records[0].intended_family
    audit = FamilyCapacityAudit(family_id, (), 1, True)
    result = BoundaryBlockResult(
        "extension_2",
        (family_id,),
        manifest,
        ConfirmationEvidence((family_id,), (family_id,), (family_id,), ()),
        (audit,),
    )

    payload = _result_payload(result)

    assert payload["confirmation_manifest"]["manifest_id"] == manifest.manifest_id
    assert payload["capacity_audits"][0]["family_id"] == family_id


def test_launch_finishes_capacity_preparation_before_remote_client(
    monkeypatch, tmp_path: Path
) -> None:
    import duraseed.runners.boundary_confirm_launch as launch

    order: list[str] = []
    source = SimpleNamespace(
        contract=SimpleNamespace(
            project_id="project", sampler_checkpoint_path="tinker://m0"
        )
    )
    snapshot = object()
    prepared = object()

    class SDK:
        sdk_version = "test"
        cookbook_version = "test"

        def get_renderer(self, *_args, **_kwargs):
            return object()

    monkeypatch.setattr(launch, "require_clean_worktree", lambda **_kwargs: None)
    monkeypatch.setattr(launch, "load_pilot_config", lambda _path: object())
    monkeypatch.setattr(launch, "load_boundary_live_source", lambda *_args: source)
    monkeypatch.setattr(
        launch,
        "validate_confirmation_resume",
        lambda *_args, **_kwargs: order.append("validate") or snapshot,
    )
    monkeypatch.setattr(launch, "_git_commit", lambda: "commit")
    monkeypatch.setattr(
        launch,
        "prepare_boundary_confirmation",
        lambda *_args, **_kwargs: order.append("capacity") or prepared,
    )
    monkeypatch.setattr(launch, "load_sdk", lambda: order.append("sdk") or SDK())
    monkeypatch.setattr(
        launch.importlib,
        "import_module",
        lambda _name: SimpleNamespace(get_tokenizer=lambda _model: object()),
    )
    monkeypatch.setattr(
        launch,
        "create_service",
        lambda *_args, **_kwargs: order.append("service") or object(),
    )

    async def create_sampler(*_args, **_kwargs):
        order.append("sampler")
        return object()

    async def execute(*_args, **_kwargs):
        order.append("execute")

    monkeypatch.setattr(launch, "create_sampler", create_sampler)
    monkeypatch.setattr(launch, "execute_boundary_confirmation_resume", execute)

    result = asyncio.run(
        launch.run_remote_boundary_confirmation_resume(
            project_id="project",
            run_id="boundary-live",
            source_root=tmp_path / "source",
            output_root=tmp_path,
            config_path=tmp_path / "config.yaml",
            human_approval=True,
        )
    )

    assert result == tmp_path / "boundary-live"
    assert order == ["validate", "capacity", "sdk", "service", "sampler", "execute"]


def test_confirmation_resume_requires_explicit_human_approval(tmp_path: Path) -> None:
    import duraseed.runners.boundary_confirm_launch as launch

    try:
        asyncio.run(
            launch.run_remote_boundary_confirmation_resume(
                project_id="project",
                run_id="boundary-live",
                source_root=tmp_path,
                output_root=tmp_path,
                config_path=tmp_path / "config.yaml",
                human_approval=False,
            )
        )
    except RunnerGateError as error:
        assert "explicit human approval" in str(error)
    else:  # pragma: no cover
        raise AssertionError("confirmation resume accepted no human approval")
