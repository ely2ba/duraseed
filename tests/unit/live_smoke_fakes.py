from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace


class RestClient:
    def __init__(self, service) -> None:  # type: ignore[no-untyped-def]
        self.service = service

    async def set_checkpoint_ttl_from_tinker_path_async(self, path, ttl):  # type: ignore[no-untyped-def]
        del path, ttl

    async def list_checkpoints_async(self, run_id):  # type: ignore[no-untyped-def]
        del run_id
        paths = [
            call[1]
            for call in self.service.calls
            if call[0] in {"save_sampler", "save_state"}
        ]
        return SimpleNamespace(
            checkpoints=[
                SimpleNamespace(
                    tinker_path=path,
                    expires_at=datetime.now(UTC) + timedelta(days=7),
                    checkpoint_type="sampler" if "/sampler/" in path else "state",
                )
                for path in paths
            ]
        )
