"""Frozen seed coordinates used by acquisition calibration."""

from duraseed.provenance import derive_namespaced_seed


def stage_a_group_seeds(step: int, group_index: int, task_id: str) -> tuple[int, ...]:
    root = derive_namespaced_seed(
        17, "tinker.stage_a.bg_rollout", step, group_index, task_id
    )
    return tuple(
        derive_namespaced_seed(root, "tinker.smoke.group_sample", sample_index)
        for sample_index in range(8)
    )


def ephemeral_sampler_id(
    run_id: str, attempt_name: str, learning_rate: float, step: int
) -> str:
    """Provide a collision-free identity when Tinker omits an ephemeral model ID."""

    return f"ephemeral:{run_id}:{attempt_name}:B-G:{learning_rate:.17g}:step-{step}"


def ephemeral_sampler_path(
    sampler: object, run_id: str, attempt_name: str, learning_rate: float, step: int
) -> str:
    return str(
        getattr(
            sampler,
            "model_id",
            ephemeral_sampler_id(run_id, attempt_name, learning_rate, step),
        )
    )


__all__ = ["ephemeral_sampler_id", "ephemeral_sampler_path", "stage_a_group_seeds"]
