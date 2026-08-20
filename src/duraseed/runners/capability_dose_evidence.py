"""Raw-observation projection for the frozen capability-dose reducer."""

from __future__ import annotations

from math import fsum

from duraseed.runtime import SampleObservation
from duraseed.runners.stage_a_amended_arms import paired_answer_tag_items
from duraseed.training.capability_dose import DosePanelEvidence


def detected_loop(token_ids: tuple[int, ...]) -> bool:
    """Apply the frozen final-1,024-token lag-agreement detector."""

    tail = token_ids[-1024:]
    for lag in range(4, min(256, len(tail) - 1) + 1):
        compared = len(tail) - lag
        matches = sum(
            left == right for left, right in zip(tail[lag:], tail[:-lag], strict=True)
        )
        if compared and matches / compared >= 0.80:
            return True
    return False


def panel_evidence(
    origin: tuple[SampleObservation, ...],
    current: tuple[SampleObservation, ...],
) -> DosePanelEvidence:
    """Retain paired gates and the complete frozen Tier-3 panel profile."""

    logprobs = tuple(value for row in current for value in row.logprobs)
    capped = tuple(row for row in current if row.generation.stop_reason == "length")
    loops = sum(
        detected_loop(row.generation.completion_token_ids or ()) for row in capped
    )
    strategies = {
        row.reward.exact_verification.strategy_family_id
        for row in current
        if row.reward.exact_verification.strategy_family_id is not None
    }
    return DosePanelEvidence(
        paired_answer_tag_items(origin, current),
        loops,
        len({row.generation.completion_text for row in current}),
        fsum(row.generation.sampled_tokens for row in current) / len(current),
        -fsum(logprobs) / len(logprobs) if logprobs else None,
        len(strategies),
    )


__all__ = ["detected_loop", "panel_evidence"]
