"""Frozen reducer for the single capability-targeted B-S dose run."""

from __future__ import annotations

from duraseed.training.capability_dose_evidence import (
    CADENCE_UPDATES,
    CONFIRMATION_SUCCESSES,
    DOSE_LEARNING_RATE,
    EPOCH_UPDATES,
    MAX_CONFIRMATIONS,
    MAX_UPDATES,
    MAXIMUM_LENGTH_STOP_RATE,
    MAXIMUM_LOOP_FRACTION,
    MAXIMUM_VALID_TAG_DROP,
    MINIMUM_FAMILY_REACHABILITY,
    THETA_SUCCESSES,
    DoseAction,
    DoseAssessment,
    DoseCriterion,
    DoseDecision,
    DoseEvaluationEvidence,
    DosePanelEvidence,
)


def assess_dose_evaluation(evidence: DoseEvaluationEvidence) -> DoseAssessment:
    """Apply only the gates authorized at this evaluation phase."""

    target = evidence.target
    tier1 = (
        DoseCriterion(
            "finite_metrics",
            1,
            True,
            evidence.finite_metrics,
            evidence.finite_metrics,
            True,
        ),
        DoseCriterion(
            "leakage_clean",
            1,
            True,
            evidence.leakage_clean,
            evidence.leakage_clean,
            True,
        ),
    )
    if evidence.phase == "cadence":
        return DoseAssessment(
            evidence,
            tier1
            + (
                DoseCriterion(
                    "cadence_tripwire_valid_tag_retention",
                    2,
                    True,
                    target.valid_tag_retention_passed,
                    target.current_valid_tag_rate,
                    target.origin_valid_tag_rate - MAXIMUM_VALID_TAG_DROP,
                ),
                DoseCriterion(
                    "cadence_tripwire_absolute_length_stop",
                    2,
                    True,
                    target.length_stop_rate <= MAXIMUM_LENGTH_STOP_RATE,
                    target.length_stop_rate,
                    MAXIMUM_LENGTH_STOP_RATE,
                ),
                DoseCriterion(
                    "cadence_loop_fraction",
                    3,
                    False,
                    None,
                    target.loop_fraction,
                    MAXIMUM_LOOP_FRACTION,
                ),
                DoseCriterion(
                    "cadence_capability_gain_lower_bound",
                    3,
                    False,
                    None,
                    target.paired_gain.approximate_95_interval[0],
                    0.0,
                ),
                DoseCriterion(
                    "cadence_family_reachability",
                    3,
                    False,
                    None,
                    target.family_reachability,
                    MINIMUM_FAMILY_REACHABILITY,
                ),
            ),
        )
    return DoseAssessment(
        evidence,
        tier1
        + (
            DoseCriterion(
                "capability_gain_approximate_95_lower_bound",
                2,
                True,
                target.paired_gain.approximate_95_interval[0] > 0,
                target.paired_gain.approximate_95_interval[0],
                0.0,
            ),
            DoseCriterion(
                "family_reachability",
                2,
                True,
                target.family_reachability >= MINIMUM_FAMILY_REACHABILITY,
                target.family_reachability,
                MINIMUM_FAMILY_REACHABILITY,
            ),
            DoseCriterion(
                "valid_tag_retention",
                2,
                True,
                target.valid_tag_retention_passed,
                target.current_valid_tag_rate,
                target.origin_valid_tag_rate - MAXIMUM_VALID_TAG_DROP,
            ),
            DoseCriterion(
                "absolute_length_stop",
                2,
                True,
                target.length_stop_rate <= MAXIMUM_LENGTH_STOP_RATE,
                target.length_stop_rate,
                MAXIMUM_LENGTH_STOP_RATE,
            ),
            DoseCriterion(
                "loop_fraction",
                2,
                True,
                target.loop_fraction <= MAXIMUM_LOOP_FRACTION,
                target.loop_fraction,
                MAXIMUM_LOOP_FRACTION,
            ),
        ),
    )


def decide_dose(evaluations: tuple[DoseEvaluationEvidence, ...]) -> DoseDecision:
    """Reduce the ordered durable evaluation history to its next frozen action."""

    if not evaluations:
        raise ValueError("dose decision requires at least one evaluation")
    latest = evaluations[-1]
    assessment = assess_dose_evaluation(latest)
    confirmations = sum(row.phase == "confirmation" for row in evaluations)
    if not assessment.tier1_passed:
        return DoseDecision(
            "tier2_degenerated",
            latest.update,
            assessment,
            confirmations,
            "Tier-1 evidence failed; the solver-SFT route ends",
        )
    if latest.phase == "cadence":
        cadence = tuple(row for row in evaluations if row.phase == "cadence")
        if len(cadence) >= 2 and all(
            not assess_dose_evaluation(row).cadence_tripwire_passed
            for row in cadence[-2:]
        ):
            return DoseDecision(
                "tier2_degenerated",
                latest.update,
                assessment,
                confirmations,
                "two consecutive cadence cost/format tripwires failed",
            )
        if (
            latest.target.success_count >= THETA_SUCCESSES
            and confirmations < MAX_CONFIRMATIONS
        ):
            return DoseDecision(
                "confirm",
                latest.update,
                assessment,
                confirmations,
                "the cadence success threshold triggered a 192-sample confirmation",
            )
        return DoseDecision(
            "continue",
            latest.update,
            assessment,
            confirmations,
            "cadence monitoring permits the frozen dose schedule to continue",
        )
    if latest.phase == "confirmation":
        if latest.target.success_count < CONFIRMATION_SUCCESSES:
            return DoseDecision(
                "continue",
                latest.update,
                assessment,
                confirmations,
                "confirmation missed 38/192; retain it and continue with stop armed",
            )
        return DoseDecision(
            "proceed_to_pilot" if assessment.tier2_passed else "tier2_degenerated",
            latest.update,
            assessment,
            confirmations,
            (
                "confirmation reached the overlap zone with clean Tier-2 evidence"
                if assessment.tier2_passed
                else "qualifying confirmation failed a Tier-2 viability gate"
            ),
        )
    if latest.target.success_count < CONFIRMATION_SUCCESSES:
        health_passed = assessment.tier1_passed and assessment.tier2_passed
        action: DoseAction = "dose_limited" if health_passed else "tier2_degenerated"
        return DoseDecision(
            action,
            latest.update,
            assessment,
            confirmations,
            "epoch cap missed the overlap zone"
            if health_passed
            else "epoch cap failed a viability gate",
        )
    return DoseDecision(
        "proceed_to_pilot" if assessment.tier2_passed else "tier2_degenerated",
        latest.update,
        assessment,
        confirmations,
        "epoch cap passed"
        if assessment.tier2_passed
        else "epoch cap failed a Tier-2 gate",
    )


__all__ = [
    "CADENCE_UPDATES",
    "CONFIRMATION_SUCCESSES",
    "DOSE_LEARNING_RATE",
    "DoseAssessment",
    "DoseCriterion",
    "DoseDecision",
    "DoseEvaluationEvidence",
    "DosePanelEvidence",
    "EPOCH_UPDATES",
    "MAX_CONFIRMATIONS",
    "MAX_UPDATES",
    "THETA_SUCCESSES",
    "assess_dose_evaluation",
    "decide_dose",
]
