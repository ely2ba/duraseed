"""Prospectively frozen coordinates and spend caps for the bounded repair."""

REPAIR_SEEDS = (17, 37)
REPAIR_DOSE = 2
REPAIR_LEARNING_RATE = 1e-4
REPAIR_CHECKPOINT_UPDATES = (4, 8, 12)
REPAIR_SELECTION_RULE = "earliest_checkpoint_passing_all_orientations"
REPAIR_SPEC = {
    "seeds": REPAIR_SEEDS,
    "dose": REPAIR_DOSE,
    "learning_rate": REPAIR_LEARNING_RATE,
    "checkpoint_updates": REPAIR_CHECKPOINT_UPDATES,
    "selection_rule": REPAIR_SELECTION_RULE,
}
REPAIR_TEACHER_TOKEN_CEILINGS = (684_720, 21_233_664, 785_664)
REPAIR_TEACHER_CAP_USD = 44.27
REPAIR_STAGE_A_CAP_USD = 155.09
REPAIR_AGGREGATE_CAP_USD = 199.36
LIFETIME_CALIBRATION_CAP_USD = 300.0
ORIGINAL_TEACHER_CAP_USD = 128.82


__all__ = [name for name in globals() if name.startswith("REPAIR_")] + [
    "LIFETIME_CALIBRATION_CAP_USD",
    "ORIGINAL_TEACHER_CAP_USD",
]
