# Progressive teacher-exposure repair

Status: accepted prospectively by Ely on 17 August 2026, after the failed
teacher-seed verification and before Stage A or Pilot 0. Implementation and
launch state are tracked separately in [`STATUS.md`](STATUS.md).

## Why this amendment exists

The first acquisition-calibration run found a passing seed-17 arm at dose 2,
learning rate `1e-4`, and 16 updates. The same coordinate then failed the
unchanged format gate in the opposite seed-37 panel orientation. Many target
and sentinel generations repeated until the shared 4096-token cap. Inspection
identified learned nontermination at that checkpoint, not a transient remote
request, checkpoint-loading, parser, or verifier failure.

The terminal run and its artifacts remain immutable diagnostic evidence. They
are not reopened, repaired in place, or treated as a Stage-A or Pilot result.
The 16-update coordinate is rejected.

## Accepted decision

Keep the selected dose 2, learning rate `1e-4`, renderer, decoder, crossed
panels, evaluation samples, and every existing teacher-seed gate unchanged.
Run exactly two progressive training trajectories: seed 17 in one frozen panel
orientation and seed 37 in the other. Evaluate the predeclared checkpoints at
updates 4, 8, and 12. Freeze the earliest update that passes all gates in both
orientations.

There is no third calibration verification pair. The two opposite orientations
are the complete calibration check for this nuisance recipe. Pilot 0 then uses
its independent seeds 11 and 29. Before either method branch runs, both Pilot
origins must pass the unchanged warm-start health gate; failure of either
origin stops the whole Pilot, with no substitute seed.

## Stopping rule and scope

If no common passing checkpoint exists by update 12, stop teacher-exposure
calibration for a new prospective decision. Do not automatically raise the
dose, change the learning rate, extend the trajectory, add seeds, weaken a
gate, or alter the completion cap.

This is a bounded repair of a shared pre-method exposure setting. It does not
compare B-S with B-G, test the DuraSeed hypothesis, change either method, or
add a broad hyperparameter sweep. Its purpose is only to find a common finite
warm start that is safe to carry into the already-planned experiment.
