# RFC: common Stage-A max-token ratification

Status: Proposed; not accepted. No paid launch is authorized by this document.

## Problem

The plan provisionally names `max_tokens=256` for Stage A and requires one
common raise if valid behavior is truncated, but never froze a separate
method-path sampling design. Creating trained method branches merely to select
their cap would be circular: Stage-A calibration itself needs a cap. It would
also consume new evidence and part of the fixed `$300` envelope.

## Existing evidence

Before any Stage-A or Pilot outcome, the fixed M0 TCES reference calibration
sampled 128 `a_validation` tasks once at 4096 tokens and evaluated shorter caps
on those same raw trajectories. It required at least 64 answer terminations,
at most 5% censoring over all 128 items, and zero censored exact successes. It
observed 73 answer terminations and eight exact successes. Caps 512, 1024, and
2048 failed with respectively 15/3, 9/2, and 4/1 censored answers/censored exact
successes; 4096 passed with 0/0. Because any trajectory censored at 512 is also
censored at 256, the provisional 256 cap necessarily fails the unchanged
exact-success guard.

The preserved summary has SHA-256
`d2505e16bec8dc4374e4a5917d80a71095018407ab5d95937c06dd7997e9ae76`;
its raw generations and rewards have SHA-256
`a3854d5cfad49a75bcdb0e95b3e0cdaddbc2cf9685f54d88eb7d5cd2cc95ff70`
and `3e91a65617e97495d8646802fdab0da5f94c4bb5ca0922b39af13b3e84b285de`.
The source is the selected step-2 M0, seed 5, with no method outcome observed.

## Proposed decision

Ratify 4096 as the one common Stage-A completion cap for every method and for
all acquisition-calibration sampling. This applies the pre-result M0 truncation
gate conservatively; it does not claim that M0 and trained method completion
lengths are equivalent. Preserve length-stop and token-count diagnostics by
method. No method-specific cap, fresh cap sampling, grid extension, or
automatic post-outcome cap change is authorized. A future change requires a
new prospective protocol decision applied to every method.

This ratification adds no spend. The complete teacher-dose and worst-case
Stage-A preflights must include 4096 for every sampled completion and fit
together within the one `$300` acquisition-calibration cap. Their deterministic
local allocations are recorded separately. The `$25` live smoke remains an engineering gate only:
real, non-fabricated data; exact online/offline reward parity; verified stop
contract; full-state resume; and weights-only branching.

## Alternatives

Keep 256 despite observed censoring (rejected); use per-method caps (rejected);
or add a new trained-method cap experiment before calibration (circular and
unbudgeted). The conservative common 4096 cap uses the strongest existing
pre-outcome evidence without pretending it establishes cross-method
equivalence.

## Approval record

Authorizer: `[pending]`

Accepted at (UTC): `[pending]`

Until both fields are filled and Status is Accepted, acquisition calibration
must stop before Tinker access.
