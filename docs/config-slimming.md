# v1 configuration carry-over

`src/duraseed/config.py` is fresh v1 code, not immutable carry-over. It retains
the concrete configuration APIs used by the carried instrument and the next
two gates: TCES/MAPS generator arguments, calibration seeds, boundary-screen
sampling, teacher-dose and Stage-A selections, the frozen Stage-B selection,
Tinker integration values, spend caps, loading, unresolved-selection reporting,
and canonical config hashing.

The v0 method/contrast enums, protocol prose models, duplicate split-size
models, publication-outcome schemas, and validators that only restated the
YAML were dropped. The complete public scientific values remain in
`duraseed_pilot_config.yaml`; no frozen value was rewritten by this slimming.
