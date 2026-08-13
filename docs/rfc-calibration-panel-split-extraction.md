# RFC: exact teacher-dose panel-split source extraction

Status: Proposed; not accepted. No implementation or paid launch is authorized
by this document.

The calibration source builder needs v0
`tinker_teacher_dose._build_panel_split_manifest` for `a_seed_train` and
`a_seed_gate`. The attempted v1 reuse of
`stage_a_prompt_pools._generate_family_records` was not equivalent: it scanned
8× rather than 32× per-family capacity and emitted different manifest identity
and metadata. That attempted builder has been removed.

Alternatives are to retain the archived runner dependency, redesign the source
contract, or extract only this deterministic builder. The smallest safe change
is the narrow extraction, with no other archived-runner dependency bundled.

If accepted, compare the old and new complete manifest objects for both splits
on the same existing fixtures and the completed frozen boundary source, using
the exact forbidden-record sequence. Record row counts, canonical byte lengths,
SHA-256 values, and zero divergence under `docs/` before enabling calibration.
Any divergence is a hard stop. Until acceptance and that evidence exist, the
real calibration launch must stop before Tinker access.
