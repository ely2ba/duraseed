# RFC: exact teacher-dose panel-split source extraction

Status: Accepted by Ely on 2026-08-13 UTC for the narrow extraction and fixture
equivalence check only. No paid launch is authorized by this document.

The calibration source builder needs v0
`tinker_teacher_dose._build_panel_split_manifest` for `a_seed_train` and
`a_seed_gate`. The attempted v1 reuse of
`stage_a_prompt_pools._generate_family_records` was not equivalent: it scanned
8× rather than 32× per-family capacity and emitted different manifest identity
and metadata. That attempted builder has been removed.

Alternatives are to retain the archived runner dependency, redesign the source
contract, or extract only this deterministic builder. The smallest safe change
is the narrow extraction, with no other archived-runner dependency bundled.

The accepted implementation scope is one deterministic builder extraction and
an old-vs-new comparison of the complete `a_seed_train` and `a_seed_gate`
manifest objects on the existing fixtures, using the exact forbidden-record
sequence. Record row counts, canonical byte lengths, SHA-256 values, and zero
divergence under `docs/`. Any divergence is a hard stop. No other archived
runner dependency is included; discovering one requires a new RFC.

Production equivalence and enablement are explicitly deferred until the
three-cohort boundary source exists. At that point, but before its output is
used for calibration, run the same old-vs-new comparison on that completed
frozen source and bind the final equivalence bytes to an authorization
artifact. Until then the production panel-split authorization hash remains
unset and the calibration launch must stop locally before Tinker access.

Authorizer: Ely

Accepted at: 2026-08-13T11:48:51Z
