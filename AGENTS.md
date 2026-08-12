# AGENTS.md — DuraSeed engineering rules

Applies to all contributors, human or agent. Read `docs/MAP.md` first.

Prime directive: small, flat, readable. A competent reader must understand any
module in one sitting. The science lives in `tasks/`, `data/`,
`evaluation/analysis.py`, and frozen artifacts—never in infrastructure. When in
doubt, write less code.

## Hard limits (enforced by `tools/check_size.py` in CI)

- No newly written v1 source or test file may exceed 400 physical lines.
- Newly written `runtime/` must remain under 1,200 physical lines total.
- Newly written phase and runner scripts must remain under 300 physical lines.
  They compose `runtime/`, `tasks/`, `data/`, and `evaluation/` and contain no
  reusable logic of their own.
- Newly written paths may not exceed directory depth 3 below `src/duraseed/`.
- The sole exemption is a Phase-3 file copied bit-for-bit from archived v0 and
  enumerated by repository-relative path and SHA-256 in
  `provenance/immutable-carryover.sha256`.
- `runtime/`, `runners/`, `tools/`, and the slimmed v1 `config.py` may never
  claim carry-over exemption.
- Before Phase 3, an absent carry-over manifest means there are no exemptions.
  The manifest is generated only after the complete carry-over set is known and
  is frozen in the same commit as those files.
- CI must fail if a manifest entry is malformed, duplicated, unsorted, missing,
  outside the repository, ineligible for exemption, or has a hash mismatch. An
  over-limit unlisted file must also fail.
- There is no size allowlist and no inline, environment-variable, or
  command-line exemption mechanism.
- Fresh-code and immutable carry-over line totals are reported separately. The
  approximately 9,000-line target applies only to newly written v1 source and
  test code.

## Build rules

1. Implement only the current phase's requirement. State which requirement you
   are implementing before writing code. No options, hooks, layers, or
   abstractions without a current caller.
2. One caller means code lives beside the caller. Two real callers means
   promotion to `runtime/`. Never design for a hypothetical third caller.
3. No abstract base classes, protocols, registries, factories, or plugin
   systems with a single implementation.
4. Validation exists to protect money and frozen scientific artifacts only:
   spend caps, sealing, leakage, checkpoint provenance, and replay
   equivalence. No per-field type paranoia and no defensive checks against the
   project's own code.
5. Prefer deletion. Any change adding more than 300 net lines must justify
   itself in the commit message.
6. Executed calibration and recovery modules are provenance: tag, archive, and
   remove them from the working tree once superseded. Scientific decisions
   live in artifacts, not code.
7. Never edit a file listed in
   `provenance/immutable-carryover.sha256`. An approved scientific change must
   first remove that file's exemption, perform any required replay, and make
   the resulting file comply with current fresh-code limits.
8. No stubs, placeholders, `NotImplementedError` for planned features, or TODO
   scaffolding. If it is not needed now, it does not exist.
9. Tests consist of property tests for verifiers, enumerators, leakage, and
   sealing, plus behavioral mock-client tests for `runtime/`. Do not write
   tests that merely restate configuration.
10. No new runtime dependency without an RFC.
11. Tinker and other remote access require explicit human launch confirmation
    and a spend cap. Never fabricate remote data, and never allow mock output
    into a scientific artifact path.

## Escape hatch

If a rule blocks real scientific work, write a five-line RFC stating the
problem, alternatives, and smallest change. Never work around a rule silently.
