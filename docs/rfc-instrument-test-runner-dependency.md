# RFC: instrument-test dependency on an archived runner

Problem: `tests/unit/test_panel_matching.py` imported `_build_broad_manifest`
from the prohibited v0 `tinker_boundary_scan` runner, so the immutable test
closure could not run in v1.

Alternatives: copy the runner; add an import-compatible shim; edit or omit the
runner-coupled test; or stop the rebuild.

Smallest change: permit one reviewed test edit that replaces the runner helper
with equivalent setup built from carried instrument APIs, while leaving all
instrument source and every test assertion byte-identical.

Status: accepted by Ely on 2026-08-12, conditional on equivalence verification.
This is the only approved Phase-3 test-edit exception. Any further dependency
on an archived runner requires a new RFC and an immediate stop.

## Equivalence evidence

- Source: archived commit `da709622ff7ee77bcdb1ac5a3a5185244ec3eeae`.
  The archived runner source SHA-256 is
  `05535cdd1af1f01e2a2b614bcc1ff235fed61da234f1cf23f3bb02d66af61655`.
- Fixture: the existing test's exact `TCESGeneratorConfig` and arguments
  (`n_operands=3`, operands 2–20, depth 3, 5 AST nodes, 256 attempts; one
  family, two items, scan ceiling 8, variant offset 16).
- Command: a one-time local equivalence script invoked
  the v0 helper read-only in the archived repository environment and the v1
  setup against the same fixture, then asserted equality of their canonical
  JSON bytes and decoded manifest objects.
- Result: both produced manifest ID
  `sha256:d349ecb6084fe23f79520ac4fd32596c893ef8672be4238ae0834f75260907ba`.
  The 2,439 canonical bytes were identical, with SHA-256
  `9143c7afe53d07c956e83a94e9e88e597ecf585eec22402e7e0b32a5696cc3fb`.
  The edited test passed all 12 cases locally.
- Scope check: the archived test was SHA-256
  `6937aa0e20df98e9a9b1b0535fb1bda32d95a9a3d736401aa1e42376c190cb73`.
  Only imports and fixture construction changed; all assertion lines remain
  byte-identical. The edited test is deliberately absent from the immutable
  carry-over manifest.
