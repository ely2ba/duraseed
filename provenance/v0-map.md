# DuraSeed v0 preservation map

This file bridges immutable v0 evidence to the greenfield v1 repository. The
scientific artifacts themselves are not rewritten.

## Source archive

- Git tag: `archive/v0`
- Tagged commit: `a9c65f92c857a9f51e26fdfe08b7ef6e30ee9597`
- Tag is present on the existing GitHub origin.
- v0 `runs/` tarball:
  `../DuraSeed-v1-preservation/archives/duraseed-v0-runs.tar.gz`
  — SHA-256
  `ead1829089149842f788c24c6acd6abb09e647f94a8bf1b1aaf81f1101c27daa`
  — 281 files.
- v0 `artifacts/` tarball:
  `../DuraSeed-v1-preservation/archives/duraseed-v0-artifacts.tar.gz`
  — SHA-256
  `82cc828b3cbae64a62b8c7941d927221a21fa68d3bd7838765d0e2d19be32f4c`
  — 18 files.
- Both archives passed gzip integrity checks and their file counts match the
  source trees.

## Ratified Stage-B decision

- Accepted ratification record (preserved outside the public repository),
  SHA-256:
  `4da07397d2b2050ab185f19449217096acd01456ac1feb07a40b1e9f17b125e7`
- Approval: Ely, 2026-08-11 UTC.
- Private decision artifact: preserved byte-for-byte in the v0 archive; its
  path-free public view is
  `frozen/v0/runs/tinker-calibration/maps-probe/maps-shortest2-20260810T144606Z/maps_recipe_decision.public.json`.
- Decision-artifact SHA-256:
  `36015e1e08a0a96b4f4b0d7dd8480a679294c1cfc2a026d3ab96073c5378acf8`
- Frozen recipe: `shortest2_cap2`, learning rate `3e-4`, step 480.
- Commit `9d9d476053f29682d6be00960f85c15098b3aace` froze the
  corrective profile and gate before execution; selector commit
  `af0b92d5251a6e571c4e9de7f2d1aaa7130f37be` also predates execution.
- Step 480 is the mechanical earliest pass under the unchanged gate; step 640
  is excluded. No cross-profile equivalence is claimed.
- The original decision JSON remains byte-for-byte unchanged.

## Phase 2 scientific artifact carry-over

- Producing modules: v0 calibration runners at commits recorded by each
  artifact; preservation date: 2026-08-12 UTC.
- Preserved root: `frozen/v0/runs/tinker-calibration/`.
- Strict sorted file manifest: `provenance/v0-artifacts.sha256` — 193 files,
  507,150,581 bytes. Each listed file matches the corresponding Phase 0 archive
  and read-only v0 source byte for byte. Three source records with machine-local
  paths are retained only in private preservation; their path-free views are
  separately hash-listed in `provenance/public-projections.sha256`.
- M0 evidence includes the raw calibration, task-agnostic format calibration,
  and selected reconstruction: manifests, generations, rewards, decisions,
  audits, checkpoint ledgers, summaries, prices, and version-bearing records.
- TCES evidence includes both cap calibrations and the broad, refine, and
  held-out confirmation stages. The confirmation evidence contains all 38
  finalists: 15 observation-gate passes and 23 failures, plus split-capacity
  audits and the exact source manifest, generations, rewards, family table,
  and summary.
- The completed Extension 1 broad and refinement evidence is preserved with
  its exact manifests, generations, rewards, family tables, summaries, prices,
  run records, and path-free public views of the two preflights required to
  authenticate the unfinished confirmation handoff. The aborted pre-output
  Extension 1 attempt is not carried.
- MAPS evidence includes both profile screens and all completed probe curves,
  their manifests, generations, rewards, metrics, checkpoint ledgers,
  summaries, prices, and the ratified `shortest2_cap2` decision.
- Four explicitly non-scientific synthetic replay fixtures are retained under
  `frozen/v0/artifacts/smoke/`: `analysis.json`, `item_transitions.csv`,
  `base_scan.json`, and `semantic_verifier_checks.json`. Their embedded
  `fixture_kind`/`scientific_claim` labels remain unchanged; they are used only
  to verify v1 reducers and verifiers against v0 outputs.
- Engineering-smoke outputs, three failed M0-recovery attempts, the aborted
  Extension 1 attempt, superseded MAPS decision snapshot, billing placeholders,
  and nonessential preflight metadata are not part of the compact scientific
  carry-over. Eight nonessential files carrying an obsolete internal protocol
  identifier were also excluded.
- The three private originals remain unchanged under their recorded SHA-256
  values: broad preflight `6597b600…c9bc89a`, refinement preflight
  `5c02110b…c46f7c`, and MAPS decision `36015e1e…78acf8`. Public projections
  remove only these non-scientific JSON fields:
  the MAPS arm's `run_directory`; the broad preflight's
  `planned_run_directory` and two prior-run directory fields; and the
  refinement preflight's `planned_run_directory` and `broad_run_directory`.
  Every other JSON value is unchanged. Boundary source authentication returns
  the same contract from the private originals and public projections.
- A case-insensitive final scan found no confidential-document reference, API key,
  password, bearer token, private key, or known credential-shaped value.

## Operational preservation evidence

- Checkpoint retention and export: all 109 non-smoke artifact-referenced paths
  are verified non-expiring. All 107 sampler checkpoints were downloaded; the
  two full training states remain remote-only because Tinker's archive endpoint
  rejects non-`sampler_weights/` checkpoint IDs. See
  `provenance/checkpoints.md` and the machine-readable inventory under
  `../DuraSeed-v1-preservation/checkpoint-inventory/`.
  The 321-file export manifest has SHA-256
  `54dc7ca7d00b1d36e40f70aaadbcce03a574db2c41843351d2fdda4fcc6209ae`.
- Billing: see `provenance/billing.md`; raw organization usage and its derived
  reconciliation are stored under
  `../DuraSeed-v1-preservation/billing/`.

`provenance/v0-artifacts.sha256` is the authoritative map for byte-identical
public v0 files; `provenance/public-projections.sha256` covers the three
derived public views. The Phase 3 code carry-over is separate and governed by
`provenance/immutable-carryover.sha256`.
