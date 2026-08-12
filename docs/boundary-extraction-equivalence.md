# Boundary extraction equivalence

Status: exact for the current Extension-1 path; three-cohort freeze pending.

On 2026-08-12, the archived v0 helpers and fresh v1 functions were run locally
against the same existing fixture and deterministic input stream.  The fixture
used three families, two items per family, seed 5, scan ceiling 16, and variant
offset 32.  The broad manifests were identical objects: 6 records, canonical
manifest ID
`sha256:211333bcb26564892f28f6258191faa466e1fd904b6a43e34b0ec521f8e51361`.
The held-out confirmation manifests were also identical objects: 8 records,
manifest ID
`sha256:5304936d7229ac999caf8898a9129e53f313926874528681f3e99b43fa9811a4`.
The refinement stream contained 35 positive and 29 zero-success families; v0
and v1 returned identical ordered outputs of 35 candidates and 12 audit
families.

The executed Extension-1 artifact was then replayed with v1.  Its authenticated
broad evidence contained 1,024 generation rows, 1,024 reward rows, and 64
family summaries.  `choose_refinement_family_ids` returned the identical
ordered 34-candidate and 12-audit lists recorded by the completed
`boundary-refine-extension1-20260811T104725Z` preflight.  The frozen broad
manifest anchor is 256 records, 64 families, manifest ID
`sha256:9ba01ecdf39103b4695122ee635e2557afe3154123d5277fc8801a7c7d46b862`,
parented by the initial manifest
`sha256:9fd4f87c4d3f2e5111c933785a0568d635cc9b1cde7a356f5c5b920e98377dd9`.
The complete v1 deterministic regeneration produced the same decoded manifest
object: 256 records, 64 families, and the exact same manifest ID.  On the 34
capacity-cleared Extension-1 finalists, archived v0 and fresh v1 independently
produced the same held-out confirmation manifest object: 136 records, manifest
ID
`sha256:23665b09e285af293d8bda200317056f35ae091014462f721226b89c46eef4a6`,
canonical-payload SHA-256
`cced04f0d3958ef9307051e0af5bb971382cb885db81f356d5b33a2c5f172d31`,
and 2,510,974 canonical bytes.

The archived and v1 Extension-1 leakage guards were run independently on the
same production manifests and returned the identical object: `status=passed`,
one prior cohort, 408 prior records, 256 new records, and the four checked
dimensions `intended_family`, `task_id`, `content_hash`, and
`numeric_identity`.  On the existing two-family capacity fixture, the archived
parallel helper and direct carried capacity instrument returned two identical
`FamilyCapacityAudit` objects in the same order.  Both families passed the
one-instance `a_validation` requirement with capacity 2 and no generation
failure.

The v1 composition points are explicit.  `validate_broad_cohort_provenance`
authenticates the fixed cohort dimensions, manifest/measurement binding, and
parent chain; `audit_new_broad_cohort` composes the carried `audit_leakage`
instrument over prior broad and confirmation manifests.
`validate_confirmation_source_contract` binds both staged runs, manifests,
sampling parameters, run IDs, sampler/state checkpoints, and training step to
one completed M0 source before confirmation.
`validate_completed_confirmation_contract` then binds the completed
confirmation run, manifest, protocol amendment, checkpoint, sampling contract,
run ID, and unique sample IDs back to that source.  The runner must call
the carried `audit_family_split_capacity` before confirmation and again with
`PANEL_SELECTED_TEST_SINGLE_MINIMUM` for selected panels.  The pure
`reduce_confirmation_evidence` intentionally accepts distinct refinement and
combined confirmation summaries so the Stage-2 and Stage-3 gates cannot be
evaluated on the wrong item grid.  The full integrated replay passed with
exactly 15 eligible and 23 ineligible finalists and ordered-decision
serialization SHA-256
`a07e06e5f11666d6b703c53ba07503d275d7efbb96edf279775669f1c9ca409f`.

No three-cohort output was compared or authorized.  Public
`freeze_three_cohort_panels` raises unconditionally while
`BOUNDARY_PANEL_FREEZE_EQUIVALENCE_STATUS` is
`pending_three_cohort_equivalence_check`.  Three-cohort reduction is deliberately
not implemented in v1 yet: there is no complete three-cohort evidence against
which to verify its evidence, candidate-covariate, capacity, matching, and
artifact bindings.  The completed-evidence boundary-freeze gate must extract
and compare that reduction exactly
against the archived implementation after Extension-2 completes.  Only a later
reviewed commit may add or enable freeze output.
