# v0 → v1 replay equivalence

Status: **PASSED** on 2026-08-12 UTC. The replay made no remote calls and
observed zero divergences.

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tools/replay_equivalence.py
```

The command first verified every consumed artifact against
`provenance/v0-artifacts.sha256`, then replayed the evidence through the
immutable v1 instrument. The path-clean public artifact manifest SHA-256 is
`6b9a30a1e8271c08dea997ea5c4a19bb6a4265de68d6265c2d37f9f3c6551a7c`.

## Exact-verifier replay

- TCES: all 2,432 held-out confirmation completions replayed, covering all 38
  finalists (the locked 15 passers and 23 failed finalists). Every complete
  structured `VerificationResult`, reward, and generation/reward linkage was
  identical. Treating one row per family as required family coverage leaves
  2,394 additional audit rows; the full census therefore strictly subsumes
  the requested random audit of at least 200 rows.
- TCES outcome coverage: 199 successes and 2,233 failures across
  `answer_too_long` (5), `ast_limit_exceeded` (145), `empty_answer` (12),
  `invalid_character` (638), `invalid_syntax` (55), `missing_answer_tag`
  (779), `multiple_answer_tags` (23), `operand_multiset_mismatch` (238), and
  `wrong_target` (338).
- MAPS: all 6,144 frozen `shortest2_cap2` validation completions replayed.
  Every complete structured result, reward, and linkage was identical: 1,006
  successes and 5,138 failures across `illegal_instruction` (48),
  `invalid_program` (56), `missing_answer_tag` (56), `multiple_answer_tags`
  (1), `program_too_long` (63), and `wrong_target` (4,914).
- The scientific `shortest2_cap2` calibration curve was rebuilt directly
  from those linked rewards at steps
  `0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480, 640`. Its 12 exact scores were
  `0.048828125, 0.05859375, 0.06640625, 0.072265625, 0.0625, 0.064453125,
  0.072265625, 0.083984375, 0.203125, 0.35546875, 0.40625, 0.470703125`.
  The resulting absolute AUC `0.30164642333984376`, raw GainAUC
  `0.25281829833984376`, and headroom-normalized GainAUC
  `0.2657966504103982` exactly match the frozen public values.

Key corpus hashes:

| Evidence | SHA-256 |
| --- | --- |
| TCES confirmation manifest | `923632195faea4e57ce841387481cca44719459c573240fc67b4563df9690e57` |
| TCES confirmation generations | `68d6c844543be6666b08f9d371bbfe1b9788092697d1ea8c372bb57117d6fb34` |
| TCES confirmation rewards | `9551d7f2e39b00adb25f530b4a8769c1d99954bcb23966af334938efb9d8500a` |
| MAPS validation manifest | `2d8a318828f37296cb4f694ad9c868988788a64b453269619dd1773188176c29` |
| MAPS generations | `cce9d4487f776b8532213875ed62938c536deb9b940601adb1be015cb1f3769c` |
| MAPS rewards | `29994553d433d23f4445195840fb7f4401567c4c9217cb330099ca48e354fbd0` |

## Boundary decision replay

- Stage 1: all 64 family aggregates were field-identical; the replay recovered
  exactly 35 success-positive candidates and the same 12-family deterministic
  zero-success audit.
- Stage 2: all 64 refined aggregates were field-identical; the frozen gate
  recovered exactly the same 38 finalists and split-capacity IDs.
- Stage 3: all 38 combined aggregates, eligibility booleans, and ordered
  exclusion-reason lists were identical. Exactly 15 passed and 23 failed.
- Ordered-decision serialization SHA-256:
  `a07e06e5f11666d6b703c53ba07503d275d7efbb96edf279775669f1c9ca409f`.
  This corrects a clerical value in the first note; rerunning the unchanged
  evidence and ordered decision tuples produced the value above with the same
  exact 15-pass/23-fail result.

## Analysis replay

The retained analysis input is explicitly labeled `synthetic_analysis_only`
and `scientific_claim=false`; this replay validates analysis semantics, not a
scientific finding. The four recorded item trajectories reproduced:

- equal-item Jeffreys posterior means: M0 `0.08823529411764706`, post-A `0.5`,
  post-B `0.38235294117647056`;
- the seven-point reported Cover@τ curve, including the artifact-recorded
  τ=0.25 values `0.04463319131698457`, `0.9610064677909834`, and
  `0.7409864741762643`;
- the complete hard-transition table and pass@4 means: M0
  `0.23750000000000002`, post-A `0.9498626373626373`, post-B
  `0.8144230769230769`;
The retained artifact directly supplies the primary Cover value, hard
transitions, and pass@4. The complete seven-point Cover curve is checked by
its archived canonical SHA-256
`eda23463171ace06ce33e7b8d3ba51518e35cc87f5ed4e5bfc68c7e0e4193b88`.
The three GainAUC variants are replayed from the scientific MAPS calibration
curve above rather than being inferred from this synthetic fixture.

## Instrument identity

The files below are hash-enforced by
`provenance/immutable-carryover.sha256`:

| Instrument | SHA-256 |
| --- | --- |
| Boundary reducers and gates | `b73491fe10e0d3ed2f49170e52e1042aad136633d34f52b692b0a23345b1960f` |
| Analysis estimands | `66b648e4be89598d01fea4988932574bb3e3f28511b5b4040593d946ba88b195` |
| TCES verifier | `c3889e09ccd0685584b184f60dbac93ddafc29c76da493ee7479c66a0287c88e` |
| MAPS verifier | `77c068d19abaf4c0499c2b82adc2de8de1fa6feb16014b4a7b625365e916515f` |

No artifact, threshold, task semantic, verifier, gate, or estimand was changed
to obtain this pass.
