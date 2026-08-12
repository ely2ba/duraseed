# RFC: canonical format-checkpoint selection

Status: accepted post-audit and before full candidate confirmation on 9 August
2026.

## Correction

The paid task-agnostic format-training run produced valid generations and
checkpoints, but its quick-screen evaluator compared the untrimmed response
with the expected response byte for byte. That was an implementation error.
It was discovered after the paid quick-screen audit and before any candidate
received the full 256-item confirmation. The historical run and its recorded
`no_qualifying_checkpoint` decision remain unchanged; selection is recovered in
a separately linked run.

For an expected literal token, the canonical rule is:

```text
canonical_response = completion.strip()
expected_response = f"<answer>{expected_token}</answer>"
exact_format_success = canonical_response == expected_response
```

Only whole-response leading and trailing whitespace is ignored. Exactly one
answer block is allowed, no non-whitespace prose may occur outside it, and the
payload must copy the expected literal exactly. Payload whitespace and content
are never normalized.

Wrapper compliance and content copying are reported separately. A checkpoint
must achieve at least 97% canonical wrapper compliance, at least 95% exact
canonical response success, a nonempty payload on every successful response,
and authenticated task-agnostic format-only training data. The literal-payload
copy rate is reported explicitly so wrapper and content failures remain
distinguishable.

## Recovery decision

All saved 64-item quick screens are replayed locally under this rule. The
replay authenticates the saved generations rather than changing the paid run.
Step 1 remains ineligible. At step 2, candidates are fully confirmed in the
predeclared order `1e-4`, `3e-4`, `1e-3`, stopping after the first pass. Each
attempt uses the complete fixed 256-item `format_eval` manifest and
`max_tokens = 128`. Selection remains earliest passing step, then smallest
learning rate.

The selected sampler-only candidate must be rematerialized as a trainable M0.
One guarded capability probe tests whether the service accepts its sampler path
for weights-only restoration. If accepted, the resulting zero-update client is
saved and independently reconfirmed. If rejected, only the selected learning
rate is reconstructed from the preserved raw zero-state weights with a fresh
optimizer and exactly the original first two canonical batches. In either
case, the rematerialized sampler must pass the complete format confirmation
before its sampler and full state are retained without expiry.

The real pinned-service probe on 9 August 2026 rejected the sampler-weights path
with a 404 during `LoadWeightsResponse`. The selected-learning-rate
reconstruction path is therefore required; no alternative restore trick is
permitted.

The first complete fixed 256-item confirmation is the selection draw. A
control-plane failure after that result is observed does not authorize a new
stochastic draw or replacement by a later result. The first recovery attempt
produced 250/256 canonical successes (97.66%) for step 2 / `1e-4`; its recorded
failure was solely the subsequently corrected nonempty-payload predicate. That
saved evidence is replayed under the accepted gate. The later service call is
used only as evidence that sampler-weight restoration is unsupported. A
subsequent diagnostic rerun is not used for selection.

## Task diagnostics

Generated TCES success is not an M0-selection gate. The raw diagnostic had only
2/128 successes and 108/128 length stops at `max_tokens = 1024`, so a claimed
10-percentage-point semantic safeguard would be vacuous. MAPS is also not a
format-selection gate. TCES completion length and difficulty, then MAPS
difficulty, are calibrated after M0 is frozen. No semantic-equivalence or
noninferiority claim is made for format training.

Using one common M0 prevents initialization from differing between Stage-A
methods. It does not imply that M0 is scientifically neutral: it can still
affect the TCES boundary composition and the MAPS difficulty distribution, both
of which are measured during subsequent calibration.

## Effect on prior evidence

At the time this estimand correction was accepted, no full-confirmation or
Pilot 0 result existed. The raw zero-update audit remains decisive because its
independently measured 60.5% format compliance required format training under
either rule. The later recovery evidence described above is calibration
evidence, not a Pilot 0 result. Pilot 0 has not started.
