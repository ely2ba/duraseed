# Stage-A answer-tag contract correction

Status: accepted on 18 August 2026, after the complete direct-M0 update-10
screen and before Pilot 0 or sealed-test access.

## Invalid terminal decision

Run `acquisition-calibration-direct-m0-20260817T203229Z` preserved complete
evidence but recorded `no_eligible_learning_rate` for both methods. That
decision is invalid because its format gate measured a different contract from
the executed task. The TCES prompt requests a concise derivation followed by
one final answer tag, the solver teacher emits that form, and the exact verifier
permits derivation outside one well-ordered `<answer>...</answer>` pair. The
screen reducer instead required the entire stripped completion to begin and end
with those tags. It therefore rejected every teacher-style B-S completion by
construction.

This is a calibration-decision error, not grounds to alter or discard the raw
generations, rewards, metrics, or checkpoint lineage. The original terminal
artifact remains preserved.

## Frozen correction

Stage-A format health is verifier-valid answer-tag retention. On the targeted
panel's paired monitor coordinates, an arm passes when its current
`valid_answer_tag` rate is at least the paired M0 rate minus 0.10. Invalid-tag
and length-stopped outputs
still receive zero reward and remain reported outcomes. The absolute and paired
length-stop gates, finite-metric requirement, leakage gate, B-G per-update mixed
group requirement, and B-G mean mixed-group floor remain unchanged.

Paired M0 valid-tag rate was `69/96`. The B-S screens produced `81/96`, `92/96`,
and `87/96` at `1e-4`, `3e-4`, and `1e-3`. The B-G screens produced `66/96`,
`65/96`, and `57/96` at `1e-5`, `3e-5`, and `1e-4`. B-G `1e-5` also remained
within the unchanged paired length-stop tolerance; the two larger B-G rates did
not. Applying the existing paired two-SE and smaller-LR selection rule once to
these sealed screens selects B-S `1e-4` and B-G `1e-5`.

No prompt, teacher target, verifier, decoder, task, panel, completion cap, or
learning-rate grid changed. This RFC supersedes only the incompatible Stage-A
outer-wrapper clauses in the direct-M0 amendment and protocol.

## One finalization and hard stop

The update-10 candidates are sampler-only and cannot resume training. Exactly
one bounded run therefore recreates only the selected B-S `1e-4` and B-G
`1e-5` branches continuously from the common M0 through update 50. It does not
rerun or extend the learning-rate screen. The final reducer uses the corrected
paired valid-tag rule and every other existing capability, sentinel,
reachability, length, rollout-health, finite-metric, and leakage criterion.

The exact two-arm authorization is `$107.84`. Conservatively charging the
completed screen at its full committed ceiling keeps the lifetime calibration
maximum at `$268.846932025`, below the unchanged `$300` cap.

If either method fails, the TCES acquisition route ends. There is no further
warm start, LR, duration, prompt, decoder, task, panel, cap, group-size, or
recovery amendment. Any continuation would be a separately motivated study
redirect, not another DuraSeed calibration loop.
