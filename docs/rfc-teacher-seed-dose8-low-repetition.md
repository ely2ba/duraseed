# Dose-8 low-repetition teacher warm start

Status: accepted prospectively by Ely on 17 August 2026, after the bounded
4/8/12 repair was stopped and before Stage A, Pilot 0, or any method outcome.

## Why one more amendment is warranted

The original dose-2 / `1e-4` recipe passed one orientation at 16 updates but
collapsed into learned nontermination in the other. The bounded continuous
4/8/12 repair then exposed a narrow tradeoff. At update 8, both orientations
passed format but failed only the mixed-group gate. At update 12, the complete
seed-17 target evaluation passed mixed-group and family-reachability gates but
missed format by three valid responses (`742/768` versus `745/768`). Once that
target gate failed, a joint update-12 pass was impossible and the run was
stopped before further paid calls.

The dose-2 trajectory contains only 24 distinct teacher traces. Sixteen
batch-32 updates present each trace 21 or 22 times. Doses 4, 8, and 16 were
never run in the original search because the seed-17 dose-2 candidate triggered
cross-orientation verification. The scientifically useful untested region is
therefore more distinct verified traces with fewer presentations per trace.
This is a nuisance-recipe search, not a claim that repetition is already known
to cause the instability. The retained-arm counts, nontermination analysis,
loss comparison, and group-size sensitivity are summarized in
[`warm-start-diagnostics.md`](warm-start-diagnostics.md).

## Frozen M1 rule

- Keep `Qwen/Qwen3.5-9B-Base`, M0, rank-32 LoRA, renderer, 4096-token cap,
  sampling settings, panels, manifests, prompts, batch size 32, learning rate
  `1e-4`, evaluation group size 8, and every existing gate unchanged.
- Use dose 8: eight exact solver demonstrations for each of the 12 targeted
  families, or 96 distinct traces per orientation.
- Run exactly one continuous weights-only-from-M0 trajectory in each existing
  orientation: seed 17 targets Panel B and seed 37 targets Panel A.
- Evaluate updates 6, 10, and 14. Canonical cyclic batching gives every trace
  exactly 2 presentations at update 6, a mean of 3.33 at update 10, and a mean
  of 4.67 at update 14. No trace exceeds 5 presentations.
- Select the earliest update at which both orientations separately pass every
  existing gate: mixed-group rate at least 0.30, family reachability at least
  0.60, all-one saturation at most 0.20, format compliance at least 0.97, the
  sentinel safeguard, finite optimization metrics, and clean leakage.
- If no checkpoint passes in both orientations, stop. Do not add a dose, LR,
  checkpoint, seed, decoding change, or gate relaxation.

The two trajectories are the complete M1 calibration. There is no third
calibration pair. If M1 succeeds, Pilot 0 supplies the next independent seeds
and still performs its pre-method warm-start health check before B-S or B-G.

## Spend boundary

The M1 workload is bounded at two 14-update trajectories and six complete gate
evaluations. Its exact token/storage upper bound is authorized with 20% dollar
headroom, capped at `$53.35`. The unchanged worst-case Stage-A reserve remains
`$155.09`.

To avoid waiting on lagging billing data, the stopped 4/8/12 repair is charged
conservatively at its entire `$44.27` teacher-action cap, rather than its smaller
provisional local usage. Its reserved Stage-A action never began and both remote
journals are clear. Together with the original run's authoritative
`$11.510623512`, the lifetime worst case is therefore
`$11.510623512 + $44.27 + $53.35 + $155.09 = $264.220623512`, below the frozen
`$300` acquisition-calibration cap. Partial vendor usage is not presented as
final billing.

## Scope and contingency

M1 changes teacher diversity and exposure together, so a success establishes a
usable shared warm start but does not identify a causal mechanism. A failure
also is not a B-S/B-G result: neither method branch has started.

Group size 16 is not automatically authorized. It is worth a separate decision
only if M1 is operationally healthy and fails specifically because mixed groups
remain too sparse. It cannot rescue format collapse or nontermination, and it
would change RL rollout volume, cost, and advantage statistics for every RL
cell.
