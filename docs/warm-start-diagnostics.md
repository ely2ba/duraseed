# Warm-start diagnostics before M1

These are read-only diagnostics of completed calibration records. They do not
change a gate or authorize an experiment. The purpose is only to decide whether
the bounded [dose-8 M1 amendment](rfc-teacher-seed-dose8-low-repetition.md) is a
sensible next nuisance-recipe check.

## Recorded gate outcomes

The original run contains seven complete trained-arm assessments. The bounded
repair adds four complete checkpoint assessments and one target-complete but
sentinel-incomplete checkpoint. Thus there are 11 complete assessments, not 11
coordinates plus a complete update-12 repair arm.

| Orientation / coordinate | Mixed groups | Families reached | All-one groups | Valid format | Sentinel change (seeded − M0) | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| seed 17, dose 1, `1e-3`, update 16 | 10/96 | 7/12 | 0/96 | 616/768 | −0.0625 | fail |
| seed 17, dose 1, `1e-4`, update 16 | 25/96 | 11/12 | 2/96 | 766/768 | −0.0313 | mixed only |
| seed 17, dose 1, `3e-4`, update 16 | 46/96 | 12/12 | 1/96 | 693/768 | +0.0313 | format |
| seed 17, dose 2, `1e-3`, update 16 | 24/96 | 9/12 | 0/96 | 614/768 | −0.0625 | mixed + format |
| seed 17, dose 2, `1e-4`, update 16 | 44/96 | 12/12 | 0/96 | 753/768 | −0.0104 | pass |
| seed 17, dose 2, `3e-4`, update 16 | 39/96 | 12/12 | 1/96 | 733/768 | +0.0313 | format |
| seed 37, dose 2, `1e-4`, update 16 | 35/96 | 11/12 | 0/96 | 260/768 | −0.0625 | format |
| seed 17, dose 2, update 4 | 26/96 | 9/12 | 0/96 | 705/768 | −0.0417 | mixed + format |
| seed 37, dose 2, update 4 | 24/96 | 9/12 | 0/96 | 598/768 | −0.0729 | mixed + format |
| seed 17, dose 2, update 8 | 25/96 | 11/12 | 0/96 | 763/768 | −0.0313 | mixed only |
| seed 37, dose 2, update 8 | 20/96 | 8/12 | 0/96 | 751/768 | −0.0625 | mixed only |
| seed 17, dose 2, update 12 | 45/96 | 9/12 | 0/96 | 742/768 | 67/96 rows only | partial; target format fails |

The three higher-learning-rate seed-17 arms all failed format. At `1e-4`,
update 8 was formatted but too sparse in both orientations; by update 12 the
complete seed-17 target panel had enough mixed groups but missed the format gate
by three responses. This supports trying a less repetitive low-LR trajectory;
it does not establish repetition as the cause.

## Exact exposure

Dose 2 contains 24 distinct traces. With canonical cyclic batch-32 updates,
the original 16-update recipe presents the first eight traces 22 times and the
remaining traces 21 times. The repair presents traces 5–6 times by update 4,
10–11 times by update 8, and exactly 16 times by update 12.

M1 contains 96 distinct traces. Every trace is presented twice by update 6;
at update 10 the first 32 appear four times and the rest three; at update 14
the first 64 appear five times and the rest four. Doses 4, 8, and 16 were never
executed in the original run because the first passing seed-17 dose triggered
verification. M1 therefore enters a genuinely unobserved high-diversity,
low-repetition region.

## Nontermination and loss

A fixed detector marked a completion as a loop when its final 1,024 token IDs
had at least 0.80 exact agreement at a lag from 4 through 256. The failed
seed-37 update-16 arm had 580 malformed or nonterminating rows, including 568
length caps and 143 detected loops. Loops affected all 12 targeted families
and both roles (123 target, 20 sentinel). The corresponding seed-17 arm had 17
malformed/nonterminating rows and one detected loop; the partial seed-17 repair
update-12 evidence had 29 malformed/nonterminating rows and two loops.

Only 1/143 seed-37 loops shared a literal 8-token span with the 24 actual
training traces. Number-normalized overlap was high but identical against a
matched corpus of unseen gate solver traces; none overlapped its own prompt.
The evidence therefore points to generic solver-style degeneration rather than
literal replay of a particular training trace. It neither proves nor rules out
an exposure effect.

Training loss was finite and smooth in both dose-2 / `1e-4` update-16 arms:
seed 17 fell from 18.423 to 1.028 and seed 37 from 18.549 to 0.794. The collapsed
checkpoint actually had the lower final training loss. There is no numerical
loss cliff, but loss shape alone cannot distinguish exposure drift, panel/data
interaction, or another branch-dependent learned instability.

## Group-size-16 sensitivity

For context only, a Jeffreys-Beta posterior predictive calculation gives the
following expected mixed-item fractions for a fresh 16-sample gate. Intervals
are exact Poisson-binomial 95% intervals across 96 future items.

| Healthy-format mixed-only failure | Observed I8 | Expected I16 | 95% future interval |
| --- | ---: | ---: | ---: |
| seed 17, dose 1 / `1e-4` | 25/96 | 0.540 | 0.448–0.625 |
| repair seed 17, update 8 | 25/96 | 0.536 | 0.448–0.625 |
| repair seed 37, update 8 | 20/96 | 0.501 | 0.406–0.594 |

The projection is strongly prior-sensitive. Across those rows, a
Beta(0.1,0.1) prior gives expected fractions of 0.297/0.291/0.238, a Beta(1,1)
prior gives 0.715/0.712/0.691, and plug-in MLE gives 0.244/0.241/0.185. Group
size 16 may help sparse mixed groups, but it is not a mechanical or guaranteed
rescue. It remains a separate conditional decision only if M1 fails mixed-only
while both orientations retain healthy format, family reachability, and
sentinel behavior.

## Decision

No further free diagnostic is needed before M1. Run exactly dose 8, LR `1e-4`,
continuous updates 6/10/14 in both orientations, select the earliest joint pass
under unchanged gates, and stop if none passes. This is a bounded calibration
decision, not a mechanism result and not evidence about B-S versus B-G.

Source evidence is under
`runs/acquisition-calibration/acquisition-calibration-20260816T082800Z/teacher-dose-arms/`
and
`runs/acquisition-calibration/acquisition-calibration-repair-20260817T092320Z/teacher-dose-arms/`.
