# Completed supervised trace-replay package

Portable local evidence for the bounded R-S/R-P follow-up. [Readout](readout.md)
and [full numerical record](readout.json); descriptive [F3 profiles](profiles/).
Nothing here triggers a gate, sampling, another run, or a public upload.

## Scope and chronology

Both source blocks originally ended `NO_MATCH`. Before any follow-up Stage-B
outcome, the owner authorized removing **historical-score eligibility only**.
The three nominees per arm, maximum 0.03 between-arm gap, and lexicographic
tie-break stayed fixed. That continuation selected seed 11 R-S@220 / R-P@20;
seed 29 remained `NO_MATCH` and has no Stage-B evidence. This is a documented
matching amendment after acquisition/selection outcomes, not the unmodified
original design. [selection.json](selection.json) preserves both decisions,
every nominee, all 12 candidate assessments, and all 120 cadence assessments.
The selected candidate scores are 1,504/4,096 and 1,503/4,096; independent
sampling on reused validation/monitor items supplies the separately reported
pre-B scores. This does not make those items an independent task holdout.

## Files and reproduction

- `readout.json` retains task IDs, per-item successes/trials at every checkpoint,
  raw and posterior trajectories, pointwise uncertainty, first-attainment rows,
  failure summaries and final validation counts. No bootstrap was recomputed by
  this export. Raw success, Jeffreys means and baseline-relative gain remain distinct.
- `profiles/` contains both complete pre-B F3 profiles and their item records.
- `selection.json` keeps selection evidence separate from the Stage-B readout.
- `corpus-lineage.jsonl` contains all chosen prompt/source identities, selection
  and order hashes, completion hashes, and frozen rendered-token lengths.
- `inventory.json` contains all four acquisition doses, corpus attrition and
  length distributions, both continuation doses, and billing summaries.
- `config.public.json` is a sanitized **non-executable projection**, not a launch
  config. It records original config/protocol/corpus hashes and the matching
  override. The frozen [protocol](../../../docs/replay-v1-protocol.md) stays unchanged.
- `learning.svg`, `retention.svg`, `failures.svg`, and `context.svg` are generated
  from saved data by the separate plotting command below, without new sampling.

From the repository root, use the existing repository `.venv` with NumPy and
Matplotlib installed. Render figures using only the compact public files:

```sh
PYTHONPATH=src .venv/bin/python tools/plot_replay_v1.py --readout artifacts/replay-v1/followup/readout.json
```

The 8 September retrospective [count breakdowns](retrospective-count-breakdowns.json)
contain both acquisition histories, full per-problem success-count histograms,
update-5-to-10 rebound counts, and fixed-grid AUC contributions. These are
descriptive additions, not new confirmation tests or changes to the registered
readout. Reproduce them without raw text or new sampling:

```sh
.venv/bin/python tools/replay_count_breakdowns.py
```

A separate local check reran the unchanged TCES verifier on all 4,608 original
R-P monitor completions at updates 5, 10, and 20 with zero structured-result
differences. Local checkpoint/sampler identities and evidence joins agree.
That check requires the private originals and does not independently establish
which weights the provider served.

The original private roots are explicit inputs if rebuilding this export:

```sh
python tools/export_replay_v1.py \
  --source-root /path/to/private/replay-v1-20260905T022302Z \
  --continuation-root /path/to/private/replay-v1-continuation-20260907T075721Z \
  --preparation-root /path/to/private/preparation \
  --output /path/to/new/public-copy
```

The completed local readout can be independently regenerated from original
generations/rewards with the repository's `duraseed.replay_results.write_report`
function, supplying the continuation root and a **new** output directory.
That operation includes the registered 50,000-resample analysis; the exporter
does not invoke it. Original text and verifier records remain available locally
for exact verifier replay without new completions. Compact counts suffice for
the reported count-based estimands and figures, but cannot reconstruct raw text,
reverify text correctness, or independently derive token/strategy fingerprints.

## Resource accounting and exclusions

The local total is $376.754818818 including $35.25 of storage allowances:
$266.620456788 for acquisition/selection plus engineering smoke, and
$110.134362030 for the continuation. These are observed token charges plus
allowances, **not settled invoice actuals**. The $191.373157272 recovery reserve
was unused; unused reserve is not authority for additional experiments.

Trace-length summaries distinguish original policy sampled tokens from rendered
supervised target tokens (including the renderer's suffix). Selected and full
training-token doses are not token-matched. Corpus selection is success-conditioned
and uses complete acquisition trajectories, not just the selected B-G checkpoint.

Excluded from this compact public projection: raw training/evaluation text and
token arrays, credentials, personal filesystem locations, provider account/project
and session identifiers, raw billing/console exports, and adapter tensors.
Originals remain privately preserved; their hashes are not substitutes for access.
Remote checkpoint locators use deterministic `opaque-sha256:` aliases. Source
task/sample IDs and original file/content hashes are retained for local joins.
No sealed tests were opened or exported. No second block is imputed, no two-block
training-seed significance is claimed, and no additional experiment is authorized.
