# AU-only baseline

**Exploratory. Requested by Tanaka on 2026-10-05, after the labels were received
on 2026-10-01.** Nothing here is part of the pre-registered analysis or changes
it, no multiplicity correction is applied, and no result may be described in
confirmatory terms. See [the analysis log](../analysis-log.md).

## Question

How much K6 information (SRS2 secondary) do the action unit intensities carry
on their own, across feature combinations, and does any combination beat
chance once the search over combinations is counted?

## What is fixed before any label is used

Everything below is in
[`config/exploratory/au_baseline.yaml`](../../config/exploratory/au_baseline.yaml)
and the code under `src/vc_multimodal/exploratory/au_baseline/`, committed and
pushed before the analysis is run on labels.

- **Inputs.** The `features.csv` sent in the 4d8a5a1 bundle and its
  `text_features.csv`, and a whole-session AU table, each refused unless its
  SHA-256 matches the plan.
- **Windows.** Speaking, listening, pooled, and speaking and listening side by
  side. *Pooled* is the participant's speaking and listening time together. It
  is not in the feature table, so `pooled.py` computes it from the per-frame
  output with aggregate's own functions: the same frames, statistics, 30 s
  minimum of measured time and QC-note blanking. The same run recomputed the
  speaking and listening windows and matched the frozen table exactly, so the
  per-frame output is the one behind it.
- **Sets.** Eleven AU sets (each of AU01, AU02, AU04, AU06, AU12; Miyamoto's
  AU01/06/12; smile AU06/12; upper face; lower face; all five means; all five
  means and SDs), each crossed with the four windows. Lower face is AU12 alone
  among the measured units, so it duplicates the AU12 sets and is evaluated once.
  References outside the family: all 30 face features, and the text features.
- **Search.** Greedy forward selection over the 15 AU means of the speaking,
  listening and pooled windows, up to five, run inside each outer training fold
  and scored by an inner 5-fold least squares R²; the outer model is then fitted
  on what it chose. It never sees the held-out sessions.
- **Models and CV.** The main run's elastic net and random forest pipelines
  (median imputation and scaling inside every fold), leave-one-out, and 20x
  repeated 5-fold. R² and Spearman ρ between predictions and outcome under both.
  Single AUs also get their plain Spearman correlation with the outcome.
- **Nulls.** 1000 label permutations shared by every set, model and target,
  scored on the first 5-fold split (as `permutation_baseline` does in the main
  run). Each set's observed split R² against its own null gives a per-set p. For
  the max statistic, each permutation's best R² across the AU sets and the
  search, under both models, forms the null for the best observed set.
- **Sensitivity.** The three best (set, model) pairs by K6 leave-one-out R²,
  re-evaluated without the session(s) passed with `--exclude-session`.

## Running it

```sh
B="$VC_OUT_ROOT/handoff/20260929_4d8a5a1eab8d"
python -m vc_multimodal.exploratory.au_baseline pooled --bundle "$B"     # no labels
python -m vc_multimodal.exploratory.au_baseline estimate --bundle "$B"   # no labels
python -m vc_multimodal.exploratory.au_baseline analyze --bundle "$B" \
    --labels <labels.csv> --exclude-session <id>
```

Measured by `estimate` on 2026-10-05 (8 workers, random targets): 16 min per
target for the observed estimates and 26.7 s per permutation per target, so
about 2.0 h at 100 permutations, 3.5 h at 200, 8.0 h at 500 and 15.4 h at the
planned 1000. A smaller count, if chosen, is passed with `--permutations` and
recorded in the results.

`analyze` refuses a dirty checkout and writes to
`$VC_OUT_ROOT/unblinded/<date>_au_baseline_<commit>/`. Nothing written there or
printed holds a label, a prediction, a residual or a per-session error.
