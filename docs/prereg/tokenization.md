# Text features without the short-lemma filter

**Status: exploratory.** Fixed on 2026-10-21, after the labels were received on
2026-10-01 (see [the analysis log](../analysis-log.md)). Nothing here changes the
confirmatory analysis run from `4d8a5a1`, and no result may be described in
confirmatory terms.

Branch `exploratory/tokenization`, from `main` at `decaa1d`. This file is
committed and pushed before any feature is recomputed and before any label is
used. Anything that reads transcripts is run by the project owner; anything that
reads labels is run by the label holder, who is also the project owner (the
labels have been held on this machine since 2026-10-01). No transcript text, label,
score, prediction or result is written to this repository.

## The source

The manuscript's 20 text features are `nlp_features.csv`, written by
`analysis.py` in Tanaka's public repository
[`hirokitan1155/virtual_client_SRS2_analysis`](https://github.com/hirokitan1155/virtual_client_SRS2_analysis)
(MIT), read at commit `ec07010`.

## The change: the length rule, and nothing else

`analysis.py` keeps a token for the lexical features only if all of these hold:

1. Janome part of speech is 名詞, 動詞 or 形容詞, and the second field is not 数.
2. Neither its base form nor its surface form is in the stopword list.
3. **`len(token.base_form) > 2`**: the lemma is at least three characters long.

The kept lemma is then normalised (`分かる` to `わかる`).

- **Original (a):** rule 3 as written. This must reproduce the existing
  `nlp_features.csv` before anything else runs (below).
- **New (b), the single primary variant:** rule 3 removed, keeping lemmas of
  every length.
- The tokenizer, its dictionary, the POS filter, the stopword list, the
  normalisation, the speaker mapping (`opposite_files`), the file order
  (`sorted(..., key=int(stem))` over the `.txt` transcripts) and the feature
  formulas are unchanged. Janome and lexicalrichness are pinned in `uv.lock`
  at the commit that adds the port.
- No other variant is computed or analysed on this branch.

## Features

Only the 8 features built from the filtered lemmas are recomputed:

1. `Jaccard`
2. `Cosine`
3. `Word_count`
4. `Word_count_doc`
5. `Word_ratio` (the CSV's name for `Word_count_ratio` in the code)
6. `Mean_words_per_turn`
7. `MTLD_patient`
8. `MTLD_doctor`

The other 12 do not use the filtered lemmas, so they are taken unchanged from
the existing `nlp_features.csv`. No OpenAI call is made and no embedding is
recomputed:

`Bert`, `Bert_turn`, `Bert_turn_max`, `Bert_turn_min`, `Bert_turn_std`,
`Agenda_social`, `Agenda_work`, `Agenda_future`, `Agenda_self`,
`Agenda_family`, `Agenda_anxiety`, `Agenda_depression`.

**Original 20** is the existing `nlp_features.csv`. **New 20** is the same
table with the 8 columns above replaced by variant (b).

## Reproduction gate

The port first computes variant (a) and compares its 8 columns with the
existing `nlp_features.csv`, row for row, within floating-point tolerance
(absolute and relative 1e-9; NaN only where the original has NaN). If any
value differs, the run fails, variant (b) is not computed, and nothing below
proceeds. A failure is recorded in the analysis log, and fixing it, if it is
fixable, is a separate change made before any label is used.

## Outcomes

K6 and SRS-2.

## Analysis

All of it run by the label holder, from a command committed on this branch
after the reproduction gate has passed.

### Models: original 20 against new 20

- **Models.** Elastic net (primary) and random forest (secondary), using the
  confirmatory pipelines unchanged (`modeling/evaluate.py` at `decaa1d`):
  `ElasticNetCV` with l1 ratios (0.1, 0.5, 0.9, 1.0), 50 alphas, `eps` 1e-2,
  inner 5-fold, `max_iter` 200,000; random forest with 500 trees. Median
  imputation and standard scaling are fitted inside every fold.
- **Validation.** Leave-one-out is primary (R², MAE, Spearman ρ of the
  predictions). Repeated 5-fold over 20 repeats is reported beside it. The
  seeds are the confirmatory ones: `runtime.seed` 20260923, and `seed + repeat`
  for the repeated splits.
- **Permutation nulls.** 1000 label permutations for each of the 8 models
  (2 feature sets × 2 models × 2 outcomes), under 5-fold on the first split,
  with the observed R² taken under the same split, as the confirmatory
  `permutation_baseline` does (ADR 15). Reported uncorrected.
- **The test.** For each outcome and model, a two-sided Wilcoxon signed-rank
  test on the per-session leave-one-out absolute errors of original 20 against
  new 20, paired by session (ADR 15).
  - Elastic net: **Holm across the two outcomes.** These two tests are the
    analysis's inference.
  - Random forest: the same two tests, reported uncorrected, as secondary.

### Correlations, matching the manuscript

- Spearman ρ of each of the 20 features with each outcome, two-sided.
- Benjamini-Hochberg FDR across the 20 features within each outcome, as
  `analysis.py` does.
- Computed for the original 20 and for the new 20, each as its own family of
  20, so the original family reproduces the manuscript's correction. The 12
  unchanged features therefore appear in both with the same ρ and p.

## Sessions

The 62 sessions of the confirmatory run, matched to the labels by session ID.
The new 20 carries a `session_id` column, written by the port from the
transcript file names, so no positional join is needed for it. No exclusion
is made: the text features have none in the confirmatory run.

## Not done

- No other length threshold, stopword list, POS set or tokenizer.
- No recomputation of the BERT or agenda features, and no OpenAI call.
- No feature selection and no combined-modality models.
- Nothing about the face, prosody or turn-taking features.
