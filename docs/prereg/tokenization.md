# Text features without the short-lemma filter

**Status: exploratory.** Fixed on 2026-10-21, after the labels were received on
2026-10-01 (see [the analysis log](../analysis-log.md)). Nothing here changes the
confirmatory analysis run from `4d8a5a1`, and no result may be described in
confirmatory terms.

**Amended on 2026-10-21**, before any label was used on this branch: a second
variant, (c). See [Amendment 1](#amendment-1-2026-10-21-variant-c); where it
and the text above differ, the amendment holds.

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

## Amendment 1, 2026-10-21: variant (c)

**When and from what.** Decided after step 1 of the run instructions had run
and passed its reproduction gate, from that step's aggregate output only: the
kept-lemma counts and vocabulary sizes, and the 30 most frequent lemmas that
(b) recovers across all sessions. No label had been used on this branch, and no
model or correlation had been run. The choice is therefore informed by the
token list, though not by any outcome.

**The variant.**

- **New (c):** rule 3 removed, as in (b), and in addition tokens whose second
  part-of-speech field is 非自立 are excluded. Rule 1 becomes: Janome part of
  speech is 名詞, 動詞 or 形容詞, and the second field is neither 数 nor
  非自立.
- 非自立 is one of the four values on a line that is commented out in
  `analysis.py` at `ec07010` (line 276,
  `#if pos2 not in {"非自立", "代名詞", "数","固有名詞"}:`). Only 非自立 is
  taken from it: 代名詞 and 固有名詞 stay kept, as in (a) and (b).
- Everything else is as for (b). (b) stays as pre-registered; (c) is added
  beside it and does not replace it.
- **New 20 (c)** is the original 20 with the same 8 columns replaced by (c).

**Computation.** (c) is computed by the same command as (b), only after (a)
has passed the reproduction gate. The same run recomputes (a) and (b), and
their four files must come out byte-identical to the first run's, whose
SHA-256 are:

| File | SHA-256 |
| --- | --- |
| `token_features_original.csv` | `c9bef307efffc15faca3a6699236a23eb60d707f812306402b6bf2f1e3ae5e19` |
| `token_features_new.csv` | `a3c2d93adda0bc4bca457624cd261f05b02365eae6a7cce1caa7b354c017b95e` |
| `text_features_original.csv` | `da984428e499df0bfd406e0139adb4329ab2aefb8bc3c0118e04e8e85b8ffd71` |
| `text_features_new.csv` | `f5f935230493880c669dcc401126d308f663fad9815170ce5609f548fbc91c3a` |

**Analysis, replacing the corresponding parts above.**

- **Models.** Original 20, new 20 (b) and new 20 (c), each with the elastic
  net and the random forest as specified above.
- **Permutation nulls.** For each of the 12 models (3 feature sets × 2 models
  × 2 outcomes), as specified above. Reported uncorrected.
- **The test.** For each outcome, model and variant, the Wilcoxon signed-rank
  test above, original 20 against that variant's new 20. (b) and (c) are not
  tested against each other.
  - Elastic net: **Holm across the four tests (2 variants × 2 outcomes).**
    These four tests are the analysis's inference.
  - Random forest: the same four tests, reported uncorrected, as secondary.
- **Correlations.** As above, with new 20 (c) as a third family of 20, BH
  within each outcome.

This supersedes, for (c) only, "the single primary variant", "No other variant
is computed or analysed on this branch" and "No other … POS set", and replaces
"Holm across the two outcomes" with the four-test family above.
