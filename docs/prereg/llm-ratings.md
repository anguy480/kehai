# LLM ratings of transcripts as features

**Status: exploratory.** Fixed on 2026-10-22, after the labels were received on
2026-10-01 (see [the analysis log](../analysis-log.md)). Nothing here changes the
confirmatory analysis run from `4d8a5a1`, and no result may be described in
confirmatory terms. [Amendment 1](#amendment-1-2026-10-22-intervals-for-the-added-value-post-hoc),
at the end, was added after the primary results had been seen and is post hoc.

Branch `exploratory/llm-ratings`, from `main` at `decaa1d`. This file and the
prompt it names are committed and pushed before any transcript is rated and
before any label is used. Anything that reads transcripts is run by the project
owner; anything that reads labels is run by the label holder, who is also the
project owner. No transcript text, model response, rating, label, score,
prediction or result is written to this repository.

## The question

Can a language model, reading the whole session dialogue, rate the constructs of
the two questionnaires as they show in the conversation, and do those ratings
track K6 and SRS-2, alone or on top of the manuscript's 20 text features?

## Input

- **Transcripts.** The `.txt` diarized transcripts that `analysis.py` in
  Tanaka's public repository
  [`hirokitan1155/virtual_client_SRS2_analysis`](https://github.com/hirokitan1155/virtual_client_SRS2_analysis)
  (MIT, read at `ec07010`) iterates, in its order,
  `sorted(..., key=lambda p: int(p.stem))`.
- **Speaker parsing, as `analysis.py` does it.** Files are read as UTF-8 with
  BOM. Only lines containing `Speaker 0` or `Speaker 1` are used; the speaker is
  the integer before the first `:`, the text is everything after it, with
  half-width and full-width spaces removed and the line break stripped.
- **Roles, by `opposite_files`.** Speaker 1 is the doctor in sessions 6, 8, 10,
  12, 18, 25, 30, 31, 32, 42, 44, 46, 50, 52, 57, 62, 107, 113, 140, 144, 158,
  194, 210, 212, 228, 245 and 261; speaker 0 is the doctor in every other
  session. The other speaker is the participant.
- **The dialogue.** Every used line, in file order, as `医師: <text>` or
  `参加者: <text>`, joined by line breaks. Nothing else: no session number,
  file name, date or other metadata reaches the model. The dialogue is not
  shortened; a session too long for the model fails (below).

## The prompt

[`config/exploratory/llm_ratings_prompt.txt`](../../config/exploratory/llm_ratings_prompt.txt),
in Japanese, committed in full with this file, SHA-256
`cf5c42bce6156c26a7788695c9416c0d90bb57347a1eb5fe73c8ec162498b882`. It holds a
system message (after `=== system ===`) and a user message (after
`=== user ===`); the dialogue replaces `{{DIALOGUE}}` in the user message, by
plain substitution. The rater refuses to run with any other prompt file. **The
prompt is not changed after this commit**; a change would be a dated amendment
here, made before any label is used.

## Scales

All 11 items are rated by the model from the dialogue alone, for the
participant only. Each has a short anchor in the prompt.

- **Set K, 6 items, integers 0 to 4:** the six K6 constructs as observable in
  the conversation, in K6 order: nervousness (`k_nervous`), hopelessness
  (`k_hopeless`), restlessness (`k_restless`), depressed mood (`k_depressed`),
  everything feels an effort (`k_effort`), worthlessness (`k_worthless`).
  0 = not shown, 4 = shown very strongly.
- **Set S, 5 items, integers 1 to 4:** the five SRS-2 treatment subscales as
  observable in the conversation: social awareness (`s_awareness`), social
  cognition (`s_cognition`), social communication (`s_communication`), social
  motivation (`s_motivation`), restricted interests and repetitive behavior
  (`s_rrb`). 1 = no difficulty seen, 4 = marked difficulty.
- An item not observable in the conversation gets the lowest value.
- **Totals.** Set K total, the sum of the 6 K features (0 to 24, the K6 range).
  Set S total, the sum of the 5 S features (5 to 20).

## Backend

A config setting, with no default: no call is made until it is set.

- `openai`: `gpt-4.1-mini` through the OpenAI API, as in Tanaka's agenda
  extraction. It sends the full dialogue to an outside service, so it is used
  only once Tanaka confirms an outside API is allowed for these transcripts.
- `local`: a local model through an OpenAI-compatible endpoint, the model named
  in the config.

The backend and model are fixed in a dated commit and a log line before the
pilot. Exactly one backend's ratings are analysed; rating with a second backend
would be a dated amendment.

## Sampling

- Temperature 0, JSON output requested, no rationale.
- **3 calls per session**, each a separate request with the same prompt and
  dialogue. Each feature is the mean of the 3 calls' integers.
- **Valid response:** a JSON object with exactly the 11 keys, every value an
  integer (`true`/`false` and `2.0` are not) within its item's range. Anything
  else is malformed.
- **Retries:** a malformed response is retried, at most 3 times. A call still
  malformed after that is failed. **A session with any failed call is failed:
  all 11 of its features are missing.** No value is ever filled in, imputed or
  taken from fewer than 3 calls.
- **Cache:** every valid response is stored, keyed by session, call index (1 to
  3), prompt SHA-256 and model name, outside the repository under
  `$VC_WORK_ROOT`. A rerun reads the cache and makes no new call for a cached
  key. Failed calls are not cached as values.

## Reliability

Test-retest reliability across the 3 calls, per item, over the rated sessions:
ICC(2,1) (two-way random effects, absolute agreement, single call) and ICC(2,3)
(the same, for the mean of 3, which is what the features are). Reported as
aggregates only. An item with no variance at all has an undefined ICC and is
reported as such.

At temperature 0 the three calls differ only by the backend's nondeterminism, so
these ICCs describe the stability of the rater, not the validity of the ratings.
A deterministic local backend may give ICC = 1 by construction.

## Pilot

Before the full run, the project owner runs the rater on **3 sessions: the three
lowest session numbers**, to check JSON parsing and runtime. The pilot reports
only parse success, retries, failures, token counts and time. It shows no rating
and uses no label. Its cached responses are reused by the full run, so the pilot
sessions are part of the full run and are not rated again.

If the pilot shows the prompt cannot be parsed, any fix is a dated amendment
here, made before the full run and before any label is used.

## Sessions

The 62 sessions of the confirmatory run, matched to the labels by session ID.
Failed sessions are left out of every analysis that uses an LLM feature,
including the original-20 arm of Analysis 3, so both arms of each comparison
use the same sessions. The number used is reported per analysis and outcome.

## Analyses

Outcomes K6 and SRS-2. The confirmatory pipelines in `modeling/evaluate.py`,
unchanged: median imputation and standardisation fitted inside each training
fold; leave-one-out as the primary estimate with repeated 5-fold (20 repeats)
beside it; 1000-permutation nulls under 5-fold; seed `20260923`.

1. **Zero-shot baseline, no fitting.** Spearman's ρ, two-sided, between the
   Set K total and K6, and between the Set S total and SRS-2.
2. **LLM features alone.** The 11 features, elastic net (primary) and random
   forest (secondary), for each outcome. The test is the elastic net's
   permutation p.
3. **Added value.** The manuscript's 20 text features against those 20 plus
   the 11 LLM features, for each outcome. The test is the paired Wilcoxon
   signed-rank test (two-sided, `zero_method="zsplit"`, `compare_errors`) on
   per-session absolute leave-one-out errors of the elastic net. The random
   forest is compared the same way.

The 20 text features are `text_features_original.csv` from step 2 of the
tokenization rerun (branch `exploratory/tokenization`), SHA-256
`da984428e499df0bfd406e0139adb4329ab2aefb8bc3c0118e04e8e85b8ffd71`. It is
`nlp_features.csv` with a `session_id` column taken from the transcript file
names, and its 8 recomputed columns matched `nlp_features.csv` within 1e-9 for
all 62 sessions, so its rows are matched by identifier and the positional order
of ADR 0014 is confirmed.

## Corrections

- **Holm across the 6 primary tests:** zero-shot (2), LLM-alone elastic net (2),
  added-value elastic net (2).
- The random forest results are uncorrected.
- **Feature-level:** Spearman's ρ between each of the 11 features and each
  outcome, two-sided, with Benjamini-Hochberg FDR within each outcome (11
  tests).

## Outputs

To a results folder outside the repository, aggregate values only: the
reliability table, the six primary tests with raw and Holm p, the model table,
the random-forest comparisons, and the feature-level table, as CSV files and a
`summary.md`, with a `run.json` recording the commit, the backend and model, the
prompt SHA-256 and the input digests. No file or terminal line holds a
per-session rating, label, prediction or error.

## Reading the results

- The 20 text features already include seven LLM ratings, the `Agenda_*`
  scores that `gpt-4.1-mini` gave to the participant's speech. Analysis 3 asks
  what these 11 ratings add beyond them, not beyond text features with no LLM in
  them.
- The model rates constructs as they show in the conversation; it never sees a
  questionnaire, and the doctor's questions shape what it can see.
- Transcription and diarization errors reach the model as they are.

## Not done

- No prompt variants, no other model, no other temperature, no chain of thought.
- No per-item model against single questionnaire items.
- No ratings of the doctor.

## Amendment 1, 2026-10-22: intervals for the added value (post hoc)

**Post hoc.** Decided on 2026-10-22, after the primary results had been seen,
to quantify how large a gain from the 11 LLM features the data can rule out. It
was not planned before any label was used. It changes no result above, adds no
test, and its intervals must be reported as post hoc. It is committed and pushed
before it is run.

- **Comparisons.** Analysis 3's pairs: the 20 text features against those 20
  plus the 11 LLM features, for K6 and SRS-2, elastic net and random forest.
  That is 4 comparisons, each with two differences, so 8 intervals.
- **Predictions.** The leave-one-out predictions of those 8 models are
  recomputed with `evaluate` from `modeling/evaluate.py`, unchanged, on the same
  two tables (the same SHA-256s), the same sessions, seed `20260923` and the
  same pipelines. Only the leave-one-out step is rerun; it does not depend on
  the repeated 5-fold or the permutations.
- **Gate.** Each recomputed leave-one-out R² and MAE must match the primary
  run's `models.csv` within 1e-9 (absolute difference), and the primary run's
  `run.json` must record the planned settings and the same input digests.
  Otherwise the run stops and writes nothing.
- **Differences.** ΔR² = R²(text + LLM) − R²(text), and ΔMAE = MAE(text + LLM)
  − MAE(text). A gain from the LLM features is a positive ΔR² and a negative
  ΔMAE. The point estimate is the difference over all sessions analysed, which
  equals the difference of the two models' values in `models.csv`.
- **Paired bootstrap over sessions.** Each resample draws n sessions with
  replacement, n being the sessions analysed for that outcome (62 if every
  session is rated and labelled). Both models are scored on the same resampled
  sessions, from their fixed leave-one-out predictions. R² in a resample is
  1 − SSE/SST, with SST taken around that resample's own mean of the outcome, as
  `score_predictions` computes it. There are 10,000 resamples. For each outcome
  the indices are drawn by `numpy.random.default_rng(20260923)`, and the elastic
  net and the random forest share them. A resample in which the outcome is
  constant has no R²: it is left out of the ΔR² interval and counted (expected
  0).
- **Interval.** The point estimate and the 95% percentile interval: the 2.5th
  and 97.5th percentiles of the 10,000 resampled differences (`numpy.quantile`,
  linear interpolation).
- **No tests.** No p-value and no multiplicity correction: these intervals
  describe, they do not test. Whether one excludes zero is not a new test of
  added value; the Wilcoxon tests of analysis 3 remain the tests.
- **Limitation.** The bootstrap resamples fixed leave-one-out predictions and
  does not refit the models on each resample, so it does not capture the
  variability from refitting. The intervals are likely somewhat too narrow.
- **Outputs.** `intervals.csv` and `intervals.md` in a new folder outside the
  repository, by default `$VC_OUT_ROOT/unblinded/llm_ratings_intervals`. For
  each interval they hold the outcome, model, metric, n, point estimate, bounds
  and resamples used, and `intervals.md` records the commit, the SHA-256 of the
  primary `models.csv`, the input digests and the seed. No file or terminal line
  holds a per-session value; the terminal shows the gate and the files written,
  not the intervals.
