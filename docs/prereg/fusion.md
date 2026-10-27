# All modalities combined, and what each one contributes

**Status: exploratory.** Fixed on 2026-10-27, after the labels were received on
2026-10-01, and after the results listed under
[What had been seen](#what-had-been-seen-when-this-was-written) had been seen
(see [the analysis log](../analysis-log.md)). Nothing here changes the
confirmatory analysis run from `4d8a5a1`, and no result may be described in
confirmatory terms. The correction below controls error within this analysis
only.

**Amended on 2026-10-27**, before any code for it was written and before any
label was used on this branch: a 14th descriptive comparison, text + nonverbal
against text. See
[Amendment 1](#amendment-1-2026-10-27-text--nonverbal-against-text); where it
and the text above differ, the amendment holds.

Asked for by Tanaka on 2026-10-27 (Slack): a combined model of all modalities,
and each modality's contribution to it.

Branch `exploratory/fusion`, from `main` at `acb2cda`, with
`exploratory/llm-ratings` at `f6bfd2c` and `exploratory/per-modality` at
`1550673` merged in, so that their checked table loading, face-table checks and
paired bootstrap are reused rather than copied. This file is committed and
pushed before any code for it is written and before any label is used on this
branch. Anything that reads labels is run by the label holder, the project
owner ([ADR 16](../decisions/0016-labels-held-by-the-project-owner.md)), from a
command committed here. No label, score, prediction, residual or result is
written to this repository.

## The question

With every modality measured so far in one model, how well are K6 and SRS-2
predicted, does that beat the manuscript's text features alone, and how much
of the combined model does each modality account for?

## What had been seen when this was written

The project owner, who is the label holder and made the choices below, had
seen the results of every analysis run on these labels before this file was
written. This analysis was designed knowing them:

- **The confirmatory run of 2026-10-01**, from `4d8a5a1`: all 40 estimates.
  They include `text_plus_all`, the closest thing to this analysis: the 20 text
  features with turns (12), prosody (12) and MediaPipe face (15 per window),
  74 features, on 62 sessions, elastic net and random forest, leave-one-out
  and repeated 5-fold. It had no permutation null and was not tested against
  text. They also include `text` alone on 62 sessions.
- **The per-modality evaluation**, runs 1 and 2 of 2026-10-22: prosody,
  turn-taking, face while speaking and face while listening, each alone, with
  the same features used here (the OpenFace face table, smoothed), with
  permutation nulls, and per-feature correlations.
- **The tokenization rerun**, step 3 of 2026-10-22: text variants (a), (b) and
  (c). Variant (a) is the text table used here.
- **The LLM ratings analysis** of 2026-10-22: the 11 LLM features alone, the
  zero-shot totals, and the 20 text features against the 20 plus the 11, on 62
  sessions; and its post hoc intervals of 2026-10-26.

So each modality alone, text alone and text + LLM have been seen, on 62
sessions (61 for the face). What has not been run is the combined model of all
six, every drop-one and grouped-drop model, and the add-one models other than
text + LLM. None of these choices can be claimed as made blind to the outcome.

## Outcomes

K6 and SRS-2 (`K6`, `SRS2` in the labels file), each analysed separately.

## Modalities and features

Six modalities, 71 features. Each is read from one table only, and that table
is refused unless its SHA-256 matches.

| Modality | Features | Table | SHA-256 |
| --- | --- | --- | --- |
| text | 20 | `text_features_original.csv` | `da984428e499df0bfd406e0139adb4329ab2aefb8bc3c0118e04e8e85b8ffd71` |
| llm | 11 | `llm_ratings_features.csv` | `050056e11feda5a1fd17aa23f17ebedb8f789163afa08509e0efa0d5489412f6` |
| prosody | 12 | `features.csv` of the `4d8a5a1` bundle | `375a38b0c8bb138b04ed07ff6504f04a39be7df7fe9df36609d394070e750f49` |
| turns | 10 | the same `features.csv` | as above |
| face_speaking | 9 | `face_openface_features.csv` | `6d67cfb0aad78a8d182972d8c77d261cd8c197a44a01de535c1c16dc133e2e3a` |
| face_listening | 9 | the same face table | as above |

### text: 20

The manuscript's 20 features, variant (a) of the tokenization rerun:
`nlp_features.csv` with a `session_id` column, its 8 recomputed columns
matching the original within 1e-9 for all 62 sessions. The same table and
SHA-256 as the LLM ratings analysis. Joined by `session_id`.

`Jaccard`, `Cosine`, `Bert`, `Bert_turn`, `Bert_turn_max`, `Bert_turn_min`,
`Bert_turn_std`, `Word_count`, `Word_count_doc`, `Word_ratio`, `MTLD_patient`,
`MTLD_doctor`, `Mean_words_per_turn`, `Agenda_social`, `Agenda_work`,
`Agenda_future`, `Agenda_self`, `Agenda_family`, `Agenda_anxiety`,
`Agenda_depression`.

### llm: 11

The LLM ratings, `openai` / `gpt-4.1-mini-2025-04-14`, each the mean of three
calls ([`prereg/llm-ratings.md`](llm-ratings.md)):

`llm__k_nervous`, `llm__k_hopeless`, `llm__k_restless`, `llm__k_depressed`,
`llm__k_effort`, `llm__k_worthless`, `llm__s_awareness`, `llm__s_cognition`,
`llm__s_communication`, `llm__s_motivation`, `llm__s_rrb`.

### prosody: 12

From the bundle, unchanged, as in the per-modality evaluation:

`prosody__f0_semitone_sd`, `prosody__f0_semitone_iqr`,
`prosody__f0_semitone_range`, `prosody__f0_semitone_mean_abs_delta`,
`prosody__voiced_fraction`, `prosody__intensity_mean_db`,
`prosody__intensity_sd_db`, `prosody__intensity_range_db`,
`prosody__jitter_local`, `prosody__shimmer_local`, `prosody__hnr_db`,
`prosody__speech_rate_proxy`.

### turns: 10

From the bundle, unchanged:

`turns__n_per_minute`, `turns__participant_speaking_ratio`,
`turns__latency_mean`, `turns__latency_median`, `turns__latency_sd`,
`turns__participant_turn_duration_mean`,
`turns__participant_turn_duration_sd`,
`turns__psychiatrist_turn_duration_mean`, `turns__pause_within_mean`,
`turns__pause_within_rate`.

**`turns__overlap_ratio` and `turns__interruption_rate` are left out.** Both
are exactly zero in all 62 sessions because whisper-diarization partitions
time ([ADR 12](../decisions/0012-tiered-analysis-not-a-feature-cap.md),
amendment of 2026-09-24). This differs from the per-modality evaluation, which
kept them in the turn-taking model at zero.

### face_speaking: 9, and face_listening: 9

From the OpenFace face table, built by `face-table` at `aef1852` and fixed in
the per-modality plan at `8c8d65a`, measured as
[`prereg/per-modality.md`](per-modality.md) describes (OpenFace 2.x, valid
frames `success == 1` and `confidence >= 0.50`, centred 1.0 s rolling mean,
5 fps grid, at least 30 s per window). For each window `w`:

`w__au01_mean`, `w__au01_sd`, `w__au01_p90`, `w__au06_mean`, `w__au06_sd`,
`w__au06_p90`, `w__au12_mean`, `w__au12_sd`, `w__au12_p90`.

**No face value comes from the bundle.** The bundle's `features.csv` also holds
MediaPipe face columns under the same names. Only the 22 prosody and turns
columns listed above are taken from it, and a test fails if any `face_*` column
of the analysed table comes from the bundle.

## Sessions

- **61 sessions**: the 62 of the confirmatory run, less session 43.
- **Session 43 is left out of every model**, the text-only ones included. Its
  QC note marks the face unavailable (camera out of focus), so both face
  modalities are blank for it. The command checks that they are. Leaving it
  out everywhere means every model and every comparison uses the same sessions.
- **Session 52: flagged, kept**, as in the per-modality evaluation (face
  degraded).
- If any of the 61 has no value for every feature of a modality, or is missing
  from any of the four tables, the run stops: the session set is fixed here,
  not decided by the data. Partly missing values are imputed inside the folds,
  as below.
- A session with no value for an outcome is left out of that outcome, for every
  model alike, and the count is reported. Expected: 61 for each.

**The 61-session numbers are not comparable with the 62-session analyses.**
Text alone and text + LLM are refitted here on 61 sessions, and their estimates
will differ somewhat from the confirmatory run, the tokenization rerun and the
LLM ratings analysis, which used 62. Prosody and turn-taking alone are not
refitted here. Compare models within this analysis only.

## Models and validation

The confirmatory pipelines in `modeling/evaluate.py`, unchanged, as in the
per-modality and LLM ratings analyses:

- **Elastic net (primary):** `ElasticNetCV` with l1 ratios (0.1, 0.5, 0.9,
  1.0), 50 alphas, `eps` 1e-2, inner 5-fold, `max_iter` 200,000.
- **Random forest (secondary):** 500 trees, other settings at sklearn's
  defaults.
- **Inside every fold:** median imputation and standard scaling, fitted on the
  training part only.
- **Leave-one-out** for every model: R², MAE and Spearman ρ of the out-of-fold
  predictions.
- **Repeated 5-fold, 20 repeats, for the full model and text alone only**,
  reported beside leave-one-out as mean R² and SD. Where leave-one-out and
  repeated 5-fold disagree about the sign of a model's R², or of the full model
  minus text, the summary says so
  ([ADR 15](../decisions/0015-cross-validation-and-the-elastic-net-path.md)).
  The contribution models get leave-one-out only, which is all the intervals
  need.
- **Seeds:** `20260923`, with `seed + repeat` for the repeated splits.
- Each estimate reports its number of non-converged fits.

### Feature sets

15 feature sets, each fitted with both models for both outcomes: 60
leave-one-out estimates.

| Set | Modalities | Features |
| --- | --- | --- |
| full | all six | 71 |
| text | text | 20 |
| full − text | llm, prosody, turns, face_speaking, face_listening | 51 |
| full − llm | text, prosody, turns, face_speaking, face_listening | 60 |
| full − prosody | text, llm, turns, face_speaking, face_listening | 59 |
| full − turns | text, llm, prosody, face_speaking, face_listening | 61 |
| full − face_speaking | text, llm, prosody, turns, face_listening | 62 |
| full − face_listening | text, llm, prosody, turns, face_speaking | 62 |
| full − face (both windows) | text, llm, prosody, turns | 53 |
| full − language (text and llm) | prosody, turns, face_speaking, face_listening | 40 |
| text + llm | text, llm | 31 |
| text + prosody | text, prosody | 32 |
| text + turns | text, turns | 30 |
| text + face_speaking | text, face_speaking | 29 |
| text + face_listening | text, face_listening | 29 |

## Primary tests

Four tests, **Holm across the four**:

1. **Full against text, K6:** a two-sided Wilcoxon signed-rank test on the
   per-session absolute leave-one-out errors of the two elastic net models,
   paired by session (`compare_errors`, `zero_method="zsplit"`, ADR 15).
2. **Full against text, SRS-2:** the same.
3. **Full model against chance, K6:** the elastic net's permutation p.
4. **Full model against chance, SRS-2:** the same.

**The permutation null is not a leave-one-out p.** It uses 1000 label
permutations under 5-fold on the first split (seed `20260923`), with the
observed R² taken under the same split, and p = (k + 1) / (1000 + 1), as the
confirmatory `permutation_baseline` does (ADR 15). Both the 5-fold observed R²
and the null are reported with the scheme named.

**Secondary, uncorrected:** the same Wilcoxon tests for the random forest, and
the random forest full model's permutation p. 4 permutation nulls in all: the
full model, 2 models × 2 outcomes. No other set gets a null.

## Contribution of each modality: descriptive

No test and no correction. These intervals describe; whether one excludes zero
is not a test, and the primary tests above remain the analysis's only
inference.

**Thirteen comparisons, each with a larger and a smaller model:**

- **Drop one (6):** full against full − m, for each of the six modalities m.
- **Grouped drops (2):** full against full − face (both windows), and full
  against full − language (text and llm).
- **Add one (5):** text + m against text, for each of the five other
  modalities m.

**Differences.** ΔR² = R²(larger) − R²(smaller), and ΔMAE = MAE(larger) −
MAE(smaller), from the leave-one-out predictions. For drop-one and the grouped
drops, the larger model is the full model; for add-one, it is text + m. In
every comparison, a contribution is a positive ΔR² and a negative ΔMAE. The
point estimate is the difference over all sessions analysed.

**Paired bootstrap over sessions**, as in Amendment 1 of
[`prereg/llm-ratings.md`](llm-ratings.md):

- Each resample draws n sessions with replacement, n being the sessions
  analysed for that outcome (61 expected). Both models of a comparison are
  scored on the same resampled sessions, from their fixed leave-one-out
  predictions.
- R² in a resample is 1 − SSE/SST, with SST taken around that resample's own
  mean of the outcome, as `score_predictions` computes it.
- 10,000 resamples. For each outcome the indices are drawn once by
  `numpy.random.default_rng(20260923)`, and all 13 comparisons and both models
  share them.
- A resample in which the outcome is constant has no R²: it is left out of the
  ΔR² interval and counted (expected 0).
- The 95% percentile interval: the 2.5th and 97.5th percentiles of the 10,000
  resampled differences (`numpy.quantile`, linear interpolation).

**104 intervals in all:** 13 comparisons × 2 outcomes × 2 models (elastic net
and random forest) × 2 differences (ΔR² and ΔMAE).

**Limitation, as in Amendment 1.** The bootstrap resamples fixed leave-one-out
predictions and does not refit the models on each resample, so it does not
capture the variability from refitting. The intervals are likely somewhat too
narrow.

## Reading the results

- **Drop-one undercounts modalities that overlap.** The 20 text features
  already include seven LLM ratings, the `Agenda_*` scores `gpt-4.1-mini` gave
  to the participant's speech, so text and llm can stand in for each other;
  the two face windows measure the same AUs of the same face. Dropping one of
  a pair may cost little because the other remains. The grouped drops measure
  each pair together.
- **Drop-one and add-one answer different questions.** Drop-one: what a
  modality adds given all the others. Add-one: what it adds to text alone.
  Neither is a modality's share of the full model's R², and the 13 differences
  are not expected to sum to anything.
- **The elastic net gives more pull to modalities with more features**: after
  scaling, every feature competes alike for the same penalty, so a 20-feature
  modality has more chances to enter than a 9-feature one. The random forest
  has a similar tilt, through more candidate splits. Contributions are not
  adjusted for this.
- **71 features on 60 training sessions per fold.** The full model is in the
  p > n regime that ADR 12 and ADR 15 describe; it rests on the elastic net's
  regularisation, and its non-converged fits are reported.
- **104 intervals** are reported, with no correction; reading one of them as a
  finding is the multiplicity problem ADR 12 exists to avoid.
- **Every result here was obtained after the labels, and the results above,
  had been seen.** Report it as exploratory.

## Commands and outputs

Two commands, committed on this branch before any label is used on it:

- `estimate`: times the run on random targets, reading the four tables but no
  label.
- `analyze`: the label holder's command. It reads the four tables by SHA-256
  and the label file, and writes to a new folder outside the repository, by
  default under `$VC_OUT_ROOT/unblinded/`:

| File | Contents |
| --- | --- |
| `primary_tests.csv` | The four primary tests, raw and Holm p, and the three secondary tests. |
| `models.csv` | One row per set, outcome and model: leave-one-out R², MAE and ρ, non-converged fits, and for full and text the repeated 5-fold R², and for full the permutation null and p. |
| `contributions.csv` | The 104 intervals: comparison, outcome, model, metric, n, point estimate, bounds and resamples used. |
| `summary.md` | The same as tables, with the statements under Reading the results. |
| `run.json` | Commit, input SHA-256s, settings, sessions analysed per outcome, timing and package versions. |
| `logs/` | The run log. |

No file or terminal line holds a label value, a feature value, a prediction, a
residual or a per-session error. The terminal shows counts and the files
written. The tests use synthetic data only, and include one that fails if any
face column comes from the bundle.

## Not done

- **No late fusion.** No model per modality stacked into a combined one, which
  would weigh modalities without regard to their size but needs nested
  cross-validation on 61 sessions.
- No weighting of modalities, group penalties or per-modality scaling.
- No feature selection, and no tuning beyond the elastic net's inner CV.
- No other feature definitions: no MediaPipe face, no tokenization variants (b)
  or (c), no AU02, AU04 or head pose.
- No test of any contribution, no permutation null for any set but the full
  model, and no repeated 5-fold for the contribution sets.
- No imputation of session 43's face.
- No other analysis is run on this branch under this pre-registration.

## Amendment 1, 2026-10-27: text + nonverbal against text

Added on 2026-10-27, after this file was pushed at `45755eb` and before any
code for it was written or any label was used on this branch. It adds one
descriptive comparison and changes nothing else: no model, test or correction.

- **Comparison 14: text + nonverbal against text.** The larger model is the
  full − llm set (text, prosody, turns, face_speaking, face_listening; 60
  features), the smaller is text (20). It asks the project's original question,
  whether the nonverbal modalities add to the manuscript's text features, with
  the OpenFace face features. Both sets are among the 15 already fitted, so no
  model is added.
- **It is neither the confirmatory test nor a rerun of `text_plus_all`.** The
  confirmatory `new_modalities_vs_text` compared the pre-registered nonverbal
  features alone against text. `text_plus_all` used the MediaPipe face, both
  constant turn features and 62 sessions, and was not compared with text.
- **The same differences, sign convention and paired bootstrap** as the other
  13, sharing their resample indices: ΔR² = R²(text + nonverbal) − R²(text) and
  ΔMAE = MAE(text + nonverbal) − MAE(text), so a contribution of the nonverbal
  modalities is a positive ΔR² and a negative ΔMAE.
- **Descriptive:** no test and no correction. The four primary tests are
  unchanged and remain the only inference.
- **112 intervals in all:** 14 comparisons × 2 outcomes × 2 models × 2
  differences. This replaces the 13 comparisons and 104 intervals above,
  wherever they appear, `contributions.csv` included.
