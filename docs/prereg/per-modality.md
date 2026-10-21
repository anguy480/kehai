# Per-modality evaluation, with OpenFace and temporal smoothing

**Status: exploratory.** Fixed on 2026-10-21, after the labels were received on
2026-10-01 (see [the analysis log](../analysis-log.md)). Nothing here changes the
confirmatory analysis run from `4d8a5a1`, and no result may be described in
confirmatory terms. The corrections below control error within this analysis
only.

Branch `exploratory/per-modality`, from `main` at `decaa1d`. This file is
committed and pushed before any model or correlation on this branch sees a
label. Everything that reads labels is run by the label holder (the project
owner, who has held the labels since 2026-10-01; see `docs/analysis-log.md`)
from a command committed here; no label, score, prediction, residual or result is
written to this repository.

## Question

How much does each modality, on its own, carry about K6 and SRS-2: prosody,
turn-taking, the face while the participant is speaking, and the face while
they are listening?

## Outcomes

K6 and SRS-2 (`K6`, `SRS2` in the labels file), each analysed separately.

## Modalities and features

Four modalities, each evaluated on its own. 42 features in all.

### Prosody: 12 features

From the `features.csv` of the `4d8a5a1` bundle (SHA-256
`375a38b0c8bb138b04ed07ff6504f04a39be7df7fe9df36609d394070e750f49`),
unchanged.

1. `prosody__f0_semitone_sd`
2. `prosody__f0_semitone_iqr`
3. `prosody__f0_semitone_range`
4. `prosody__f0_semitone_mean_abs_delta`
5. `prosody__voiced_fraction`
6. `prosody__intensity_mean_db`
7. `prosody__intensity_sd_db`
8. `prosody__intensity_range_db`
9. `prosody__jitter_local`
10. `prosody__shimmer_local`
11. `prosody__hnr_db`
12. `prosody__speech_rate_proxy`

### Turn-taking: 12 features

From the same `features.csv`, unchanged.

1. `turns__n_per_minute`
2. `turns__participant_speaking_ratio`
3. `turns__overlap_ratio`
4. `turns__latency_mean`
5. `turns__latency_median`
6. `turns__latency_sd`
7. `turns__interruption_rate`
8. `turns__participant_turn_duration_mean`
9. `turns__participant_turn_duration_sd`
10. `turns__psychiatrist_turn_duration_mean`
11. `turns__pause_within_mean`
12. `turns__pause_within_rate`

`turns__overlap_ratio` and `turns__interruption_rate` are exactly zero in all 62
sessions, because whisper-diarization partitions time (ADR 12, amendment of
2026-09-24). They stay in the turn-taking model, where they contribute nothing,
so the model uses the same set the confirmatory `turns` set did. They have no
Spearman correlation and are left out of Analysis A (below).

### Face while speaking: 9 features

### Face while listening: 9 features

OpenFace only. No MediaPipe value enters either face modality, and nothing is
compared against MediaPipe. For each window `w` in `face_speaking`,
`face_listening`:

1. `w__au01_mean`
2. `w__au01_sd`
3. `w__au01_p90`
4. `w__au06_mean`
5. `w__au06_sd`
6. `w__au06_p90`
7. `w__au12_mean`
8. `w__au12_sd`
9. `w__au12_p90`

These are the three AUs Miyamoto et al. 2025 found predictive of social
performance (ADR 13), each with a mean, an SD and a 90th percentile. AU02, AU04
and head pose are not used.

How they are measured:

- **Extraction.** OpenFace 2.x `FeatureExtraction` with action unit output, run
  on the participant's tile, cropped by the existing rule (letterbox detection
  and tile geometry, as `vc face` does now). Intensities are the `AU01_r`,
  `AU06_r` and `AU12_r` columns.
- **Valid frames.** `success == 1` and `confidence >= 0.50` (the existing
  `face.min_confidence`). Every other frame is dropped as a frame with no usable
  face, as now. No head-pose filter or other new filter is added.
- **Smoothing, fixed here.** A centred rolling mean over 1.0 s, applied to each
  AU intensity on the full timeline at the video's native frame rate. The window
  in frames is 1.0 s × the video's fps, rounded to the nearest integer and
  raised by one if even so that it centres; at 25 fps that is 25 frames. It is
  applied separately to each contiguous run of valid frames and never spans a
  dropped frame. At the edges of a run the mean is taken over the valid frames
  inside the window (minimum one), so no frame is dropped for sitting near a gap.
- **Order.** Smooth first, on the whole session, then split into speaking and
  listening.
- **Frame grid.** After smoothing, the statistics use the existing 5 fps grid
  (every 5th frame at 25 fps, `face.sample_fps: 5.0`).
- **Windows.** The speaking and listening timeline that `vc turns` already
  produced, unchanged. A frame belongs to a window when its timestamp falls
  inside it.
- **Minimum time.** A window with less than 30 s of measured time
  (`aggregate.min_speaking_s`, `min_listening_s`) gives no features for that
  window, as now.
- **Statistics.** Mean, SD (ddof = 1) and 90th percentile over the window's
  valid smoothed frames, as `vc aggregate` computes them now.

## Sessions

- **62 sessions** for prosody and turn-taking.
- **Session 43: face withheld.** Its QC note marks the face unavailable (camera
  out of focus, 99.2% of frames without a usable face). Both face modalities
  are blank for it, so it is left out of both face analyses (61 sessions there)
  and kept in prosody and turn-taking.
- **Session 52: flagged, kept.** Its QC note marks the face degraded (32.4% of
  frames dropped). It stays in every analysis and its flag travels with the
  results.
- In each modality, a session with no value for any of that modality's features
  is left out of that modality's model and its correlations, and the count is
  reported. Partly missing values are imputed inside the folds, as below.
- No other exclusion is made.

## Analysis A: one feature at a time

- Spearman ρ of each feature with each outcome, two-sided, over the sessions
  that have both values.
- **Benjamini-Hochberg FDR across all features within each outcome**: one family
  per outcome, holding every feature of all four modalities that has a
  correlation. That is 40 per outcome: 12 prosody, 10 turn-taking (the two
  constant features are reported as untestable and are not in the family) and
  9 + 9 face. A feature found constant over the analysed sessions when the
  analysis runs is handled the same way and the change in family size is
  reported.
- Reported: ρ, raw p, BH q, and n per feature and outcome.
- Implemented in `vc model` (a per-feature table) and used by the per-modality
  command.

## Analysis B: one model per modality

- **Models.** Elastic net (primary) and random forest (secondary), using the
  confirmatory pipelines unchanged (`modeling/evaluate.py` at `decaa1d`):
  - Elastic net: `ElasticNetCV` with l1 ratios (0.1, 0.5, 0.9, 1.0), 50 alphas,
    `eps` 1e-2, inner 5-fold, `max_iter` 200,000.
  - Random forest: 500 trees, other settings at sklearn's defaults.
- **Inside every fold.** Median imputation and standard scaling are pipeline
  steps fitted on the training part only, and the elastic net's
  hyperparameters come from its inner CV.
- **Validation.**
  - Leave-one-out is primary, with R², MAE and Spearman ρ of out-of-fold
    predictions against the outcome.
  - Repeated 5-fold over 20 repeats is reported beside it, as mean R² and its
    SD.
  - The seeds are the confirmatory ones: `runtime.seed` 20260923, with
    `seed + repeat` for the repeated splits.
- **Null.** 1000 label permutations per model, under 5-fold on the first split
  (as the confirmatory `permutation_baseline` does, ADR 15). The observed R² is
  taken under the same split, and p = (k + 1) / (1000 + 1).
- **Correction.**
  - **Holm across the 8 elastic net tests** (4 modalities × 2 outcomes): the
    permutation p of each elastic net model against chance.
  - Random forest permutation p-values are reported uncorrected, as secondary.
- Each estimate reports the number of non-converged fits, as `vc model` does.
- 16 models and 16 nulls in all: 4 modalities × 2 outcomes × 2 model families.

## Order of running

The face features have to be extracted before the face modalities can be
analysed, so the analysis is handed off in two runs:

1. Prosody and turn-taking, from the `4d8a5a1` bundle.
2. Face while speaking and face while listening, from the OpenFace table.

The first run reports raw values only: ρ and raw p for Analysis A, the estimates
and uncorrected permutation p for Analysis B. **No q-value and no Holm-adjusted p
is computed over a partial family.** The second run reads the first run's
output, checks it came from the same plan, and computes BH across all 40
features per outcome and Holm across all 8 elastic net tests.

## Not done

- No comparison with MediaPipe, and no comparison with the text baseline.
- No combined-modality models, no feature selection, no tuning beyond the inner
  CV above.
- No other smoothing window, frame grid or validity rule.
- No other analysis is run on this branch under this pre-registration.
