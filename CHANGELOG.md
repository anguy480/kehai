# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Every stage that writes a per-session table overwrote it instead of merging,
  so a run over a subset deleted the rows for every session it did not touch.
  Nothing failed while it happened: `vc --force --sessions 130 face` reported
  success and left `face_qc.csv` holding one row where it had held three. With 62
  sessions and individual reruns this destroyed coverage on every run. The rows a
  run did not compute are now kept, in `face`, `vad`, `turns`, `prosody`,
  `diarize`, `verify_layout`, `assign-speakers` and `aggregate`.
  (`extract-audio` was already safe: it rebuilds its table from every sidecar on
  disk.)
- A session skipped as already done could get a row with every column empty,
  assembled from its backend sidecar alone. Empty is indistinguishable from a
  failed measurement, so it corrupted every rate computed over the table - the
  face summary reported "letterbox corrected in 2 of 3 sessions" when the answer
  was 2 of 2 measured. A skipped session's previous row is now carried forward,
  and a session whose row is missing is no longer considered done, so it is
  re-measured rather than stubbed. Rates in the summaries count only sessions
  that have a value, and say how many do not.
- `verify-layout` now records that rows kept from an earlier run were decided
  against that run's cohort: the recurring label is identified across whichever
  sessions are in a run, so only a full run settles it for the whole cohort.

- Head pose from MediaPipe was returned in the order `(yaw, roll, pitch)` while
  labelled `(pitch, yaw, roll)`. Each angle was individually recoverable, so a
  test that round-tripped through the same convention passed; rotating an image
  in its own plane is a rotation about the optical axis and appeared as yaw
  rather than roll. Fixed, and now verified by driving the real model with
  images rotated by known angles. The same error made `head_pitch` mean a
  different axis under each backend, since the OpenFace mapping was correct.
- The face crop was measured from a single frame one second in, which can be a
  fade-in or a title card, and would miss a layout that changes mid-session. It
  is now checked at three points through the recording and flagged
  `face_crop_unstable` when they disagree.
- The facial backend was recorded only in the QC table, so deleting that table
  lost the one fact that makes a later table safe to assemble. It is now
  written beside the measurements, and the mixing check covers every session
  with data on disk rather than only those in the current run - so
  re-extracting a subset with the other backend is refused.

- Label OCR found nothing in any of the 62 recordings. They carry 180px
  letterbox bars top and bottom - the content is two 640x360 tiles side by side
  inside a 720p frame - so a label region expressed as a fraction of the whole
  frame landed inside the bottom bar. Tile fractions are now interpreted within
  the detected content area (`video.letterbox_detection: auto`).
- `speakers.label_ocr.min_recurrence` lowered from 0.50 to 0.10, set from the
  data rather than guessed: two recurring labels appear across the cohort, each
  covering roughly half of it, so any threshold above about 0.42 recognises
  neither. With the letterbox fix and this threshold, 59 of 62 sessions are
  settled by OCR, all agreeing that the psychiatrist is on the left, with no
  session contradicting it.

- A per-session failure logged a full traceback at normal level, so 62 expected
  failures (a prerequisite stage not yet run) buried the summary. The traceback
  is now at DEBUG and the error line stays.
- Tests asserting that something is *never* logged used pytest's `caplog`,
  which captures nothing from the package logger because `configure_logging`
  sets `propagate = False`. Those assertions were passing without checking
  anything. A `package_logs` fixture now captures from the package logger
  itself, and the tests assert that something *was* logged before asserting
  what was not.
- `onnxruntime` was missing from the dependencies: `silero-vad` 6.x imports it
  at package import time, so it is a hard requirement rather than an optional
  accelerator.

- `vc extract-audio` accepted a truncated recording as if it were whole. ffmpeg
  exits 0 on a partially copied file, reporting the problem on stderr and simply
  stopping early, so a returncode check alone is not enough. The decoded
  duration is now compared against the container's stated duration, and a
  shortfall beyond `audio.max_duration_shortfall_s` is flagged `audio_truncated`
  with the shortfall quantified. `vc inventory` cannot detect this: a truncated
  file keeps its original metadata, so the stated duration looks normal.
- `vc extract-audio` crashed instead of reporting when every session failed: the
  resulting empty table's typed columns landed as `object` and failed their own
  contract.
- Synthetic recordings are now muxed with `+faststart`, matching how Zoom writes
  them, so truncating one leaves its metadata intact and its media data short -
  which is the real failure mode - rather than making the file unreadable.

- `vc inventory` crashed with `KeyError: 'session_id'` when an unrelated file
  was already at the output path (a headerless CSV from a manual ffprobe loop).
  The merge path assumed any existing file was one this pipeline wrote. It is
  now validated against the inventory schema first, before any probing, so an
  unusable file fails in a second rather than after 62 ffprobe calls. Without
  `--force` the run stops and explains how to resolve it, leaving the file
  untouched; with `--force` the file is moved aside to
  `inventory.csv.bak-<timestamp>` rather than overwritten.
- Inventory dtype coercion is now shared between building a table and reading
  one back. A CSV round-trip returns a nullable `Int64` column as `int64` and an
  entirely missing column as `object`, so a table this pipeline wrote failed its
  own contract on the next run. Boolean columns are parsed rather than cast,
  since `astype(bool)` maps the string `"False"` to `True`.

### Changed

- The outcome check now flags column names describing affective constructs
  (`anxiety`, `depress`, `distress`), and a column is exempt only where a
  confirmation is recorded against it in `model.text_features.confirmed_predictors`
  with a statement, an attribution and a date. This replaces a hardcoded allowlist
  that named the manuscript's two agenda columns beside patterns that never
  matched those names, so the exemption read as deliberate while never firing -
  and would have exempted them silently had such a pattern been added later.

  Prof. Tanaka's confirmation that `Agenda_anxiety` and `Agenda_depression` are
  LLM-rated topic scores derived from participant transcripts only, and not from
  the K6 or SRS-2 questionnaires, is recorded there, quoted into the run manifest,
  and quoted in the handoff README beside the row-ordering provenance. The
  baseline's independence from the outcomes is documented rather than assumed,
  which matters because every confirmatory test compares against that baseline.

- `turns__overlap_ratio` is no longer a confirmatory feature. The lab's
  whisper-diarization output partitions time - every moment belongs to exactly
  one speaker - so pairwise overlap is exactly 0.0000 s in all 62 sessions, and
  both overlap-derived features are structurally constant. The confirmatory slot
  went to `turns__n_per_minute`. Both features stay in the table at zero, which
  records the limitation rather than hiding it, and become live unchanged under a
  diarizer that permits overlap. Made before any questionnaire score was seen;
  see `docs/decisions/0012`.
- `diarization.import_dir` now points at the diarization run supplied by the lab
  (`diarization/diarizations_original` under `$VC_WORK_ROOT`), so the import
  backend works with no override.

- The facial features follow the lab's own published precedent rather than
  being chosen here (docs/decisions/0013). The action units are the house set
  used in Tanaka et al. 2025 (JMIR Form Res 9:e59261) - AU01, AU02, AU04, AU06
  and AU12 - and the confirmatory features are the three that Miyamoto et al.
  2025 (Acta Psychologica 254:104782) found positively associated with social
  performance: AU01, AU06 and AU12, in both the speaking and listening
  windows. Features are named by AU so they line up with the literature and
  with OpenFace, with the backend recorded per session because MediaPipe
  blendshape scores are not AU intensities.
- The previous blendshape list could not measure AU02 or AU06. AU06 is one of
  the three the precedent found predictive, so that was a silent gap in the
  planned feature set.
- `mediapipe` pinned below 1.0: 1.0.1 aborts the process on macOS arm64 inside
  the face detector subgraph, with or without an explicit CPU delegate. It is a
  hard abort rather than an exception, so it cannot be handled. 0.10.x runs and
  returns all 52 blendshapes.
- The face landmarker model asset is pinned by SHA-256, verified to emit every
  blendshape the configured action units need.

- The feature budget is enforced by tiering rather than by a hard cap
  (docs/decisions/0012). `aggregate.max_features` is now a sanity ceiling
  against a bug generating hundreds of columns; what makes the analysis
  defensible is `model.tiers`: a confirmatory set of three features per family
  chosen on prior literature, four named tests with Holm correction, and
  everything else reported as exploratory with its comparison count stated.
  Nested feature selection remains inside the folds but is reported as a
  sensitivity analysis, since it addresses leakage rather than multiplicity.

- Duration flags now use a 300-1000 s window with `mad_k` 2.5, so a flag means
  "genuinely odd" rather than "not 10 to 12 minutes". The previous 8-15 min
  window flagged 10 of 62 sessions, mostly ordinary variation.
- `face.sample_fps` is 5.0 and `speakers.mouth_crosscheck.sample_fps` is 12.5:
  every 5th and every 2nd frame at the confirmed 25 fps, so both are exactly
  evenly spaced.

### Added

- The handoff README shows where each session sits on facial tracking quality:
  the distribution of `qc__face_dropped_fraction` across the cohort, the sessions
  well clear of the rest named individually with their confirmed cause, and a
  pointer to the QC report so any session can be looked up rather than inferred
  from a summary. A new `qc__face_dropped_fraction` column carries the value that
  the face stage already flagged on, so the number that explains a session's
  facial features travels with them.
- A limitation note stating that face tracking quality varies with the
  participant's own home setup - lighting, focus, camera distance, framing - and
  that this is a known cost of remote recording which makes the facial results not
  directly comparable with Miyamoto et al. 2025, whose lab conditions included
  controlled lighting, a fixed camera and an eye tracker. Where a facial result
  here is weaker than the equivalent there, recording conditions are a live
  explanation that cannot be ruled out.

- `vc qc-note` records a finding a person confirmed by watching a recording, so
  it travels with the data instead of living in their head. The pipeline can say
  that a session produced almost no usable faces; it cannot say whether the crop
  was wrong, the detector failed, or the camera was out of focus, and those call
  for different responses.

  A note is scoped to a modality, which is what makes it safe to act on
  automatically: `face=unavailable` withholds that session's facial features and
  leaves its audio features untouched, because a blurry camera says nothing about
  the audio. Withholding rather than leaving the values in place matters too - a
  measurement nobody should use is more dangerous than an absent one, since
  whoever did not read the note will model it.

  The finding then appears in `qc__annotations` and `qc__annotation_reason` on the
  feature table and QC report, as `qc_notes.csv` in the handoff bundle, in a
  README section saying the blanks are deliberate and must not be imputed, and in
  the manifest in full, so a bundle is an immutable snapshot of what was annotated
  and by whom. A note that cannot be parsed stops the stage rather than being
  skipped: a finding that silently fails to travel is the thing this prevents.

  Recorded for session 43, whose participant's camera is too out of focus for
  face tracking - 99.2% of sampled frames produced no usable face, confirmed by
  watching the recording. Its audio features are unaffected and remain in use.

- `vc model` runs the analysis on the machine that holds the questionnaire
  scores: leave-one-participant-out as the primary estimate for comparability
  with the manuscript, repeated 5-fold beside it with sign disagreements
  reported, the pre-registered comparisons as paired per-session Wilcoxon tests
  corrected by Holm, a permutation null under a named scheme, and a markdown
  summary written for someone who did not write the code. No label, prediction
  or residual is written to any output. See `docs/decisions/0015`.

### Fixed

- The elastic net explored penalties a thousand times weaker than the one that
  zeroes every coefficient, which at 54 features on 62 sessions is effectively
  unpenalised least squares. Coordinate descent does not converge there:
  measured on that shape, the default settings produced 124 non-converged fits
  per outer fold, each returning whatever coefficients the iteration limit left
  behind, with no error raised and no sign of it in the output. The path is now
  bounded at `eps = 1e-2`, which still spans two orders of magnitude, converges
  everywhere tested, and is 16 times faster in that regime.

- `vc assign-speakers` decides which diarized speaker is the participant, which
  every later stage depends on and nothing downstream can detect if it is wrong.
  Speaker embeddings (ECAPA-TDNN, pinned) against the psychiatrist reference
  clips decide it; mouth movement per tile and the Zoom name labels corroborate
  it where they can, and disagreement is flagged rather than used to break a
  tie. Only the ranking is used, never an absolute similarity, because absolute
  scores differ systematically between the two recruitment waves. The stage also
  compares the reference clips against each other, with a within-clip control,
  which is what answers whether the two recurring Zoom labels are two people.

- `vc handoff` builds the bundle the label holder receives: the feature table,
  a dictionary describing every column, the quality columns in a separate file
  because they are not predictors, the text baseline with explicit session IDs, a
  manifest, and a README written for someone who will not read the code. It
  refuses to build from a modified working tree unless `--allow-dirty`, since the
  commit is the bundle's main provenance; refuses a column that looks like a
  questionnaire outcome or anything derived from a transcript; refuses to ship a
  column it cannot describe; and stages the bundle in a temporary directory so an
  interrupted build leaves nothing that could be sent by mistake.

- The manuscript's text features are aligned to sessions under an explicit,
  documented ordering rule rather than an assumption (docs/decisions/0014). The
  table has no identifier column, so the rule is quoted from the code that wrote
  the file, reproduced from the transcript files that code iterated rather than
  from our own inventory, and guarded by counts that must match exactly: 62
  transcript files and 62 rows, with either one moving stopping the join. The
  rule, its provenance, the file's digest and the full resolved session order are
  recorded in the run manifest. A table with no identifier and no stated rule is
  still refused.

- `vc aggregate` detects features with no variance across sessions and reports
  them, escalating when a constant feature holds a confirmatory slot: a
  pre-registered slot that cannot support or refute anything needs replacing
  before the analysis runs, not after.
- The manuscript's text features are validated before use. The table must carry
  an explicit session identifier - row order is refused, because a mismatched
  order would attach each participant's text features to someone else and no
  metric would reveal it - and any column whose name suggests a questionnaire
  outcome stops the load rather than leaking a label into the predictor set.

- Project skeleton: `src` layout, `pyproject.toml`, uv with a committed
  lockfile, ruff, mypy (strict), pytest with coverage, Makefile.
- Data-safety controls: `.gitignore` covering media, audio, transcripts, feature
  tables and `.env`; pre-commit hooks for large files, secrets and a custom
  media-blocking hook, with tests for the hook itself.
- GitHub Actions CI: lint, strict type check and tests on synthetic data only,
  with a cached uv environment.
- `vc` CLI entry point with global options (`--config`, `--overlay`,
  `--sessions`, `--workers`, `--force`, `--log-level`) and exit codes that
  distinguish a setup problem from failing sessions.
- Configuration layer: pydantic models, `config/default.yaml` documenting every
  open question, YAML overlays and dotted CLI overrides, snapshotted per run.
- Roots and session discovery from the environment only, reporting layout
  problems rather than stopping at the first one.
- Logging to stderr and a per-run file under `$VC_OUT_ROOT/logs`.
- pandera data contracts whose failure messages summarise by column and check
  instead of quoting values derived from clinical recordings.
- Atomic output writes and a shared per-session runner with parallelism,
  idempotent skipping and per-session failure isolation.
- `vc extract-audio`: mono 16 kHz WAV per session, plus a left/right channel
  comparison measured during the same decode. Every recording carries one mixed
  stereo stream, so any genuine channel separation would be a speaker cue
  independent of diarization; if the channels are duplicates, the summary says
  so plainly rather than leaving later stages to assume separation exists.
  Statistics are accumulated over chunks, which keeps memory flat on long
  sessions while remaining numerically exact.
- Mandatory integer frame-step resolution for video sampling. A configured
  sample rate that does not divide a recording's native rate is rejected with
  the rates that do, rather than silently rounded: 10 fps at 25 fps native
  alternates 2- and 3-frame steps, and uneven spacing distorts anything derived
  from differences between frames with nothing to notice.
- `layout.csv` records `recurring_label_key`: which recurring speaker settled
  each session, as a cohort-local ordinal (`PSY_A`, `PSY_B`) that is not
  derived from the label text in any way, not even by hashing. The summary
  cross-tabulates it against recruitment wave and says whether the split is
  clean by wave, which decides how many psychiatrist reference clips are needed
  and which sessions each covers.
- `vc verify-layout --debug-region`: reports the detected letterbox, tile boxes
  and computed label regions in both fractional and pixel coordinates, with
  observation counts, how many passed the confidence threshold and how many
  survived normalisation, per region per session. Written to
  `layout_debug.csv`, which has no text column by construction.
- `vc preview --label-regions` (on by default): draws the detected content area
  and the computed label regions on each sheet, so what OCR reads can be
  checked by eye rather than inferred from it finding nothing.
- `speakers.label_ocr.upscale`: enlarges the label patch before recognition,
  since the label text is only a dozen or so pixels tall.
- Letterbox detection and region resolution as pure geometry
  (`features/geometry.py`).
- `vc aggregate`: joins the turn, prosodic and facial measures into one row
  per session - the table the handoff bundle carries. Facial action units are
  summarised separately over the participant-speaking and
  participant-listening windows, which is this project's addition to the lab's
  prior work and the reason the stage exists: the windows come from `vc turns`
  and the measures from `vc face`, and neither knows about the other. 54
  features in four families. Every discovered session gets a row, with any
  missing upstream stage named rather than the session dropped; a window with
  too little measured time contributes nothing rather than statistics resting
  on a handful of frames; and per-window coverage is recorded so a thin summary
  is visible. The stage reports the confirmatory/exploratory split and refuses
  a table whose facial measures come from two backends.
- `aggregate.peak_action_units` and `aggregate.pose_measures`: depth where the
  precedent points and breadth nowhere else. Every action unit gets a mean and
  a standard deviation; the three units Miyamoto et al. 2025 found predictive
  additionally get a 90th percentile, since these distributions are
  zero-inflated and a mean and a peak answer different questions; head pose
  gets only a standard deviation, because its mean records where the camera sat.
- `vc face`: facial action units from the participant's tile. Frames are
  sampled at a rate that divides the recording's own frame rate exactly,
  cropped after correcting for the letterbox bars these recordings carry, and
  measured; frames are decoded, measured and discarded. Undetected frames stay
  in the table as rows, so the dropped fraction accounts for every frame looked
  at. Head pose is recorded as head pose and never as gaze.
- Two face backends behind one interface: MediaPipe (default) and an OpenFace
  CSV importer that is first-class rather than a fallback, since OpenFace is
  the lab's house pipeline. The backend and its version are recorded per
  session, the MediaPipe model is verified against its pinned hash before use,
  and a table whose sessions do not share one backend is refused with an
  explanation that a switch is a full rerun, not a top-up.
- `handoff_text.py`: the notes the handoff README must carry, kept as tested
  constants so a note earned by a design decision cannot be lost when the
  handoff stage is written. Two so far: why a backend change requires a full
  rerun, and why there are no gaze features.
- `vc prosody`: participant prosody via Praat, with overlapping speech
  excluded and stretches too short to measure discarded. Twelve features: F0
  variability, IQR, range and frame-to-frame movement, all in semitones
  relative to that speaker's own median so absolute pitch differences cannot
  dominate (docs/decisions/0005); voiced fraction; intensity mean, spread and
  range; jitter; shimmer; harmonics-to-noise; and an intensity-peak speech-rate
  proxy, named as a proxy. Statistics are pooled over frames rather than
  averaged over spans, so a twenty-second answer does not count the same as a
  half-second interjection. A missing measure is None, never zero.
  `prosody.opensmile` is a documented extension point that reports why it is
  not implemented: eGeMAPSv02 is 88 features, which would overrun the budget
  for 62 sessions several times over.
- `vc vad`: recovers the spans that actually contain speech. One detector pass
  over the recording, intersected with the diarized segments, with a
  `per_segment` mode kept for comparison (docs/decisions/0011). Records per
  session what fraction of diarized segment time survived as speech, which is
  the measurement justifying the stage. Detected speech is clipped to the audio
  that actually decoded, so a truncated recording cannot claim speech past its
  end.
- `vc turns`: turns, response latency, within-turn pauses, speaking-time ratio,
  overlap, and the speaking/listening timeline the facial stages consume.
  Twelve features, expressed as ratios and per-minute rates rather than raw
  totals, since session length varies from 4 to 16 minutes. Interruptions
  (negative latency) are counted separately and never averaged into response
  times, and a long silence is not counted as a response at all.
- Role mapping is loaded from recorded evidence, or from a hand-written
  `roles.csv` for piloting, and is never guessed; sessions using the manual
  table are flagged.
- Pure interval arithmetic (`features/spans.py`) and turn mathematics
  (`features/turn_math.py`), both with no I/O, tested against hand-worked
  examples.
- `vc diarize`, with all three backends behind one interface: `import` (the
  preferred path, reading whisper-diarization SRT or RTTM produced elsewhere),
  `pyannote` (a local fallback, behind the optional extra), and the upstream
  `whisper-diarization` tool invoked as an external subprocess whose output the
  importer then reads. Speaker labels are normalised to one form, segments are
  written to `$VC_WORK_ROOT` as Parquet, and the QC table written to
  `$VC_OUT_ROOT` has no text column. The import backend reports every file it
  could not match to a session, which is how a naming difference gets noticed.
- `vc doctor`: environment and raw-layout check, including whether label OCR
  is usable.
- `speakers.assumed_psychiatrist_side` (default `left`), used only as a fallback
  where label OCR is unavailable or inconclusive.
- `vc verify-layout`: read the Zoom name label in each tile to determine which
  side the psychiatrist is on, across every session. The psychiatrist is
  identified without being named, by the fact that their label recurs across
  sessions while each participant's appears once. OCR is primary; a disagreement
  with the assumed side is recorded as a QC flag and never silently overrides
  what OCR found. Output is counts, sides and session IDs: recognised text is
  compared in memory and never printed, logged or written.
- Pluggable OCR backends: `apple_vision` (default, on-device macOS Vision via
  the optional `ocr` extra, supports Japanese), `tesseract` (external binary if
  present), and `none`. An unavailable backend is a reported state, not an
  error.
- `vc inventory`: ffprobe every recording, detect the audio stream layout and
  variable frame rate, flag duration outliers robustly, and print a
  metadata-only summary. Running a subset merges with the existing table.
- `vc preview`: one contact sheet per session showing the whole frame with the
  configured crop boxes drawn plus each tile as it will be cropped, at several
  timestamps, so the layout can be confirmed by eye.
- Eight architecture decision records and `docs/data.md`.
