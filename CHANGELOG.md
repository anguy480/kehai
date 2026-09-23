# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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
