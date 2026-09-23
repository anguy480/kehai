# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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
