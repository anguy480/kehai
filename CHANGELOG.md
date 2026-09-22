# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `vc doctor`: environment and raw-layout check.
- `vc inventory`: ffprobe every recording, detect the audio stream layout and
  variable frame rate, flag duration outliers robustly, and print a
  metadata-only summary. Running a subset merges with the existing table.
- `vc preview`: one contact sheet per session showing the whole frame with the
  configured crop boxes drawn plus each tile as it will be cropped, at several
  timestamps, so the layout can be confirmed by eye.
- Eight architecture decision records and `docs/data.md`.
