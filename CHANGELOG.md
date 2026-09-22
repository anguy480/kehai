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
- `vc` CLI entry point (`vc version`).
