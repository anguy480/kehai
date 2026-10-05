# CLAUDE.md

Rules for Claude Code sessions in this repository. This repo is public: keep
anything about participants, sessions, results or manuscripts out of every
file, commit message and log entry.

## Data rules

- Raw data, intermediates and outputs live outside the repo, under the roots
  named by `VC_DATA_ROOT`, `VC_WORK_ROOT` and `VC_OUT_ROOT` (set in `.env`).
- Never open, read, list, print or copy anything under those roots, including
  previews, contact sheets, logs and feature tables. Run the scripts and look
  only at their summary output on the terminal.
- Never read, print or commit label files or label values. `.gitignore` and the
  `block-media-files` pre-commit hook are a safety net, not permission.
- Tests use synthetic data only (`tests/synth/`). Never point a test at a real
  root or a real file.
- Don't make external API calls (LLMs, cloud services, uploads) with anything
  derived from the recordings, even summaries, without asking first.

## Analysis integrity

- `main` holds the frozen analysis that the label holder runs. On `main`, never
  change analysis code (`modeling/`, feature extraction, `aggregate`), features,
  the confirmatory plan (`modeling/tiers.py`, `config/default.yaml`), tests, or
  the Holm correction settings.
- New analysis goes on an `exploratory/*` branch. Commit and push its feature
  sets and methods before any label is used, then add a dated line to
  `docs/analysis-log.md` on that branch recording what was run and from which
  commit. Never write a label value, score, prediction or residual to the log.
- Usability fixes (CLI, handoff, docs) go in their own commits, never mixed with
  anything else. Open the body with "Usability fix. It does not change the
  analysis:" and say what is untouched.

## Design decisions

ADRs live in `docs/decisions/` (numbered `NNNN-title.md`). Before changing a
stage, read the ADRs that cover it and keep the change consistent with them. If
a change contradicts one, stop and ask; a reversed decision needs a new ADR.

## Workflow

Python 3.11 managed by `uv`. Every command runs through `uv run` (or `make`), so
an active conda environment doesn't matter.

```sh
make setup        # uv sync --all-groups, install pre-commit hooks, write .env
make lint         # ruff check + ruff format --check on src tests scripts
make format       # ruff autofix + format
make typecheck    # mypy, strict, src/
make test         # pytest with coverage (what CI runs)
make test-fast    # skips tests marked slow or integration
uv run pre-commit run --all-files   # all hooks: ruff, mypy, detect-secrets, media block
```

CI (`.github/workflows/ci.yml`) runs lint, typecheck and `make test`; keep all
three green before committing.

Stages are `vc` subcommands, run one at a time in this order. Global options go
before the subcommand (`uv run vc --sessions 3,17 prosody`; also `--config`,
`--overlay`, `--workers`, `--force`):

```sh
uv run vc doctor          # check config, ffmpeg and the roots first
uv run vc inventory
uv run vc verify-layout
uv run vc extract-audio
uv run vc diarize
uv run vc assign-speakers
uv run vc vad
uv run vc turns
uv run vc prosody
uv run vc face
uv run vc aggregate
uv run vc handoff
```

`vc model` is the label holder's step: don't run it with real labels unless
asked. `make pilot` calls `vc run-all`, which doesn't exist; don't use it.

Commit messages follow `type(scope): summary`, e.g. `docs(analysis-log): ...`.
Commit or push only when asked.
