# vc-multimodal

Prosodic, turn-taking and facial feature extraction from recorded clinical Zoom
sessions, with a deliberately label-free handoff to statistical analysis.

> **Status: in development.** The tooling skeleton is in place. Pipeline stages
> are landing in order; see [CHANGELOG.md](CHANGELOG.md).

## Research question

A lab manuscript (under review) extracted *text* features from 62 Japanese
psychiatrist–participant Zoom sessions (lexical overlap, Sentence-BERT
similarity, MTLD, turn statistics, LLM-rated agenda scores) and related them to
two questionnaire scores — K6 (psychological distress) and SRS-2 (social
communication traits) — using leave-one-participant-out cross-validation with
elastic net. Text features predicted K6 moderately but failed on SRS-2
(negative cross-validated R²).

This project asks whether **prosodic, turn-taking and facial** features from the
same sessions predict SRS-2 and K6 better than text does, alone and combined.

## The handoff model

The questionnaire scores are held by the project's supervising professor and are
never present on the extraction machine or in this repository. The pipeline is
therefore split in two:

```
  this machine (no labels, ever)          |   label holder's machine
  ---------------------------------------- | ---------------------------
  raw video -> ... -> features.csv         |   features.csv + labels.csv
                     handoff bundle  ======>   -> vc model
                                          |   -> results.csv + summary.md
```

Everything in this repo is designed to make that bundle trustworthy and
self-explaining: it carries a feature dictionary, a per-session QC report, and a
manifest recording the exact commit, config, package and model versions used.
See [docs/decisions/0001-handoff-split-no-labels-on-student-machine.md](docs/decisions/0001-handoff-split-no-labels-on-student-machine.md).

## Data handling rules

These are lab requirements, not preferences. The data is clinical research data
collected under ethics review and it does not leave the lab.

1. **Raw data lives outside the repository**, under `$VC_DATA_ROOT`. Paths are
   read from the environment or a gitignored `.env`; none are hardcoded.
2. **No data file is ever opened, printed or displayed** during development.
   Summary output only: counts, shapes, durations, timings, error messages.
3. **Transcripts are the most sensitive artifact.** They stay in
   `$VC_WORK_ROOT`, are never printed, and are never included in a handoff
   bundle.
4. **No frames or annotated video are ever written**, and none are ever
   displayed. The face stage decodes, measures and discards. The single
   exception is `vc preview`, which writes one cropped still per session so
   the participant crop can be confirmed by eye. This rule is enforced by a
   static test over `src/` rather than by packaging: `cv2` arrives via
   MediaPipe's `opencv-contrib-python` and is GUI-capable, so the guard is
   [tests/unit/test_no_frame_output.py](tests/unit/test_no_frame_output.py).
5. `.gitignore` excludes all media, audio, transcripts, feature tables, `.env`,
   and data/output directories. A **pre-commit hook**
   ([scripts/block_media_files.py](scripts/block_media_files.py)) blocks them
   even when `git add -f` bypasses `.gitignore`, alongside a large-file check
   and a secret scan.
6. **Tests use only synthetic data** generated inside the test run. No real data
   reaches CI.

## Setup

Requires macOS or Linux, Python 3.11 (fetched automatically by `uv`), and
`ffmpeg` / `ffprobe` on `PATH` or configured in `.env`.

```bash
git clone <this repo> && cd vc-multimodal
make setup          # uv sync, install pre-commit hooks, write .env
```

Then edit `.env` and confirm the three data roots are right.

### A note on conda and virtualenvs

`uv` creates and owns `.venv` in this directory, and every `make` target runs
through `uv run`, so the project environment always takes precedence. Even so,
**deactivate any conda base environment and any older venv before working
here** (`conda deactivate`), so that a stray `python` or `pip` cannot pick up the
wrong interpreter.

One wrinkle on the development machine: `ffmpeg` and `ffprobe` are installed via
conda-forge and are only on `PATH` while conda is active. `make setup` records
their absolute paths into `.env` as `VC_FFMPEG` / `VC_FFPROBE`, so the pipeline
keeps working after `conda deactivate`. Homebrew cannot install ffmpeg on this
macOS version; the project never tries to install it, and fails at startup with
a clear message if it is missing.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `VC_DATA_ROOT` | Read-only raw media: five date folders of `<session_id>.mp4`. |
| `VC_WORK_ROOT` | Intermediates: audio, diarization segments, transcripts. |
| `VC_OUT_ROOT` | Outputs: inventory, previews, logs, features, handoff bundles. |
| `VC_FFMPEG`, `VC_FFPROBE` | Absolute binary paths; fall back to `PATH`. |
| `HF_TOKEN` | Only for the optional local `pyannote` diarization backend. |

## Pipeline

Each stage is a `vc` subcommand, runs per session in parallel, is idempotent
(skips finished sessions unless `--force`), writes outputs atomically, isolates
per-session failures and reports a summary, and accepts `--sessions` to run a
subset.

| Stage | What it does |
| --- | --- |
| `vc doctor` | Check the environment before anything else: config validity, ffmpeg/ffprobe versions, the three roots, and whether the raw layout matches expectations. |
| `vc inventory` | ffprobe every mp4; validate count, IDs, readability; flag duration outliers and variable frame rate. |
| `vc preview` | Write one cropped frame per session for visual confirmation of the participant tile. |
| `vc extract-audio` | Mono 16 kHz WAV per session (one per audio stream). |
| `vc diarize` | Pluggable: `import` (existing whisper-diarization output), `pyannote`, or external `whisper-diarization`. |
| `vc assign-speakers` | Map diarized speakers to psychiatrist/participant via embedding similarity, cross-checked against mouth movement. |
| `vc vad` | Silero VAD *inside* diarized segments to recover true speech boundaries. |
| `vc turns` | Turns, response latency, pauses, speaking-time ratio, overlap, speaking/listening timeline. |
| `vc prosody` | Participant speech only, overlaps excluded: F0 in semitones re: own median, intensity, jitter, shimmer, speech-rate proxy. |
| `vc face` | Sample, crop, landmark (MediaPipe by default; OpenFace CSV importer available); drop low-confidence frames. |
| `vc aggregate` | One row per session, facial features split by participant-speaking vs -listening. |
| `vc handoff` | Build the bundle: features, feature dictionary, QC report, manifest, professor-facing README. |
| `vc model` | Run by the label holder: leave-one-participant-out CV across feature sets and targets. |

Stage documentation fills in as each lands.

### Piloting

Never run 62 sessions first. `config/pilot.yaml` lists a small set of session
IDs; `make pilot` runs every stage over just those.

```bash
uv run vc doctor                            # is the environment ready?
uv run vc inventory                         # all sessions, metadata only
uv run vc --sessions 3,17,28 inventory      # a subset
uv run vc --sessions 3,17,28 preview        # then look at the sheets yourself
make pilot                                  # every stage, pilot sessions only
```

Global options (`--config`, `--overlay`, `--sessions`, `--workers`, `--force`,
`--log-level`) come *before* the stage name. Exit codes distinguish a setup
problem (2) from a stage that ran but had failing sessions (1).

## Development

```bash
make lint typecheck test
```

`ruff` for lint and format, `mypy --strict` over `src/`, `pytest` with coverage.
Data contracts are enforced with `pandera` at stage boundaries. Random seeds and
model versions are pinned and recorded in every run's manifest.

## Open questions

Tracked here until resolved; each is configurable rather than guessed.

- **Participant identity.** *Decided, pending confirmation:* each session is
  one participant, so leave-one-participant-out is leave-one-session-out. An
  optional `participant_map.csv` overrides this, and the grouping used is
  recorded in the manifest. See
  [ADR 7](docs/decisions/0007-session-as-participant-grouping.md).
- **Psychiatrist identity and reference clips.** Not confirmed that the same
  psychiatrist ran every session, especially across waves, so several reference
  clips are supported with an optional session-to-psychiatrist map. Clips are
  made by hand into `$VC_WORK_ROOT/reference/`. See
  [ADR 8](docs/decisions/0008-speaker-assignment-embedding-with-two-tile-crosscheck.md).
- **Audio stream layout.** One mixed stream or two; determined per file by
  `vc inventory`.
- **Video layout.** Gallery view (psychiatrist left, participant right) is
  expected but not assumed; the participant crop is configurable in fractional
  coordinates and confirmed by eye via `vc preview`.
- **Diarization source.** Reusing the original whisper-diarization output keeps
  the comparison against the manuscript's text features apples-to-apples;
  re-diarizing locally would confound modality with transcript changes. See
  [ADR 2](docs/decisions/0002-pluggable-diarization-backends.md).

## Design decisions

The reasoning behind the choices that would otherwise be invisible in the code
is recorded in [docs/decisions/](docs/decisions/): the handoff split, pluggable
diarization, MediaPipe versus OpenFace, VAD inside diarized segments, semitone
normalisation, the feature budget for N=62, participant grouping, and speaker
assignment. [docs/data.md](docs/data.md) documents the expected external
layout.

## Licensing

No license file yet — licensing awaits lab approval. Until then, treat this
repository as all-rights-reserved and internal to the lab.
