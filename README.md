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

`vc verify-layout` additionally needs the optional on-device OCR extra, which is
macOS-only:

```bash
uv sync --extra ocr
```

Without it, the layout check reports every session as inconclusive and falls
back to `speakers.assumed_psychiatrist_side`, which it tells you about. `vc
doctor` reports whether OCR is usable.

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
| `VC_PSYCHIATRIST_LABELS` | Optional. Fragments of the psychiatrist's Zoom name label, for `vc verify-layout`. Normally unnecessary, and deliberately *not* in any committed config: a real name must never enter this repository. |

## Pipeline

Each stage is a `vc` subcommand, runs per session in parallel, is idempotent
(skips finished sessions unless `--force`), writes outputs atomically, isolates
per-session failures and reports a summary, and accepts `--sessions` to run a
subset.

| Stage | What it does |
| --- | --- |
| `vc doctor` | Check the environment before anything else: config validity, ffmpeg/ffprobe versions, the three roots, and whether the raw layout matches expectations. |
| `vc inventory` | ffprobe every mp4; validate count, IDs, readability; flag duration outliers and variable frame rate. |
| `vc preview` | One contact sheet per session: the whole frame with the detected content area and the computed label regions drawn on it, plus each tile as it will be cropped. `--no-label-regions` for the plain version. |
| `vc verify-layout` | Read the Zoom name label in each tile to check which side the psychiatrist is on, across every session. Reports counts and session IDs only; recognised text is never printed, logged or written. `--debug-region` reports where every region was looked for, in fractional and pixel coordinates, with how much OCR saw there. |
| `vc extract-audio` | Mono 16 kHz WAV per session, and a left/right channel comparison: the one stream is stereo, and any real separation would be a speaker cue that owes nothing to diarization. |
| `vc diarize` | Who spoke when. Pluggable: `import` (existing whisper-diarization output; the preferred path), `pyannote` (local fallback), or the external `whisper-diarization` tool. Writes normalised segments to `$VC_WORK_ROOT` and a text-free QC table to `$VC_OUT_ROOT`. |
| `vc assign-speakers` | Map diarized speakers to psychiatrist/participant via embedding similarity, cross-checked against mouth movement. |
| `vc vad` | Silero VAD to recover true speech boundaries: one pass over the recording, intersected with the diarized segments. Reports how much segment time was actually silence. |
| `vc turns` | Turns, response latency, pauses, speaking-time ratio, overlap, and the speaking/listening timeline the facial stages consume. Requires a role mapping and refuses to guess one. |
| `vc prosody` | Participant speech only, overlapping speech excluded: F0 in semitones relative to that speaker's own median, intensity, jitter, shimmer, harmonics-to-noise and a speech-rate proxy, via Praat. Requires a role mapping. |
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
uv run vc verify-layout                     # all 62: which side is the psychiatrist?
uv run vc extract-audio                     # mono 16 kHz + the stereo probe
uv run vc --sessions 28 diarize             # check one session before all 62
uv run vc vad                               # refine segments into real speech
uv run vc turns                             # needs a role mapping; see below
uv run vc prosody                           # participant prosody, same mapping
make pilot                                  # every stage, pilot sessions only
```

Global options (`--config`, `--overlay`, `--sessions`, `--workers`, `--force`,
`--log-level`) come *before* the stage name. Exit codes distinguish a setup
problem (2) from a stage that ran but had failing sessions (1).

### Diagnosing the label regions

If `vc verify-layout` reports sessions as inconclusive, the question is almost
always whether OCR was looking in the right place:

```bash
uv run vc --sessions 28 verify-layout --debug-region   # where it looked
uv run vc --force preview --label-regions              # and what that looks like
```

`--debug-region` prints the detected letterbox, the tile boxes and the computed
label regions in both fractional and pixel coordinates, with the number of
observations, how many passed the confidence threshold, and how many survived
normalisation — never the recognised text. The same numbers are written to
`$VC_OUT_ROOT/layout_debug.csv`, one row per region per session.

On the preview sheets the orange box is the detected content area and the green
boxes are the label regions. A green box that is not over a name is the reason
OCR found nothing.

### Role assignment, and piloting before it exists

`vc turns` and everything after it need to know which diarized speaker is the
participant. Getting that backwards would not crash anything: it would measure
prosody on the wrong voice and invert the speaking/listening split. So nothing
guesses. A mapping comes from one of two places:

1. `$VC_WORK_ROOT/roles/<session_id>.json`, written by `vc assign-speakers`
   along with the evidence behind it. This needs a psychiatrist reference clip
   in `$VC_WORK_ROOT/reference/`.
2. `$VC_WORK_ROOT/roles.csv`, written by hand, so a few sessions can be piloted
   before that clip exists. It is an explicit human judgement rather than an
   assumption, and sessions using it are flagged `turns_manual_role_mapping`.

```csv
session_id,speaker,role
28,SPEAKER_00,psychiatrist
28,SPEAKER_01,participant
```

The speaker labels come from `diarization_qc.csv`. A recorded assignment always
takes precedence over the hand-written table.

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
- **Which side the psychiatrist is on.** Sessions checked by hand were all
  LEFT, recorded as `speakers.assumed_psychiatrist_side` and used only as a
  fallback. `vc verify-layout` checks it per session by reading the Zoom name
  labels; a disagreement is flagged, not applied. If all 62 come back left the
  assumption is confirmed. See
  [ADR 9](docs/decisions/0009-verify-tile-layout-by-label-ocr.md).
- **Audio stream layout.** *Resolved:* one mixed AAC stream, stereo, 48 kHz, in
  all 62 recordings, so diarization is required. Whether the stereo channels
  carry any usable separation is measured by `vc extract-audio`; see
  [ADR 10](docs/decisions/0010-measure-stereo-channel-separation.md).
- **Video layout.** *Resolved.* All 62 recordings are 1280x720 at a constant
  25 fps, with **180px letterbox bars top and bottom**: the content is two
  640x360 tiles side by side, psychiatrist on the left. Tile fractions are
  interpreted within the detected content area
  (`video.letterbox_detection: auto`), because treating them as fractions of
  the whole frame put every crop in the wrong place. Confirmed for 59 of 62
  sessions by label OCR, with no session contradicting it; the remaining three
  fall back to `speakers.assumed_psychiatrist_side` and are flagged.
- **Diarization source.** Requested from the lab, and required either way: the
  single mixed audio stream means there is no per-speaker audio, so every
  speaker attribution rests on diarization. Both paths are built — `import` for
  the original output (preferred, since it keeps the comparison against the
  manuscript's text features apples-to-apples) and `pyannote` as a local
  fallback (which would confound modality with transcript changes). Set
  `diarization.import_dir` once the files arrive. See
  [ADR 2](docs/decisions/0002-pluggable-diarization-backends.md).
- **Text features from the manuscript.** Requested. `vc model` joins them by
  `session_id` when supplied; the comparison runs without them otherwise.

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
