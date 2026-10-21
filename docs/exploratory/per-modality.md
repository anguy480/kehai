# Per-modality evaluation: running it

**Exploratory.** What is run, and why, is fixed in
[the pre-registration](../prereg/per-modality.md) and, as data, in
[`config/exploratory/per_modality.yaml`](../../config/exploratory/per_modality.yaml).
This page covers only how to run it. Every step that reads labels is run by
the label holder (the project owner, who has held the labels since
2026-10-01), on the machine that holds them, or by Claude Code at the label
holder's explicit request. Claude Code passes the labels' path to the command
and never reads the labels or the outputs.

The analysis runs in two parts, because the face features exist only after
OpenFace extraction:

1. **Prosody and turn-taking**, from the bundle already sent (`4d8a5a1`).
2. **Face while speaking and face while listening**, from the OpenFace face
   table, once it is extracted and its digest is fixed in the plan.

The first run writes raw values only. The second reads the first run's results
folder and computes BH q and Holm p over the complete families. Keep the first
run's results folder: the second run needs it.

## What you need

- A clean checkout of branch `exploratory/per-modality` at or after the commit
  recorded for the run in `docs/analysis-log.md`. The command refuses to run
  from a checkout with uncommitted or untracked files; a detached worktree at
  the pushed commit is the simplest way to get one.
- `uv` (it installs the environment on first run). No `.env`, no data folders
  and no ffmpeg are needed.
- The bundle folder from 2026-09-29, `20260929_4d8a5a1eab8d`. The command checks
  that its `features.csv` is the one sent and refuses any other.
- The labels as a CSV with a `session_id` column (`session` or `id` also work)
  and columns named `K6` and `SRS2`. Other columns are ignored.

## Run 1: prosody and turn-taking

```sh
cd <your checkout of the repository>
git fetch origin
git checkout exploratory/per-modality
git pull --ff-only

uv run python -m vc_multimodal.exploratory.per_modality analyze --run first \
    --bundle <path to 20260929_4d8a5a1eab8d> \
    --labels <path to the labels CSV> \
    --out <new folder for run 1 results>
```

It prints the sessions used per modality and the files it wrote, and logs each
model as it finishes. `--jobs` sets the number of parallel workers (default:
every CPU).

What it writes to `--out`:

| File | Contents |
| --- | --- |
| `univariate.csv` | Analysis A: Spearman rho, raw p and n for each of the 24 features and both outcomes. `q` is empty. |
| `models.csv` | Analysis B: one row per modality, outcome and model, with LOO R², MAE and rho, the repeated 5-fold R², the permutation null and p, and non-converged fits. `p_holm` is empty. |
| `summary.md` | The same, as readable tables. |
| `run.json` | Commit, plan and input digests, sessions per modality, timing and package versions. |
| `logs/` | The run log. |

No file holds a label value, a prediction or a residual.

## Run 2: the face modalities

Run it after run 1, from a clean checkout at or after the commit recorded for
run 2 in `docs/analysis-log.md`.
Besides the labels and run 1's results folder, it needs the OpenFace face table,
`face_openface_features.csv`, which `face-table` wrote to
`$VC_OUT_ROOT/exploratory/per_modality/`. It holds facial features
only, no label. The command checks its SHA-256 against the one fixed in
[the plan](../../config/exploratory/per_modality.yaml) (`6d67cfb0aad7...`) and
refuses any other file.

```sh
cd <your checkout of the repository>
git fetch origin
git checkout exploratory/per-modality
git pull --ff-only

uv run python -m vc_multimodal.exploratory.per_modality analyze --run second \
    --face-table <path to face_openface_features.csv> \
    --labels <path to the labels CSV> \
    --prior <the run 1 results folder> \
    --out <new folder for run 2 results>
```

`--prior` must be run 1's folder as run 1 wrote it. The command refuses a folder
from another analysis, another bundle or another permutation count, since its
results could not then share a correction.

What it writes to `--out`:

| File | Contents |
| --- | --- |
| `univariate.csv` | Analysis A for the 18 face features, raw values as in run 1. |
| `models.csv` | Analysis B for the two face modalities, raw values as in run 1. |
| `final_univariate.csv` | Analysis A for all four modalities, with BH q within each outcome across the whole family. |
| `final_models.csv` | Analysis B for all four modalities, with Holm p across the 8 elastic-net tests. |
| `summary.md` | The final tables, readable. |
| `run.json` | Commit, plan and input digests, run 1's commit and inputs, sessions per modality, timing and package versions. |
| `logs/` | The run log. |

No file holds a label value, a prediction or a residual. Session 43's row in the
face table is blank, by a QC note, so it is left out of the face modalities.

## Runtime

Run 1, as made on 2026-10-22 on an Apple M2 with 8 workers: about **0.5 h**
(31 min) at the planned 1000 permutations. The 2.0 h first written here, from
`estimate` on 2026-10-21, was about four times too long.

Run 2, made the same day on the same machine: about **0.5 h** (30 min), as
`estimate` predicted.

To measure it on your own machine, `estimate` fits on random targets and reads
no labels. Its figure for run 1 is an overestimate:

```sh
uv run python -m vc_multimodal.exploratory.per_modality estimate --run first \
    --bundle <path to 20260929_4d8a5a1eab8d>
uv run python -m vc_multimodal.exploratory.per_modality estimate --run second \
    --face-table <path to face_openface_features.csv>
```
