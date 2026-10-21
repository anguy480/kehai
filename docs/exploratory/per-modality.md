# Per-modality evaluation: running it

**Exploratory.** What is run, and why, is fixed in
[the pre-registration](../prereg/per-modality.md) and, as data, in
[`config/exploratory/per_modality.yaml`](../../config/exploratory/per_modality.yaml).
This page covers only how to run it. Every step that reads labels is run by
the label holder, on the label holder's machine.

The analysis runs in two parts, because the face features exist only after
OpenFace extraction:

1. **Prosody and turn-taking**, from the bundle already sent (`4d8a5a1`).
2. **Face while speaking and face while listening**, from the OpenFace face
   table, once it is extracted and its digest is fixed in the plan.

The first run writes raw values only. The second reads the first run's results
folder and computes BH q and Holm p over the complete families. Keep the first
run's results folder: the second run needs it.

## What you need

- A checkout of this repository on branch `exploratory/per-modality`, at the
  commit named in the message that sent you here. The command refuses to run
  from a checkout with uncommitted changes.
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

Not ready yet. It needs the OpenFace face table and a commit that fixes that
table's digest in the plan. The command will be:

```sh
uv run python -m vc_multimodal.exploratory.per_modality analyze --run second \
    --face-table <path to the face table> \
    --labels <path to the labels CSV> \
    --prior <the run 1 results folder> \
    --out <new folder for run 2 results>
```

## Runtime

Run 1, measured on 2026-10-21 on an Apple M2 with 8 workers: about **2.0 h** at
the planned 1000 permutations. The random forests dominate, at about 7 s per
permutation each against about 1.8 s for the elastic nets. Run 2 has the same
number of models on fewer features, so expect about the same.

To measure it on your own machine, `estimate` fits on random targets and reads
no labels:

```sh
uv run python -m vc_multimodal.exploratory.per_modality estimate --run first \
    --bundle <path to 20260929_4d8a5a1eab8d>
```
