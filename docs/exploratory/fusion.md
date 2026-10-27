# All modalities combined: running it

**Exploratory.** What is fitted, tested and resampled is fixed in
[the pre-registration](../prereg/fusion.md) and its Amendment 1. This page
covers only how to run it. The analysis reads the labels, so the label holder,
who is the project owner, runs it, or Claude Code at their explicit request,
passing the labels' path without reading the labels or any result
([ADR 16](../decisions/0016-labels-held-by-the-project-owner.md)).

There are two steps:

1. **Estimate:** time the run on random targets. It reads the four feature
   tables and no label.
2. **Analyze:** the pre-registered analysis. It reads the labels.

## What you need

- A clean checkout of branch `exploratory/fusion`, at or after the commit
  recorded for it in `docs/analysis-log.md`. `analyze` refuses a checkout with
  uncommitted or untracked files.
- `.env` with `VC_OUT_ROOT`, for the default paths below. Every path can also
  be given on the command line.
- The four feature tables. Each is refused unless its SHA-256 is the one
  already fixed for it, so none can be swapped by accident:

| Table | Default place, under `$VC_OUT_ROOT` | Option | Fixed in |
| --- | --- | --- | --- |
| `text_features_original.csv` | `exploratory/tokenization/` | `--text-features` | the LLM ratings plan |
| `llm_ratings_features.csv` | `exploratory/llm_ratings/` | `--llm-features` | the LLM ratings plan |
| the bundle's `features.csv` | `handoff/20260929_4d8a5a1eab8d/` (the folder) | `--bundle` | the per-modality plan |
| `face_openface_features.csv` | `exploratory/per_modality/` | `--face-table` | the per-modality plan |

- The labels, a CSV with a `session_id` column (`session` or `id` also work)
  and columns named `K6` and `SRS2`.

Only the prosody and turn-taking columns are taken from the bundle. Its
MediaPipe face columns, which share the OpenFace names, are never read into
the analysis.

## 1. Estimate

```sh
cd <your checkout of the repository>
git fetch origin
git checkout exploratory/fusion
git pull --ff-only

uv run python -m vc_multimodal.exploratory.fusion estimate
```

It times each of the 15 feature sets with both models on random targets, and
the full model's permutation nulls on 4 permutations, then extrapolates to
1000. It prints the seconds per set, the time per permutation, and the total
for the number of workers (`--jobs`, default every CPU). The four nulls are
the long calls, so the total cannot fall below one null however many workers
there are.

## 2. Analyze

```sh
uv run python -m vc_multimodal.exploratory.fusion analyze \
    --labels <path to the labels CSV>
```

`--out` defaults to `$VC_OUT_ROOT/unblinded/<date>_fusion_<commit>`, and must
be a new folder. It prints the sessions analysed per outcome (61 expected),
the session left out (43) and the one flagged (52), the counts of models,
nulls and intervals, and the files it wrote; no result.

| File | Contents |
| --- | --- |
| `primary_tests.csv` | The four primary tests with raw and Holm p, then the four random-forest tests, uncorrected. |
| `models.csv` | 60 rows, one per set, outcome and model: leave-one-out R², MAE and ρ, non-converged fits; repeated 5-fold for full and text; the permutation null for full. |
| `contributions.csv` | The 112 intervals: 14 comparisons × 2 outcomes × 2 models × ΔR² and ΔMAE. |
| `summary.md` | The same as tables, with the plan's caveats. |
| `run.json` | Commit, input SHA-256s, settings, sessions, timing and package versions. |
| `logs/` | The run log. |

No file holds a label value, a feature value, a prediction, a residual or a
per-session error.
