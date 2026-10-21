# Text features without the short-lemma filter: running it

**Exploratory.** What is run, and why, is fixed in
[the pre-registration](../prereg/tokenization.md). This page covers only how to
run it.

There are two steps, both run by the project owner, who holds the transcripts
and the labels:

1. **Recompute the features.** Reads the transcripts; no label is read.
2. **Analyse them.** Reads the labels. Prepared once step 1 has passed its
   reproduction check.

## Step 1: recompute the eight lemma-based features

The token features are ported from `analysis.py` in Tanaka's public repository
(MIT, commit `ec07010`); see `src/vc_multimodal/exploratory/tokenization/`.
Janome 0.5.0 and lexicalrichness 0.5.1 are pinned in the `text` dependency
group.

From a clean checkout of `exploratory/tokenization` with this machine's `.env`
(the main checkout, or a copy of `.env` beside this one):

```sh
uv sync --all-groups
uv run python -m vc_multimodal.exploratory.tokenization features
```

It reads the `.txt` transcripts in `$VC_WORK_ROOT/diarization/diarizations_original/`
(the files `analysis.py` read, sorted by session number as it sorted them) and
`$VC_WORK_ROOT/nlp_features.csv`. `--transcripts`, `--nlp-features` and `--out`
override the three locations.

**The reproduction check comes first.** The original rule must reproduce the
eight columns of `nlp_features.csv`, row for row, to within 1e-9. If it does
not, the command prints which columns differ and by how much at most, writes
nothing, does not compute the new variant, and exits with status 1. Nothing
else proceeds until that is resolved.

When it passes, it computes (b) and then (c) (Amendment 1 of the
pre-registration), and writes to `$VC_OUT_ROOT/exploratory/tokenization/`:

| File | Contents |
| --- | --- |
| `token_features_original.csv` | `session_id` and the 8 features, original rule. |
| `token_features_new.csv` | `session_id` and the 8 features, (b): no length rule. |
| `token_features_new_c.csv` | `session_id` and the 8 features, (c): no length rule, no 非自立. |
| `text_features_original.csv` | `session_id` and all 20, the 8 from the original rule. |
| `text_features_new.csv` | `session_id` and all 20, the 8 from (b). |
| `text_features_new_c.csv` | `session_id` and all 20, the 8 from (c). |

The 12 BERT and agenda features are copied unchanged into every 20-feature
table. It prints, across all sessions only: kept lemmas and vocabulary size
under each variant; what (b) recovers against (a), in total and per session
(minimum, median, maximum), with its 30 most frequent recovered lemmas; what
(c) drops against (b) and its net change against (a), with the 30 most
frequent dropped lemmas; and each file's SHA-256. No per-session text is
written or printed.

The (a) and (b) files must come out byte-identical to the first run's; their
SHA-256 are in Amendment 1. The SHA-256 of `text_features_new_c.csv` is then
fixed in `config/exploratory/tokenization.yaml`, in a commit of its own, before
step 2 can run.

## Step 2: the analysis

Reads the labels, so it is run by the label holder, from a clean checkout of
`exploratory/tokenization` at or after the commit that fixes (c)'s SHA-256.
The command refuses a checkout with uncommitted or untracked files, any step 1
table whose SHA-256 is not the planned one, and a results folder already in
use.

```sh
uv run python -m vc_multimodal.exploratory.tokenization analyze \
    --labels <path to the labels CSV>
```

- **Labels.** A CSV with a session column (`session_id`, `session` or `id`)
  and columns named `K6` and `SRS2`. If it has other names, it stops and
  prints the column names only.
- **Inputs.** The three 20-feature tables in
  `$VC_OUT_ROOT/exploratory/tokenization/`; `--features-dir` points elsewhere.
- **Results.** `$VC_OUT_ROOT/unblinded/<date>_tokenization_<commit>/` unless
  `--out` says otherwise. `--jobs` sets the parallel workers (default: every
  CPU).

What it writes:

| File | Contents |
| --- | --- |
| `comparisons.csv` | The test: Wilcoxon of original 20 against (b) and against (c), by outcome and model; `p_holm` across the four elastic-net tests. |
| `models.csv` | The 12 models: LOO R², MAE and Spearman, the repeated 5-fold R², the permutation null and p, and non-converged fits. |
| `univariate.csv` | Spearman ρ and p of each of the 20 features with each outcome, and BH q within each feature set and outcome. |
| `summary.md` | The same, readable. |
| `run.json` | Commit, input SHA-256, settings, sessions modelled, timing and package versions. |
| `logs/` | The run log. |

No file holds a label value, a prediction, a residual or a per-session error.
The terminal shows only the sessions modelled and the files written.

To time it first, on random targets and without reading a label:

```sh
uv run python -m vc_multimodal.exploratory.tokenization estimate
```
