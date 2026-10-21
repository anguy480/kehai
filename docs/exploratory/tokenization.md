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

When it passes, it writes to `$VC_OUT_ROOT/exploratory/tokenization/`:

| File | Contents |
| --- | --- |
| `token_features_original.csv` | `session_id` and the 8 features, original rule. |
| `token_features_new.csv` | `session_id` and the 8 features, no length rule. |
| `text_features_original.csv` | `session_id` and all 20, the 8 from the original rule. |
| `text_features_new.csv` | `session_id` and all 20, the 8 from the new variant. The 12 BERT and agenda features are copied unchanged. |

and prints, across all sessions only: kept lemmas and vocabulary size under
each variant, the lemmas recovered in total and per session (minimum, median,
maximum), the 30 most frequent recovered lemmas, and each file's SHA-256. No
per-session text is written or printed.

`text_features_original.csv` should equal the bundle's `text_features.csv`;
step 2 checks that before it runs.
