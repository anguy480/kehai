# Analysis log

Dated record of events that bear on what the results can claim. No label
value, per-session score, prediction or residual is ever written here.

## 2026-10-01 - Labels received; the analysis is unblinded

- **Received** from Tanaka on 2026-10-01: the questionnaire scores for all 62
  sessions (`demo_2.csv`), with a request that the analysis be run on this
  machine rather than his. This departs from
  [ADR 1](decisions/0001-handoff-split-no-labels-on-student-machine.md): from
  this date the extraction machine has held the labels.
- **Frozen commit**: `4d8a5a1eab8d3316c9645be818957d8d25562faa`, the commit
  recorded in the manifest of the bundle sent to Tanaka
  (`20260929_4d8a5a1eab8d`, `features.csv` SHA-256 `375a38b0c8bb...`). HEAD at
  receipt was `775efdd`; between the two only the bundle README's wording and its
  test changed (`stages/handoff.py`, `tests/unit/test_handoff_stage.py`). The
  analysis code, `modeling/tiers.py`, `config/` and `uv.lock` are identical to
  the frozen commit, and the confirmatory run was made from a checkout of it.
- **Earlier model runs** (2026-09-28 and -29, five in all) used randomly
  generated labels to exercise the stage; see commit `9008984`. No real label
  had been on this machine before 2026-10-01.
- **The file as received** is UTF-8 (CRLF), not the cp932 it was described as;
  its only non-ASCII bytes are in `Gender`. It was converted to `session_id,K6,
  SRS2`, with `Gender` and `Age` dropped, and is kept under
  `$VC_WORK_ROOT/labels/`, outside this repository. Received file SHA-256 begins
  `24cb199bca42a97a`.
- **Session IDs** match the feature table exactly: 62 in each, none missing on
  either side.
- `.gitignore` gained label-file patterns after receipt. That change touches no
  analysis.
- **Confirmatory run**, 2026-10-01 13:57-15:05 JST: `vc model` from a clean
  checkout of `4d8a5a1eab8d`, environment matching the manifest's 11 recorded
  package versions, on the bundle's `features.csv`; 62 sessions modelled, exit
  0, `n_fits_not_converged` 0 in all 40 estimates. Outputs are under
  `$VC_OUT_ROOT/unblinded/20261001_4d8a5a1eab8d/results/`. Holm-adjusted p for
  the four pre-registered tests: audio vs text, K6 0.0046 (text closer);
  all vs text, K6 0.0039 (text closer); audio vs text, SRS2 0.241; all vs
  text, SRS2 0.241.
- The bundle README's command (`uv run vc model ...` from the bundle directory)
  fails there with exit 2, `config file not found: config/default.yaml`,
  before the labels are read. The run above is that command issued from the
  repository root with absolute paths to the bundle and the labels.

**From this point on, anything changed in features, aggregation, the
confirmatory feature list, the confirmatory tests or their correction is
exploratory**, whatever it is called, and must be reported as a change made
after the labels were seen. The confirmatory result is the run from
`4d8a5a1eab8d` described above.

2026-10-01, post-unblinding usability fix, no change to the analysis: `vc` finds `config/default.yaml` when run from outside the repository, and the bundle README sets `UV_PROJECT` so its command runs from the bundle folder.

2026-10-01, post-unblinding usability fix, no change to the analysis: `vc model` runs with no `.env` or `VC_*` variables, reading the bundle's `text_features.csv` (joined table checked identical to the work-root one) and logging under `--out`; `handoff` and `qc-note` no longer require the data root.

2026-10-02, recorded after unblinding: with the random smoke-test labels, `model_results.csv` and `model_comparisons.csv` from `4d8a5a1` (repository root, work-root text features) and from `2c777757e8b4` (clean environment, bundle text features) are byte-identical; the sent `features.csv` has SHA-256 `375a38b0c8bb138b04ed07ff6504f04a39be7df7fe9df36609d394070e750f49`, which the bundle manifest did not record.

2026-10-22, exploratory, after unblinding, no confirmatory claim: LLM ratings of the session transcripts as features (6 K6 constructs and 5 SRS-2 treatment subscales as observable in the conversation, rated by a language model from the whole dialogue) pre-registered in [`prereg/llm-ratings.md`](prereg/llm-ratings.md) on branch `exploratory/llm-ratings` from `decaa1d`, with the full Japanese prompt (`config/exploratory/llm_ratings_prompt.txt`, SHA-256 `cf5c42bce615...`), committed and pushed before any transcript is rated or any label is used. The backend is a config setting with no default, so no call is made until the project owner sets it; `openai` waits on Tanaka's confirmation that an outside API is allowed.

2026-10-22, exploratory, no transcript read, no call made, no label used: the rater for [`prereg/llm-ratings.md`](prereg/llm-ratings.md) set up at `e96e605`, with the project owner's commands in [`exploratory/llm-ratings.md`](exploratory/llm-ratings.md): a pilot on the three lowest session numbers, then the full run. It refuses any prompt but the pre-registered one (SHA-256 `cf5c42bce615...`), and with no backend set (as committed) it stops before reading a transcript.

2026-10-22, exploratory, no transcript read, no call made, no label used: the analysis for [`prereg/llm-ratings.md`](prereg/llm-ratings.md) (zero-shot, LLM features alone, added value over the manuscript's 20 text features; Holm across the six primary tests) set up at `81416ba`, with the label holder's command in [`exploratory/llm-ratings.md`](exploratory/llm-ratings.md). It reads `text_features_original.csv` from the tokenization rerun by its SHA-256, and refuses to run until the rater's feature table has been made and its SHA-256 fixed in the plan in a commit of its own.
