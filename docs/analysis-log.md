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
  package versions, on the bundle's `features.csv`, over the 62 sessions; exit
  0. Outputs are under
  `$VC_OUT_ROOT/unblinded/20261001_4d8a5a1eab8d/results/`. No result is
  recorded here.
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

2026-10-21, exploratory, after unblinding, no confirmatory claim: per-modality evaluation (prosody, turn-taking, face while speaking, face while listening, the face measured with OpenFace and smoothed by a centred 1.0 s rolling mean) pre-registered in [`prereg/per-modality.md`](prereg/per-modality.md) on branch `exploratory/per-modality` from `decaa1d`, committed and pushed before any label is used. Every run that reads labels is made by the label holder, the project owner (who has held the labels since 2026-10-01), from a command committed on that branch; this line first said Tanaka, corrected on 2026-10-21.

2026-10-21, exploratory, no label used: run 1 of the per-modality evaluation (prosody and turn-taking, Analyses A and B, raw values only) set up at `bedc438ac048`, with the label holder's command in [`exploratory/per-modality.md`](exploratory/per-modality.md) for the label holder (the project owner; first written as Tanaka, corrected on 2026-10-21) to run; it refuses any `features.csv` but the 4d8a5a1 bundle's. BH q and Holm p are left to run 2, over the complete families.

2026-10-21, exploratory, no label used: run 2 of the per-modality evaluation (face while speaking and face while listening, Analyses A and B, then BH q and Holm p over the complete families with run 1's results) set up at `d593cfe36513`, with the label holder's command in [`exploratory/per-modality.md`](exploratory/per-modality.md) for the label holder (the project owner; first written as Tanaka, corrected on 2026-10-21) to run. Its face table comes from OpenFace extraction of all 62 sessions at `aef1852`; the plan fixes its SHA-256 (`8c8d65a`) and the command refuses any other file.

2026-10-22, exploratory, after unblinding, no confirmatory claim: runs 1 and 2 of the per-modality evaluation were made, in that order, from a clean detached worktree at `9f4a5ca0d85c` with the commands in [`exploratory/per-modality.md`](exploratory/per-modality.md), with the label holder's label file, `labels_20261001_tanaka.csv`. Run 1 read the 4d8a5a1 bundle's `features.csv`; run 2 read the OpenFace face table whose SHA-256 is fixed at `8c8d65a` and took run 1's results folder as `--prior`. Claude Code ran both on the label holder's machine at the label holder's explicit request, passing the label file's path to the commands without reading it or any result; both exited 0. No result is recorded here.

2026-10-22, exploratory, after unblinding, no confirmatory claim: LLM ratings of the session transcripts as features (6 K6 constructs and 5 SRS-2 treatment subscales as observable in the conversation, rated by a language model from the whole dialogue) pre-registered in [`prereg/llm-ratings.md`](prereg/llm-ratings.md) on branch `exploratory/llm-ratings` from `decaa1d`, with the full Japanese prompt (`config/exploratory/llm_ratings_prompt.txt`, SHA-256 `cf5c42bce615...`), committed and pushed before any transcript is rated or any label is used. The backend is a config setting with no default, so no call is made until the project owner sets it; `openai` waits on Tanaka's confirmation that an outside API is allowed.

2026-10-22, exploratory, no transcript read, no call made, no label used: the rater for [`prereg/llm-ratings.md`](prereg/llm-ratings.md) set up at `e96e605`, with the project owner's commands in [`exploratory/llm-ratings.md`](exploratory/llm-ratings.md): a pilot on the three lowest session numbers, then the full run. It refuses any prompt but the pre-registered one (SHA-256 `cf5c42bce615...`), and with no backend set (as committed) it stops before reading a transcript.

2026-10-22, exploratory, no transcript read, no call made, no label used: the analysis for [`prereg/llm-ratings.md`](prereg/llm-ratings.md) (zero-shot, LLM features alone, added value over the manuscript's 20 text features; Holm across the six primary tests) set up at `81416ba`, with the label holder's command in [`exploratory/llm-ratings.md`](exploratory/llm-ratings.md). It reads `text_features_original.csv` from the tokenization rerun by its SHA-256, and refuses to run until the rater's feature table has been made and its SHA-256 fixed in the plan in a commit of its own.

2026-10-22, exploratory, no transcript read, no call made, no label used: the backend for [`prereg/llm-ratings.md`](prereg/llm-ratings.md) set to `openai` at `b6fbd48`, after Tanaka approved the OpenAI API for these transcripts, with the model pinned to the dated snapshot `gpt-4.1-mini-2025-04-14` rather than the `gpt-4.1-mini` alias. The API key is read from `OPENAI_API_KEY` at run time and is not recorded anywhere. These are the only ratings the pre-registration analyses; rating with another backend or model would be a dated amendment.

2026-10-22, exploratory, no label used: the full LLM ratings run for [`prereg/llm-ratings.md`](prereg/llm-ratings.md) made by the project owner with `openai` / `gpt-4.1-mini-2025-04-14` (62 of 62 sessions rated, 186 valid calls, none failed), and the SHA-256 of its `llm_ratings_features.csv`, `050056e11feda5a1...`, fixed in the plan at `696dde6`, before the analysis. The analysis refuses any other table.

2026-10-22, exploratory, POST HOC, decided after the primary results of the LLM ratings analysis had been seen: Amendment 1 to [`prereg/llm-ratings.md`](prereg/llm-ratings.md), committed at `b71bd93` before it is run. It adds paired-bootstrap 95% percentile intervals (10,000 resamples over sessions, seed `20260923`) for the change in leave-one-out R² and MAE from adding the 11 LLM features to the 20 text features, for K6 and SRS-2, elastic net and random forest, to quantify how large a gain the data can rule out. The leave-one-out predictions are recomputed and must match the primary `models.csv` within 1e-9. No p-value and no correction: descriptive intervals, not tests. It changes no existing result.

2026-10-22, exploratory, POST HOC, no label used here: the `intervals` command for Amendment 1 of [`prereg/llm-ratings.md`](prereg/llm-ratings.md) set up at `2c5af78`, with the label holder's command in [`exploratory/llm-ratings.md`](exploratory/llm-ratings.md) (step 5). It stops unless the recomputed leave-one-out estimates match the primary `models.csv` within 1e-9, and writes descriptive intervals only, no p-value.

2026-10-26, exploratory, after unblinding, no confirmatory claim, recorded late: the primary analysis of [`prereg/llm-ratings.md`](prereg/llm-ratings.md) (step 4 of [`exploratory/llm-ratings.md`](exploratory/llm-ratings.md)) was made by the label holder, the project owner, on 2026-10-22 at about 17:13 JST from `846b144`, with their label file, `$VC_WORK_ROOT/labels/labels_20261001_tanaka.csv`, writing to `$VC_OUT_ROOT/unblinded/llm_ratings/`. It is the run whose results Amendment 1 was decided after. This line belonged before the Amendment 1 line and was added late, on 2026-10-26. No result is recorded here.

2026-10-26, exploratory, POST HOC, after unblinding, no confirmatory claim: the `intervals` command for Amendment 1 of [`prereg/llm-ratings.md`](prereg/llm-ratings.md) (step 5 of [`exploratory/llm-ratings.md`](exploratory/llm-ratings.md)) was run twice on 2026-10-26 by the label holder, the project owner, with their label file, `$VC_WORK_ROOT/labels/labels_20261001_tanaka.csv`, and the primary analysis's results folder, `$VC_OUT_ROOT/unblinded/llm_ratings/`. The first run was from `6da5e05`, a local commit that was never pushed and was then amended into `b97c35b`; its output was moved to `$VC_OUT_ROOT/unblinded/llm_ratings_intervals_6da5e05/`. It was repeated so that the recorded commit would be one on the remote: the second run was from `c582f79`, writing to `$VC_OUT_ROOT/unblinded/llm_ratings_intervals/`, but `c582f79` was pushed only after it, at about 15:58 JST. Both commits differ from the method set up and pushed at `2c5af78` only in `README.md` and `docs/`. The label holder reports the two outputs identical. No result is recorded here.

2026-10-26, exploratory, POST HOC, after unblinding, no confirmatory claim, correcting the line above (added at `12c9b88`): that line was written before the rerun it describes, and its account of the rerun is wrong; there was no run from `c582f79`. What happened: the label holder, the project owner, ran the `intervals` command for Amendment 1 of [`prereg/llm-ratings.md`](prereg/llm-ratings.md) first from `6da5e05`, a local commit never pushed and since amended into `b97c35b`, and moved its output to `$VC_OUT_ROOT/unblinded/llm_ratings_intervals_6da5e05/`. They then reran it at about 16:01 JST from `12c9b88`, which had been pushed at 15:59 JST, with the same label file and results folder, writing to `$VC_OUT_ROOT/unblinded/llm_ratings_intervals/`. `12c9b88` differs from the method at `2c5af78` only in `README.md` and `docs/`. `diff` shows the two runs' `intervals.csv` files identical. No result is recorded here.

2026-10-27, exploratory, after unblinding, no confirmatory claim: a combined model of all six modalities (text, LLM ratings, prosody, turn-taking, face while speaking, face while listening; 71 features, 61 sessions) and the contribution of each, asked for by Tanaka on 2026-10-27, pre-registered in [`prereg/fusion.md`](prereg/fusion.md) at `45755eb` on branch `exploratory/fusion`, from `main` at `acb2cda` with `exploratory/llm-ratings` (`f6bfd2c`) and `exploratory/per-modality` (`1550673`) merged in; committed and pushed before any code for it is written and before any label is used on this branch. It was designed after the label holder had seen the results of the confirmatory run, the `text_plus_all` estimate among them, and of the per-modality, tokenization and LLM ratings analyses, as the pre-registration lists.

2026-10-27, exploratory, no label used on this branch: Amendment 1 to [`prereg/fusion.md`](prereg/fusion.md), committed at `d160db1` before any code for the plan was written. It adds a 14th descriptive comparison, text + nonverbal (the full − llm set: text, prosody, turns and the two OpenFace face windows) against text, both sets already among those fitted, so 112 intervals instead of 104. No model, test or correction changes.

2026-10-27, exploratory, no label used on this branch: the analysis for [`prereg/fusion.md`](prereg/fusion.md) and its Amendment 1 set up at `99291c8`, with the label holder's commands in [`exploratory/fusion.md`](exploratory/fusion.md). It reads the four tables by the SHA-256 already fixed in the LLM ratings and per-modality plans, takes only prosody and turn-taking from the bundle, leaves session 43 out of every model, and writes summaries only.

2026-10-27, exploratory, after unblinding, no confirmatory claim: the fusion analysis of [`prereg/fusion.md`](prereg/fusion.md), with its Amendment 1, was made from a clean checkout at `62746cd` (on GitHub before the run), from 14:30 to 15:05 JST, with the command in [`exploratory/fusion.md`](exploratory/fusion.md) and the label holder's label file, `$VC_WORK_ROOT/labels/labels_20261001_tanaka.csv`, writing to `$VC_OUT_ROOT/unblinded/20261027_fusion_62746cdf3a67/`; `estimate` was run first, at 14:28 JST; 61 sessions per outcome, session 43 left out; exit 0. Claude Code ran both at the label holder's explicit request, passing the label file's path without reading it. Afterwards, at the label holder's explicit request, Claude Code read the run's `summary.md` and no other output file, as [ADR 17](decisions/0017-claude-code-reads-summaries-on-request.md) allows; these results therefore count as seen by Claude Code as well as by the label holder. No result is recorded here.

2026-10-27, exploratory, after unblinding: Correction 1 to [`prereg/fusion.md`](prereg/fusion.md), made after the run from `62746cd`. The outputs table said "the three secondary tests"; the plan's Primary tests section defines four (the random forest's two Wilcoxon tests and its two permutation p's), and the code at `99291c8` computes those four. Wording only: no model, test or correction changes.
