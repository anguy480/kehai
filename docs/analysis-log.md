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

2026-10-05, exploratory, requested by Tanaka after unblinding, with no multiplicity correction and no confirmatory claim: AU-only baseline on branch `exploratory/au-baseline` ([design](exploratory/au-baseline.md)); every feature set, the nested search, models, CV schemes and nulls are fixed in `config/exploratory/au_baseline.yaml` at commit `7d28d3088d4ec19a83aec04654476973e7c1b006`, pushed before any model saw a label (pooled AU table SHA-256 `ef17e07913557d32e07c191da9507db3f171d61601e34ea099d1b5555e681403`).

2026-10-06, exploratory and uncorrected, no confirmatory claim: the AU-only baseline (plan fixed at `7d28d30`, run from `07543e5`, both pushed before any label was used) was run on 62 sessions with 1000 permutations, from 2026-10-05 20:08 to 2026-10-06 06:19 JST, exit 0 with no leave-one-out fit failing to converge; results in `$VC_OUT_ROOT/unblinded/20261005_au_baseline_07543e5ec044/`.
