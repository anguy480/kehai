# LLM ratings of transcripts: running it

**Exploratory.** What is rated, how, and how it is analysed is fixed in
[the pre-registration](../prereg/llm-ratings.md), and the prompt in
[`config/exploratory/llm_ratings_prompt.txt`](../../config/exploratory/llm_ratings_prompt.txt).
This page covers only how to run it. Rating reads the transcripts and no label,
so the project owner runs it; the analysis reads the labels, so the label
holder, who is also the project owner, runs it. Claude Code never reads the
transcripts, the model's responses, the ratings, the labels or any result.

There are four steps:

1. **Choose the backend**, in a commit of its own. Until then no call can be
   made.
2. **Pilot:** the three lowest session numbers, to check parsing and runtime.
   It prints counts only, no rating.
3. **Full run:** every session, reusing the pilot's calls. It writes the
   feature table the analysis reads.
4. **Analysis:** once the feature table's SHA-256 is fixed in the plan. It
   reads the labels.

## What you need

- A clean checkout of branch `exploratory/llm-ratings`. The command refuses a
  checkout with uncommitted or untracked files, so the backend and prompt used
  are the committed ones.
- The `.txt` transcripts that `analysis.py` reads, in
  `$VC_WORK_ROOT/diarization/diarizations_original/` (the same folder as step
  2 of the tokenization rerun), or anywhere else with `--transcripts`. There
  must be 62.
- `.env` with `VC_WORK_ROOT` and `VC_OUT_ROOT`.

## 1. Choose the backend

In [`config/exploratory/llm_ratings.yaml`](../../config/exploratory/llm_ratings.yaml),
set `backend` to one of:

- **`local`**: also set `backends.local.base_url` (for example
  `http://127.0.0.1:8000/v1`) and `backends.local.model` to the name the server
  serves. Any OpenAI-compatible server works: vLLM, llama.cpp, Ollama, LM
  Studio. If it needs a key, set `api_key_env` to the name of the variable
  holding it. The server must accept `response_format: {"type": "json_object"}`
  and a context long enough for a whole session; the pilot shows both.
- **`openai`**: `gpt-4.1-mini`, pinned to a dated snapshot, only once Tanaka
  has confirmed that an outside API is allowed for these transcripts. It sends
  each full session dialogue to OpenAI. Put the key in `OPENAI_API_KEY`,
  either in your shell or in `.env` in the folder you run the command from (it
  is gitignored, and a worktree has its own). Never put it in a tracked file.

The backend is set: `openai` with `gpt-4.1-mini-2025-04-14`, at `b6fbd48`.

Commit that change on its own, with a dated line in `docs/analysis-log.md`
naming the backend and model, and push it before the pilot. The pre-registration
analyses one backend's ratings; rating with a second is an amendment.

## 2. Pilot

```sh
cd <your checkout of the repository>
git checkout exploratory/llm-ratings
git pull --ff-only

uv run python -m vc_multimodal.exploratory.llm_ratings pilot
```

It rates the three lowest session numbers, three calls each, and prints:

```text
log: <path>
llm ratings, pilot: backend <backend>, model <model>, prompt cf5c42bce615, commit <commit>
sessions: 3 of 62 (the three lowest session numbers)
calls: 9 (9 made now, 0 from the cache); valid <n>, malformed after retries <n>, rejected <n>
responses: <n>, parse success <x>%, valid at the first attempt <x>%, retries <n>
sessions rated: <n>; failed: <n>
tokens: prompt <n> (mean <n>, max <n> per response), completion <n>
time: <n> s, <n> s per call made now
wrote <out>/llm_ratings_pilot.json
```

Paste that output back. A malformed response is counted by reason (not JSON,
wrong keys, and so on) on a line of its own; its text is never shown. A
`rejected` call usually means the session is longer than the model's context.

If parsing fails, the fix is a dated amendment to the pre-registration, made
before the full run; the prompt cannot simply be edited, because the command
refuses any prompt whose SHA-256 is not the pre-registered one.

## 3. Full run

```sh
uv run python -m vc_multimodal.exploratory.llm_ratings rate
```

It reuses the pilot's 9 calls and makes the other 177. `--jobs N` rates N
sessions at once (default 1); with the OpenAI API, 4 is reasonable. If the
backend stops answering (network, rate limit, server error, after 5 retries
with backoff), the run stops; run the same command again and it continues from
where it stopped, making no call twice.

It prints the same counts as the pilot, then each item's ICC(2,1) and ICC(2,3)
across the three calls, and writes, under `$VC_OUT_ROOT/exploratory/llm_ratings/`
(or `--out`):

| File | Contents |
| --- | --- |
| `llm_ratings_features.csv` | `session_id` and the 11 features `llm__k_*`, `llm__s_*`, each the mean of the three calls. A failed session's row is blank. |
| `llm_ratings_summary.json` | The counts, the ICCs, the commit, backend, model, prompt and plan SHA-256. |
| `logs/` | The run log. |

Every call's response is cached under
`$VC_WORK_ROOT/exploratory/llm_ratings/cache/`, by prompt SHA-256, model,
session and call. Keep it: a rerun reads it and makes no call. A malformed call
is cached as malformed, so a rerun does not try it again; a rejected request is
not cached, so a rerun does.

Then paste the output back, and the feature table's SHA-256
(`shasum -a 256 <out>/llm_ratings_features.csv`). The SHA-256 is fixed in the
plan in a commit of its own before the analysis, which refuses any other table.

## 4. Analysis

Run it after the feature table's SHA-256 has been fixed in
`config/exploratory/llm_ratings.yaml` (`analysis.llm_features.sha256`), in a
commit of its own, pushed, from a clean checkout at or after that commit. Until
then it refuses to run. Besides the labels it reads:

- `llm_ratings_features.csv` from step 3, by default in
  `$VC_OUT_ROOT/exploratory/llm_ratings/` (`--features` for another path).
- `text_features_original.csv`, the manuscript's 20 text features with a
  `session_id` column, written by step 2 of the tokenization rerun, by default
  in `$VC_OUT_ROOT/exploratory/tokenization/` (`--text-features` for another
  path). Its SHA-256 is already fixed in the plan (`da984428e499...`).

The labels are a CSV with a `session_id` column (`session` or `id` also work)
and columns named `K6` and `SRS2`.

To time it first, on random targets and without labels:

```sh
uv run python -m vc_multimodal.exploratory.llm_ratings estimate
```

Then:

```sh
uv run python -m vc_multimodal.exploratory.llm_ratings analyze \
    --labels <path to the labels CSV> \
    --out <new folder for the results>
```

`--out` defaults to `$VC_OUT_ROOT/unblinded/<date>_llm_ratings_<commit>`, and
must be a new folder. `--jobs` sets the number of parallel workers (default:
every CPU). It prints the sessions analysed per outcome, any session the rater
did not rate (left out of everything, both arms of the comparison included),
and the files it wrote:

| File | Contents |
| --- | --- |
| `primary_tests.csv` | The six primary tests (zero-shot, LLM alone, added value; K6 and SRS-2), with raw p and Holm p across the six. |
| `models.csv` | One row per feature set (`llm`, `text`, `text_llm`), outcome and model: LOO R², MAE and rho, the repeated 5-fold R², the permutation null and p, and non-converged fits. |
| `comparisons.csv` | Added value for both models: Wilcoxon on per-session LOO absolute errors, the random forest uncorrected. |
| `univariate.csv` | Spearman rho, p and BH q (within each outcome, across the 11) for each LLM feature. |
| `summary.md` | The same, as readable tables. |
| `run.json` | Commit, input digests, settings, sessions analysed, timing and package versions. |
| `logs/` | The run log. |

No file holds a label value, a rating, a prediction or a residual.

## 5. Post hoc intervals for the added value (Amendment 1)

**Post hoc**: [Amendment 1](../prereg/llm-ratings.md#amendment-1-2026-10-22-intervals-for-the-added-value-post-hoc)
was decided after the primary results had been seen. Run it from a clean
checkout at or after the commit that added this section, giving it the primary
analysis's results folder:

```sh
uv run python -m vc_multimodal.exploratory.llm_ratings intervals \
    --labels <path to the labels CSV> \
    --results <the primary analysis's results folder>
```

It reads the same two tables as step 4 (`--features` and `--text-features` as
there), checks that the primary run's `run.json` records the planned settings
and the same input SHA-256s, and recomputes the leave-one-out predictions of the
8 added-value models (text and text + LLM; K6 and SRS-2; elastic net and random
forest). Unless every recomputed n, R² and MAE matches the primary `models.csv`
within 1e-9, it stops and writes nothing. It then draws 10,000 paired bootstrap
resamples of the sessions per outcome (seed `20260923`) and prints the gate, the
sessions and the files it wrote, not the intervals. The recomputation is the
slow part, a few minutes; `--jobs` (default: every CPU) runs the 8 models in
parallel.

`--out` defaults to `$VC_OUT_ROOT/unblinded/llm_ratings_intervals` and must be a
new folder.

| File | Contents |
| --- | --- |
| `intervals.csv` | 8 rows: for each outcome and model, ΔR² and ΔMAE ((text + LLM) minus text), the point estimate, the 95% percentile interval, and the resamples used. |
| `intervals.md` | The same as a table, the post hoc statement, the limitation, and the commit, seed and SHA-256s of the primary `models.csv` and the inputs. |
| `logs/` | The run log. |

A gain from the LLM features is a positive ΔR² and a negative ΔMAE. There is no
p-value: the intervals describe, and the Wilcoxon tests of step 4 remain the
tests. The bootstrap does not refit the models, so the intervals are likely
somewhat too narrow.
