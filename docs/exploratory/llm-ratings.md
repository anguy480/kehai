# LLM ratings of transcripts: running it

**Exploratory.** What is rated, how, and how it is analysed is fixed in
[the pre-registration](../prereg/llm-ratings.md), and the prompt in
[`config/exploratory/llm_ratings_prompt.txt`](../../config/exploratory/llm_ratings_prompt.txt).
This page covers only how to run it. Rating reads the transcripts and no label,
so the project owner runs it. Claude Code never reads the transcripts, the
model's responses, the ratings or any result.

There are three steps:

1. **Choose the backend**, in a commit of its own. Until then no call can be
   made.
2. **Pilot:** the three lowest session numbers, to check parsing and runtime.
   It prints counts only, no rating.
3. **Full run:** every session, reusing the pilot's calls. It writes the
   feature table the analysis reads.

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
- **`openai`**: `gpt-4.1-mini`, only once Tanaka has confirmed that an outside
  API is allowed for these transcripts. It sends each full session dialogue to
  OpenAI. Put the key in `OPENAI_API_KEY` in your shell, never in a file in
  the repository.

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
