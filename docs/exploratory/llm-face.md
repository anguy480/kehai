# LLM-rated face features

**Exploratory. Designed on 2026-10-02, after the labels were received on
2026-10-01.** Nothing here is part of the pre-registered analysis or changes it,
and no result from it may be described in confirmatory terms. See
[the analysis log](../analysis-log.md).

## What it does

Mirrors the manuscript's LLM-rated agenda scores, for the face instead of the
transcript.

1. **Describe** (`python -m vc_multimodal.exploratory.llm_face describe`). Each
   session's per-frame facial measures become a short description, one
   paragraph for the participant's speaking windows and one for their listening
   windows - the same windows, and the same 30 s minimum of measured time, that
   `vc aggregate` uses. For each of AU1, AU2, AU4, AU6 and AU12 it says how often
   the unit was present, how strongly, how long episodes lasted and whether that
   changed between the first and last third of the conversation; then how much
   the head moved. The template is deterministic and has no model in it
   (`src/vc_multimodal/exploratory/llm_face/template.py`). Session 43 (face
   unavailable) gets no description; session 52 (face degraded) is described and
   flagged.
2. **Rate** (`... rate --runs 3`). A local, open-weight model rates each
   description on five 1-7 scales: positive affect, expressivity, flat affect,
   tension/negative affect, and engagement while listening. Prompt:
   `src/vc_multimodal/exploratory/llm_face/rating_prompt.txt`. Three identical
   requests per session.

The descriptions and ratings are derived from clinical recordings. They are
written under `$VC_WORK_ROOT/llm_face/`, are never printed, and are not in this
repository: only their digests are, below. `rate.py` refuses any host but this
machine, and checks the model digest before the first request.

## Frozen before any label was joined

Produced on 2026-10-02 with no label file opened by this code.

| Item | Value |
| --- | --- |
| Model | `qwen2.5:3b-instruct-q4_K_M` via Ollama 0.24.0, on `127.0.0.1` |
| Model digest | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` |
| Decoding | temperature 0, seed 0, `num_ctx` 4096, `num_predict` 200, output constrained to a JSON schema |
| Template (`template.py`) SHA-256 | `f17bb67945191a7765d586ef45f10ab61bff19ee87e07a0bad0e767b1285206f` |
| Prompt (`rating_prompt.txt`) SHA-256 | `38eee35d5f08fe6572676089cfb027beeeaebd2410c0c24e6ef6314e48e5fc6f` |
| `descriptions.csv` SHA-256 | `609f42c5e944ed3cdf8adc66f24c6ab4d137bb37b1b03f74a139a60ae50728ef` |
| `scores_runs.csv` SHA-256 (61 sessions x 3 runs, 0 failed) | `3a868788a7c9bc816b23de298729626731289708b66203672fd102722e4dddb8` |

Anything that changes a digest above changes the analysis, and is a new,
separately reported exploratory analysis.

## Known limits, stated before the results

* **AU6 is uninformative.** MediaPipe's cheek-raising blendshape is effectively
  zero in every session (the cohort's 95th percentile of its mean is 0.000), so
  every description says it "essentially never appeared". This also affects the
  two confirmatory AU06 features, which are near-constant.
* **The thresholds were set from this cohort's feature table** - label-free,
  but the same data. They are in `template.py` with their reasoning.
* **Three runs at temperature 0 measure determinism**, not rater reliability:
  identical requests to a greedy decoder should agree perfectly.
* **A 3B model on a laptop.** Chosen because it runs on this machine (8 GB)
  and nothing leaves it. A larger model might rate differently.
* **The description is built from the same numbers as the numeric face
  features.** Whether the ratings add anything beyond re-describing them is the
  first question the analysis asks.
