"""LLM-rated natural-language face features. Exploratory, designed after unblinding.

Three steps, each a subcommand of `python -m vc_multimodal.exploratory.llm_face`:

* `describe` turns each session's per-frame facial measures into a short
  description, separately for speaking and listening, with a fixed template
  (`template.py`). No model is involved.
* `rate` has a pinned, local, open-weight model rate each description on five
  1-7 scales, at temperature 0, several times (`rate.py`, `rating_prompt.txt`).
  It refuses to send anything to a host other than this machine.
* `analyze` relates the ratings to the numeric face features and the outcomes.

Descriptions and ratings are derived from clinical recordings, so they are
written under `$VC_WORK_ROOT`, never printed, and never committed.
"""
