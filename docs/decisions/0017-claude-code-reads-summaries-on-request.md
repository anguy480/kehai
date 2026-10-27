# 17. Claude Code may read an analysis summary at the label holder's request

Date: 2026-10-27

## Status

Accepted. Amends [ADR 16](0016-labels-held-by-the-project-owner.md), which
said that Claude Code sessions never read the results.

## Context

Under ADR 16 a Claude Code session may run a label step at the label holder's
explicit request, but reads neither the labels nor the results. The label
holder reads the results and reports what is needed.

The label holder wants to ask Claude Code to read and explain the results of
exploratory runs. Each exploratory analysis writes its aggregate results as a
Markdown summary beside its tables under `$VC_OUT_ROOT/unblinded/`: `summary.md`
for the analyses, `intervals.md` for the LLM ratings intervals. These hold
aggregate statistics only (R², MAE, ρ, p-values, intervals), never a label
value, a prediction, a residual or a per-session error. That is what each
analysis's run documentation, under `docs/exploratory/`, promises.

On 2026-10-27 the label holder asked Claude Code to read the fusion run's
`summary.md` (see the analysis log on `exploratory/fusion`, 2026-10-27), and
asked for that to become the rule.

## Decision

- At the label holder's explicit request, for a named run, a Claude Code
  session may read that run's `summary.md` or `intervals.md` under
  `$VC_OUT_ROOT/unblinded/`, and nothing else there.
- Everything else under the roots stays closed: the labels, per-session files,
  feature tables, the other output files (`*.csv`, `run.json`, the logs), and
  the confirmatory run's `model_summary.md`. The rule names files, not "any
  summary".
- A request covers the run it names. It does not carry over to later runs.
- If a summary turns out to hold anything per-session, Claude Code stops
  reading, says so, and the analysis that wrote it is fixed on its branch.
- Nothing read goes into the repository: no result, figure or paraphrase of
  one in a file, commit message or log line. The repository is public.
- The run's line in the analysis log records that Claude Code read the summary
  and at whose request.

## Consequences

- What Claude Code has read counts as seen. Any later pre-registration Claude
  Code drafts lists those results under what had been seen when it was
  written, as the label holder's own reading already must.
- Reading a summary sends its contents to the model provider. The label
  holder's explicit request is the permission that `CLAUDE.md` requires before
  anything derived from the recordings, even a summary, leaves the machine.
- The labels themselves stay as ADR 16 has them: Claude Code passes their path
  to a command and never reads them.
