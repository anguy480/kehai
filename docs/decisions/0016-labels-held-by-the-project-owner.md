# 16. The labels are held on the project owner's machine

Date: 2026-10-22

## Status

Accepted. Supersedes, in part,
[ADR 1](0001-handoff-split-no-labels-on-student-machine.md): the labels are no
longer kept off the extraction machine. The split of the code into a label-free
extraction and a separate label holder's step stands.

## Context

ADR 1 kept the questionnaire scores (K6, SRS-2) with the supervising professor,
Tanaka, and off the machine that extracts the features, so that the features
could not be tuned against the outcome.

On 2026-10-01 Tanaka sent the scores for all 62 sessions to the project owner,
with a request that the analysis be run on the project owner's machine, which
is the extraction machine (see the [analysis log](../analysis-log.md),
2026-10-01). The labels have been on that machine since then. Tanaka approved
this arrangement in writing on 2026-10-22, on Slack.

## Decision

- The project owner is the label holder. The labels are kept on their machine,
  under `$VC_WORK_ROOT/labels/`, outside this repository, and are never
  committed.
- The code keeps the split of ADR 1. Extraction, `aggregate` and `handoff` read
  no label. `vc model` and the exploratory analysis commands take the labels
  file as an explicit argument and write aggregate results only, under
  `$VC_OUT_ROOT/unblinded/`.
- The confirmatory result is the `vc model` run of 2026-10-01 from the frozen
  commit `4d8a5a1`, on the bundle built before any label was on this machine.
- Every analysis after that is exploratory: it goes on an `exploratory/*`
  branch, its methods are committed and pushed before any label is used, and a
  dated line in the analysis log records the commit it ran from (see
  `CLAUDE.md`).
- Claude Code sessions may run a label step at the label holder's explicit
  request, passing the labels' path to the command, but never read the labels
  or the results.

## Consequences

- ADR 1's guarantee that features cannot be tuned against the outcome now holds
  by construction only for the confirmatory features, which were frozen before
  2026-10-01. For later work it rests on process: methods pushed before the
  labels are used, and the dated log.
- Every result after 2026-10-01 must be reported as made after the labels were
  seen.
- The written approval is dated 2026-10-22. From 2026-10-01 until then, the
  arrangement rested on the request that came with the labels.
