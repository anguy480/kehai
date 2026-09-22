# 1. Split the pipeline so labels never reach the extraction machine

- Status: accepted
- Date: 2026-09-23

## Context

The questionnaire scores (K6, SRS-2) are held by the supervising professor and
will not be shared. Extraction work happens on a student machine that must never
hold them. A conventional single pipeline that reads features and labels
together cannot exist here.

There is also a scientific reason to want this separation even if the labels
were available: a researcher who can see the outcome while building features is
free to iterate towards a better-looking result. Keeping the label holder's step
separate makes that impossible by construction.

## Decision

Two halves with a file as the only interface.

- Extraction (`vc inventory` ... `vc handoff`) runs with no labels present. Its
  final product is a handoff bundle: a feature table, a feature dictionary, a
  per-session QC report, a manifest, and a short README.
- Analysis (`vc model`) runs on the label holder's machine. It takes the feature
  table plus a private labels CSV and emits aggregate cross-validated metrics
  only.

The bundle is refused from a dirty git tree unless `--allow-dirty` is passed, and
the manifest records the commit, resolved config, package versions and model
versions, so any result can be traced to the code that produced it.

## Consequences

- Feature extraction cannot be tuned against the outcome, even accidentally.
- The bundle must be self-explaining: the professor should not need to read code
  to run the analysis or interpret the QC flags.
- Results come back as metrics, never per-session labels, so nothing sent back
  can reconstruct the questionnaire scores.
- Iteration is slower: a feature bug found during analysis means a new bundle.
  That is an acceptable price for the guarantee.
