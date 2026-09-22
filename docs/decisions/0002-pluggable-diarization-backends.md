# 2. Put diarization behind a backend interface, defaulting to import

- Status: accepted
- Date: 2026-09-23

## Context

The lab manuscript's text features were extracted from transcripts produced by
whisper-diarization. This project adds prosodic, turn-taking and facial features
to the same sessions and compares them against those text features.

Three sources of diarization are possible: importing the original output,
running pyannote locally, or running whisper-diarization locally. The upstream
whisper-diarization repository has heavy dependencies that do not install
cleanly on Apple silicon.

## Decision

One abstract interface producing a normalised segment table (`session_id`,
`speaker`, `start`, `end`, optional `text`), with three backends:

- `import` (default): parse existing whisper-diarization output from a folder.
- `pyannote`: diarize locally, behind the optional `pyannote` extra, with the
  model and revision pinned and recorded.
- `whisper_diarization`: invoke the upstream tool as an external subprocess in
  its own environment, then read its output with the importer. It is
  deliberately not a Python dependency of this project.

`import` is the default because reusing the manuscript's exact transcripts keeps
the comparison apples-to-apples. Re-diarizing would confound "new modality" with
"new transcripts", and a difference in results could not be attributed.

## Consequences

- The comparison against the manuscript is direct when the original output is
  available, which is the intended path.
- A local fallback exists if it is not, at the cost of confounding.
- Adding a backend means implementing one interface, not touching later stages.
- The heavy dependency stays outside this project's lockfile and CI.
