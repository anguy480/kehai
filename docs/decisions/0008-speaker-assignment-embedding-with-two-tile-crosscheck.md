# 8. Assign roles by voice embedding, cross-checked against both video tiles

- Status: accepted
- Date: 2026-09-23

## Context

Diarization returns anonymous labels (`SPEAKER_00`, `SPEAKER_01`). Every later
feature depends on knowing which is the participant: prosody is participant-only,
and the speaking/listening split for facial features inverts if the roles are
swapped. A silent role swap would not crash anything; it would just produce
confidently wrong features.

Two sources of evidence are available. Acoustically, the psychiatrist's voice can
be matched against a reference clip. Visually, a speaker's mouth moves while they
talk, and in Zoom gallery view *both* faces are visible.

It is not confirmed that the same psychiatrist ran every session, particularly
across the two recruitment waves.

## Decision

- Primary method: speaker-embedding similarity against a reference clip of the
  psychiatrist. Several reference clips are supported, one per psychiatrist, with
  an optional `session_id,psychiatrist_id` map restricting which clip applies to
  which session. Reference clips live in `$VC_WORK_ROOT/reference/`.
- Independent cross-check: correlate each diarized speaker's speech timeline
  against jaw-open movement in *every* configured video tile, not only the
  participant's. A correct assignment shows each speaker correlating with exactly
  one tile, and the two speakers with different tiles. Correlating against a
  single tile could only ever weakly confirm one role; correlating against both
  can positively identify each.
- The embedding decides. The similarity margin, the per-tile correlations and
  whether the two methods agree are recorded as QC columns. Disagreement, or a
  margin below `speakers.min_margin`, raises a flag for manual review rather
  than silently choosing.

## Consequences

- Role assignment has two independent lines of evidence, and the cases where
  they conflict are visible in the QC report rather than buried.
- The cross-check depends on the video layout being per-tile, so it is only
  meaningful once `vc preview` has confirmed gallery view. In active-speaker
  view it must be disabled, and the embedding stands alone.
- A reference clip must be produced by hand before this stage can run. It is
  derived from a recording, so it stays in the work tree and never enters the
  repository or a handoff bundle.
