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

## Amendment, 2026-09-26: the method, validated

The backend is ECAPA-TDNN via SpeechBrain, pinned to revision
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, in the optional `speaker` extra. It
needs no access token and runs on CPU. Before adopting it, it was validated on
this cohort with controls in both directions:

| comparison | what it is | observed |
| --- | --- | --- |
| two halves of one reference clip | the same voice, twice | 0.72, 0.79 |
| psychiatrist vs participant, same session | two voices, same recording | 0.18 - 0.68 |
| psychiatrist vs psychiatrist reference | what the method is for | 0.70 - 0.88 |

Two things follow, and both are built into the rule.

**Only the ranking can be trusted.** A participant scores 0.18-0.27 against the
psychiatrist reference in the summer wave but 0.46-0.60 in the winter wave.
That is a channel effect, not a fact about the participants, and any absolute
threshold would therefore mean something different in each wave. The rule takes
the highest-scoring speaker and reports the margin; a small margin is a flag,
not a refusal, because weak evidence is not wrong evidence.

**The ranking is nevertheless robust.** Across the sessions checked by hand,
both reference clips ranked the same speaker first in every case, including the
winter sessions where absolute scores are compressed. Every session is
therefore scored against *every* clip, not only its mapped one: once the
session's own speakers are embedded, comparing against another clip is a dot
product, and a clip that disagrees is information whether or not it was
expected to apply.

### What OCR agreement actually means

Label OCR determines which *side of the frame* the psychiatrist occupies.
Diarization produces *anonymous voices*. These live in different spaces, and
nothing compares them until something says which voice belongs to which tile.
The mouth-movement cross-check is that bridge and the only one: stereo panning
would have been another, but all 62 recordings have bit-identical channels.

So embedding-vs-OCR agreement is computable exactly where the mouth evidence
exists, and is reported as `unavailable` elsewhere. `unavailable` is not a soft
`agree`: it means the assignment stands uncorroborated by the labels.

### Three speakers

Two sessions diarize into three speakers. The psychiatrist is the best acoustic
match; the participant is the *remaining speaker who talked most*, chosen by
speech rather than by similarity so that a sliver of a third voice cannot be
handed the participant's role. Any further speaker is assigned `unknown`, which
excludes their audio from every feature, and the session is flagged.
