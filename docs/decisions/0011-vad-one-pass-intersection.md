# 11. Detect voice activity in one pass, then intersect

- Status: accepted
- Date: 2026-09-23
- Refines: [ADR 4](0004-vad-inside-diarized-segments.md)

## Context

ADR 4 settled that diarized segment boundaries cannot be trusted for timing,
because they follow transcription units and whisper-diarization segments span
long silences. It described running voice activity detection "inside each
diarized segment", which suggests slicing the audio per segment and detecting
within each slice.

Silero is a recurrent model. Its accuracy at a boundary depends on the audio
surrounding that boundary, which a slice by definition removes. An 11-minute
session holds a few hundred segments, so per-segment detection also means a few
hundred short model invocations instead of one.

## Decision

Two modes, with `vad.mode: intersect` as the default.

- **intersect**: detect once over the whole recording, then intersect the
  detected speech with each diarized segment. Diarization decides *who*,
  detection decides *when*, and the intersection carries both.
- **per_segment**: detect inside each segment's audio separately, shifting the
  results back into recording time. The literal reading, kept so the two can be
  compared on real data.

Intersection satisfies the actual requirement of ADR 4 — segment boundaries are
not used as speech boundaries — while giving the detector the context it needs.
Speech that straddles a speaker change is split at the diarization boundary and
attributed to both, which is the correct reading of overlapping talk.

Detected speech is also clipped to the audio that actually decoded, so nothing
can claim speech beyond the end of a truncated recording.

## Consequences

- One model invocation per session rather than hundreds: about 110x realtime on
  the development machine, so all 62 sessions take a few minutes.
- Boundary accuracy does not depend on where diarization happened to cut.
- The two modes can disagree, and which was used is recorded per session in
  `vad_qc.csv` alongside the fraction of segment time that survived as speech.
- The `retained_fraction` column is the measurement that justifies this whole
  stage: a median well below 1.0 confirms that segment boundaries really were
  spanning silence.
