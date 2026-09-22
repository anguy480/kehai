# 4. Run voice activity detection inside diarized segments

- Status: accepted
- Date: 2026-09-23

## Context

Turn-taking features depend on accurate speech boundaries: response latency is
the participant's speech onset minus the psychiatrist's preceding offset, and an
error of a few hundred milliseconds matters.

whisper-diarization segments are known to span long silent stretches, because
they follow transcription units rather than acoustic activity. Taking segment
boundaries as speech boundaries would systematically understate pauses and
distort every latency measure.

## Decision

Diarization decides *who* speaks and *roughly when*. Silero VAD then runs
*within* each diarized segment to find where speech actually starts and stops,
and all timing features are computed from the VAD boundaries.

## Consequences

- Latency, pause and speaking-ratio features reflect acoustic speech rather than
  transcription units.
- The result depends on VAD thresholds, which are configurable and snapshotted
  into every run.
- VAD is constrained to segments rather than run over the whole recording, so it
  inherits diarization's speaker attribution instead of having to re-derive it.
- A segment containing no detected speech yields no speech spans, which is
  recorded rather than treated as zero-length speech.
