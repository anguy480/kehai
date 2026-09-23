# 10. Measure left/right channel separation while extracting audio

- Status: accepted
- Date: 2026-09-23

## Context

`vc inventory` over all 62 recordings found a single mixed AAC stream, **stereo**,
48 kHz, in every file. Later stages want mono 16 kHz, so the obvious thing is to
downmix and move on.

That would discard a question worth answering. Zoom sometimes pans participants
across the stereo field. If the two channels genuinely differ, that partial
separation is information about who is speaking that owes nothing to
diarization — which matters here, because there is exactly one mixed stream and
therefore no per-speaker audio, making every downstream speaker attribution
dependent on diarization being right.

The measurement is also cheap: the audio has to be decoded anyway.

## Decision

`vc extract-audio` decodes each recording once and, in the same pass, compares
the channels. It writes `audio_qc.csv` with the Pearson correlation between
channels, the interaural level difference, per-channel RMS and peaks, the active
fraction, and QC flags.

The correlation is computed **over active frames only**. A 12-minute session is
mostly silence, and silence carries no panning information, so including it
would dilute the measurement toward whatever the noise floor happens to do.

The statistics are **accumulated over chunks** rather than computed on a fully
decoded recording. Decoding 12 minutes of stereo to float64 is roughly 180 MB,
and sessions run in parallel. Accumulating sums keeps memory flat regardless of
length, and the correlation it produces is exact, not an approximation — tests
assert it matches `numpy.corrcoef` to within 1e-9 at chunk sizes from 1 frame to
100,000.

Flags distinguish the cases that matter: `stereo_correlated` (the channels carry
the same signal, so there is nothing to exploit), `stereo_partial_separation`,
`stereo_strong_separation`, `stereo_identical` (bit-for-bit equal),
`stereo_channel_imbalance` and `audio_clipping`. Thresholds are configurable
under `audio.stereo_probe`.

`stereo_identical` is reported but is not the primary signal: lossy AAC normally
prevents bit-equality even for a genuinely mono source, so the correlation is
what to trust.

## Consequences

- If the channels turn out to be duplicates, the summary says so plainly, and
  nothing downstream is built on separation that does not exist. That is a
  useful negative result rather than a wasted stage.
- If some sessions do show separation, they are named, and a channel-based
  speaker cue becomes available as a third line of evidence alongside the voice
  embedding and the mouth-movement cross-check (ADR 8).
- The mono output is unaffected either way: the downmix is the mean of the two
  channels, matching ffmpeg's `-ac 1`.
- Per-session statistics are written as a sidecar beside each WAV, so a rerun
  that skips completed sessions can still assemble the complete QC table without
  re-decoding anything.
