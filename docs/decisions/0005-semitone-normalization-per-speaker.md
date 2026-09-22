# 5. Express F0 in semitones relative to each speaker's own median

- Status: accepted
- Date: 2026-09-23

## Context

Absolute fundamental frequency differs substantially between speakers,
predominantly by sex: typical adult male and female medians differ by roughly an
octave. With 62 participants, raw Hz features would be dominated by that
difference. A model could then appear to predict a questionnaire score while
actually keying on speaker sex, which is both a confound and a fairness problem.

Hertz is also perceptually non-linear: a 20 Hz change is large for a low voice
and small for a high one.

## Decision

Convert F0 to semitones relative to the speaker's own median within the session:

    semitones = 12 * log2(f0 / median_f0)

Variability and range features are computed in that space. The median is taken
per speaker per session, so each participant is their own reference.

## Consequences

- Pitch *variability* and *range* become comparable across speakers, which is
  what the research question is about.
- Absolute pitch level is deliberately discarded. If it is ever wanted, it must
  be added as an explicit, separately named feature.
- The transformation is undefined for non-positive or unvoiced F0 values, which
  are excluded before conversion rather than clamped.
- A session with too little voiced speech to estimate a stable median is flagged
  in QC rather than given an unreliable reference.
