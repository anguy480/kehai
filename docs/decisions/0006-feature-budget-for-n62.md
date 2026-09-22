# 6. Cap the feature count, with families and a naming convention

- Status: accepted
- Date: 2026-09-23

## Context

The dataset has 62 sessions. Prosodic, turn-taking and facial extraction can
produce hundreds of candidate features almost for free: every blendshape times
every statistic times speaking and listening periods. With N=62 and
leave-one-participant-out cross-validation, a large feature set overfits, and
comparisons between feature sets stop meaning anything.

The manuscript this work is compared against reported a negative cross-validated
R² for SRS-2 from text features. Adding a wide feature matrix would make a
similar result more likely, not less.

## Decision

- A hard ceiling on the number of features, configured as
  `aggregate.max_features` (60) and enforced as an error rather than a warning.
- Roughly 50 features, allocated as: turns ~12, prosody ~14, face while the
  participant speaks ~12, face while listening ~12.
- Every feature is named `family__name`, with the family drawn from a fixed list
  (`turns`, `prosody`, `face_speaking`, `face_listening`, `text`), enforced by
  the feature schema. Feature sets for model comparison are then selected by
  family, with no hand-maintained column lists to drift out of date.
- Scaling and any feature selection happen inside each training fold only, which
  is asserted by a dedicated leakage test.

## Consequences

- Around 1.2 samples per feature, so regularisation carries real weight; elastic
  net remains the default, matching the manuscript.
- Adding a feature is a deliberate act that may require removing another.
- The blendshape subset is chosen in config rather than "all of them".
- Family-based selection makes the audio/face/text comparison mechanical.
