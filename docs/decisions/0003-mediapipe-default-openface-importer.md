# 3. MediaPipe by default, with an OpenFace CSV importer

- Status: accepted
- Date: 2026-09-23

## Context

Facial features are needed from 62 recordings of roughly 11 minutes. The lab
uses OpenFace 2.0, and a final run may happen on a lab machine. Development
happens on an Apple silicon MacBook Air, where OpenFace is awkward to build,
while MediaPipe runs natively.

## Decision

A pluggable face backend with two implementations:

- `mediapipe` (default): Face Landmarker, giving blendshapes and head pose. The
  model asset is downloaded once, pinned, and recorded in the manifest.
- `openface`: read OpenFace 2.0 CSV output produced elsewhere.

Aggregation consumes a common per-frame representation, so the choice of backend
does not change the shape of the feature table.

## Version pin

MediaPipe is pinned below 1.0. Version 1.0.1 aborts the process on macOS arm64
inside the face detector subgraph (`DrishtiMetalHelper ... Check failed:
service_ Service is unavailable`), with or without an explicit CPU delegate. It
is a hard abort, not an exception, so it cannot be caught and handled. 0.10.x
runs and returns all 52 blendshapes. The model asset's SHA-256 is pinned in
config and was verified to contain every blendshape the configured action units
need.

That MediaPipe is the fragile half of this pair, while OpenFace is the lab's
own house pipeline, strengthens the case for the importer being a first-class
path rather than a fallback. See
[ADR 13](0013-face-features-follow-the-lab-precedent.md).

## Consequences

- Development is unblocked on the development machine.
- A lab-machine OpenFace run can be swapped in without touching aggregation,
  though the two backends' measures are not numerically interchangeable: a
  comparison across backends is a change of instrument, and any mixed run must
  be recorded as such in the manifest.
- Blendshapes are a deliberately small, named subset, which protects the feature
  budget (see ADR 6).
- No frames or annotated video are written by either backend.
