# 13. Follow the lab's own precedent for the facial features

- Status: accepted
- Date: 2026-09-23
- Related: [ADR 3](0003-mediapipe-default-openface-importer.md) (which backend),
  [ADR 12](0012-tiered-analysis-not-a-feature-cap.md) (tiered analysis)

## Context

The facial features could be chosen from the several hundred measures a
landmarker produces. They should not be: the lab has a published multimodal
study of the same construct, with the same house pipeline, and choosing
anything else would throw away the only directly relevant prior.

**Miyamoto, K., Tanaka, H., Hamet Bagnou, J., Clavel, C., Prigent, E.,
Benamara, A., Le Scanff, C., Martin, J.-C., & Nakamura, S. (2025). Impact of
eye movements and facial expressions on social performance during a
collaborative problem-solving task. *Acta Psychologica*, 254, 104782.**
doi:10.1016/j.actpsy.2025.104782, PMID 39923549.

Two findings, quoted from the abstract:

> "The lower the intensity of expression is in inner brow raiser, the cheek
> raiser, and the lip corner puller, the lower is the social performance."

> "The lower the percentage of looking at the conversation partner is and the
> higher the percentage of looking outside is, the lower is the social
> performance."

The first names **AU01 (inner brow raiser), AU06 (cheek raiser) and AU12 (lip
corner puller)**, with expressivity in those three positively associated with
social performance.

A companion study from the same group corroborates the same AU set and gives
the clinical direction:

**Tanaka, H., Miyamoto, K., Hamet Bagnou, J., Prigent, E., Clavel, C., Martin,
J.-C., & Nakamura, S. (2025). Analysis of Social Performance and Action Units
During Social Skills Training. *JMIR Formative Research*, 9, e59261.**
PMID 39801481.

That study extracted AU01, AU02, AU04, AU06 and AU12 with OpenFace, and found
AU06 and AU12 significantly *deactivated* in schizophrenia relative to
controls, and AU02 significantly *activated* in autism spectrum disorder
relative to the other groups.

Both findings are close to this project's targets: social performance is the
construct SRS-2 indexes, and reduced AU06/AU12 in a clinical group is the kind
of blunted expressivity K6 is meant to capture.

## Decision

### Extract the lab's AU set, not a set of our own

The five AUs of the house pipeline: **AU01, AU02, AU04, AU06, AU12**. Plus jaw
opening, which is not an expression feature but is needed by the
mouth-movement cross-check in [ADR 8](0008-speaker-assignment-embedding-with-two-tile-crosscheck.md),
and eye blink as a tracking-quality signal.

MediaPipe blendshapes are mapped onto those AUs, and the configured blendshape
list is exactly what the mapping needs. Before this decision the list omitted
the outer brow raiser and the cheek raiser, which would have left AU02 and
AU06 unmeasurable: AU06 is one of the three the precedent found predictive.

### Name features by AU, and record the backend

Features are named `face_speaking__au12_mean` rather than by blendshape, so
that they line up with the literature and with the lab's own OpenFace output.

**MediaPipe blendshape scores are not AU intensities.** They are a different
parameterisation on a different scale, and a value from one backend is not
comparable with a value from the other. Naming them by AU is a claim about
*what is being measured*, not about the units. So `qc__face_backend` is
recorded per session, the manifest records the backend and its version, and a
table mixing the two backends is detectable rather than silently pooled. If
the final run happens on a lab machine with OpenFace, it must be a full rerun,
not a top-up.

### The confirmatory features are the three the precedent found

Fixing the slots ADR 12 left open, and still before any questionnaire score
has been seen:

| Family | Feature | Precedent |
| --- | --- | --- |
| face_speaking | `face_speaking__au12_mean` | lip corner puller, Miyamoto 2025 |
| face_speaking | `face_speaking__au06_mean` | cheek raiser, Miyamoto 2025 |
| face_speaking | `face_speaking__au01_mean` | inner brow raiser, Miyamoto 2025 |
| face_listening | `face_listening__au12_mean` | as above, in the listening window |
| face_listening | `face_listening__au06_mean` | as above |
| face_listening | `face_listening__au01_mean` | as above |

The same three AUs in both windows. The AUs come from the precedent; the
speaking/listening split is this project's contribution, and putting the same
three in both windows is what makes the contrast interpretable rather than
confounded with a change of measure. AU02 and AU04 are extracted and sit in
the exploratory tier: AU02 has a published association with autism spectrum
disorder and so is a strong exploratory candidate for SRS-2, but it was not
among the three predictive of social performance, and promoting it on the
strength of a different study's different outcome is exactly the move the
tiering exists to prevent.

## Consequences

* **Their stronger finding cannot be replicated here.** The gaze result rests
  on Tobii eye tracking. These are Zoom recordings with no eye tracker, and
  head pose is not gaze: a participant can look at the partner's tile without
  moving their head, and the tile is not where the camera is. Head pose is
  extracted and reported as head pose, never as looking-at-partner, and the
  write-up has to state plainly that the gaze channel of the precedent is
  absent rather than negative.
* **The speaking/listening split is not theirs.** That study used separate
  speaking-time annotations; here the split is derived from diarization plus
  voice activity detection, so it carries the error of both. `vc turns` records
  the timeline it produced and the QC flags behind it, and a session whose role
  assignment was uncertain is visible in the feature table.
* The comparison against the lab's prior work is now direct for the facial
  channel and absent for the gaze channel, which is a cleaner position than a
  broad feature set that matches nothing.
* Adding an AU later is possible, in the exploratory tier. Moving one into the
  confirmatory tier after a label has been seen is not, and the manifest's
  commit makes the ordering checkable.
