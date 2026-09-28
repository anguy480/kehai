"""Prose fragments the handoff README is assembled from.

Kept here rather than inline in the handoff stage so that a note earned by a
design decision cannot be lost when that stage is rewritten, and so each one
can be tested for the facts it has to state.

The handoff README is read by the person running the analysis, who will not
read the code. Anything they need in order to avoid a mistake belongs in it.
"""

from __future__ import annotations

from typing import Final

#: Why a change of facial backend invalidates a partial rerun. Required
#: reading before anyone re-extracts a subset of sessions.
FACE_BACKEND_NOTE: Final = """\
### If the facial measurements are re-extracted

Facial features come from one of two backends, and the one used is recorded in
`qc__face_backend` on every row of `features.csv`, and in `manifest.json`.

**MediaPipe blendshape scores are not OpenFace action unit intensities.** They
measure the same things - the same action units, named the same way - on
different scales. A value from one cannot be compared with a value from the
other, and a feature table holding both describes nothing.

So changing backend means **re-extracting every session, not topping up the
ones that are missing**. The pipeline refuses to build a bundle whose sessions
do not all share one backend, and says so rather than averaging them. If you
receive a bundle where `qc__face_backend` is not identical across all rows,
something has gone wrong: ask for it to be rebuilt rather than dropping the odd
rows.
"""

#: What the gaze channel of the lab's prior work means for this bundle.
GAZE_ABSENCE_NOTE: Final = """\
### There are no gaze features

The lab's prior work on this construct found gaze measures predictive - how
much of the time a participant looked at their conversation partner. Those
measures came from a Tobii eye tracker.

These are Zoom recordings with no eye tracker. Head pose is available and is
reported as head pose; it is **not** gaze, and must not be read as a proxy for
it: a participant can look at their partner's tile without moving their head,
and the tile is not where the camera is.

The gaze channel is therefore **absent from this bundle, not negative**. A
result showing that the facial features here underperform the prior work's
combined model is not evidence that gaze does not matter.
"""


#: Why facial measurement quality is not uniform across this cohort, and why
#: that is not comparable with the lab study the feature choices came from.
REMOTE_RECORDING_NOTE: Final = """\
### Face tracking quality varies with the participant's own setup

These are recordings of people in their own homes, joining on their own
equipment. Lighting, focus, camera distance and framing are whatever each
participant had, and they are not controllable after the fact. Facial
measurement quality therefore varies between sessions in a way that has nothing
to do with the participants themselves.

`qc__face_dropped_fraction` in the QC report is where this is visible: the share
of sampled frames in which no usable face was found. Most sessions are a
fraction of a percent. A few are much higher, and where someone has watched the
recording and confirmed the cause, it is recorded in `qc_notes.csv` with the
reason.

**This is a known cost of remote recording, and it is one of the ways this study
is not comparable with the lab work its feature choices came from.** Miyamoto et
al. 2025 (Acta Psychologica 254:104782) recorded a collaborative task under lab
conditions - controlled lighting, a fixed camera at a known distance, and an eye
tracker - and reported facial and gaze measures from that. The action units here
are the same measures by name, taken in conditions that study did not have to
contend with. Where a facial result here is weaker than the equivalent there,
the recording conditions are a live explanation and cannot be ruled out.

What follows from that for the analysis:

* Treat `qc__face_dropped_fraction` as information about the recording, not
  about the participant. It is not a behavioural measure and must not be
  modelled as one.
* A facial feature computed from few usable frames is noisy however clean the
  number looks. `qc__face_frames_speaking` and `qc__face_frames_listening` are
  how you see that per window.
* If you exclude sessions on tracking quality, decide the threshold once, from
  the QC report, and state it. Choosing it after seeing how it moves a result is
  a different analysis.
"""


def notes() -> tuple[str, ...]:
    """Every note the handoff README must carry, in order."""
    return (FACE_BACKEND_NOTE, GAZE_ABSENCE_NOTE, REMOTE_RECORDING_NOTE)
