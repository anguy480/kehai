"""Facial measurement arithmetic: blendshape combination, head pose, coverage.

Pure functions, no video and no model, so the parts that can quietly be wrong
are checkable: combining a left and a right blendshape into one action unit,
turning a 4x4 transform into head angles, and computing how much of a session
was actually measured.

A note that governs the whole module. MediaPipe blendshape scores and OpenFace
action unit intensities measure the same constructs on different scales, and a
value from one is not comparable with a value from the other
(docs/decisions/0013). Nothing here converts between them. The backend is
recorded alongside every measurement so a table that mixes them can be refused
rather than silently pooled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

# A face either was or was not found. MediaPipe applies its detection threshold
# internally and returns nothing below it, so presence is all it reports;
# OpenFace gives a graded confidence in its own column.
PRESENT: Final = 1.0
ABSENT: Final = 0.0

_ROTATION_SIZE: Final = 3
_TRANSFORM_SIZE: Final = 4
# Beyond this, the pitch is close enough to vertical that yaw and roll are not
# separable and the decomposition is degenerate.
_GIMBAL_LIMIT: Final = 1.0 - 1e-6


def combine_blendshapes(scores: Mapping[str, float], names: Sequence[str]) -> float | None:
    """Combine the blendshapes standing in for one action unit.

    Averaged rather than summed, and averaged over the names that were actually
    present: a left/right pair averages to something on the same scale as a
    single-sided shape, so action units stay comparable with each other.

    Returns:
        The combined value, or None when none of the names was reported.
    """
    values = [float(scores[name]) for name in names if name in scores]
    if not values:
        return None
    return sum(values) / len(values)


def head_pose_from_matrix(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[float, float, float] | None:
    """Extract head pitch, yaw and roll in degrees from a 4x4 transform.

    MediaPipe returns the head's orientation as a 4x4 row-major rigid
    transform: an orthonormal rotation in the top-left 3x3, the translation in
    the last column. The rotation is decomposed as `R = Rz @ Ry @ Rx`, and each
    angle is named for the axis it turns about in the camera's frame, where X
    points right, Y points up and Z lies along the optical axis:

    * **pitch** is rotation about X, which is nodding.
    * **yaw** is rotation about Y, which is turning to look left or right.
    * **roll** is rotation about Z, which is tilting towards a shoulder.

    The mapping from expression to axis is the part that is easy to get wrong,
    so it is verified rather than asserted: rotating an image in its own plane
    is a rotation about the optical axis and must therefore appear as roll, and
    a test does exactly that against the real model. An earlier version of this
    function returned the three angles in the order `(yaw, roll, pitch)` while
    labelling them `(pitch, yaw, roll)`, which no amount of internal
    round-tripping would have caught.

    The same naming is used for the OpenFace importer, whose `pose_Rx`,
    `pose_Ry` and `pose_Rz` are already per-axis, so the two backends agree on
    what each column means even though their values are on different scales.

    These angles describe head orientation relative to the camera and **not**
    where the person is looking: gaze needs an eye tracker, which these
    recordings do not have (docs/decisions/0013).

    Returns:
        `(pitch, yaw, roll)` in degrees, or None if the matrix is unusable or
        the decomposition is degenerate.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape not in {(_TRANSFORM_SIZE, _TRANSFORM_SIZE), (_ROTATION_SIZE, _ROTATION_SIZE)}:
        return None
    rotation = array[:_ROTATION_SIZE, :_ROTATION_SIZE]
    if not np.all(np.isfinite(rotation)):
        return None

    # For R = Rz(roll) @ Ry(yaw) @ Rx(pitch): R[2,0] = -sin(yaw), so yaw is
    # the angle that goes degenerate at +/-90 degrees, where the other two are
    # no longer separable.
    sin_yaw = -float(rotation[2, 0])
    if abs(sin_yaw) >= _GIMBAL_LIMIT:
        return None
    yaw = math.asin(sin_yaw)
    pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


@dataclass(frozen=True, slots=True)
class FrameMeasure:
    """What was measured in one sampled frame.

    Attributes:
        frame_index: Index in the original recording, not in the sample.
        timestamp_s: Time in the recording.
        detected: Whether a face was found and kept.
        confidence: Graded where the backend reports one, otherwise presence.
        units: Action unit key to value, on the backend's own scale.
        jaw: Jaw opening, for the speaker cross-check rather than as a feature.
        blink: Eye closure, as a tracking-quality signal.
        head: `(pitch, yaw, roll)` in degrees, where available.
    """

    frame_index: int
    timestamp_s: float
    detected: bool
    confidence: float
    units: Mapping[str, float | None] = field(default_factory=dict)
    jaw: float | None = None
    blink: float | None = None
    head: tuple[float, float, float] | None = None

    @classmethod
    def missing(cls, frame_index: int, timestamp_s: float) -> FrameMeasure:
        """A frame in which no usable face was found."""
        return cls(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            detected=False,
            confidence=ABSENT,
        )


def dropped_fraction(measures: Sequence[FrameMeasure]) -> float | None:
    """Share of sampled frames with no usable face.

    Returns None for no frames at all, which is a different problem from every
    frame being unusable and should not read as 0% dropped.
    """
    if not measures:
        return None
    dropped = sum(1 for measure in measures if not measure.detected)
    return dropped / len(measures)


def kept(measures: Sequence[FrameMeasure]) -> tuple[FrameMeasure, ...]:
    """The frames with a usable face."""
    return tuple(measure for measure in measures if measure.detected)


def smoothing_window(fps: float, seconds: float) -> int:
    """A centred window of `seconds` at `fps`, in frames.

    Rounded to the nearest whole frame and raised by one if even, so that it
    has a middle frame: at 25 fps, 1.0 s is 25 frames.

    Raises:
        ValueError: if either value is not positive.
    """
    if fps <= 0.0 or seconds <= 0.0:
        msg = f"a smoothing window needs a positive fps and duration, got {fps} and {seconds}"
        raise ValueError(msg)
    frames = max(1, round(fps * seconds))
    return frames if frames % 2 == 1 else frames + 1


def smooth_within_runs(
    frame_index: Sequence[int] | np.ndarray,
    valid: Sequence[bool] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    window: int,
) -> np.ndarray:
    """Centred rolling mean, separately within each run of valid frames.

    A run is a stretch of consecutive frame indices that are all valid and
    finite. The mean never reaches across an invalid frame or a gap in the
    frame numbering, so a dropped face cannot leak a neighbour's expression
    into the frames beside it. At the edges of a run the mean is taken over the
    frames of the run inside the window, so no frame is lost for sitting next
    to a gap.

    Args:
        frame_index: Frame numbers, in ascending order.
        valid: Which frames had a usable face.
        values: One measure per frame.
        window: Window length in frames, odd.

    Returns:
        The smoothed values, NaN wherever the frame is not valid.

    Raises:
        ValueError: if the inputs disagree in length, the frames are not in
            ascending order, or the window is not a positive odd number.
    """
    if window < 1 or window % 2 == 0:
        msg = f"the window must be a positive odd number of frames, got {window}"
        raise ValueError(msg)
    index = np.asarray(frame_index, dtype=np.int64)
    data = np.asarray(values, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool) & np.isfinite(data)
    if not (index.shape == data.shape == usable.shape):
        msg = "frame_index, valid and values must have the same length"
        raise ValueError(msg)
    if index.size > 1 and np.any(np.diff(index) <= 0):
        msg = "frame_index must be strictly ascending"
        raise ValueError(msg)

    smoothed = np.full(data.shape, np.nan, dtype=np.float64)
    # A run starts wherever a usable frame does not directly follow another.
    follows = np.zeros(index.size, dtype=bool)
    follows[1:] = usable[1:] & usable[:-1] & (np.diff(index) == 1)
    starts = np.flatnonzero(usable & ~follows)
    if starts.size == 0:
        return smoothed
    ends = np.append(starts[1:], index.size)
    half = window // 2
    for start, limit in zip(starts, ends, strict=True):
        # The run continues while frames follow on; it cannot pass the next start.
        stop = start + 1
        while stop < limit and follows[stop]:
            stop += 1
        run = data[start:stop]
        totals = np.concatenate(([0.0], np.cumsum(run)))
        position = np.arange(run.size)
        low = np.maximum(position - half, 0)
        high = np.minimum(position + half + 1, run.size)
        smoothed[start:stop] = (totals[high] - totals[low]) / (high - low)
    return smoothed


def apply_confidence_threshold(
    measures: Sequence[FrameMeasure], threshold: float
) -> tuple[FrameMeasure, ...]:
    """Mark frames below `threshold` as undetected.

    Applied after measurement rather than during it, so that the dropped
    fraction counts every frame that was looked at. A backend that filters
    internally reports presence, and presence always clears a threshold in
    [0, 1], so this is a no-op for it rather than a second filter.
    """
    return tuple(
        measure
        if measure.detected and measure.confidence >= threshold
        else FrameMeasure.missing(measure.frame_index, measure.timestamp_s)
        for measure in measures
    )
