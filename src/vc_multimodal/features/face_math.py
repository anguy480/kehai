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

    MediaPipe returns the head's transformation as a 4x4 row-major matrix. The
    rotation is decomposed in ZYX order, which gives the intuitive reading:
    pitch is nodding, yaw is turning, roll is tilting. Angles are in the
    camera's frame, so they describe head orientation relative to the camera
    and **not** where the person is looking: gaze needs an eye tracker, which
    these recordings do not have (docs/decisions/0013).

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

    # ZYX decomposition: -R[2,0] is sin(pitch).
    sin_pitch = -float(rotation[2, 0])
    if abs(sin_pitch) >= _GIMBAL_LIMIT:
        return None
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
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
