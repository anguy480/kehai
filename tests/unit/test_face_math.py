"""Facial measurement arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vc_multimodal.features.face_math import (
    ABSENT,
    PRESENT,
    FrameMeasure,
    apply_confidence_threshold,
    combine_blendshapes,
    dropped_fraction,
    head_pose_from_matrix,
    kept,
)


def rotation(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """A 4x4 transform for the given head angles, in degrees.

    Built from the axes themselves rather than from the extraction's own
    formulae, so this cannot agree with a mislabelled decomposition. In the
    camera frame X points right, Y up and Z along the optical axis, so pitch
    turns about X, yaw about Y and roll about Z.
    """
    a, b, c = (math.radians(angle) for angle in (pitch, yaw, roll))
    about_x = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    about_y = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    about_z = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = about_z @ about_y @ about_x
    return matrix


# ---------------------------------------------------------------------------
# combining blendshapes into an action unit
# ---------------------------------------------------------------------------
def test_a_left_right_pair_averages():
    """Averaged, not summed, so a paired AU stays on the same scale as a single."""
    value = combine_blendshapes(
        {"mouthSmileLeft": 0.4, "mouthSmileRight": 0.6},
        ["mouthSmileLeft", "mouthSmileRight"],
    )
    assert value == pytest.approx(0.5)


def test_a_single_sided_unit_passes_through():
    assert combine_blendshapes({"browInnerUp": 0.3}, ["browInnerUp"]) == pytest.approx(0.3)


def test_one_missing_side_averages_over_what_is_there():
    """Rather than treating the absent side as zero, which would halve it."""
    value = combine_blendshapes({"mouthSmileLeft": 0.4}, ["mouthSmileLeft", "mouthSmileRight"])
    assert value == pytest.approx(0.4)


def test_an_entirely_absent_unit_is_none_not_zero():
    assert combine_blendshapes({}, ["mouthSmileLeft"]) is None


def test_extra_scores_are_ignored():
    value = combine_blendshapes({"browInnerUp": 0.3, "jawOpen": 0.9}, ["browInnerUp"])
    assert value == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# head pose
# ---------------------------------------------------------------------------
def test_an_identity_transform_is_a_level_head():
    pitch, yaw, roll = head_pose_from_matrix(np.eye(4))
    assert (pitch, yaw, roll) == pytest.approx((0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        ({"pitch": 20.0}, (20.0, 0.0, 0.0)),
        ({"yaw": 30.0}, (0.0, 30.0, 0.0)),
        ({"roll": -15.0}, (0.0, 0.0, -15.0)),
        ({"pitch": 10.0, "yaw": 20.0, "roll": 5.0}, (10.0, 20.0, 5.0)),
    ],
)
def test_each_axis_lands_in_its_own_channel(
    angles: dict[str, float], expected: tuple[float, float, float]
):
    """A rotation about one axis must move one angle and leave the others.

    An earlier version returned the three in the order (yaw, roll, pitch) while
    labelling them (pitch, yaw, roll). Every axis was individually recoverable,
    so a test that round-tripped through the same convention passed anyway.
    """
    result = head_pose_from_matrix(rotation(**angles))
    assert result == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("axis", "index"),
    [("pitch", 0), ("yaw", 1), ("roll", 2)],
)
def test_no_other_channel_moves(axis: str, index: int):
    """Stated separately from the values, since this is the property that broke."""
    result = head_pose_from_matrix(rotation(**{axis: 25.0}))
    assert result is not None
    assert result[index] == pytest.approx(25.0, abs=1e-6)
    others = [value for position, value in enumerate(result) if position != index]
    assert others == pytest.approx([0.0, 0.0], abs=1e-6)


def test_a_three_by_three_rotation_is_accepted():
    assert head_pose_from_matrix(rotation(yaw=45.0)[:3, :3]) == pytest.approx(
        (0.0, 45.0, 0.0), abs=1e-6
    )


def test_a_degenerate_decomposition_returns_nothing():
    """At 90 degrees of yaw, pitch and roll are not separable, so none is claimed."""
    assert head_pose_from_matrix(rotation(yaw=90.0)) is None
    assert head_pose_from_matrix(rotation(yaw=-90.0)) is None


def test_the_rotation_matrix_is_a_proper_rotation():
    """Guards the assumption that the top-left 3x3 is orthonormal, not scaled."""
    matrix = rotation(pitch=10.0, yaw=20.0, roll=30.0)[:3, :3]
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_a_malformed_matrix_returns_nothing():
    assert head_pose_from_matrix(np.zeros((2, 2))) is None
    assert head_pose_from_matrix(np.full((4, 4), np.nan)) is None


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
def measures(detected: list[bool], confidence: float = PRESENT) -> list[FrameMeasure]:
    return [
        FrameMeasure(
            frame_index=index * 5,
            timestamp_s=index * 0.2,
            detected=found,
            confidence=confidence if found else ABSENT,
            units={"au12": 0.5} if found else {},
        )
        for index, found in enumerate(detected)
    ]


def test_the_dropped_fraction_counts_every_frame_looked_at():
    assert dropped_fraction(measures([True, True, False, False])) == pytest.approx(0.5)


def test_no_frames_is_not_zero_percent_dropped():
    """A different problem from every frame being unusable."""
    assert dropped_fraction([]) is None


def test_all_frames_dropped_is_one():
    assert dropped_fraction(measures([False, False])) == pytest.approx(1.0)


def test_kept_returns_only_measured_frames():
    assert len(kept(measures([True, False, True]))) == 2


# ---------------------------------------------------------------------------
# the confidence threshold
# ---------------------------------------------------------------------------
def test_a_frame_below_the_threshold_becomes_undetected():
    low = measures([True], confidence=0.2)
    result = apply_confidence_threshold(low, 0.5)
    assert not result[0].detected
    assert result[0].confidence == ABSENT
    # Still counted as a frame that was looked at.
    assert dropped_fraction(result) == pytest.approx(1.0)


def test_a_frame_at_the_threshold_is_kept():
    assert apply_confidence_threshold(measures([True], confidence=0.5), 0.5)[0].detected


def test_presence_always_clears_a_threshold_in_range():
    """MediaPipe filters internally and reports presence, so this is a no-op."""
    result = apply_confidence_threshold(measures([True], confidence=PRESENT), 0.99)
    assert result[0].detected


def test_the_frame_grid_survives_thresholding():
    original = measures([True, True], confidence=0.1)
    result = apply_confidence_threshold(original, 0.9)
    assert [m.frame_index for m in result] == [m.frame_index for m in original]
    assert [m.timestamp_s for m in result] == [m.timestamp_s for m in original]


def test_a_missing_frame_carries_its_position_and_nothing_else():
    missing = FrameMeasure.missing(25, 1.0)
    assert missing.frame_index == 25
    assert missing.timestamp_s == pytest.approx(1.0)
    assert not missing.detected
    assert missing.units == {}
    assert missing.head is None
