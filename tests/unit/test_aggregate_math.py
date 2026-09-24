"""Windowed summarisation arithmetic.

The speaking/listening contrast is what this project adds to the lab's prior
work, so which frames land in which window is load-bearing and is tested
directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from vc_multimodal.features.aggregate_math import (
    BASE_STATS,
    PEAK_STAT,
    feature_names,
    mask_in_spans,
    statistic,
    stats_plan,
    summarise_window,
    window_coverage,
)
from vc_multimodal.features.spans import Span


def spans(*pairs: tuple[float, float]) -> list[Span]:
    return [Span(start, end) for start, end in pairs]


# ---------------------------------------------------------------------------
# which frames fall in a window
# ---------------------------------------------------------------------------
def test_frames_inside_a_span_are_selected():
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    assert mask_in_spans(times, spans((0.4, 1.2))).tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]


def test_a_frame_on_the_start_is_in_and_on_the_end_is_out():
    """Half-open spans, so adjacent windows cannot both claim a frame."""
    assert mask_in_spans([1.0], spans((1.0, 2.0))).tolist() == [True]
    assert mask_in_spans([2.0], spans((1.0, 2.0))).tolist() == [False]


def test_several_spans_are_handled():
    times = np.array([0.1, 0.6, 1.1, 1.6, 2.1])
    result = mask_in_spans(times, spans((0.0, 0.5), (1.0, 1.5), (2.0, 2.5)))
    assert result.tolist() == [True, False, True, False, True]


def test_overlapping_spans_do_not_double_count():
    """They are merged first, so a frame matches at most one."""
    times = np.array([1.0])
    assert mask_in_spans(times, spans((0.0, 2.0), (0.5, 3.0))).tolist() == [True]


def test_frames_outside_every_span_are_excluded():
    assert not mask_in_spans([10.0, 20.0], spans((0.0, 1.0))).any()


def test_no_spans_selects_nothing():
    assert not mask_in_spans([0.5, 1.0], []).any()


def test_no_frames_gives_an_empty_mask():
    assert mask_in_spans([], spans((0.0, 1.0))).size == 0


def test_unsorted_spans_are_handled():
    times = np.array([0.5, 2.5])
    assert mask_in_spans(times, spans((2.0, 3.0), (0.0, 1.0))).tolist() == [True, True]


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
def test_coverage_counts_frames_and_measured_frames():
    times = np.array([0.0, 0.5, 1.0, 1.5])
    detected = np.array([True, True, False, True])

    coverage = window_coverage(times, detected, spans((0.0, 1.2)))

    assert coverage.seconds == pytest.approx(1.2)
    assert coverage.n_frames == 3
    assert coverage.n_measured == 2


def test_measured_seconds_is_pro_rata():
    """Two of three frames usable in a 30 s window is 20 s of measured time."""
    times = np.array([0.0, 10.0, 20.0])
    coverage = window_coverage(times, np.array([True, True, False]), spans((0.0, 30.0)))
    assert coverage.measured_seconds == pytest.approx(20.0)
    assert coverage.measured_fraction == pytest.approx(2 / 3)


def test_a_window_with_no_frames_has_no_measured_fraction():
    """Not zero: nothing was looked at, which is different from nothing found."""
    coverage = window_coverage([100.0], [True], spans((0.0, 1.0)))
    assert coverage.n_frames == 0
    assert coverage.measured_fraction is None
    assert coverage.measured_seconds == pytest.approx(0.0)


def test_an_empty_window_has_no_seconds():
    coverage = window_coverage([0.0], [True], [])
    assert coverage.seconds == pytest.approx(0.0)
    assert coverage.n_frames == 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_the_mean_and_standard_deviation():
    values = np.array([1.0, 2.0, 3.0])
    assert statistic(values, "mean") == pytest.approx(2.0)
    assert statistic(values, "sd") == pytest.approx(1.0)


def test_the_peak_separates_from_the_mean_when_expression_is_frequent_enough():
    """Zero-inflated distributions: the mean says how much, the peak how strong."""
    values = np.array([0.0] * 80 + [1.0] * 20)
    assert statistic(values, "mean") == pytest.approx(0.2)
    assert statistic(values, PEAK_STAT) == pytest.approx(1.0)


def test_the_peak_adds_nothing_for_a_rarely_expressed_unit():
    """A property worth knowing rather than a defect.

    The 90th percentile can only see expression occupying more than a tenth of
    the window. Below that it collapses towards the mean, and both read near
    zero - which is itself the interpretable answer: the unit was not expressed.
    """
    rare = np.array([0.0] * 95 + [1.0] * 5)
    assert statistic(rare, PEAK_STAT) == pytest.approx(0.0)
    assert statistic(rare, "mean") == pytest.approx(0.05)


def test_a_single_value_has_no_spread():
    assert statistic(np.array([5.0]), "sd") is None
    assert statistic(np.array([5.0]), "mean") == pytest.approx(5.0)


def test_statistics_of_nothing_are_none():
    for name in (*BASE_STATS, PEAK_STAT):
        assert statistic(np.zeros(0), name) is None


def test_non_finite_values_are_dropped():
    values = np.array([1.0, np.nan, 3.0])
    assert statistic(values, "mean") == pytest.approx(2.0)


def test_an_all_missing_measure_is_none_not_zero():
    assert statistic(np.array([np.nan, np.nan]), "mean") is None


def test_an_unknown_statistic_is_refused():
    with pytest.raises(ValueError, match="unknown statistic"):
        statistic(np.array([1.0]), "mode")


# ---------------------------------------------------------------------------
# summarising a window
# ---------------------------------------------------------------------------
def test_only_frames_both_inside_and_measured_are_summarised():
    values = {"au12": np.array([1.0, 2.0, 100.0, 4.0])}
    inside = np.array([True, True, True, False])
    detected = np.array([True, True, False, True])

    summary = summarise_window(values, inside, detected, stats_by_measure={"au12": ("mean",)})

    # The third frame is inside but unmeasured; the fourth is measured but out.
    assert summary["au12_mean"] == pytest.approx(1.5)


def test_a_window_with_nothing_selected_yields_none():
    summary = summarise_window(
        {"au12": np.array([1.0, 2.0])},
        np.array([False, False]),
        np.array([True, True]),
        stats_by_measure={"au12": ("mean", "sd")},
    )
    assert summary == {"au12_mean": None, "au12_sd": None}


def test_each_measure_gets_the_statistics_it_was_assigned():
    values = {"au12": np.array([1.0, 2.0, 3.0]), "head_yaw": np.array([1.0, 2.0, 3.0])}
    summary = summarise_window(
        values,
        np.array([True, True, True]),
        np.array([True, True, True]),
        stats_by_measure={"au12": ("mean", "sd", PEAK_STAT), "head_yaw": ("sd",)},
    )
    assert set(summary) == {"au12_mean", "au12_sd", "au12_p90", "head_yaw_sd"}


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def test_every_unit_gets_the_base_statistics():
    plan = stats_plan(("au01", "au02"), ())
    assert plan["au01"] == BASE_STATS
    assert plan["au02"] == BASE_STATS


def test_the_predictive_units_additionally_get_a_peak():
    plan = stats_plan(("au01", "au02", "au12"), ("au01", "au12"))
    assert PEAK_STAT in plan["au01"]
    assert PEAK_STAT in plan["au12"]
    assert PEAK_STAT not in plan["au02"]


def test_head_pose_gets_only_a_spread():
    """Its mean records where the camera sat, not anything about the person."""
    plan = stats_plan(("au12",), (), ("head_yaw",))
    assert plan["head_yaw"] == ("sd",)


def test_the_feature_names_follow_the_window_and_convention():
    names = feature_names("face_speaking", ("au12",), ("au12",), ("head_yaw",))
    assert names == (
        "face_speaking__au12_mean",
        "face_speaking__au12_sd",
        "face_speaking__au12_p90",
        "face_speaking__head_yaw_sd",
    )


def test_the_confirmatory_names_are_produced_by_the_plan():
    """ADR 13 fixed these, so the generator has to emit exactly them."""
    for window in ("face_speaking", "face_listening"):
        names = feature_names(
            window, ("au01", "au02", "au04", "au06", "au12"), ("au01", "au06", "au12")
        )
        for unit in ("au01", "au06", "au12"):
            assert f"{window}__{unit}_mean" in names
