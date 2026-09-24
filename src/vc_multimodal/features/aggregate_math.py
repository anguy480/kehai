"""Summarising per-frame facial measures over the speaking and listening windows.

The contrast this project adds to the lab's prior work is that the same action
units are summarised separately over the time the participant is speaking and
the time they are listening (docs/decisions/0013). That makes the windowing
arithmetic load-bearing, so it lives here as pure functions over arrays and
spans, with no filesystem and no configuration objects.

Three choices worth stating:

* **A frame belongs to a window if its timestamp falls inside it.** Frames are
  instants sampled every 200 ms, not intervals, so no frame is split between
  windows and none is counted twice. Speaking and listening never overlap by
  construction, and mutual silence belongs to neither.
* **Only measured frames count.** A frame where no face was found contributes
  nothing rather than a zero; the share that were dropped is reported
  separately, per window, so a statistic resting on very little is visible.
* **A window with too little measured time yields None, not a number.** Ten
  usable frames of listening can produce a mean, and it would be noise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from vc_multimodal.features.spans import Span, merge

# Statistics computed for every action unit in every window.
BASE_STATS: Final = ("mean", "sd")

# An additional statistic for the action units the precedent found predictive.
# These distributions are heavily zero-inflated - most frames are neutral - so
# the mean answers "how much expression overall" while a high percentile
# answers "how strong it gets", and the two can diverge sharply.
#
# It has a floor worth knowing: the 90th percentile can only see expression
# occupying more than a tenth of the window. Below that it collapses towards
# the mean and both read near zero, which is the interpretable answer rather
# than a defect - the unit was not expressed.
PEAK_STAT: Final = "p90"
PEAK_PERCENTILE: Final = 90.0

_MIN_FOR_SD: Final = 2


def mask_in_spans(timestamps: Sequence[float] | np.ndarray, spans: Sequence[Span]) -> np.ndarray:
    """Which timestamps fall inside any of `spans`.

    Spans are merged first, so they are disjoint and sorted and a timestamp can
    match at most one. Uses a binary search rather than a scan over pairs: a
    session holds a few thousand frames and a few hundred spans.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    if times.size == 0:
        return np.zeros(0, dtype=bool)

    merged = merge(spans)
    if not merged:
        return np.zeros(times.size, dtype=bool)

    starts = np.array([span.start for span in merged], dtype=np.float64)
    ends = np.array([span.end for span in merged], dtype=np.float64)

    # The last span starting at or before each timestamp is the only one that
    # can contain it.
    index = np.searchsorted(starts, times, side="right") - 1
    inside = index >= 0
    candidate = np.clip(index, 0, len(merged) - 1)
    return inside & (times < ends[candidate])


@dataclass(frozen=True, slots=True)
class WindowCoverage:
    """How much of a window was actually measured.

    Attributes:
        seconds: Length of the window itself.
        n_frames: Frames sampled inside it.
        n_measured: Frames inside it with a usable face.
    """

    seconds: float
    n_frames: int
    n_measured: int

    @property
    def measured_seconds(self) -> float:
        """Window time backed by a measured frame, pro rata."""
        if self.n_frames == 0:
            return 0.0
        return self.seconds * self.n_measured / self.n_frames

    @property
    def measured_fraction(self) -> float | None:
        """Share of the window's frames that were measured."""
        if self.n_frames == 0:
            return None
        return self.n_measured / self.n_frames


def window_coverage(
    timestamps: Sequence[float] | np.ndarray,
    detected: Sequence[bool] | np.ndarray,
    spans: Sequence[Span],
) -> WindowCoverage:
    """Measure how much of a window the sampled frames cover."""
    inside = mask_in_spans(timestamps, spans)
    found = np.asarray(detected, dtype=bool)
    return WindowCoverage(
        seconds=sum(span.duration for span in merge(spans)),
        n_frames=int(inside.sum()),
        n_measured=int((inside & found).sum()),
    )


def statistic(values: np.ndarray, name: str) -> float | None:
    """One summary statistic, or None where the values cannot support it.

    Raises:
        ValueError: if the statistic is not one this module computes.
    """
    usable = values[np.isfinite(values)]
    if usable.size == 0:
        return None
    if name == "mean":
        return float(np.mean(usable))
    if name == "sd":
        if usable.size < _MIN_FOR_SD:
            return None
        return float(np.std(usable, ddof=1))
    if name == PEAK_STAT:
        return float(np.percentile(usable, PEAK_PERCENTILE))
    msg = f"unknown statistic {name!r}"
    raise ValueError(msg)


def summarise_window(
    values_by_measure: Mapping[str, np.ndarray],
    inside: np.ndarray,
    detected: np.ndarray,
    *,
    stats_by_measure: Mapping[str, Sequence[str]],
) -> dict[str, float | None]:
    """Summarise every measure over one window.

    Args:
        values_by_measure: Measure name to its per-frame values.
        inside: Which frames fall in the window.
        detected: Which frames had a usable face.
        stats_by_measure: Which statistics to compute for each measure.

    Returns:
        `"<measure>_<stat>"` to value, with None wherever the window gives no
        basis for it.
    """
    keep = inside & detected
    summary: dict[str, float | None] = {}
    for measure, values in values_by_measure.items():
        selected = np.asarray(values, dtype=np.float64)[keep]
        for name in stats_by_measure.get(measure, BASE_STATS):
            summary[f"{measure}_{name}"] = statistic(selected, name)
    return summary


def stats_plan(
    unit_keys: Sequence[str],
    peak_units: Sequence[str],
    pose_measures: Sequence[str] = (),
) -> dict[str, tuple[str, ...]]:
    """Which statistics each measure gets.

    Depth where the precedent points and breadth nowhere else: every action
    unit gets a mean and a standard deviation, the units found predictive of
    social performance additionally get a high percentile, and head pose gets
    only a standard deviation, since it is a measure of movement rather than of
    expression and its mean is an artifact of where the camera sat.
    """
    plan: dict[str, tuple[str, ...]] = {}
    peaks = set(peak_units)
    for key in unit_keys:
        plan[key] = (*BASE_STATS, PEAK_STAT) if key in peaks else BASE_STATS
    for measure in pose_measures:
        plan[measure] = ("sd",)
    return plan


def feature_names(
    window: str,
    unit_keys: Sequence[str],
    peak_units: Sequence[str],
    pose_measures: Sequence[str] = (),
) -> tuple[str, ...]:
    """The feature column names one window contributes, in order."""
    plan = stats_plan(unit_keys, peak_units, pose_measures)
    return tuple(f"{window}__{measure}_{name}" for measure, names in plan.items() for name in names)
