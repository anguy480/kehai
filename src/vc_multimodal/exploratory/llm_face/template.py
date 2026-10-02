"""The description template: per-frame facial measures to plain sentences.

Deterministic by construction - thresholds and wording are fixed here, and the
same frames always give the same text. This file is hashed into every output
it produces, so a change to any threshold or word is visible downstream.

What a description says about each action unit in a window:

* **How often** it was present: the share of measured frames above that unit's
  threshold, in words and as a percentage.
* **How strongly**, when present: how far above the threshold, scaled to what
  was left of the 0-1 range.
* **How long** episodes lasted: the median run of consecutive present frames.
* **How it changed** over the conversation: last third against first third.

And about the head: how much pitch and yaw varied, and how often it moved
sharply between consecutive frames.

A description never contains a session identifier, a time of day or anything
else about the person: only these measures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

#: How each unit is named in a description: FACS name, with the unit number.
UNIT_NAMES: Final[Mapping[str, str]] = {
    "au01": "inner brow raising (AU1)",
    "au02": "outer brow raising (AU2)",
    "au04": "brow lowering, as in frowning (AU4)",
    "au06": "cheek raising (AU6)",
    "au12": "lip corner pulling, as in smiling (AU12)",
}

#: Score above which a frame counts as showing the unit. MediaPipe blendshape
#: scores run 0-1 but rest at different levels: across sessions the brow
#: raisers average about 0.27 with a typical 90th percentile near 0.46, while
#: brow lowering averages under 0.01 and smiling about 0.04. One threshold for
#: all would call a resting brow raised, or never see a frown. Fixed on
#: 2026-10-02 from the session-level feature table, which holds no label, before
#: any description was rated.
PRESENT_ABOVE: Final[Mapping[str, float]] = {
    "au01": 0.5,
    "au02": 0.5,
    "au04": 0.1,
    "au06": 0.1,
    "au12": 0.2,
}

#: Share of measured frames present -> word. Upper bounds, checked in order.
FREQUENCY_WORDS: Final = (
    (0.01, "essentially never"),
    (0.05, "rarely"),
    (0.20, "occasionally"),
    (0.50, "often"),
    (math.inf, "most of the time"),
)

#: Mean excess over the threshold while present, as a share of the headroom
#: above it -> word.
STRENGTH_WORDS: Final = ((0.20, "weak"), (0.45, "moderate"), (math.inf, "strong"))

#: Median episode length in seconds -> word.
DURATION_WORDS: Final = ((1.0, "brief"), (3.0, "short"), (math.inf, "sustained"))

#: A change in presence smaller than this between the first and last thirds is
#: described as no change.
TREND_MARGIN: Final = 0.05

#: Fewest present-or-absent frames in a third for its share to be compared.
MIN_FRAMES_PER_THIRD: Final = 10

#: Mean of the pitch and yaw standard deviations, in degrees -> phrase.
HEAD_WORDS: Final = (
    (3.0, "kept their head quite still"),
    (6.0, "moved their head moderately"),
    (math.inf, "moved their head a lot"),
)

#: A change in pitch or yaw between consecutive frames larger than this, in
#: degrees, counts as one sharp head movement.
SHARP_MOVE_DEG: Final = 5.0

#: Consecutive frames further apart than this many sampling steps are not
#: treated as continuous: a gap ends an episode and is not a head movement.
CONTINUITY_STEPS: Final = 1.5

_MIN_FOR_SD: Final = 2

#: How each window is introduced.
WINDOW_PHRASES: Final[Mapping[str, str]] = {
    "speaking": "While speaking",
    "listening": "While listening",
}


@dataclass(frozen=True, slots=True)
class UnitStats:
    """One action unit over one window."""

    unit: str
    present_fraction: float
    strength: float | None
    median_episode_s: float | None
    trend: float | None


@dataclass(frozen=True, slots=True)
class WindowStats:
    """Everything a description says about one window."""

    window: str
    measured_seconds: float
    measured_fraction: float | None
    units: tuple[UnitStats, ...]
    head_sd_deg: float | None
    sharp_moves_per_min: float | None


def _word(value: float, scale: Sequence[tuple[float, str]]) -> str:
    for bound, word in scale:
        if value < bound:
            return word
    return scale[-1][1]


def _continuous(times: np.ndarray, step_s: float) -> np.ndarray:
    """For each frame after the first, whether it follows on from the last."""
    return np.diff(times) <= CONTINUITY_STEPS * step_s


def unit_stats(unit: str, times: np.ndarray, values: np.ndarray, step_s: float) -> UnitStats:
    """Describe one unit from its measured frames, in time order."""
    threshold = PRESENT_ABOVE[unit]
    present = values > threshold
    n = values.size
    fraction = float(present.mean()) if n else 0.0

    strength: float | None = None
    if present.any():
        excess = (values[present] - threshold) / (1.0 - threshold)
        strength = float(np.mean(excess))

    episodes: list[int] = []
    run = 0
    linked = np.concatenate([[False], _continuous(times, step_s)]) if n else np.zeros(0, bool)
    for is_present, follows in zip(present, linked, strict=True):
        if is_present and (follows or run == 0):
            run += 1
        elif is_present:
            episodes.append(run)
            run = 1
        else:
            if run:
                episodes.append(run)
            run = 0
    if run:
        episodes.append(run)
    median_episode = float(np.median(episodes)) * step_s if episodes else None

    trend: float | None = None
    if n:
        start, end = float(times[0]), float(times[-1])
        first = times < start + (end - start) / 3
        last = times >= start + 2 * (end - start) / 3
        if first.sum() >= MIN_FRAMES_PER_THIRD and last.sum() >= MIN_FRAMES_PER_THIRD:
            trend = float(present[last].mean() - present[first].mean())

    return UnitStats(unit, fraction, strength, median_episode, trend)


def head_stats(
    times: np.ndarray, pitch: np.ndarray, yaw: np.ndarray, step_s: float, measured_seconds: float
) -> tuple[float | None, float | None]:
    """Head variability in degrees, and sharp movements per minute."""
    usable = ~(np.isnan(pitch) | np.isnan(yaw))
    times, pitch, yaw = times[usable], pitch[usable], yaw[usable]
    if pitch.size < _MIN_FOR_SD or measured_seconds <= 0:
        return None, None
    variability = float((np.std(pitch, ddof=1) + np.std(yaw, ddof=1)) / 2)
    linked = _continuous(times, step_s)
    jumps = np.maximum(np.abs(np.diff(pitch)), np.abs(np.diff(yaw))) > SHARP_MOVE_DEG
    per_minute = float((jumps & linked).sum()) / (measured_seconds / 60.0)
    return variability, per_minute


def _unit_sentence(stats: UnitStats) -> str:
    name = UNIT_NAMES[stats.unit]
    name = name[0].upper() + name[1:]
    if stats.present_fraction < FREQUENCY_WORDS[0][0]:
        return f"{name} essentially never appeared."
    parts = [
        f"{name} appeared {_word(stats.present_fraction, FREQUENCY_WORDS)} "
        f"({stats.present_fraction:.0%} of the time)"
    ]
    if stats.strength is not None:
        parts.append(f"{_word(stats.strength, STRENGTH_WORDS)} when present")
    if stats.median_episode_s is not None:
        parts.append(
            f"in {_word(stats.median_episode_s, DURATION_WORDS)} episodes "
            f"(typically {stats.median_episode_s:.1f} s)"
        )
    if stats.trend is not None:
        if abs(stats.trend) < TREND_MARGIN:
            parts.append("about as often late in the conversation as early")
        elif stats.trend > 0:
            parts.append("more often later in the conversation")
        else:
            parts.append("less often later in the conversation")
    return ", ".join(parts) + "."


def render_window(stats: WindowStats) -> str:
    """One paragraph for one window."""
    coverage = (
        f", {stats.measured_fraction:.0%} of that time"
        if stats.measured_fraction is not None
        else ""
    )
    lines = [
        f"{WINDOW_PHRASES[stats.window]} ({stats.measured_seconds / 60:.1f} min "
        f"of the face measured{coverage}):"
    ]
    lines.extend(_unit_sentence(unit) for unit in stats.units)
    if stats.head_sd_deg is not None and stats.sharp_moves_per_min is not None:
        lines.append(
            f"They {_word(stats.head_sd_deg, HEAD_WORDS)} (pitch and yaw varied by "
            f"about {stats.head_sd_deg:.1f} degrees), with about "
            f"{stats.sharp_moves_per_min:.1f} sharp head movements per minute."
        )
    return " ".join(lines)


def render_insufficient(window: str) -> str:
    """The paragraph for a window with too little measured time."""
    return f"{WINDOW_PHRASES[window]}: too little of the face was measured to describe."
