"""Prosodic arithmetic: semitone normalisation, robust statistics, pooling.

Kept apart from anything that touches audio so the decisions that matter can be
checked directly. Three of them are worth stating, since none is the only
reasonable choice:

* **F0 is expressed in semitones relative to the speaker's own median**, not in
  hertz. Absolute pitch differs between speakers by roughly an octave along
  sex, which with 62 participants would dominate any pitch feature and let a
  model appear to predict a questionnaire score while keying on speaker sex.
  Semitones are also perceptually linear, where hertz is not: 20 Hz is a large
  change for a low voice and a small one for a high voice.
  See docs/decisions/0005.
* **Statistics are pooled across a session's speech, not averaged per span.**
  A session is dozens of utterances of very unequal length; averaging per-span
  means would weight a half-second interjection the same as a twenty-second
  answer. Frame-level values are therefore concatenated, and the per-span
  measures that cannot be pooled that way (jitter, shimmer) are combined
  weighted by the analysed duration behind each.
* **A missing measure is None, never zero.** Zero jitter is a claim about a
  voice; no jitter measurement is the absence of one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

SEMITONES_PER_OCTAVE: Final = 12.0

# Percentiles used for a range that ignores the occasional tracking error.
RANGE_LOW_PERCENTILE: Final = 5.0
RANGE_HIGH_PERCENTILE: Final = 95.0

# Praat reports an unvoiced or undefined frame as these sentinels rather than
# as a gap, so they have to be filtered rather than trusted.
UNVOICED_F0: Final = 0.0
UNDEFINED_HARMONICITY: Final = -200.0

_MIN_FOR_SPREAD: Final = 2


def to_semitones(frequencies: Sequence[float] | np.ndarray, reference_hz: float) -> np.ndarray:
    """Convert frequencies to semitones relative to `reference_hz`.

    Args:
        frequencies: Frequencies in hertz. Non-positive values are dropped
            rather than clamped: the conversion is undefined for them, and a
            clamp would invent a pitch that was never measured.
        reference_hz: The speaker's own median F0.

    Returns:
        Semitones, empty when nothing was usable.

    Raises:
        ValueError: if the reference is not positive.
    """
    if reference_hz <= 0.0:
        msg = f"the semitone reference must be positive, got {reference_hz}"
        raise ValueError(msg)
    values = np.asarray(frequencies, dtype=np.float64)
    usable = values[np.isfinite(values) & (values > 0.0)]
    if usable.size == 0:
        return np.zeros(0, dtype=np.float64)
    return SEMITONES_PER_OCTAVE * np.log2(usable / reference_hz)


def voiced_frequencies(frequencies: Sequence[float] | np.ndarray) -> np.ndarray:
    """Keep only the frames Praat actually tracked a pitch in."""
    values = np.asarray(frequencies, dtype=np.float64)
    return values[np.isfinite(values) & (values > UNVOICED_F0)]


def defined_harmonicity(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Keep only the harmonicity frames Praat defined."""
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array) & (array > UNDEFINED_HARMONICITY)]


def median_or_none(values: Sequence[float] | np.ndarray) -> float | None:
    """Median, or None when there is nothing to take one of."""
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)) if array.size else None


def mean_or_none(values: Sequence[float] | np.ndarray) -> float | None:
    """Mean, or None when there is nothing to average."""
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)) if array.size else None


def sd_or_none(values: Sequence[float] | np.ndarray) -> float | None:
    """Sample standard deviation, or None for fewer than two values."""
    array = np.asarray(values, dtype=np.float64)
    if array.size < _MIN_FOR_SPREAD:
        return None
    return float(np.std(array, ddof=1))


def iqr_or_none(values: Sequence[float] | np.ndarray) -> float | None:
    """Interquartile range, robust to the occasional tracking error."""
    array = np.asarray(values, dtype=np.float64)
    if array.size < _MIN_FOR_SPREAD:
        return None
    return float(np.percentile(array, 75.0) - np.percentile(array, 25.0))


def percentile_range(values: Sequence[float] | np.ndarray) -> float | None:
    """Spread from the 5th to the 95th percentile.

    Used instead of max minus min, which on a pitch contour usually reports one
    octave-jump tracking error rather than the speaker's range.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size < _MIN_FOR_SPREAD:
        return None
    low = float(np.percentile(array, RANGE_LOW_PERCENTILE))
    high = float(np.percentile(array, RANGE_HIGH_PERCENTILE))
    return high - low


def mean_absolute_delta(values: Sequence[float] | np.ndarray) -> float | None:
    """Mean absolute change between consecutive values.

    Captures how much a contour moves moment to moment, which is a different
    thing from how widely it ranges: a monotone drifting slowly and a lively
    voice can share a standard deviation.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size < _MIN_FOR_SPREAD:
        return None
    return float(np.mean(np.abs(np.diff(array))))


def weighted_mean(values: Sequence[float | None], weights: Sequence[float]) -> float | None:
    """Mean of `values` weighted by `weights`, ignoring missing values.

    Used for measures that cannot be pooled frame by frame, so that a long
    utterance counts for more than a brief one.
    """
    pairs = [
        (value, weight)
        for value, weight in zip(values, weights, strict=True)
        if value is not None and math.isfinite(value) and weight > 0.0
    ]
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0.0:  # pragma: no cover - guarded by the filter
        return None
    return sum(value * weight for value, weight in pairs) / total_weight


def syllable_nuclei(
    intensity_db: Sequence[float] | np.ndarray,
    *,
    frame_step_s: float,
    threshold_db_below_max: float = 25.0,
    min_spacing_s: float = 0.1,
) -> int:
    """Count intensity peaks as a proxy for syllables.

    A simplified form of the standard syllable-nuclei method: peaks in the
    intensity contour, above a threshold set relative to the loudest frame, and
    no closer together than a plausible syllable. It is a proxy and named as
    one: it does not use voicing, so it will count a loud unvoiced burst, and a
    proper rate would need a syllabifier for Japanese.

    Args:
        intensity_db: The intensity contour in decibels.
        frame_step_s: Seconds between contour frames.
        threshold_db_below_max: How far below the loudest frame a peak must
            still reach to be counted.
        min_spacing_s: Minimum separation between counted peaks.

    Returns:
        The number of peaks.
    """
    values = np.asarray(intensity_db, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < _MIN_FOR_SPREAD + 1 or frame_step_s <= 0.0:
        return 0

    floor = float(np.max(values)) - threshold_db_below_max
    min_gap = max(1, round(min_spacing_s / frame_step_s))

    peaks: list[int] = []
    for index in range(1, values.size - 1):
        value = values[index]
        if value < floor:
            continue
        # A strict rise on the left and no fall on the right: a plateau counts
        # once, at its onset, and a flat contour counts nothing at all. Testing
        # only for "not lower than either neighbour" would call every frame of
        # a sustained passage a syllable.
        if not (value > values[index - 1] and value >= values[index + 1]):
            continue
        if peaks and index - peaks[-1] < min_gap:
            # Keep whichever of the two is louder.
            if value > values[peaks[-1]]:
                peaks[-1] = index
            continue
        peaks.append(index)
    return len(peaks)


@dataclass(frozen=True, slots=True)
class SpanMeasures:
    """Raw prosodic measures from one analysed stretch of speech.

    Frame-level arrays are kept so that session statistics can be pooled over
    frames rather than over spans.
    """

    duration_s: float
    f0_hz: np.ndarray
    intensity_db: np.ndarray
    harmonicity_db: np.ndarray
    intensity_frame_step_s: float
    jitter_local: float | None = None
    shimmer_local: float | None = None

    @property
    def n_voiced(self) -> int:
        """How many frames carried a tracked pitch."""
        return int(voiced_frequencies(self.f0_hz).size)

    @property
    def n_pitch_frames(self) -> int:
        """How many pitch frames were analysed, voiced or not."""
        return int(np.asarray(self.f0_hz).size)


def pool(measures: Sequence[SpanMeasures]) -> dict[str, np.ndarray]:
    """Concatenate the frame-level arrays across spans."""
    if not measures:
        empty = np.zeros(0, dtype=np.float64)
        return {"f0_hz": empty, "intensity_db": empty, "harmonicity_db": empty}
    return {
        "f0_hz": np.concatenate([voiced_frequencies(m.f0_hz) for m in measures]),
        "intensity_db": np.concatenate(
            [np.asarray(m.intensity_db, dtype=np.float64) for m in measures]
        ),
        "harmonicity_db": np.concatenate([defined_harmonicity(m.harmonicity_db) for m in measures]),
    }


FEATURE_NAMES: Final = (
    "prosody__f0_semitone_sd",
    "prosody__f0_semitone_iqr",
    "prosody__f0_semitone_range",
    "prosody__f0_semitone_mean_abs_delta",
    "prosody__voiced_fraction",
    "prosody__intensity_mean_db",
    "prosody__intensity_sd_db",
    "prosody__intensity_range_db",
    "prosody__jitter_local",
    "prosody__shimmer_local",
    "prosody__hnr_db",
    "prosody__speech_rate_proxy",
)


def prosody_features(measures: Sequence[SpanMeasures]) -> dict[str, float | None]:
    """Compute the session-level prosodic features.

    The semitone reference is the median of every voiced frame in the session,
    so each speaker is measured against themselves.

    Returns:
        Feature name to value, with None wherever the session gives no basis
        for the measure. Names follow the `prosody__` convention.
    """
    pooled = pool(measures)
    f0_hz = pooled["f0_hz"]
    intensity = pooled["intensity_db"]
    harmonicity = pooled["harmonicity_db"]

    reference = median_or_none(f0_hz)
    semitones = (
        to_semitones(f0_hz, reference) if reference is not None and reference > 0.0 else np.zeros(0)
    )

    analysed_s = sum(m.duration_s for m in measures)
    pitch_frames = sum(m.n_pitch_frames for m in measures)
    voiced_frames = sum(m.n_voiced for m in measures)

    nuclei = sum(
        syllable_nuclei(m.intensity_db, frame_step_s=m.intensity_frame_step_s) for m in measures
    )

    return {
        "prosody__f0_semitone_sd": sd_or_none(semitones),
        "prosody__f0_semitone_iqr": iqr_or_none(semitones),
        "prosody__f0_semitone_range": percentile_range(semitones),
        "prosody__f0_semitone_mean_abs_delta": mean_absolute_delta(semitones),
        "prosody__voiced_fraction": voiced_frames / pitch_frames if pitch_frames else None,
        "prosody__intensity_mean_db": mean_or_none(intensity),
        "prosody__intensity_sd_db": sd_or_none(intensity),
        "prosody__intensity_range_db": percentile_range(intensity),
        "prosody__jitter_local": weighted_mean(
            [m.jitter_local for m in measures], [m.duration_s for m in measures]
        ),
        "prosody__shimmer_local": weighted_mean(
            [m.shimmer_local for m in measures], [m.duration_s for m in measures]
        ),
        "prosody__hnr_db": mean_or_none(harmonicity),
        "prosody__speech_rate_proxy": nuclei / analysed_s if analysed_s > 0.0 else None,
    }
