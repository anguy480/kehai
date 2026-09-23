"""Prosodic measurement through Praat, via parselmouth.

Praat is the reference implementation for these measures, which matters for a
manuscript: F0, intensity, harmonicity, jitter and shimmer computed here are
the same quantities the literature reports, computed the same way.

Jitter and shimmer need a point process derived from the pitch track, so they
are measured per span rather than per frame. A span too short or too unvoiced
to support them yields None rather than a number, and the session-level pooling
weights what survives by duration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
import parselmouth
from parselmouth.praat import call

from vc_multimodal.features.prosody_math import SpanMeasures
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.prosody.base import ProsodyBackend, ProsodyError

if TYPE_CHECKING:
    from vc_multimodal.config import ProsodyConfig

logger = get_logger(__name__)

# Praat's defaults for the period-to-period measures: shortest and longest
# credible period, and the largest ratio between neighbouring periods.
_PERIOD_FLOOR_S: Final = 0.0001
_PERIOD_CEILING_S: Final = 0.02
_MAX_PERIOD_FACTOR: Final = 1.3
_MAX_AMPLITUDE_FACTOR: Final = 1.6

# Praat needs a few pitch periods to measure anything at all.
_MIN_PERIODS: Final = 6


class ParselmouthBackend(ProsodyBackend):
    """Measures prosody with Praat's own algorithms."""

    name = "parselmouth"

    def available(self) -> bool:
        """Always true: parselmouth is a required dependency."""
        return True

    def unavailable_reason(self) -> str:
        """Empty: this backend is always available."""
        return ""

    def version(self) -> str:
        """Parselmouth's version, which pins the Praat build behind it."""
        return f"parselmouth/{parselmouth.__version__}"

    def measure(
        self, samples: np.ndarray, sample_rate: int, *, config: ProsodyConfig
    ) -> SpanMeasures:
        """Measure one stretch of speech.

        Raises:
            ProsodyError: if Praat cannot analyse the audio at all.
        """
        duration_s = samples.size / sample_rate if sample_rate else 0.0
        if samples.size == 0 or duration_s <= 0.0:
            msg = "cannot measure an empty span"
            raise ProsodyError(msg)

        sound = parselmouth.Sound(
            np.ascontiguousarray(samples, dtype=np.float64), sampling_frequency=sample_rate
        )
        floor, ceiling = config.f0_floor_hz, config.f0_ceiling_hz

        try:
            pitch = sound.to_pitch(pitch_floor=floor, pitch_ceiling=ceiling)
            f0_hz = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
        except Exception as exc:
            msg = f"Praat could not track pitch: {exc}"
            raise ProsodyError(msg) from exc

        intensity_db, intensity_step = self._intensity(sound, floor)
        harmonicity_db = self._harmonicity(sound, floor)
        jitter, shimmer = self._voice_quality(sound, floor, ceiling, duration_s)

        return SpanMeasures(
            duration_s=duration_s,
            f0_hz=f0_hz,
            intensity_db=intensity_db,
            harmonicity_db=harmonicity_db,
            intensity_frame_step_s=intensity_step,
            jitter_local=jitter,
            shimmer_local=shimmer,
        )

    def _intensity(self, sound: Any, floor: float) -> tuple[np.ndarray, float]:
        """The intensity contour and its frame step.

        A span shorter than the analysis window yields no contour, which is not
        an error: the pitch track from the same span may still be usable.
        """
        try:
            intensity = sound.to_intensity(minimum_pitch=floor)
        except Exception:
            return np.zeros(0, dtype=np.float64), 0.0
        values = np.asarray(intensity.values[0], dtype=np.float64)
        return values, float(intensity.time_step)

    def _harmonicity(self, sound: Any, floor: float) -> np.ndarray:
        """The harmonics-to-noise contour, empty when it cannot be computed."""
        try:
            harmonicity = sound.to_harmonicity_cc(minimum_pitch=floor)
        except Exception:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(harmonicity.values[0], dtype=np.float64)

    def _voice_quality(
        self, sound: Any, floor: float, ceiling: float, duration_s: float
    ) -> tuple[float | None, float | None]:
        """Jitter and shimmer, or None where the span cannot support them.

        Praat returns NaN rather than raising when there are too few periods,
        so the result is checked rather than trusted.
        """
        if duration_s < _MIN_PERIODS / floor:
            return None, None
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", floor, ceiling)
            jitter = call(
                point_process,
                "Get jitter (local)",
                0,
                0,
                _PERIOD_FLOOR_S,
                _PERIOD_CEILING_S,
                _MAX_PERIOD_FACTOR,
            )
            shimmer = call(
                [sound, point_process],
                "Get shimmer (local)",
                0,
                0,
                _PERIOD_FLOOR_S,
                _PERIOD_CEILING_S,
                _MAX_PERIOD_FACTOR,
                _MAX_AMPLITUDE_FACTOR,
            )
        except Exception:
            return None, None
        return _finite(jitter), _finite(shimmer)


def _finite(value: object) -> float | None:
    """Coerce a Praat result to a float, or None when it is not a number."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
