"""Stereo channel statistics, computed in one streaming pass.

Every recording carries a single mixed AAC stream in stereo at 48 kHz. Zoom
sometimes pans speakers across the stereo field, so the two channels may not be
identical, and any real separation would be worth having: it would give a cheap,
diarization-independent signal about who is speaking.

The statistics are accumulated over chunks rather than computed on a whole
decoded recording, because a 12-minute session decoded to stereo float64 is
around 180 MB and several sessions run in parallel. Accumulating sums keeps
memory constant regardless of session length, and the Pearson correlation it
produces is exact, not an approximation.

Pure functions and one accumulator: no filesystem, no subprocess, no ffmpeg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

# Amplitude below which a frame is treated as silence rather than signal.
# Silence correlates arbitrarily, so including it would swamp the measurement.
DEFAULT_ACTIVITY_FLOOR: Final = 0.002

# A sample at or above this fraction of full scale counts as clipped.
CLIPPING_CEILING: Final = 0.999

FLAG_IDENTICAL: Final = "stereo_identical"
FLAG_CORRELATED: Final = "stereo_correlated"
FLAG_PARTIAL_SEPARATION: Final = "stereo_partial_separation"
FLAG_STRONG_SEPARATION: Final = "stereo_strong_separation"
FLAG_CHANNEL_IMBALANCE: Final = "stereo_channel_imbalance"
FLAG_CLIPPING: Final = "audio_clipping"
FLAG_NOT_STEREO: Final = "audio_not_stereo"
FLAG_NO_ACTIVE_AUDIO: Final = "audio_no_active_signal"
FLAG_TRUNCATED: Final = "audio_truncated"
FLAG_DECODE_WARNINGS: Final = "audio_decode_warnings"

_STEREO_CHANNELS: Final = 2
# Correlation needs at least two frames to have any variance to divide by.
_MIN_FRAMES_FOR_CORRELATION: Final = 2


@dataclass(frozen=True, slots=True)
class StereoStats:
    """What one recording's two channels look like relative to each other.

    Attributes:
        n_samples: Frames seen, per channel.
        n_active: Frames loud enough to be treated as signal.
        correlation: Pearson correlation between the channels over active
            frames, in [-1, 1]. None when there was nothing active to measure.
        rms_left: Root-mean-square amplitude of the left channel.
        rms_right: Root-mean-square amplitude of the right channel.
        ild_db: Interaural level difference, `20*log10(rms_left/rms_right)`.
            Positive means the left channel is louder. None when either channel
            is silent.
        peak_left: Largest absolute sample in the left channel.
        peak_right: Largest absolute sample in the right channel.
        bit_identical: Whether the two channels were sample-for-sample equal.
            Definitive when true, but lossy encoding usually prevents it even
            for a mono source, so `correlation` is the signal to trust.
    """

    n_samples: int
    n_active: int
    correlation: float | None
    rms_left: float
    rms_right: float
    ild_db: float | None
    peak_left: float
    peak_right: float
    bit_identical: bool

    @property
    def active_fraction(self) -> float:
        """Fraction of frames that carried signal."""
        return self.n_active / self.n_samples if self.n_samples else 0.0

    @property
    def separation(self) -> float | None:
        """How far the channels are from identical, as `1 - correlation`.

        Zero for a mono source duplicated across both channels; larger where the
        channels genuinely differ.
        """
        return None if self.correlation is None else 1.0 - self.correlation


class StereoAccumulator:
    """Accumulates stereo statistics over successive chunks.

    Sums are kept in float64 while chunks arrive as int16 or float32, so long
    sessions do not lose precision.

    Args:
        activity_floor: Amplitude below which a frame counts as silence. A frame
            is active when either channel reaches it.
    """

    def __init__(self, *, activity_floor: float = DEFAULT_ACTIVITY_FLOOR) -> None:
        """Start an empty accumulator."""
        self.activity_floor = activity_floor
        self._n = 0
        self._n_active = 0
        self._sum_l = 0.0
        self._sum_r = 0.0
        self._sum_ll = 0.0
        self._sum_rr = 0.0
        self._sum_lr = 0.0
        self._sq_l = 0.0
        self._sq_r = 0.0
        self._peak_l = 0.0
        self._peak_r = 0.0
        self._identical = True

    def update(self, chunk: np.ndarray) -> None:
        """Add one chunk of interleaved-then-split stereo samples.

        Args:
            chunk: Array shaped `(n, 2)` of samples in [-1, 1].

        Raises:
            ValueError: if the chunk is not two-channel.
        """
        if chunk.ndim != _STEREO_CHANNELS or chunk.shape[1] != _STEREO_CHANNELS:
            msg = f"expected a (n, 2) stereo chunk, got shape {chunk.shape}"
            raise ValueError(msg)
        if chunk.shape[0] == 0:
            return

        data = chunk.astype(np.float64, copy=False)
        left, right = data[:, 0], data[:, 1]

        self._n += left.size
        # RMS and peaks cover the whole recording, not just the active part.
        self._sq_l += float(np.dot(left, left))
        self._sq_r += float(np.dot(right, right))
        self._peak_l = max(self._peak_l, float(np.max(np.abs(left))))
        self._peak_r = max(self._peak_r, float(np.max(np.abs(right))))
        if self._identical and not np.array_equal(left, right):
            self._identical = False

        # Correlation is measured over active frames only: silence carries no
        # panning information and would dominate a 12-minute recording.
        active = (np.abs(left) >= self.activity_floor) | (np.abs(right) >= self.activity_floor)
        if not active.any():
            return
        a_left, a_right = left[active], right[active]
        self._n_active += a_left.size
        self._sum_l += float(a_left.sum())
        self._sum_r += float(a_right.sum())
        self._sum_ll += float(np.dot(a_left, a_left))
        self._sum_rr += float(np.dot(a_right, a_right))
        self._sum_lr += float(np.dot(a_left, a_right))

    def result(self) -> StereoStats:
        """Finalise the accumulated sums into statistics."""
        n, n_active = self._n, self._n_active

        correlation: float | None = None
        if n_active >= _MIN_FRAMES_FOR_CORRELATION:
            mean_l = self._sum_l / n_active
            mean_r = self._sum_r / n_active
            cov = self._sum_lr / n_active - mean_l * mean_r
            var_l = self._sum_ll / n_active - mean_l * mean_l
            var_r = self._sum_rr / n_active - mean_r * mean_r
            if var_l > 0.0 and var_r > 0.0:
                correlation = float(min(1.0, max(-1.0, cov / math.sqrt(var_l * var_r))))

        rms_l = math.sqrt(self._sq_l / n) if n else 0.0
        rms_r = math.sqrt(self._sq_r / n) if n else 0.0
        ild = 20.0 * math.log10(rms_l / rms_r) if rms_l > 0.0 and rms_r > 0.0 else None

        return StereoStats(
            n_samples=n,
            n_active=n_active,
            correlation=correlation,
            rms_left=rms_l,
            rms_right=rms_r,
            ild_db=ild,
            peak_left=self._peak_l,
            peak_right=self._peak_r,
            bit_identical=self._identical and n > 0,
        )


def downmix_to_mono(chunk: np.ndarray) -> np.ndarray:
    """Average a stereo chunk to mono, matching ffmpeg's `-ac 1` downmix."""
    if chunk.ndim == 1:
        return chunk
    mono: np.ndarray = chunk.mean(axis=1)
    return mono


def stereo_flags(
    stats: StereoStats,
    *,
    correlated_above: float,
    strong_separation_below: float,
    imbalance_db: float,
) -> list[str]:
    """Describe a recording's channel layout as QC flags.

    Args:
        stats: Accumulated statistics.
        correlated_above: Correlation at or above which the channels are treated
            as carrying the same signal, so no separation is available.
        strong_separation_below: Correlation below which separation is strong
            enough to be worth exploiting.
        imbalance_db: Absolute interaural level difference worth flagging.

    Returns:
        Flags, in a stable order.
    """
    flags: list[str] = []

    if stats.n_samples == 0:
        return [FLAG_NO_ACTIVE_AUDIO]
    if stats.correlation is None:
        flags.append(FLAG_NO_ACTIVE_AUDIO)
    else:
        if stats.bit_identical:
            flags.append(FLAG_IDENTICAL)
        if stats.correlation >= correlated_above:
            flags.append(FLAG_CORRELATED)
        else:
            flags.append(FLAG_PARTIAL_SEPARATION)
            if stats.correlation < strong_separation_below:
                flags.append(FLAG_STRONG_SEPARATION)

    if stats.ild_db is not None and abs(stats.ild_db) >= imbalance_db:
        flags.append(FLAG_CHANNEL_IMBALANCE)
    if max(stats.peak_left, stats.peak_right) >= CLIPPING_CEILING:
        flags.append(FLAG_CLIPPING)

    return flags
