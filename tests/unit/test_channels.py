"""Stereo channel statistics.

The streaming accumulator has to give exactly what a whole-array computation
would, since it exists only to bound memory. That equivalence is asserted
directly, against numpy, at several chunk sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from vc_multimodal.features.channels import (
    CLIPPING_CEILING,
    DEFAULT_ACTIVITY_FLOOR,
    FLAG_CHANNEL_IMBALANCE,
    FLAG_CLIPPING,
    FLAG_CORRELATED,
    FLAG_IDENTICAL,
    FLAG_NO_ACTIVE_AUDIO,
    FLAG_PARTIAL_SEPARATION,
    FLAG_STRONG_SEPARATION,
    StereoAccumulator,
    StereoStats,
    downmix_to_mono,
    stereo_flags,
)

SAMPLE_RATE = 16000
FLAG_KWARGS = {
    "correlated_above": 0.98,
    "strong_separation_below": 0.5,
    "imbalance_db": 3.0,
}


def _tone(frequency: float, seconds: float = 2.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def _accumulate(
    left: np.ndarray, right: np.ndarray, *, chunk: int = 4096, floor: float | None = None
) -> StereoStats:
    accumulator = StereoAccumulator(
        activity_floor=DEFAULT_ACTIVITY_FLOOR if floor is None else floor
    )
    data = np.stack([left, right], axis=1)
    for start in range(0, len(data), chunk):
        accumulator.update(data[start : start + chunk])
    return accumulator.result()


def _numpy_correlation(left: np.ndarray, right: np.ndarray, floor: float) -> float:
    active = (np.abs(left) >= floor) | (np.abs(right) >= floor)
    return float(np.corrcoef(left[active], right[active])[0, 1])


# ---------------------------------------------------------------------------
# the cases the probe exists to tell apart
# ---------------------------------------------------------------------------
def test_a_mono_source_duplicated_across_channels_correlates_perfectly():
    signal = _tone(140.0)
    stats = _accumulate(signal, signal.copy())
    assert stats.correlation == pytest.approx(1.0)
    assert stats.bit_identical
    assert stats.separation == pytest.approx(0.0)


def test_two_speakers_hard_panned_are_uncorrelated():
    """The best case: one speaker per channel would be complete separation."""
    stats = _accumulate(_tone(140.0), _tone(230.0))
    assert abs(stats.correlation or 1.0) < 0.05
    assert not stats.bit_identical


def test_partial_panning_lands_between_the_two():
    """What Zoom panning would actually look like."""
    left_voice, right_voice = _tone(140.0), _tone(230.0)
    stats = _accumulate(0.7 * left_voice + 0.3 * right_voice, 0.3 * left_voice + 0.7 * right_voice)
    assert stats.correlation is not None
    assert 0.3 < stats.correlation < 0.95


def test_an_inverted_channel_correlates_negatively():
    signal = _tone(140.0)
    stats = _accumulate(signal, -signal)
    assert stats.correlation == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# the accumulator must match a whole-array computation exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk", [1, 7, 997, 4096, 100_000])
def test_streaming_matches_numpy_at_any_chunk_size(chunk: int):
    left_voice, right_voice = _tone(140.0), _tone(230.0)
    left = 0.7 * left_voice + 0.3 * right_voice
    right = 0.3 * left_voice + 0.7 * right_voice

    streamed = _accumulate(left, right, chunk=chunk).correlation
    assert streamed == pytest.approx(
        _numpy_correlation(left, right, DEFAULT_ACTIVITY_FLOOR), abs=1e-9
    )


def test_chunk_size_does_not_change_the_result():
    left, right = _tone(140.0), 0.5 * _tone(140.0) + 0.5 * _tone(230.0)
    results = {_accumulate(left, right, chunk=size).correlation for size in (64, 1024, 65536)}
    assert max(results) - min(results) < 1e-9  # type: ignore[type-var]


def test_rms_matches_a_direct_computation():
    left, right = _tone(140.0, amplitude=0.4), _tone(230.0, amplitude=0.1)
    stats = _accumulate(left, right)
    assert stats.rms_left == pytest.approx(float(np.sqrt(np.mean(left**2))), rel=1e-9)
    assert stats.rms_right == pytest.approx(float(np.sqrt(np.mean(right**2))), rel=1e-9)


# ---------------------------------------------------------------------------
# silence handling
# ---------------------------------------------------------------------------
def test_silence_is_excluded_from_the_correlation():
    """Most of a session is silence, which correlates arbitrarily."""
    speech = _tone(140.0, seconds=1.0)
    silence = np.zeros(SAMPLE_RATE * 5)
    left = np.concatenate([speech, silence])
    right = np.concatenate([speech.copy(), silence])

    stats = _accumulate(left, right)
    assert stats.correlation == pytest.approx(1.0)
    assert stats.n_active == pytest.approx(speech.size, rel=0.05)
    assert stats.active_fraction < 0.25


def test_noise_below_the_floor_does_not_count_as_active():
    rng = np.random.default_rng(0)
    quiet = rng.normal(0.0, 0.0001, size=SAMPLE_RATE)
    stats = _accumulate(quiet, quiet.copy())
    assert stats.n_active == 0
    assert stats.correlation is None


def test_a_fully_silent_recording_has_no_correlation_and_is_flagged():
    silence = np.zeros(SAMPLE_RATE)
    stats = _accumulate(silence, silence)
    assert stats.correlation is None
    assert FLAG_NO_ACTIVE_AUDIO in stereo_flags(stats, **FLAG_KWARGS)


def test_one_silent_channel_yields_no_correlation():
    """A constant channel has no variance, so correlation is undefined."""
    stats = _accumulate(_tone(140.0), np.zeros(SAMPLE_RATE * 2))
    assert stats.correlation is None
    assert stats.rms_right == 0.0
    assert stats.ild_db is None


def test_no_samples_at_all():
    stats = StereoAccumulator().result()
    assert stats.n_samples == 0
    assert stats.correlation is None
    assert not stats.bit_identical
    assert stats.active_fraction == 0.0
    assert stats.separation is None
    assert stereo_flags(stats, **FLAG_KWARGS) == [FLAG_NO_ACTIVE_AUDIO]


# ---------------------------------------------------------------------------
# level difference
# ---------------------------------------------------------------------------
def test_a_louder_left_channel_gives_a_positive_ild():
    stats = _accumulate(_tone(140.0, amplitude=0.4), _tone(140.0, amplitude=0.2))
    assert stats.ild_db == pytest.approx(20.0 * np.log10(2.0), abs=0.01)


def test_balanced_channels_give_a_zero_ild():
    signal = _tone(140.0)
    assert _accumulate(signal, signal.copy()).ild_db == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(10),
        np.zeros((10, 1)),
        np.zeros((10, 3)),
        np.zeros((2, 10, 2)),
    ],
)
def test_a_non_stereo_chunk_is_rejected(bad: np.ndarray):
    with pytest.raises(ValueError, match="stereo chunk"):
        StereoAccumulator().update(bad)


def test_an_empty_chunk_is_ignored():
    accumulator = StereoAccumulator()
    accumulator.update(np.zeros((0, 2)))
    assert accumulator.result().n_samples == 0


# ---------------------------------------------------------------------------
# downmix
# ---------------------------------------------------------------------------
def test_downmix_averages_the_channels():
    left, right = _tone(140.0), _tone(230.0)
    mono = downmix_to_mono(np.stack([left, right], axis=1))
    assert mono.shape == left.shape
    assert mono == pytest.approx((left + right) / 2.0)


def test_downmixing_mono_is_a_no_op():
    signal = _tone(140.0)
    assert downmix_to_mono(signal) is signal


# ---------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------
def _stats(**overrides: object) -> StereoStats:
    base: dict[str, object] = {
        "n_samples": 1000,
        "n_active": 800,
        "correlation": 1.0,
        "rms_left": 0.1,
        "rms_right": 0.1,
        "ild_db": 0.0,
        "peak_left": 0.5,
        "peak_right": 0.5,
        "bit_identical": False,
    }
    base.update(overrides)
    return StereoStats(**base)  # type: ignore[arg-type]


def test_identical_channels_are_flagged_as_both_identical_and_correlated():
    flags = stereo_flags(_stats(bit_identical=True), **FLAG_KWARGS)
    assert FLAG_IDENTICAL in flags
    assert FLAG_CORRELATED in flags
    assert FLAG_PARTIAL_SEPARATION not in flags


def test_a_correlation_just_below_the_threshold_counts_as_separation():
    flags = stereo_flags(_stats(correlation=0.979), **FLAG_KWARGS)
    assert FLAG_PARTIAL_SEPARATION in flags
    assert FLAG_STRONG_SEPARATION not in flags


def test_a_low_correlation_is_flagged_as_strong_separation():
    flags = stereo_flags(_stats(correlation=0.1), **FLAG_KWARGS)
    assert FLAG_PARTIAL_SEPARATION in flags
    assert FLAG_STRONG_SEPARATION in flags


def test_the_threshold_is_inclusive():
    assert FLAG_CORRELATED in stereo_flags(_stats(correlation=0.98), **FLAG_KWARGS)


def test_an_imbalanced_pair_of_channels_is_flagged():
    assert FLAG_CHANNEL_IMBALANCE in stereo_flags(_stats(ild_db=-6.0), **FLAG_KWARGS)
    assert FLAG_CHANNEL_IMBALANCE not in stereo_flags(_stats(ild_db=-1.0), **FLAG_KWARGS)


def test_clipping_is_flagged():
    assert FLAG_CLIPPING in stereo_flags(_stats(peak_right=CLIPPING_CEILING), **FLAG_KWARGS)
    assert FLAG_CLIPPING not in stereo_flags(_stats(peak_right=0.9), **FLAG_KWARGS)


def test_flags_are_stable_in_order():
    stats = _stats(correlation=0.2, ild_db=-9.0, peak_left=1.0, bit_identical=False)
    assert stereo_flags(stats, **FLAG_KWARGS) == [
        FLAG_PARTIAL_SEPARATION,
        FLAG_STRONG_SEPARATION,
        FLAG_CHANNEL_IMBALANCE,
        FLAG_CLIPPING,
    ]
