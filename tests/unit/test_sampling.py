"""Frame sampling arithmetic.

Every recording is a constant 25 fps. A sample rate that does not divide that
exactly cannot be honoured, and rounding it silently would give uneven frame
spacing with nothing to notice, so it is rejected instead.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from vc_multimodal.features.sampling import (
    SamplingError,
    resolve_sampling,
)


# ---------------------------------------------------------------------------
# rates that divide the native rate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("native", "requested", "step"),
    [
        (25.0, 5.0, 5),  # the configured default
        (25.0, 12.5, 2),  # the mouth cross-check rate
        (25.0, 25.0, 1),
        (25.0, 2.5, 10),
        (25.0, 1.0, 25),
        (30.0, 10.0, 3),
        (24.0, 8.0, 3),
    ],
)
def test_an_exact_divisor_resolves_to_that_step(native: float, requested: float, step: int):
    sampling = resolve_sampling(native, requested)
    assert sampling.step == step
    assert sampling.effective_fps == pytest.approx(requested)


def test_the_effective_rate_is_what_was_asked_for():
    """QC records this, so it must never differ from the request."""
    sampling = resolve_sampling(25.0, 5.0)
    assert sampling.effective_fps == sampling.requested_fps
    assert sampling.native_fps == 25.0


def test_a_broadcast_rate_expressed_as_a_ratio_is_accepted():
    """29.97 fps is 30000/1001; an exact float test would reject it."""
    native = 30000.0 / 1001.0
    sampling = resolve_sampling(native, native / 3.0)
    assert sampling.step == 3


# ---------------------------------------------------------------------------
# rates that do not
# ---------------------------------------------------------------------------
def test_ten_fps_from_twenty_five_is_rejected():
    """The exact case that prompted this: it alternates 2- and 3-frame steps."""
    with pytest.raises(SamplingError, match="does not divide"):
        resolve_sampling(25.0, 10.0)


def test_the_rejection_explains_itself_and_offers_valid_rates():
    with pytest.raises(SamplingError) as caught:
        resolve_sampling(25.0, 10.0)
    message = str(caught.value)
    assert "2.5000th frame" in message
    assert "alternate" in message
    assert "5.0" in message  # a suggested exact rate
    assert "12.5" in message


@pytest.mark.parametrize("requested", [7.0, 9.0, 11.0, 13.0, 24.0])
def test_other_non_divisors_are_rejected(requested: float):
    with pytest.raises(SamplingError, match="does not divide"):
        resolve_sampling(25.0, requested)


def test_a_rate_above_the_native_rate_is_rejected():
    with pytest.raises(SamplingError, match="not that many frames"):
        resolve_sampling(25.0, 30.0)


@pytest.mark.parametrize("requested", [0.0, -1.0])
def test_a_non_positive_rate_is_rejected(requested: float):
    with pytest.raises(SamplingError, match="must be positive"):
        resolve_sampling(25.0, requested)


@pytest.mark.parametrize("native", [None, 0.0, -25.0])
def test_an_unknown_native_rate_is_an_error_not_a_guess(native: float | None):
    """Guessing a step would produce timestamps that disagree with the video."""
    with pytest.raises(SamplingError, match="without the recording's frame rate"):
        resolve_sampling(native, 5.0)


def test_the_unknown_rate_error_points_at_inventory():
    with pytest.raises(SamplingError) as caught:
        resolve_sampling(None, 5.0)
    assert "vc inventory" in str(caught.value)
    assert "variable-frame-rate" in str(caught.value)


# ---------------------------------------------------------------------------
# using a resolved plan
# ---------------------------------------------------------------------------
def test_frame_indices_are_evenly_spaced():
    sampling = resolve_sampling(25.0, 5.0)
    indices = list(sampling.frame_indices(26))
    assert indices == [0, 5, 10, 15, 20, 25]
    gaps = {b - a for a, b in pairwise(indices)}
    assert gaps == {5}


def test_the_sampled_count_matches_the_indices():
    sampling = resolve_sampling(25.0, 5.0)
    assert sampling.n_sampled(100) == len(list(sampling.frame_indices(100)))


def test_an_empty_or_negative_recording_samples_nothing():
    sampling = resolve_sampling(25.0, 5.0)
    assert sampling.n_sampled(0) == 0
    assert sampling.n_sampled(-10) == 0


def test_timestamps_follow_the_native_rate_not_the_sample_rate():
    """A sampled frame's time is where it sits in the original recording."""
    sampling = resolve_sampling(25.0, 5.0)
    assert sampling.timestamp(0) == 0.0
    assert sampling.timestamp(25) == pytest.approx(1.0)
    assert sampling.timestamp(5) == pytest.approx(0.2)


def test_a_twelve_minute_recording_at_five_fps():
    """The real shape of the work: about 3,500 frames per session."""
    sampling = resolve_sampling(25.0, 5.0)
    n_frames = int(25.0 * 60 * 11.6)
    assert sampling.n_sampled(n_frames) == pytest.approx(3480, abs=2)


# ---------------------------------------------------------------------------
# the shipped configuration must satisfy its own rule
# ---------------------------------------------------------------------------
def test_the_configured_rates_divide_the_real_frame_rate(default_config):
    """All 62 recordings are a constant 25 fps, as `vc inventory` confirmed.

    Both configured rates take every 5th frame. The mouth cross-check was
    halved to this rate once the correlation it measures was clear: a mouth
    against a speech timeline made of multi-second turns, not against
    individual syllables, so the extra frames bought nothing and cost a
    landmarker call each.
    """
    native = 25.0
    assert resolve_sampling(native, default_config.face.sample_fps).step == 5
    assert resolve_sampling(native, default_config.speakers.mouth_crosscheck.sample_fps).step == 5
