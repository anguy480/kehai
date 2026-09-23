"""Prosodic arithmetic.

The decisions worth testing directly are the semitone normalisation, the choice
to pool over frames rather than average over spans, and the rule that a missing
measure is None rather than zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from vc_multimodal.features.prosody_math import (
    FEATURE_NAMES,
    SpanMeasures,
    defined_harmonicity,
    iqr_or_none,
    mean_absolute_delta,
    mean_or_none,
    median_or_none,
    percentile_range,
    pool,
    prosody_features,
    sd_or_none,
    syllable_nuclei,
    to_semitones,
    voiced_frequencies,
    weighted_mean,
)


def measures(
    *,
    duration: float = 5.0,
    f0: list[float] | None = None,
    intensity: list[float] | None = None,
    harmonicity: list[float] | None = None,
    jitter: float | None = 0.01,
    shimmer: float | None = 0.05,
    step: float = 0.01,
) -> SpanMeasures:
    return SpanMeasures(
        duration_s=duration,
        f0_hz=np.array(f0 if f0 is not None else [150.0] * 100, dtype=np.float64),
        intensity_db=np.array(
            intensity if intensity is not None else [70.0] * 100, dtype=np.float64
        ),
        harmonicity_db=np.array(
            harmonicity if harmonicity is not None else [15.0] * 100, dtype=np.float64
        ),
        intensity_frame_step_s=step,
        jitter_local=jitter,
        shimmer_local=shimmer,
    )


# ---------------------------------------------------------------------------
# semitone normalisation
# ---------------------------------------------------------------------------
def test_an_octave_is_twelve_semitones():
    assert to_semitones([300.0], 150.0)[0] == pytest.approx(12.0)
    assert to_semitones([75.0], 150.0)[0] == pytest.approx(-12.0)


def test_the_reference_itself_is_zero():
    assert to_semitones([150.0], 150.0)[0] == pytest.approx(0.0)


def test_two_speakers_an_octave_apart_get_identical_variability():
    """The whole point: pitch range must not encode speaker sex.

    One speaker centred at 110 Hz and another at 220 Hz, each varying by the
    same musical interval, must come out the same.
    """
    low = np.array([110.0, 110.0 * 2 ** (3 / 12), 110.0 * 2 ** (-3 / 12)])
    high = low * 2.0

    low_st = to_semitones(low, float(np.median(low)))
    high_st = to_semitones(high, float(np.median(high)))

    assert sd_or_none(low_st) == pytest.approx(sd_or_none(high_st))
    # In hertz they would differ by a factor of two.
    assert sd_or_none(low) != pytest.approx(sd_or_none(high))


def test_semitones_are_perceptually_linear_where_hertz_is_not():
    """A fixed interval is a fixed number of semitones at any pitch."""
    step_low = to_semitones([120.0], 100.0)[0]
    step_high = to_semitones([360.0], 300.0)[0]
    assert step_low == pytest.approx(step_high)


def test_unvoiced_and_impossible_frequencies_are_dropped_not_clamped():
    """Clamping would invent a pitch that was never measured."""
    result = to_semitones([150.0, 0.0, -5.0, np.nan, 300.0], 150.0)
    assert result.size == 2
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(12.0)


def test_an_all_unvoiced_contour_yields_nothing():
    assert to_semitones([0.0, 0.0], 150.0).size == 0


def test_a_non_positive_reference_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        to_semitones([150.0], 0.0)


def test_voiced_frames_are_selected_by_praats_sentinel():
    assert voiced_frequencies([0.0, 150.0, 0.0, 200.0]).tolist() == [150.0, 200.0]


def test_undefined_harmonicity_is_dropped():
    assert defined_harmonicity([-200.0, 15.0, -300.0]).tolist() == [15.0]


# ---------------------------------------------------------------------------
# robust statistics
# ---------------------------------------------------------------------------
def test_the_percentile_range_ignores_a_tracking_error():
    """Max minus min on a pitch contour usually reports one octave jump."""
    clean = [0.0] * 50 + [2.0] * 50
    with_glitch = [*clean, 40.0]
    assert percentile_range(with_glitch) == pytest.approx(percentile_range(clean), abs=0.6)
    assert max(with_glitch) - min(with_glitch) == pytest.approx(40.0)


def test_the_iqr_is_the_middle_half():
    assert iqr_or_none([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(2.0)


def test_mean_absolute_delta_separates_movement_from_spread():
    """A slow drift and a lively contour can share a standard deviation.

    A uniform ramp over [-3, 3] has an SD of 6/sqrt(12); a two-level
    alternation matches it at that amplitude, so the two contours differ only
    in how fast they move.
    """
    amplitude = 6.0 / np.sqrt(12.0)
    drift = np.linspace(-3.0, 3.0, 200)
    lively = np.tile([-amplitude, amplitude], 100)

    assert sd_or_none(drift) == pytest.approx(sd_or_none(lively), rel=0.02)
    assert mean_absolute_delta(lively) > 50 * (mean_absolute_delta(drift) or 0.0)


@pytest.mark.parametrize(
    "statistic",
    [median_or_none, mean_or_none, sd_or_none, iqr_or_none, percentile_range, mean_absolute_delta],
)
def test_every_statistic_of_nothing_is_none(statistic):
    assert statistic([]) is None


def test_spread_of_a_single_value_is_none_not_zero():
    """Zero spread is a claim; one observation cannot support it."""
    assert sd_or_none([5.0]) is None
    assert iqr_or_none([5.0]) is None
    assert mean_absolute_delta([5.0]) is None
    # A location statistic is fine from one value.
    assert median_or_none([5.0]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# weighted pooling
# ---------------------------------------------------------------------------
def test_a_long_utterance_counts_for_more_than_a_brief_one():
    assert weighted_mean([0.0, 1.0], [9.0, 1.0]) == pytest.approx(0.1)


def test_missing_values_are_skipped_not_treated_as_zero():
    assert weighted_mean([None, 4.0], [5.0, 5.0]) == pytest.approx(4.0)


def test_all_missing_values_pool_to_none():
    assert weighted_mean([None, None], [1.0, 1.0]) is None


def test_zero_weights_are_ignored():
    assert weighted_mean([1.0, 99.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_frames_are_pooled_across_spans():
    pooled = pool([measures(f0=[100.0] * 10), measures(f0=[200.0] * 30)])
    assert pooled["f0_hz"].size == 40
    # The median follows the frames, not the span count: 30 of 40 are 200.
    assert float(np.median(pooled["f0_hz"])) == pytest.approx(200.0)


def test_pooling_nothing():
    pooled = pool([])
    assert pooled["f0_hz"].size == 0


# ---------------------------------------------------------------------------
# the speech rate proxy
# ---------------------------------------------------------------------------
def test_intensity_peaks_are_counted():
    contour = np.tile([50.0, 70.0], 20)  # twenty peaks
    assert syllable_nuclei(contour, frame_step_s=0.05, min_spacing_s=0.05) == pytest.approx(
        20, abs=1
    )


def test_quiet_peaks_below_the_threshold_are_ignored():
    contour = np.array([80.0, 20.0, 30.0, 20.0, 80.0])
    assert syllable_nuclei(contour, frame_step_s=0.05, threshold_db_below_max=25.0) == 0


def test_peaks_closer_than_a_syllable_are_merged():
    contour = np.array([50.0, 70.0, 50.0, 72.0, 50.0])
    one = syllable_nuclei(contour, frame_step_s=0.05, min_spacing_s=0.5)
    two = syllable_nuclei(contour, frame_step_s=0.05, min_spacing_s=0.05)
    assert one == 1
    assert two == 2


def test_a_flat_contour_has_no_peaks():
    """Otherwise a sustained loud passage reads as a stream of syllables."""
    assert syllable_nuclei(np.full(50, 70.0), frame_step_s=0.01) == 0


def test_a_plateau_counts_once_at_its_onset():
    contour = np.array([50.0, 70.0, 70.0, 70.0, 70.0, 50.0])
    assert syllable_nuclei(contour, frame_step_s=0.05, min_spacing_s=0.05) == 1


def test_too_short_a_contour_has_no_peaks():
    assert syllable_nuclei([70.0, 80.0], frame_step_s=0.01) == 0


def test_a_zero_frame_step_is_handled():
    assert syllable_nuclei(np.tile([50.0, 70.0], 10), frame_step_s=0.0) == 0


# ---------------------------------------------------------------------------
# the feature set
# ---------------------------------------------------------------------------
def test_the_feature_set_is_small_and_named_by_convention():
    assert len(FEATURE_NAMES) <= 14
    assert all(name.startswith("prosody__") for name in FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_every_feature_is_produced_for_a_normal_session():
    features = prosody_features([measures(), measures()])
    assert set(features) == set(FEATURE_NAMES)
    assert all(value is not None for value in features.values())


def test_the_semitone_reference_is_the_speakers_own_median():
    """So the features describe variation, not which speaker it is."""
    quiet_low = prosody_features([measures(f0=[100.0, 105.0, 95.0] * 10)])
    same_but_higher = prosody_features([measures(f0=[200.0, 210.0, 190.0] * 10)])
    assert quiet_low["prosody__f0_semitone_sd"] == pytest.approx(
        same_but_higher["prosody__f0_semitone_sd"], rel=1e-6
    )


def test_the_voiced_fraction_reflects_unvoiced_frames():
    features = prosody_features([measures(f0=[150.0] * 60 + [0.0] * 40)])
    assert features["prosody__voiced_fraction"] == pytest.approx(0.6)


def test_features_are_none_rather_than_zero_when_unmeasurable():
    features = prosody_features([])
    assert all(value is None for value in features.values())


def test_an_entirely_unvoiced_session_has_no_pitch_features():
    features = prosody_features([measures(f0=[0.0] * 100)])
    assert features["prosody__f0_semitone_sd"] is None
    assert features["prosody__voiced_fraction"] == pytest.approx(0.0)
    # Intensity does not need voicing, so it survives.
    assert features["prosody__intensity_mean_db"] is not None


def test_a_span_with_no_jitter_measurement_does_not_zero_the_session():
    features = prosody_features([measures(jitter=None), measures(jitter=0.02)])
    assert features["prosody__jitter_local"] == pytest.approx(0.02)


def test_the_speech_rate_is_per_second_of_analysed_speech():
    ten_peaks = np.tile([50.0, 70.0], 5)
    features = prosody_features([measures(duration=1.0, intensity=list(ten_peaks), step=0.1)])
    assert features["prosody__speech_rate_proxy"] == pytest.approx(5.0, abs=1.0)
