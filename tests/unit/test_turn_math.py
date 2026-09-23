"""Turn-taking and response-latency mathematics.

This is the arithmetic the project's audio findings will rest on, so it is
tested against examples worked out by hand, with the awkward cases named
explicitly: interruptions, long silences, a speaker who never stops, and a
session in which one role never speaks at all.

The scenarios use round numbers so every expected value can be read off.
"""

from __future__ import annotations

import pytest

from vc_multimodal.features.spans import Span, overlap_duration
from vc_multimodal.features.turn_math import (
    FEATURE_NAMES,
    ROLE_PARTICIPANT,
    ROLE_PSYCHIATRIST,
    build_turns,
    count_interruptions,
    overlap_ratio,
    response_latencies,
    speaking_ratio,
    speaking_timeline,
    turn_features,
    usable_latencies,
    within_turn_pauses,
)

P = ROLE_PARTICIPANT
D = ROLE_PSYCHIATRIST


def spans(*pairs: tuple[float, float]) -> list[Span]:
    return [Span(start, end) for start, end in pairs]


def clean_exchange() -> dict[str, list[Span]]:
    """Four turns, alternating, with exactly 0.5 s between each."""
    return {
        D: spans((0.5, 2.0), (4.5, 6.0)),
        P: spans((2.5, 4.0), (6.5, 8.0)),
    }


# ---------------------------------------------------------------------------
# building turns
# ---------------------------------------------------------------------------
def test_alternating_speech_becomes_alternating_turns():
    turns = build_turns(clean_exchange())
    assert [turn.role for turn in turns] == [D, P, D, P]
    assert [turn.index for turn in turns] == [0, 1, 2, 3]
    assert turns[0].start == pytest.approx(0.5)
    assert turns[0].end == pytest.approx(2.0)


def test_consecutive_spans_by_one_role_form_a_single_turn():
    """Two utterances with nobody else between them are one turn, not two."""
    turns = build_turns({D: spans((0.0, 1.0), (2.0, 3.0)), P: spans((5.0, 6.0))})
    assert len(turns) == 2
    assert turns[0].role == D
    assert turns[0].start == pytest.approx(0.0)
    assert turns[0].end == pytest.approx(3.0)
    assert turns[0].n_spans == 2


def test_a_short_gap_is_bridged_before_turns_are_built():
    """A breath must not end a turn."""
    by_role = {P: spans((0.0, 1.0), (1.2, 2.0))}
    assert len(build_turns(by_role, merge_gap=0.3)) == 1
    # Without bridging they are still one turn, since nobody else spoke.
    assert build_turns(by_role, merge_gap=0.0)[0].n_spans == 2


def test_the_other_role_speaking_ends_a_turn():
    turns = build_turns({P: spans((0.0, 1.0), (3.0, 4.0)), D: spans((1.5, 2.5))})
    assert [turn.role for turn in turns] == [P, D, P]


def test_a_session_where_only_one_person_speaks():
    turns = build_turns({P: spans((0.0, 5.0))})
    assert len(turns) == 1
    assert turns[0].role == P


def test_no_speech_makes_no_turns():
    assert build_turns({}) == ()
    assert build_turns({P: [], D: []}) == ()


def test_turn_duration():
    assert build_turns({P: spans((1.0, 4.0))})[0].duration == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# response latency
# ---------------------------------------------------------------------------
def test_latency_is_the_gap_from_the_psychiatrist_to_the_participant():
    latencies = response_latencies(build_turns(clean_exchange()))
    assert [pytest.approx(x.seconds) for x in latencies] == [0.5, 0.5]


def test_latency_is_measured_only_in_that_direction():
    """A psychiatrist turn following a participant turn is not a response."""
    latencies = response_latencies(build_turns(clean_exchange()))
    assert len(latencies) == 2  # not 3, though there are three transitions


def test_the_first_turn_has_no_latency():
    latencies = response_latencies(build_turns({P: spans((5.0, 6.0))}))
    assert latencies == ()


def test_an_interruption_gives_a_negative_latency():
    """The participant began before the psychiatrist finished."""
    latencies = response_latencies(build_turns({D: spans((0.0, 3.0)), P: spans((2.5, 6.0))}))
    assert latencies[0].seconds == pytest.approx(-0.5)
    assert latencies[0].is_interruption


def test_interruptions_are_counted_not_averaged_into_response_times():
    latencies = response_latencies(build_turns({D: spans((0.0, 3.0)), P: spans((2.5, 6.0))}))
    assert count_interruptions(latencies) == 1
    assert usable_latencies(latencies, max_latency_s=10.0) == ()


def test_a_long_silence_is_not_a_response():
    """After 30 s the participant is starting something, not answering."""
    latencies = response_latencies(build_turns({D: spans((0.0, 1.0)), P: spans((31.0, 32.0))}))
    assert latencies[0].seconds == pytest.approx(30.0)
    assert usable_latencies(latencies, max_latency_s=10.0) == ()


def test_a_latency_exactly_at_the_limit_is_kept():
    latencies = response_latencies(build_turns({D: spans((0.0, 1.0)), P: spans((11.0, 12.0))}))
    assert usable_latencies(latencies, max_latency_s=10.0) == (pytest.approx(10.0),)


def test_a_zero_latency_is_a_response_not_an_interruption():
    latencies = response_latencies(build_turns({D: spans((0.0, 1.0)), P: spans((1.0, 2.0))}))
    assert count_interruptions(latencies) == 0
    assert usable_latencies(latencies, max_latency_s=10.0) == (pytest.approx(0.0),)


def test_latencies_carry_the_turn_they_belong_to():
    latencies = response_latencies(build_turns(clean_exchange()))
    assert [x.turn_index for x in latencies] == [1, 3]


# ---------------------------------------------------------------------------
# within-turn pauses
# ---------------------------------------------------------------------------
def test_a_pause_inside_a_turn_is_measured():
    by_role = {P: spans((0.0, 1.0), (1.5, 2.5)), D: spans((5.0, 6.0))}
    turns = build_turns(by_role, merge_gap=0.3)
    pauses = within_turn_pauses(turns, by_role[P], role=P, min_pause_s=0.18)
    assert pauses == (pytest.approx(0.5),)


def test_a_pause_shorter_than_the_threshold_is_not_a_pause():
    by_role = {P: spans((0.0, 1.0), (1.1, 2.0))}
    turns = build_turns(by_role, merge_gap=0.3)
    assert within_turn_pauses(turns, by_role[P], role=P, min_pause_s=0.18) == ()


def test_the_gap_between_speakers_is_not_a_within_turn_pause():
    """It is a response latency, and mixing the two would blur both."""
    by_role = clean_exchange()
    turns = build_turns(by_role)
    assert within_turn_pauses(turns, by_role[P], role=P, min_pause_s=0.18) == ()


def test_pauses_are_only_counted_for_the_requested_role():
    by_role = {P: spans((0.0, 1.0), (2.0, 3.0)), D: spans((5.0, 6.0), (7.0, 8.0))}
    turns = build_turns(by_role)
    assert len(within_turn_pauses(turns, by_role[P], role=P, min_pause_s=0.1)) == 1
    assert len(within_turn_pauses(turns, by_role[D], role=D, min_pause_s=0.1)) == 1


# ---------------------------------------------------------------------------
# ratios
# ---------------------------------------------------------------------------
def test_speaking_ratio_of_an_even_exchange_is_a_half():
    assert speaking_ratio(clean_exchange()) == pytest.approx(0.5)


def test_speaking_ratio_reflects_an_uneven_exchange():
    by_role = {P: spans((0.0, 1.0)), D: spans((2.0, 5.0))}
    assert speaking_ratio(by_role) == pytest.approx(0.25)


def test_speaking_ratio_is_none_when_nobody_speaks():
    """Not zero, which would claim the participant was silent while talked at."""
    assert speaking_ratio({P: [], D: []}) is None


def test_speaking_ratio_is_one_when_only_the_participant_speaks():
    assert speaking_ratio({P: spans((0.0, 1.0)), D: []}) == pytest.approx(1.0)


def test_overlap_ratio_counts_simultaneous_speech():
    by_role = {D: spans((0.0, 3.0)), P: spans((2.0, 5.0))}
    # Covered time is 0-5 = 5 s; overlap is 2-3 = 1 s.
    assert overlap_ratio(by_role) == pytest.approx(0.2)


def test_a_clean_exchange_has_no_overlap():
    assert overlap_ratio(clean_exchange()) == pytest.approx(0.0)


def test_overlap_ratio_is_none_without_speech():
    assert overlap_ratio({}) is None


# ---------------------------------------------------------------------------
# the speaking/listening timeline
# ---------------------------------------------------------------------------
def test_the_timeline_splits_speaking_from_listening():
    timeline = speaking_timeline(clean_exchange())
    assert timeline.speaking_seconds == pytest.approx(3.0)
    assert timeline.listening_seconds == pytest.approx(3.0)


def test_listening_excludes_time_the_participant_is_also_speaking():
    by_role = {D: spans((0.0, 10.0)), P: spans((4.0, 6.0))}
    timeline = speaking_timeline(by_role)
    assert timeline.speaking_seconds == pytest.approx(2.0)
    assert timeline.listening_seconds == pytest.approx(8.0)
    assert [(s.start, s.end) for s in timeline.listening] == [(0.0, 4.0), (6.0, 10.0)]


def test_mutual_silence_belongs_to_neither_state():
    by_role = {D: spans((0.0, 1.0)), P: spans((5.0, 6.0))}
    timeline = speaking_timeline(by_role)
    assert timeline.speaking_seconds + timeline.listening_seconds == pytest.approx(2.0)


def test_speaking_and_listening_never_overlap():
    by_role = {D: spans((0.0, 10.0)), P: spans((2.0, 4.0), (6.0, 8.0))}
    timeline = speaking_timeline(by_role)
    assert overlap_duration(timeline.speaking, timeline.listening) == pytest.approx(0.0)


def test_a_session_with_no_psychiatrist_has_no_listening_time():
    timeline = speaking_timeline({P: spans((0.0, 5.0))})
    assert timeline.speaking_seconds == pytest.approx(5.0)
    assert timeline.listening_seconds == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the feature set
# ---------------------------------------------------------------------------
def _features(by_role: dict[str, list[Span]], duration: float = 8.5) -> dict[str, float | None]:
    return turn_features(
        by_role,
        duration_s=duration,
        merge_gap_s=0.3,
        min_pause_s=0.18,
        max_latency_s=10.0,
    )


def test_the_feature_set_is_small_and_named_by_convention():
    """With 62 sessions every feature costs power (docs/decisions/0006)."""
    assert len(FEATURE_NAMES) <= 14
    assert all(name.startswith("turns__") for name in FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_every_feature_is_produced_for_a_normal_session():
    features = _features(clean_exchange())
    assert set(features) == set(FEATURE_NAMES)
    assert features["turns__latency_mean"] == pytest.approx(0.5)
    assert features["turns__latency_median"] == pytest.approx(0.5)
    assert features["turns__latency_sd"] == pytest.approx(0.0)
    assert features["turns__participant_speaking_ratio"] == pytest.approx(0.5)
    assert features["turns__overlap_ratio"] == pytest.approx(0.0)


def test_rates_are_per_minute_of_recording():
    """Session length varies from 4 to 16 minutes, so raw counts would mostly
    measure duration rather than behaviour."""
    features = _features(clean_exchange(), duration=60.0)
    assert features["turns__n_per_minute"] == pytest.approx(4.0)


def test_turn_durations_are_reported_per_role():
    features = _features({D: spans((0.0, 4.0)), P: spans((5.0, 6.0))})
    assert features["turns__psychiatrist_turn_duration_mean"] == pytest.approx(4.0)
    assert features["turns__participant_turn_duration_mean"] == pytest.approx(1.0)


def test_a_standard_deviation_needs_two_turns():
    features = _features({P: spans((0.0, 1.0))})
    assert features["turns__participant_turn_duration_sd"] is None


def test_features_are_none_rather_than_zero_when_unmeasurable():
    """Zero would be a claim; None is the absence of one."""
    features = _features({}, duration=600.0)
    assert features["turns__latency_mean"] is None
    assert features["turns__participant_speaking_ratio"] is None
    assert features["turns__overlap_ratio"] is None
    assert features["turns__pause_within_rate"] is None


def test_a_session_with_no_participant_speech_yields_no_participant_features():
    features = _features({D: spans((0.0, 5.0))})
    assert features["turns__participant_turn_duration_mean"] is None
    assert features["turns__participant_speaking_ratio"] == pytest.approx(0.0)
    assert features["turns__latency_mean"] is None


def test_interruption_rate_is_per_minute():
    by_role = {D: spans((0.0, 3.0), (10.0, 13.0)), P: spans((2.5, 6.0), (12.5, 16.0))}
    features = _features(by_role, duration=60.0)
    assert features["turns__interruption_rate"] == pytest.approx(2.0)


def test_pause_features_describe_the_participant():
    by_role = {
        P: spans((0.0, 1.0), (1.5, 2.5), (3.0, 4.0)),
        D: spans((10.0, 11.0)),
    }
    features = _features(by_role, duration=60.0)
    assert features["turns__pause_within_mean"] == pytest.approx(0.5)
    # Two pauses over three seconds of participant speech.
    assert features["turns__pause_within_rate"] == pytest.approx(2.0 / (3.0 / 60.0))


def test_a_zero_length_recording_does_not_divide_by_zero():
    features = _features(clean_exchange(), duration=0.0)
    assert features["turns__n_per_minute"] is None
    assert features["turns__interruption_rate"] is None


def test_the_features_match_a_hand_worked_session():
    """A complete worked example: every value below was computed by hand."""
    by_role = {
        D: spans((0.0, 10.0), (20.0, 30.0)),
        P: spans((11.0, 19.0), (32.0, 40.0)),
    }
    features = _features(by_role, duration=60.0)

    # Four turns in one minute.
    assert features["turns__n_per_minute"] == pytest.approx(4.0)
    # Participant speaks 8 + 8 = 16 s of 36 s total.
    assert features["turns__participant_speaking_ratio"] == pytest.approx(16.0 / 36.0)
    # Two responses: 11 - 10 = 1 s, and 32 - 30 = 2 s.
    assert features["turns__latency_mean"] == pytest.approx(1.5)
    assert features["turns__latency_median"] == pytest.approx(1.5)
    assert features["turns__latency_sd"] == pytest.approx(0.7071, abs=1e-4)
    assert features["turns__interruption_rate"] == pytest.approx(0.0)
    assert features["turns__participant_turn_duration_mean"] == pytest.approx(8.0)
    assert features["turns__psychiatrist_turn_duration_mean"] == pytest.approx(10.0)
