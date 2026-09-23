"""Parsing diarization output produced elsewhere.

The file the lab can supply is not yet known, so the parsers are tested against
the shapes whisper-diarization and pyannote actually write, plus the formatting
variations that arrive with files passed between machines.
"""

from __future__ import annotations

import pytest

from tests.synth import generators as gen
from vc_multimodal.diarization.base import (
    Segment,
    canonical_speaker,
    covered_time,
    overlap_time,
    sort_segments,
    speakers_in,
    strip_text,
    total_speech,
)
from vc_multimodal.diarization.srt import (
    DiarizationError,
    parse_srt_cues,
    parse_timestamp,
    segments_from_rttm,
    segments_from_srt,
    split_speaker,
)


# ---------------------------------------------------------------------------
# speaker labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Speaker 0", "SPEAKER_00"),
        ("SPEAKER_00", "SPEAKER_00"),
        ("speaker 1", "SPEAKER_01"),
        ("SPEAKER_1", "SPEAKER_01"),
        ("spk 2", "SPEAKER_02"),
        ("Speaker 10", "SPEAKER_10"),
    ],
)
def test_speaker_labels_normalise_to_one_form(raw: str, expected: str):
    """Backends spell these differently; everything downstream sees one form."""
    assert canonical_speaker(raw) == expected


def test_a_label_with_no_number_falls_back():
    assert canonical_speaker("interviewer") == "SPEAKER_00"
    assert canonical_speaker("interviewer", fallback_index=1) == "SPEAKER_01"


@pytest.mark.parametrize(
    ("cue", "label", "text"),
    [
        ("Speaker 0: hello", "Speaker 0", "hello"),
        ("SPEAKER_00: hello", "SPEAKER_00", "hello"),
        ("[SPEAKER_01]: hello", "SPEAKER_01", "hello"),
        ("(Speaker 1) : hello", "Speaker 1", "hello"),
        ("spk 2 : hello", "spk 2", "hello"),
        ("Speaker 0：こんにちは", "Speaker 0", "こんにちは"),
        ("speaker_3:hello", "speaker_3", "hello"),
    ],
)
def test_a_speaker_prefix_is_split_off_the_cue(cue: str, label: str, text: str):
    assert split_speaker(cue) == (label, text)


@pytest.mark.parametrize("cue", ["plain text", "話者: hello", "0: hello", ": hello"])
def test_text_with_no_recognised_prefix_is_left_alone(cue: str):
    found, text = split_speaker(cue)
    assert found is None
    assert text == cue.strip()


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("00:00:00,000", 0.0),
        ("00:00:00,500", 0.5),
        ("00:01:01,250", 61.25),
        ("01:01:01,001", 3661.001),
        ("00:00:01.500", 1.5),  # a full stop instead of a comma
        ("01:30,000", 90.0),  # hours omitted
        ("00:00:01,5", 1.5),  # a single fractional digit
        ("  00:00:02,000  ", 2.0),
        # MM:SS with no hours and no fraction is still a timestamp.
        ("00:00", 0.0),
    ],
)
def test_timestamps_are_parsed(raw: str, seconds: float):
    assert parse_timestamp(raw) == pytest.approx(seconds)


@pytest.mark.parametrize("raw", ["", "abc", "1:2:3:4", "00;00;01,000", "--", "00:00:"])
def test_a_malformed_timestamp_is_an_error(raw: str):
    with pytest.raises(DiarizationError, match="malformed timestamp"):
        parse_timestamp(raw)


# ---------------------------------------------------------------------------
# SRT, as whisper-diarization writes it
# ---------------------------------------------------------------------------
def test_the_synthetic_generator_round_trips_through_the_parser():
    """The generators encode the ground truth the later stages are tested on."""
    session = gen.alternating_session(28, n_turns=6, turn_s=1.5, gap_s=0.5, lead_in_s=0.5)
    segments = segments_from_srt(gen.srt_text(session))

    assert len(segments) == len(session.utterances)
    for segment, utterance in zip(segments, session.utterances, strict=True):
        assert segment.start == pytest.approx(utterance.start, abs=0.001)
        assert segment.end == pytest.approx(utterance.end, abs=0.001)
        assert segment.speaker == canonical_speaker(utterance.speaker)


def test_text_is_kept_or_dropped_at_the_parse_boundary():
    session = gen.alternating_session(28, n_turns=2)
    with_text = segments_from_srt(gen.srt_text(session), keep_text=True)
    without = segments_from_srt(gen.srt_text(session), keep_text=False)

    assert all(segment.text for segment in with_text)
    assert all(segment.text is None for segment in without)


def test_a_missing_cue_index_is_tolerated():
    content = "00:00:00,000 --> 00:00:01,000\nSpeaker 0: hi\n"
    assert len(segments_from_srt(content)) == 1


def test_crlf_line_endings_and_a_byte_order_mark_are_tolerated():
    """Files passed between machines arrive like this."""
    content = "﻿1\r\n00:00:00,000 --> 00:00:01,000\r\nSpeaker 0: hi\r\n"
    assert len(segments_from_srt(content)) == 1


def test_multi_line_cue_text_is_joined():
    content = "1\n00:00:00,000 --> 00:00:02,000\nSpeaker 0: first line\nsecond line\n"
    segments = segments_from_srt(content)
    assert segments[0].text == "first line second line"


def test_blank_blocks_are_skipped():
    content = (
        "1\n00:00:00,000 --> 00:00:01,000\nSpeaker 0: a\n\n\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nSpeaker 1: b\n"
    )
    assert len(segments_from_srt(content)) == 2


def test_a_cue_with_no_prefix_continues_the_previous_speaker():
    """Continuation cues are written without repeating the speaker."""
    content = (
        "1\n00:00:00,000 --> 00:00:01,000\nSpeaker 0: first\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nstill the same person\n"
    )
    segments = segments_from_srt(content)
    assert [segment.speaker for segment in segments] == ["SPEAKER_00", "SPEAKER_00"]


def test_cues_before_the_first_label_are_dropped():
    content = (
        "1\n00:00:00,000 --> 00:00:01,000\nunattributable\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nSpeaker 0: attributed\n"
    )
    segments = segments_from_srt(content)
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(1.0)


def test_a_file_with_no_labels_at_all_is_refused():
    """Treating it as one speaker would destroy the distinction silently."""
    content = "1\n00:00:00,000 --> 00:00:01,000\njust text\n"
    with pytest.raises(DiarizationError, match="no speaker labels"):
        segments_from_srt(content)


def test_the_refusal_explains_what_was_expected():
    with pytest.raises(DiarizationError) as caught:
        segments_from_srt("1\n00:00:00,000 --> 00:00:01,000\njust text\n")
    assert "Speaker 0:" in str(caught.value)


def test_zero_length_cues_are_dropped():
    content = (
        "1\n00:00:01,000 --> 00:00:01,000\nSpeaker 0: instant\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nSpeaker 0: real\n"
    )
    assert len(segments_from_srt(content)) == 1


def test_an_empty_file_is_an_error():
    with pytest.raises(DiarizationError, match="no cues"):
        segments_from_srt("")


def test_a_file_of_only_zero_length_cues_is_an_error():
    content = "1\n00:00:01,000 --> 00:00:01,000\nSpeaker 0: x\n"
    with pytest.raises(DiarizationError, match="positive duration"):
        segments_from_srt(content)


def test_cues_are_parsed_without_attribution_too():
    cues = parse_srt_cues("1\n00:00:00,000 --> 00:00:01,000\nSpeaker 0: hi\n")
    assert len(cues) == 1
    assert cues[0].text == "Speaker 0: hi"


# ---------------------------------------------------------------------------
# RTTM, as pyannote writes it
# ---------------------------------------------------------------------------
def test_rttm_lines_are_parsed():
    content = (
        "SPEAKER file 1 0.500 1.500 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER file 1 2.500 1.500 <NA> <NA> speaker_1 <NA> <NA>\n"
    )
    segments = segments_from_rttm(content)
    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert segments[0].start == pytest.approx(0.5)
    assert segments[0].end == pytest.approx(2.0)


def test_rttm_carries_no_text():
    """Which makes it the safer format to be handed."""
    content = "SPEAKER file 1 0.0 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    assert segments_from_rttm(content)[0].text is None


def test_non_speaker_rttm_lines_are_ignored():
    content = (
        "SPKR-INFO file 1 <NA> <NA> <NA> unknown SPEAKER_00 <NA> <NA>\n"
        "SPEAKER file 1 0.0 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    )
    assert len(segments_from_rttm(content)) == 1


def test_zero_duration_rttm_lines_are_dropped():
    content = (
        "SPEAKER file 1 0.0 0.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER file 1 1.0 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    )
    assert len(segments_from_rttm(content)) == 1


def test_a_short_rttm_line_is_an_error():
    with pytest.raises(DiarizationError, match="malformed RTTM line"):
        segments_from_rttm("SPEAKER file 1 0.0\n")


def test_non_numeric_rttm_timings_are_an_error():
    content = "SPEAKER file 1 start dur <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    with pytest.raises(DiarizationError, match="malformed RTTM timing"):
        segments_from_rttm(content)


def test_an_rttm_with_no_speaker_lines_is_an_error():
    with pytest.raises(DiarizationError, match="no SPEAKER lines"):
        segments_from_rttm("# a comment\n")


# ---------------------------------------------------------------------------
# segment arithmetic
# ---------------------------------------------------------------------------
def _segments(*spans: tuple[str, float, float]) -> tuple[Segment, ...]:
    return tuple(Segment(speaker=s, start=a, end=b) for s, a, b in spans)


def test_segments_sort_by_time():
    unsorted = _segments(("SPEAKER_01", 5.0, 6.0), ("SPEAKER_00", 1.0, 2.0))
    assert [s.start for s in sort_segments(unsorted)] == [1.0, 5.0]


def test_speakers_are_listed_in_order_of_appearance():
    segments = _segments(
        ("SPEAKER_01", 0.0, 1.0), ("SPEAKER_00", 1.0, 2.0), ("SPEAKER_01", 2.0, 3.0)
    )
    assert speakers_in(segments) == ("SPEAKER_01", "SPEAKER_00")


def test_total_speech_counts_overlap_twice():
    segments = _segments(("SPEAKER_00", 0.0, 2.0), ("SPEAKER_01", 1.0, 3.0))
    assert total_speech(segments) == pytest.approx(4.0)


def test_covered_time_counts_overlap_once():
    segments = _segments(("SPEAKER_00", 0.0, 2.0), ("SPEAKER_01", 1.0, 3.0))
    assert covered_time(segments) == pytest.approx(3.0)


def test_overlap_time_is_the_difference():
    segments = _segments(("SPEAKER_00", 0.0, 2.0), ("SPEAKER_01", 1.0, 3.0))
    assert overlap_time(segments) == pytest.approx(1.0)


def test_disjoint_segments_have_no_overlap():
    segments = _segments(("SPEAKER_00", 0.0, 1.0), ("SPEAKER_01", 2.0, 3.0))
    assert covered_time(segments) == pytest.approx(2.0)
    assert overlap_time(segments) == pytest.approx(0.0)


def test_a_segment_entirely_inside_another():
    segments = _segments(("SPEAKER_00", 0.0, 10.0), ("SPEAKER_01", 2.0, 3.0))
    assert covered_time(segments) == pytest.approx(10.0)
    assert overlap_time(segments) == pytest.approx(1.0)


def test_three_way_overlap():
    segments = _segments(
        ("SPEAKER_00", 0.0, 3.0), ("SPEAKER_01", 1.0, 4.0), ("SPEAKER_02", 2.0, 5.0)
    )
    assert covered_time(segments) == pytest.approx(5.0)
    assert total_speech(segments) == pytest.approx(9.0)


def test_arithmetic_on_no_segments():
    assert covered_time(()) == 0.0
    assert total_speech(()) == 0.0
    assert overlap_time(()) == 0.0


def test_text_can_be_stripped_from_segments():
    segments = (Segment("SPEAKER_00", 0.0, 1.0, "secret words"),)
    stripped = strip_text(segments)
    assert stripped[0].text is None
    assert stripped[0].speaker == "SPEAKER_00"
    # The originals are untouched, being frozen.
    assert segments[0].text == "secret words"


def test_segment_duration():
    assert Segment("SPEAKER_00", 1.0, 3.5).duration == pytest.approx(2.5)
