"""The synthetic generators are test infrastructure, so they are tested too.

Later stages are checked against the ground truth these generators encode. If
the generators and the ground truth disagree, every downstream test is
meaningless, so the agreement is asserted here directly.
"""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from tests.synth import generators as gen


def test_alternating_session_encodes_exact_turn_timing():
    session = gen.alternating_session(1, n_turns=4, turn_s=1.5, gap_s=0.5, lead_in_s=0.5)

    assert len(session.utterances) == 4
    assert [u.speaker for u in session.utterances] == [
        gen.PSYCHIATRIST,
        gen.PARTICIPANT,
        gen.PSYCHIATRIST,
        gen.PARTICIPANT,
    ]
    # First turn starts after the lead-in; each later turn after exactly one gap.
    assert session.utterances[0].start == pytest.approx(0.5)
    for earlier, later in zip(session.utterances, session.utterances[1:], strict=False):
        assert later.start - earlier.end == pytest.approx(0.5)


def test_alternating_session_has_no_overlap():
    session = gen.alternating_session(1, n_turns=6)
    for earlier, later in zip(session.utterances, session.utterances[1:], strict=False):
        assert later.start >= earlier.end


def test_overlapping_session_contains_a_known_overlap():
    session = gen.overlapping_session(1)
    psychiatrist_end = session.utterances[0].end
    participant_start = session.utterances[1].start
    assert psychiatrist_end - participant_start == pytest.approx(0.5)


def test_speech_totals_match_the_schedule():
    session = gen.alternating_session(1, n_turns=4, turn_s=1.5)
    assert session.total_speech(gen.PSYCHIATRIST) == pytest.approx(3.0)
    assert session.total_speech(gen.PARTICIPANT) == pytest.approx(3.0)


def test_speaking_at_agrees_with_the_spans():
    session = gen.alternating_session(1, n_turns=2, turn_s=1.0, gap_s=1.0, lead_in_s=1.0)
    assert session.speaking_at(gen.PSYCHIATRIST, 1.5)
    assert not session.speaking_at(gen.PSYCHIATRIST, 0.5)
    assert not session.speaking_at(gen.PARTICIPANT, 1.5)
    assert session.speaking_at(gen.PARTICIPANT, 3.5)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def test_waveform_length_matches_the_session_duration():
    session = gen.alternating_session(1, n_turns=2)
    samples = gen.session_waveform(session)
    assert len(samples) == round(session.duration * session.sample_rate)


def test_waveform_is_loud_during_speech_and_quiet_between():
    session = gen.alternating_session(1, n_turns=2, turn_s=1.0, gap_s=1.0, lead_in_s=1.0)
    samples = gen.session_waveform(session)
    sr = session.sample_rate

    during = np.abs(samples[int(1.2 * sr) : int(1.8 * sr)]).mean()
    between = np.abs(samples[int(2.2 * sr) : int(2.8 * sr)]).mean()
    assert during > between * 20


def test_a_single_speaker_stream_omits_the_other_speaker():
    session = gen.alternating_session(1, n_turns=2, turn_s=1.0, gap_s=1.0, lead_in_s=1.0)
    only_psychiatrist = gen.session_waveform(session, speakers=[gen.PSYCHIATRIST])
    sr = session.sample_rate
    participant_window = np.abs(only_psychiatrist[int(3.2 * sr) : int(3.8 * sr)]).mean()
    psychiatrist_window = np.abs(only_psychiatrist[int(1.2 * sr) : int(1.8 * sr)]).mean()
    assert psychiatrist_window > participant_window * 20


def test_speakers_are_rendered_at_different_pitches():
    assert gen.SPEAKER_TONE_HZ[gen.PSYCHIATRIST] != gen.SPEAKER_TONE_HZ[gen.PARTICIPANT]


def test_wav_is_written_as_mono_16_bit(tmp_path: Path):
    session = gen.alternating_session(1, n_turns=2)
    path = gen.write_session_wav(tmp_path / "1.wav", session)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == session.sample_rate


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
def test_video_has_the_expected_geometry_and_frame_count(tmp_path: Path):
    session = gen.alternating_session(1, n_turns=2, turn_s=1.0, gap_s=0.5)
    path = gen.write_session_video(tmp_path / "1.mp4", session)

    capture = cv2.VideoCapture(str(path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == session.width
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == session.height
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(session.fps, abs=0.5)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    assert frames == pytest.approx(round(session.duration * session.fps), abs=1)


def test_mouth_movement_tracks_the_correct_tile(tmp_path: Path):
    """The video ground truth for the speaker/tile cross-check.

    Mouth-region variability must be higher in the tile of whoever is speaking.
    Without this, the `assign-speakers` cross-check could pass on a video that
    does not actually encode the association.
    """
    session = gen.alternating_session(1, n_turns=2, turn_s=2.0, gap_s=0.0, lead_in_s=0.0, fps=10.0)
    path = gen.write_session_video(tmp_path / "1.mp4", session)

    capture = cv2.VideoCapture(str(path))
    half, mid = session.width // 2, session.height // 2
    rows = slice(mid + session.height // 10, mid + session.height // 10 + 25)
    series: dict[str, list[float]] = {"left": [], "right": []}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            series["left"].append(float(grey[rows, :half].mean()))
            series["right"].append(float(grey[rows, half:].mean()))
    finally:
        capture.release()

    n = len(series["left"])
    first_half, second_half = slice(0, n // 2), slice(n // 2, n)
    # The psychiatrist (left tile) speaks first, the participant (right) second.
    assert np.std(series["left"][first_half]) > np.std(series["left"][second_half])
    assert np.std(series["right"][second_half]) > np.std(series["right"][first_half])


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (0.5, "00:00:00,500"),
        (61.25, "00:01:01,250"),
        (3661.001, "01:01:01,001"),
    ],
)
def test_srt_timestamps(seconds: float, expected: str):
    assert gen._srt_timestamp(seconds) == expected


def test_negative_srt_timestamp_is_rejected():
    with pytest.raises(ValueError, match="negative timestamp"):
        gen._srt_timestamp(-1.0)


def test_srt_carries_speaker_labels_in_the_cue_text():
    session = gen.alternating_session(1, n_turns=2, turn_s=1.5, gap_s=0.5, lead_in_s=0.5)
    blocks = gen.srt_text(session).strip().split("\n\n")
    assert len(blocks) == 2
    first = blocks[0].splitlines()
    assert first[0] == "1"
    assert first[1] == "00:00:00,500 --> 00:00:02,000"
    assert first[2].startswith(f"{gen.PSYCHIATRIST}:")


def test_srt_is_written_as_utf8(tmp_path: Path):
    session = gen.alternating_session(1, n_turns=2)
    path = gen.write_srt(tmp_path / "1.srt", session)
    assert gen.PARTICIPANT in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# muxing: what ffprobe will actually see
# ---------------------------------------------------------------------------
def _probe_streams(ffprobe: str, path: Path) -> list[dict[str, object]]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    streams: list[dict[str, object]] = json.loads(result.stdout)["streams"]
    return streams


@pytest.mark.slow
def test_muxed_mp4_has_one_video_and_one_audio_stream(
    tmp_path: Path, ffmpeg_bin: str, ffprobe_bin: str
):
    session = gen.alternating_session(28, n_turns=4)
    path = gen.write_session_mp4(tmp_path / "28.mp4", session, tmp_dir=tmp_path, ffmpeg=ffmpeg_bin)
    streams = _probe_streams(ffprobe_bin, path)
    assert sum(s["codec_type"] == "video" for s in streams) == 1
    assert sum(s["codec_type"] == "audio" for s in streams) == 1


@pytest.mark.slow
def test_per_speaker_audio_produces_two_streams(tmp_path: Path, ffmpeg_bin: str, ffprobe_bin: str):
    """The 'one mixed stream or two' question must be testable both ways."""
    session = gen.alternating_session(29, n_turns=4)
    path = gen.write_session_mp4(
        tmp_path / "29.mp4",
        session,
        tmp_dir=tmp_path,
        per_speaker_audio=True,
        ffmpeg=ffmpeg_bin,
    )
    streams = _probe_streams(ffprobe_bin, path)
    assert sum(s["codec_type"] == "audio" for s in streams) == 2


@pytest.mark.slow
def test_mux_failure_is_reported(tmp_path: Path, ffmpeg_bin: str):
    missing = tmp_path / "absent.mp4"
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        gen.mux(tmp_path / "out.mp4", missing, [], ffmpeg=ffmpeg_bin)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def test_labels_frame_has_one_row_per_session():
    frame = gen.labels_frame([1, 2, 3])
    assert list(frame.columns) == ["session_id", "K6", "SRS2"]
    assert len(frame) == 3


def test_labels_frame_is_reproducible():
    pd.testing.assert_frame_equal(
        gen.labels_frame([1, 2], seed=5), gen.labels_frame([1, 2], seed=5)
    )
