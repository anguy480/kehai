"""ffmpeg/ffprobe discovery and metadata parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vc_multimodal.ffmpeg import (
    FFMPEG_ENV,
    FFPROBE_ENV,
    FfmpegError,
    FfmpegTools,
    parse_media_info,
    parse_rational,
)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def test_discovery_prefers_the_configured_absolute_path(
    monkeypatch: pytest.MonkeyPatch, ffmpeg_bin: str, ffprobe_bin: str
):
    """Pinned paths survive `conda deactivate`, which removes them from PATH."""
    monkeypatch.setenv(FFMPEG_ENV, ffmpeg_bin)
    monkeypatch.setenv(FFPROBE_ENV, ffprobe_bin)
    monkeypatch.setenv("PATH", "")
    tools = FfmpegTools.discover()
    assert tools.ffmpeg == Path(ffmpeg_bin)
    assert tools.ffprobe == Path(ffprobe_bin)


def test_discovery_falls_back_to_path(
    monkeypatch: pytest.MonkeyPatch, ffmpeg_bin: str, ffprobe_bin: str
):
    monkeypatch.delenv(FFMPEG_ENV, raising=False)
    monkeypatch.delenv(FFPROBE_ENV, raising=False)
    assert FfmpegTools.discover().ffmpeg.name.startswith("ffmpeg")


def test_a_configured_path_that_is_not_executable_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake = tmp_path / "ffmpeg"
    fake.write_text("not a binary", encoding="utf-8")
    monkeypatch.setenv(FFMPEG_ENV, str(fake))
    with pytest.raises(FfmpegError, match="not an executable file"):
        FfmpegTools.discover()


def test_a_missing_binary_explains_the_conda_situation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(FFMPEG_ENV, raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FfmpegError) as caught:
        FfmpegTools.discover()
    message = str(caught.value)
    assert "does not install it" in message
    assert "conda" in message
    assert FFMPEG_ENV in message


def test_versions_are_recorded_for_the_manifest(ffmpeg_bin: str, ffprobe_bin: str):
    versions = FfmpegTools(ffmpeg=Path(ffmpeg_bin), ffprobe=Path(ffprobe_bin)).versions()
    assert "ffmpeg version" in versions["ffmpeg"]
    assert "ffprobe version" in versions["ffprobe"]


def test_probing_a_missing_file_is_an_error(ffmpeg_bin: str, ffprobe_bin: str, tmp_path: Path):
    tools = FfmpegTools(ffmpeg=Path(ffmpeg_bin), ffprobe=Path(ffprobe_bin))
    with pytest.raises(FfmpegError, match="ffprobe failed"):
        tools.probe(tmp_path / "absent.mp4")


def test_probing_a_non_media_file_is_an_error(ffmpeg_bin: str, ffprobe_bin: str, tmp_path: Path):
    """A placeholder file with an mp4 name must fail loudly, not silently."""
    junk = tmp_path / "28.mp4"
    junk.write_bytes(b"not a real recording")
    tools = FfmpegTools(ffmpeg=Path(ffmpeg_bin), ffprobe=Path(ffprobe_bin))
    with pytest.raises(FfmpegError, match="ffprobe failed"):
        tools.probe(junk)


def test_extracting_a_frame_from_a_missing_file_is_an_error(
    ffmpeg_bin: str, ffprobe_bin: str, tmp_path: Path
):
    tools = FfmpegTools(ffmpeg=Path(ffmpeg_bin), ffprobe=Path(ffprobe_bin))
    with pytest.raises(FfmpegError, match="could not extract a frame"):
        tools.extract_frame(tmp_path / "absent.mp4", 1.0, tmp_path / "out.png")


# ---------------------------------------------------------------------------
# rationals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("25/1", 25.0),
        ("30000/1001", 29.97002997002997),
        ("25", 25.0),
        (25, 25.0),
        (29.97, 29.97),
    ],
)
def test_rationals_are_parsed(value: object, expected: float):
    assert parse_rational(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "", "  ", "0/0", "abc", "1/abc", "abc/1", "/"])
def test_unusable_rationals_become_none(value: object):
    assert parse_rational(value) is None


# ---------------------------------------------------------------------------
# media info
# ---------------------------------------------------------------------------
def _probe(
    *,
    video: dict[str, Any] | None = None,
    audio: list[dict[str, Any]] | None = None,
    container: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default_video = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "r_frame_rate": "25/1",
        "avg_frame_rate": "25/1",
    }
    streams: list[dict[str, Any]] = []
    if video is not None or video is None:
        merged = dict(default_video)
        merged.update(video or {})
        streams.append(merged)
    streams.extend(audio or [])
    return {
        "format": {"duration": "640.5", "size": "123456789", **(container or {})},
        "streams": streams,
    }


def _audio(**overrides: Any) -> dict[str, Any]:
    stream = {
        "codec_type": "audio",
        "codec_name": "aac",
        "channels": 2,
        "sample_rate": "48000",
        "index": 1,
    }
    stream.update(overrides)
    return stream


def test_media_info_reads_container_and_stream_metadata():
    info = parse_media_info(_probe(audio=[_audio()]))
    assert info.duration_s == pytest.approx(640.5)
    assert info.size_bytes == 123456789
    assert info.video_codec == "h264"
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == pytest.approx(25.0)
    assert info.fps_variable is False
    assert info.n_audio_streams == 1
    assert info.primary_audio is not None
    assert info.primary_audio.channels == 2
    assert info.primary_audio.sample_rate == 48000


def test_two_audio_streams_are_both_reported():
    """The open question of one mixed stream versus two is answered here."""
    info = parse_media_info(_probe(audio=[_audio(index=1), _audio(index=2, channels=1)]))
    assert info.n_audio_streams == 2
    assert [s.channels for s in info.audio_streams] == [2, 1]
    assert info.primary_audio is not None
    assert info.primary_audio.index == 1


def test_a_file_with_no_audio_is_described_not_rejected():
    info = parse_media_info(_probe(audio=[]))
    assert info.n_audio_streams == 0
    assert info.primary_audio is None


def test_variable_frame_rate_is_detected_from_the_rate_mismatch():
    info = parse_media_info(_probe(video={"r_frame_rate": "30/1", "avg_frame_rate": "24000/1001"}))
    assert info.fps_variable is True
    # The average rate is the one worth recording for timing.
    assert info.fps == pytest.approx(23.976, abs=0.001)


def test_a_tiny_rate_difference_is_not_called_variable():
    info = parse_media_info(_probe(video={"r_frame_rate": "25/1", "avg_frame_rate": "24.999/1"}))
    assert info.fps_variable is False


def test_a_missing_average_rate_falls_back_to_the_base_rate():
    info = parse_media_info(_probe(video={"avg_frame_rate": "0/0"}))
    assert info.fps == pytest.approx(25.0)
    assert info.fps_variable is False


def test_duration_falls_back_to_the_video_stream():
    probe = _probe(video={"duration": "12.5"}, container={})
    del probe["format"]["duration"]
    assert parse_media_info(probe).duration_s == pytest.approx(12.5)


def test_missing_metadata_becomes_none_rather_than_raising():
    info = parse_media_info({"format": {}, "streams": []})
    assert info.duration_s is None
    assert info.size_bytes is None
    assert info.video_codec is None
    assert info.width is None
    assert info.fps is None
    assert info.n_audio_streams == 0


def test_an_empty_probe_is_handled():
    info = parse_media_info({})
    assert info.duration_s is None
    assert info.fps_variable is False


@pytest.mark.slow
def test_probing_a_real_synthetic_recording(
    make_real_media: Any, ffmpeg_bin: str, ffprobe_bin: str
):
    path, session = make_real_media(28)
    tools = FfmpegTools(ffmpeg=Path(ffmpeg_bin), ffprobe=Path(ffprobe_bin))
    info = parse_media_info(tools.probe(path))

    assert info.duration_s == pytest.approx(session.duration, abs=0.5)
    assert (info.width, info.height) == (session.width, session.height)
    assert info.fps == pytest.approx(session.fps, abs=0.1)
    assert info.n_audio_streams == 1
    assert info.primary_audio is not None
    assert info.primary_audio.sample_rate == session.sample_rate


@pytest.mark.slow
def test_probe_output_is_valid_json_for_a_real_file(
    make_real_media: Any, ffmpeg_bin: str, ffprobe_bin: str
):
    path, _ = make_real_media(28)
    tools = FfmpegTools(ffmpeg=Path(ffmpeg_bin), ffprobe=Path(ffprobe_bin))
    probe = tools.probe(path)
    assert json.dumps(probe)
    assert "format" in probe
    assert "streams" in probe
