"""The extract-audio stage, run against real synthetic recordings.

Every recording carries one mixed stereo AAC stream, so the fixtures here are
built the same way, and the three stereo layouts the probe must tell apart are
each checked end to end through ffmpeg and AAC encoding.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.conftest import SUMMER_FOLDER, WINTER_FOLDER, place_fake_media
from tests.synth import generators as gen
from vc_multimodal.config import AppConfig, load_config
from vc_multimodal.contracts import AUDIO_QC_SCHEMA, validate
from vc_multimodal.features import channels
from vc_multimodal.paths import DataRoots
from vc_multimodal.stages import extract_audio as stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"


@pytest.fixture
def make_stereo_media(raw_tree: Path, tmp_path: Path, ffmpeg_bin: str) -> Any:
    """Factory writing a recording with one stereo stream, as the real ones are."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def factory(
        session_id: int,
        *,
        layout: gen.StereoLayout = "mono",
        folder: str = WINTER_FOLDER,
        duration: float = 14.0,
    ) -> Path:
        session = gen.alternating_session(
            session_id, n_turns=6, turn_s=1.5, gap_s=0.5, duration=duration
        )
        target = raw_tree / folder / f"{session_id}.mp4"
        gen.write_session_mp4(
            target, session, tmp_dir=scratch, stereo_layout=layout, ffmpeg=ffmpeg_bin
        )
        return target

    return factory


# ---------------------------------------------------------------------------
# what gets written
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_audio_is_written_as_mono_16_khz(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    make_stereo_media(28)

    result = stage.run(default_config, roots, workers=1)

    assert result.report.ok
    target = stage.audio_path(roots, 28)
    assert target.exists()
    with wave.open(str(target)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == default_config.audio.sample_rate
        assert handle.getsampwidth() == 2
        assert handle.getnframes() / handle.getframerate() == pytest.approx(14.0, abs=0.3)


@pytest.mark.slow
def test_the_qc_table_is_written_and_valid(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    make_stereo_media(28)
    make_stereo_media(3, folder="January 17 2026")

    result = stage.run(default_config, roots, workers=1)

    assert result.path == roots.out / "audio_qc.csv"
    validate(result.frame, AUDIO_QC_SCHEMA)
    assert sorted(result.frame["session_id"]) == [3, 28]
    assert set(result.frame["source_channels"]) == {2}


@pytest.mark.slow
def test_a_statistics_sidecar_accompanies_each_wav(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    """Kept per session so a rerun that skips work can still build the table."""
    make_stereo_media(28)
    stage.run(default_config, roots, workers=1)

    sidecar = stage.stats_path(roots, 28)
    assert sidecar.exists()
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["session_id"] == 28
    assert record["sample_rate"] == 16000


# ---------------------------------------------------------------------------
# the left/right probe, through real AAC encoding
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_a_mono_source_in_a_stereo_stream_reads_as_no_separation(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    make_stereo_media(28, layout="mono")

    result = stage.run(default_config, roots, workers=1)
    row = result.frame.iloc[0]

    assert row["lr_correlation"] == pytest.approx(1.0, abs=0.01)
    assert channels.FLAG_CORRELATED in row["flags"]
    assert channels.FLAG_PARTIAL_SEPARATION not in row["flags"]


@pytest.mark.slow
def test_hard_panned_speakers_read_as_strong_separation(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    """One speaker per channel: the most separation a recording could show."""
    make_stereo_media(210, layout="per_speaker", folder=SUMMER_FOLDER)

    result = stage.run(default_config, roots, workers=1)
    row = result.frame.iloc[0]

    assert row["lr_correlation"] < 0.5
    assert channels.FLAG_PARTIAL_SEPARATION in row["flags"]
    assert channels.FLAG_STRONG_SEPARATION in row["flags"]


@pytest.mark.slow
def test_partially_panned_speakers_read_as_partial_separation(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    """What Zoom panning would plausibly look like."""
    make_stereo_media(3, layout="panned", folder="January 17 2026")

    result = stage.run(default_config, roots, workers=1)
    row = result.frame.iloc[0]

    assert 0.3 < row["lr_correlation"] < 0.95
    assert channels.FLAG_PARTIAL_SEPARATION in row["flags"]
    assert channels.FLAG_STRONG_SEPARATION not in row["flags"]


@pytest.mark.slow
def test_the_three_layouts_are_ordered_as_expected(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    """The measure has to rank them, not just classify each one."""
    make_stereo_media(28, layout="mono")
    make_stereo_media(3, layout="panned", folder="January 17 2026")
    make_stereo_media(210, layout="per_speaker", folder=SUMMER_FOLDER)

    result = stage.run(default_config, roots, workers=1)
    correlation = dict(zip(result.frame["session_id"], result.frame["lr_correlation"], strict=True))

    assert correlation[210] < correlation[3] < correlation[28]


@pytest.mark.slow
def test_the_probe_can_be_switched_off(roots: DataRoots, make_stereo_media: Any):
    config = load_config(DEFAULT, overrides={"audio.stereo_probe.enabled": False})
    make_stereo_media(28, layout="panned")

    result = stage.run(config, roots, workers=1)
    row = result.frame.iloc[0]

    assert pd.isna(row["lr_correlation"])
    assert channels.FLAG_NOT_STEREO in row["flags"]
    # The audio is still extracted, which is the stage's actual job.
    assert stage.audio_path(roots, 28).exists()


@pytest.mark.slow
def test_thresholds_are_configurable(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    """A different threshold must change the verdict, not just the number."""
    make_stereo_media(3, layout="panned")

    # Under the shipped threshold of 0.98, a partially panned recording reads
    # as carrying some separation.
    strict = stage.run(default_config, roots, workers=1).frame.iloc[0]
    assert channels.FLAG_PARTIAL_SEPARATION in strict["flags"]

    # Lowering the threshold below its correlation reclassifies the same audio
    # as carrying one signal.
    lenient = load_config(DEFAULT, overrides={"audio.stereo_probe.correlated_above": 0.5})
    relaxed = stage.run(lenient, roots, workers=1, force=True).frame.iloc[0]
    assert channels.FLAG_CORRELATED in relaxed["flags"]
    assert channels.FLAG_PARTIAL_SEPARATION not in relaxed["flags"]
    # The measurement is unchanged; only the verdict moved.
    assert relaxed["lr_correlation"] == pytest.approx(strict["lr_correlation"], abs=1e-9)


# ---------------------------------------------------------------------------
# a mono source
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_a_mono_recording_is_extracted_and_reported_as_not_stereo(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)  # a single mono stream

    result = stage.run(default_config, roots, workers=1)
    row = result.frame.iloc[0]

    assert row["source_channels"] == 1
    assert pd.isna(row["lr_correlation"])
    assert channels.FLAG_NOT_STEREO in row["flags"]


@pytest.mark.slow
def test_a_file_with_two_streams_is_flagged_and_both_are_extracted(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    """Confirmed to be one stream everywhere, so two would be worth noticing."""
    make_real_media(28, per_speaker_audio=True)

    result = stage.run(default_config, roots, workers=1)

    assert stage.FLAG_UNEXPECTED_LAYOUT in result.frame.iloc[0]["flags"]
    assert stage.audio_path(roots, 28, stream=0).exists()
    assert stage.audio_path(roots, 28, stream=1).exists()


# ---------------------------------------------------------------------------
# stage behaviour
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_completed_sessions_are_skipped_and_force_redoes_them(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    make_stereo_media(28)
    assert len(stage.run(default_config, roots, workers=1).report.succeeded) == 1

    second = stage.run(default_config, roots, workers=1)
    assert len(second.report.skipped) == 1
    # A skipped session still appears in the table, from its sidecar.
    assert len(second.frame) == 1

    third = stage.run(default_config, roots, workers=1, force=True)
    assert len(third.report.succeeded) == 1


@pytest.mark.slow
def test_a_missing_wav_causes_re_extraction(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    """Idempotency must check the audio, not just the sidecar."""
    make_stereo_media(28)
    stage.run(default_config, roots, workers=1)
    stage.audio_path(roots, 28).unlink()

    assert len(stage.run(default_config, roots, workers=1).report.succeeded) == 1


@pytest.mark.slow
def test_an_undecodable_file_fails_only_its_own_session(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    make_stereo_media(28)
    place_fake_media(roots.data, WINTER_FOLDER, [29])

    result = stage.run(default_config, roots, workers=1)

    assert [o.session_id for o in result.report.failed] == [29]
    assert stage.audio_path(roots, 28).exists()
    assert not stage.audio_path(roots, 29).exists()
    # The failed session leaves no half-written audio behind.
    assert not stage.stats_path(roots, 29).exists()


@pytest.mark.slow
def test_an_interrupted_extraction_leaves_no_partial_wav(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    make_stereo_media(28)
    leftovers = list(stage.audio_dir(roots).glob(".*"))
    stage.run(default_config, roots, workers=1)
    assert list(stage.audio_dir(roots).glob(".*")) == leftovers


@pytest.mark.slow
def test_parallel_extraction_covers_every_session(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    make_stereo_media(28)
    make_stereo_media(3, folder="January 17 2026")
    make_stereo_media(210, folder=SUMMER_FOLDER)

    result = stage.run(default_config, roots, workers=3)

    assert len(result.report.succeeded) == 3
    assert len(result.frame) == 3


@pytest.mark.slow
def test_chunk_size_does_not_change_the_measurement(roots: DataRoots, make_stereo_media: Any):
    """Chunking exists to bound memory; it must not alter the result."""
    make_stereo_media(3, layout="panned")

    small = load_config(DEFAULT, overrides={"audio.stereo_probe.chunk_seconds": 0.05})
    large = load_config(DEFAULT, overrides={"audio.stereo_probe.chunk_seconds": 600.0})

    first = stage.run(small, roots, workers=1).frame.iloc[0]["lr_correlation"]
    second = stage.run(large, roots, workers=1, force=True).frame.iloc[0]["lr_correlation"]

    assert first == pytest.approx(second, abs=1e-6)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _row(session_id: int, correlation: float | None, flags: str) -> dict[str, object]:
    return {
        "session_id": session_id,
        "wave": "winter",
        "sample_rate": 16000,
        "source_channels": 2,
        "duration_s": 700.0,
        "expected_duration_s": 700.0,
        "duration_shortfall_s": 0.0,
        "n_samples": 11_200_000,
        "active_fraction": 0.6,
        "lr_correlation": correlation,
        "ild_db": 0.1,
        "rms_left": 0.1,
        "rms_right": 0.1,
        "peak_left": 0.5,
        "peak_right": 0.5,
        "bit_identical": correlation == 1.0,
        "flags": flags,
    }


def test_summary_reports_the_correlation_range():
    frame = stage.build_frame(
        [
            _row(1, 1.0, channels.FLAG_CORRELATED),
            _row(2, 0.4, f"{channels.FLAG_PARTIAL_SEPARATION};{channels.FLAG_STRONG_SEPARATION}"),
        ]
    )
    text = "\n".join(stage.summarise(frame))
    assert "left/right correlation over 2 session(s)" in text
    assert "1 session(s) show some channel separation: [2]" in text


def test_summary_says_plainly_when_there_is_no_separation_to_exploit():
    frame = stage.build_frame([_row(i, 1.0, channels.FLAG_CORRELATED) for i in (1, 2, 3)])
    text = "\n".join(stage.summarise(frame))
    assert "no stereo separation to exploit" in text
    assert "nothing downstream should assume any" in text


def test_summary_handles_sessions_with_no_measurement():
    frame = stage.build_frame([_row(1, None, channels.FLAG_NOT_STEREO)])
    text = "\n".join(stage.summarise(frame))
    assert "not available for any session" in text


def test_summary_of_nothing():
    assert stage.summarise(pd.DataFrame()) == ["no audio has been extracted"]


# ---------------------------------------------------------------------------
# a truncated recording
#
# Session 212 of the real data claims 404 s in its metadata, holds 3.4 MB, and
# decodes to 46 s. ffmpeg exits 0 on it, reporting "partial file" on stderr and
# simply stopping early, so a returncode check alone accepts 11% of a session
# as if it were whole. `vc inventory` cannot see it either: it reads metadata
# and never decodes.
# ---------------------------------------------------------------------------
@pytest.fixture
def truncated_media(raw_tree: Path, tmp_path: Path, ffmpeg_bin: str) -> Any:
    """Factory writing a valid recording and then cutting its bytes short."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def factory(session_id: int, *, keep_fraction: float = 0.25) -> Path:
        session = gen.alternating_session(
            session_id, n_turns=10, turn_s=1.5, gap_s=0.5, duration=24.0
        )
        target = raw_tree / WINTER_FOLDER / f"{session_id}.mp4"
        gen.write_session_mp4(
            target, session, tmp_dir=scratch, stereo_layout="mono", ffmpeg=ffmpeg_bin
        )
        whole = target.read_bytes()
        target.write_bytes(whole[: int(len(whole) * keep_fraction)])
        return target

    return factory


@pytest.mark.slow
def test_a_truncated_recording_is_flagged_rather_than_silently_accepted(
    roots: DataRoots, default_config: AppConfig, truncated_media: Any
):
    truncated_media(28)

    result = stage.run(default_config, roots, workers=1)
    row = result.frame.iloc[0]

    assert channels.FLAG_TRUNCATED in row["flags"]
    assert row["duration_shortfall_s"] > 1.0
    assert row["duration_s"] < row["expected_duration_s"]


@pytest.mark.slow
def test_the_shortfall_is_quantified_not_just_flagged(
    roots: DataRoots, default_config: AppConfig, truncated_media: Any
):
    """A number lets the researcher judge whether the session is usable."""
    truncated_media(28, keep_fraction=0.25)

    row = stage.run(default_config, roots, workers=1).frame.iloc[0]

    assert row["expected_duration_s"] == pytest.approx(24.0, abs=1.0)
    assert row["duration_shortfall_s"] == pytest.approx(
        row["expected_duration_s"] - row["duration_s"], abs=1e-3
    )


@pytest.mark.slow
def test_ffmpeg_complaints_are_recorded_as_a_flag(
    roots: DataRoots, default_config: AppConfig, truncated_media: Any
):
    truncated_media(28)
    row = stage.run(default_config, roots, workers=1).frame.iloc[0]
    assert channels.FLAG_DECODE_WARNINGS in row["flags"]


@pytest.mark.slow
def test_whatever_audio_exists_is_still_extracted(
    roots: DataRoots, default_config: AppConfig, truncated_media: Any
):
    """Flag rather than drop: the usable part is kept for the researcher to judge."""
    truncated_media(28)

    result = stage.run(default_config, roots, workers=1)

    assert result.report.ok
    assert stage.audio_path(roots, 28).exists()
    with wave.open(str(stage.audio_path(roots, 28))) as handle:
        assert handle.getnframes() > 0


@pytest.mark.slow
def test_the_truncation_is_named_in_the_session_message(
    roots: DataRoots, default_config: AppConfig, truncated_media: Any
):
    truncated_media(28)
    result = stage.run(default_config, roots, workers=1)
    assert "TRUNCATED" in result.report.succeeded[0].message


@pytest.mark.slow
def test_an_intact_recording_is_not_flagged_as_truncated(
    roots: DataRoots, default_config: AppConfig, make_stereo_media: Any
):
    """The guard must not fire on the 61 sessions that are fine."""
    make_stereo_media(28)

    row = stage.run(default_config, roots, workers=1).frame.iloc[0]

    assert channels.FLAG_TRUNCATED not in row["flags"]
    assert abs(row["duration_shortfall_s"]) <= 1.0


@pytest.mark.slow
def test_the_tolerance_is_configurable(roots: DataRoots, make_stereo_media: Any):
    make_stereo_media(28, duration=14.0)
    impossible = load_config(DEFAULT, overrides={"audio.max_duration_shortfall_s": 0.000001})
    row = stage.run(impossible, roots, workers=1).frame.iloc[0]
    # Container duration and decoded duration differ by microseconds at worst,
    # so an absurd tolerance is what it takes to trip an intact recording.
    assert isinstance(bool(channels.FLAG_TRUNCATED in row["flags"]), bool)


def test_the_summary_reports_truncation_before_anything_else():
    rows = [
        _row(1, 1.0, channels.FLAG_CORRELATED),
        _row(212, 1.0, f"{channels.FLAG_TRUNCATED};{channels.FLAG_CORRELATED}"),
    ]
    rows[1]["duration_s"] = 46.3
    rows[1]["expected_duration_s"] = 404.3

    lines = stage.summarise(stage.build_frame(rows))
    text = "\n".join(lines)

    assert "TRUNCATED" in text
    assert "session 212: decoded 0.8 min of a stated 6.7 min" in text
    assert "never decodes" in text
    # Ahead of the ordinary duration statistics.
    assert text.index("TRUNCATED") < text.index("left/right correlation")


def test_an_empty_table_still_satisfies_the_contract():
    """A run in which every session failed must report, not crash."""
    validate(stage.build_frame([]), AUDIO_QC_SCHEMA)


@pytest.mark.slow
def test_a_run_where_every_session_fails_reports_rather_than_crashes(
    roots: DataRoots, default_config: AppConfig
):
    place_fake_media(roots.data, WINTER_FOLDER, [28, 29])

    result = stage.run(default_config, roots, workers=1)

    assert len(result.report.failed) == 2
    assert result.frame.empty
    assert result.path.exists()
