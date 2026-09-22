"""The inventory stage: flagging, typing, merging and the printed summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.conftest import SUMMER_FOLDER, WINTER_FOLDER, place_fake_media
from vc_multimodal.config import AppConfig, DurationChecks, load_config
from vc_multimodal.contracts import INVENTORY_SCHEMA, validate
from vc_multimodal.ffmpeg import AudioStreamInfo, MediaInfo
from vc_multimodal.paths import DataRoots, RawSession
from vc_multimodal.stages import inventory as stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"

CHECKS = DurationChecks(min_seconds=480.0, max_seconds=900.0, mad_k=4.0)


# ---------------------------------------------------------------------------
# robust outlier detection
# ---------------------------------------------------------------------------
def test_mad_flags_a_single_short_recording_among_many():
    durations = dict.fromkeys(range(1, 21), 660.0)
    durations[210] = 30.0
    assert stage.mad_outliers(durations, k=4.0) == (210,)


def test_mad_is_not_fooled_by_the_outlier_it_is_looking_for():
    """A mean/SD rule would let one extreme value inflate the threshold."""
    durations = dict.fromkeys(range(1, 11), 660.0)
    durations[11] = 700.0
    durations[210] = 5.0
    assert 210 in stage.mad_outliers(durations, k=4.0)


def test_mad_returns_nothing_when_every_duration_is_identical():
    assert stage.mad_outliers(dict.fromkeys(range(1, 11), 660.0), k=4.0) == ()


def test_outlier_detection_survives_a_zero_median_deviation():
    """Twenty identical recordings and one short one: the MAD itself is zero.

    A plain MAD rule divides by zero here and, guarded, flags nothing at all -
    failing in precisely the case it is meant to catch.
    """
    durations = dict.fromkeys(range(1, 21), 660.0)
    durations[210] = 30.0
    assert stage.mad_outliers(durations, k=4.0) == (210,)


def test_mad_needs_a_cohort():
    assert stage.mad_outliers({1: 10.0, 2: 700.0}, k=4.0) == ()


def test_mad_flags_long_recordings_too():
    durations = dict.fromkeys(range(1, 21), 660.0)
    durations[5] = 690.0
    durations[7] = 3600.0
    assert 7 in stage.mad_outliers(durations, k=4.0)


# ---------------------------------------------------------------------------
# duration flags
# ---------------------------------------------------------------------------
def test_absolute_window_flags_short_and_long():
    flags = stage.duration_flags({1: 100.0, 2: 660.0, 3: 5000.0}, CHECKS)
    assert stage.FLAG_SHORT in flags[1]
    assert 2 not in flags
    assert stage.FLAG_LONG in flags[3]


def test_known_short_session_is_marked_as_already_known():
    flags = stage.duration_flags({210: 30.0}, CHECKS, known_short=[210])
    assert stage.FLAG_SHORT in flags[210]
    assert stage.FLAG_KNOWN_SHORT in flags[210]


def test_known_short_marking_does_not_invent_a_flag():
    """A known-short session of normal length should not be flagged at all."""
    flags = stage.duration_flags({210: 660.0}, CHECKS, known_short=[210])
    assert 210 not in flags


def test_unreadable_files_are_not_duration_flagged():
    assert stage.duration_flags({1: None}, CHECKS) == {}


def test_both_tests_can_flag_the_same_session():
    durations = dict.fromkeys(range(1, 21), 660.0)
    durations[210] = 30.0
    flags = stage.duration_flags(durations, CHECKS, known_short=[210])
    assert set(flags[210]) == {stage.FLAG_SHORT, stage.FLAG_OUTLIER, stage.FLAG_KNOWN_SHORT}


# ---------------------------------------------------------------------------
# stream flags
# ---------------------------------------------------------------------------
def _info(**overrides: Any) -> MediaInfo:
    defaults: dict[str, Any] = {
        "duration_s": 660.0,
        "size_bytes": 1000,
        "video_codec": "h264",
        "width": 1920,
        "height": 1080,
        "fps": 25.0,
        "fps_variable": False,
        "audio_streams": (AudioStreamInfo(index=1, codec="aac", channels=1, sample_rate=48000),),
    }
    defaults.update(overrides)
    return MediaInfo(**defaults)


def test_a_normal_file_has_no_stream_flags():
    assert stage.stream_flags(_info()) == []


def test_two_audio_streams_are_flagged_for_attention():
    info = _info(
        audio_streams=(
            AudioStreamInfo(1, "aac", 1, 48000),
            AudioStreamInfo(2, "aac", 1, 48000),
        )
    )
    assert stage.FLAG_MULTI_AUDIO in stage.stream_flags(info)


def test_missing_audio_and_video_are_flagged():
    assert stage.FLAG_NO_AUDIO in stage.stream_flags(_info(audio_streams=()))
    assert stage.FLAG_NO_VIDEO in stage.stream_flags(_info(width=None, height=None))


def test_variable_frame_rate_is_flagged():
    assert stage.FLAG_VFR in stage.stream_flags(_info(fps_variable=True))


# ---------------------------------------------------------------------------
# table construction
# ---------------------------------------------------------------------------
def _session(session_id: int = 28) -> RawSession:
    return RawSession(
        session_id=session_id,
        wave="winter",
        date_folder=WINTER_FOLDER,
        path=Path(f"/nowhere/{WINTER_FOLDER}/{session_id}.mp4"),
    )


def test_a_probed_row_carries_every_column(default_config: AppConfig):
    frame = stage.build_frame([stage._row(_session(), _info(), None)])
    validate(frame, INVENTORY_SCHEMA)
    assert list(frame.columns) == list(stage.COLUMN_ORDER)
    assert frame.loc[0, "readable"]
    assert frame.loc[0, "relpath"] == f"{WINTER_FOLDER}/28.mp4"


def test_an_unreadable_file_still_gets_a_row_with_the_reason():
    frame = stage.build_frame([stage._row(_session(), None, "FfmpegError: moov atom not found")])
    validate(frame, INVENTORY_SCHEMA)
    assert not frame.loc[0, "readable"]
    assert frame.loc[0, "flags"].startswith(stage.FLAG_UNREADABLE)
    assert "moov atom" in frame.loc[0, "flags"]
    assert pd.isna(frame.loc[0, "duration_s"])


def test_an_all_unreadable_table_still_satisfies_the_contract():
    """Columns that are entirely missing must still carry their declared dtype."""
    rows = [stage._row(_session(i), None, "boom") for i in (1, 2, 3)]
    frame = stage.build_frame(rows)
    validate(frame, INVENTORY_SCHEMA)
    assert str(frame["duration_s"].dtype) == "float64"
    assert str(frame["width"].dtype) == "Int64"


def test_rows_are_sorted_by_session_id():
    rows = [stage._row(_session(i), _info(), None) for i in (210, 3, 28)]
    assert list(stage.build_frame(rows)["session_id"]) == [3, 28, 210]


def test_duration_flags_are_appended_to_existing_flags(default_config: AppConfig):
    info = _info(
        duration_s=30.0,
        audio_streams=(AudioStreamInfo(1, "aac", 1, 48000), AudioStreamInfo(2, "aac", 1, 48000)),
    )
    frame = stage.apply_duration_flags(
        stage.build_frame([stage._row(_session(210), info, None)]), default_config
    )
    flags = frame.loc[0, "flags"].split(";")
    assert stage.FLAG_MULTI_AUDIO in flags
    assert stage.FLAG_SHORT in flags
    assert stage.FLAG_KNOWN_SHORT in flags


# ---------------------------------------------------------------------------
# summary: metadata only
# ---------------------------------------------------------------------------
def test_summary_reports_counts_durations_and_stream_layout(default_config: AppConfig):
    rows = [
        stage._row(_session(1), _info(), None),
        stage._row(
            _session(2),
            _info(audio_streams=(AudioStreamInfo(1, "aac", 1, 48000),) * 2),
            None,
        ),
    ]
    lines = stage.summarise(stage.build_frame(rows), default_config)
    text = "\n".join(lines)
    assert "sessions: 2 (expected 62)" in text
    assert "1920x1080" in text
    assert "audio streams per file" in text
    assert "1 (1), 2 (1)" in text


def test_summary_names_unreadable_sessions(default_config: AppConfig):
    rows = [stage._row(_session(1), None, "boom"), stage._row(_session(2), _info(), None)]
    lines = stage.summarise(stage.build_frame(rows), default_config)
    assert any("UNREADABLE: [1]" in line for line in lines)


def test_summary_groups_flags_with_the_sessions_that_raised_them(default_config: AppConfig):
    rows = [stage._row(_session(210), _info(duration_s=30.0), None)]
    frame = stage.apply_duration_flags(stage.build_frame(rows), default_config)
    lines = stage.summarise(frame, default_config)
    assert any(stage.FLAG_SHORT in line and "[210]" in line for line in lines)


def test_summary_of_an_empty_inventory(default_config: AppConfig):
    assert stage.summarise(pd.DataFrame(), default_config) == ["inventory is empty"]


def test_summary_never_contains_a_filesystem_path(default_config: AppConfig):
    """Summaries go to stdout; they carry metadata, not locations on disk."""
    rows = [stage._row(_session(1), _info(), None)]
    text = "\n".join(stage.summarise(stage.build_frame(rows), default_config))
    assert "/nowhere" not in text


# ---------------------------------------------------------------------------
# running the stage
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_run_probes_real_recordings_and_writes_the_table(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)
    make_real_media(3, folder="January 17 2026")
    make_real_media(210, folder=SUMMER_FOLDER, per_speaker_audio=True)

    result = stage.run(default_config, roots, workers=1)

    assert result.path == roots.out / "inventory.csv"
    assert len(result.frame) == 3
    assert list(result.frame["session_id"]) == [3, 28, 210]
    assert set(result.frame["wave"]) == {"winter", "summer"}
    assert result.report.ok
    validate(result.frame, INVENTORY_SCHEMA)


@pytest.mark.slow
def test_run_detects_the_two_stream_session(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)
    make_real_media(3, folder="January 17 2026", per_speaker_audio=True)
    result = stage.run(default_config, roots, workers=1)
    streams = dict(zip(result.frame["session_id"], result.frame["n_audio_streams"], strict=True))
    assert streams[28] == 1
    assert streams[3] == 2
    flags = dict(zip(result.frame["session_id"], result.frame["flags"], strict=True))
    assert stage.FLAG_MULTI_AUDIO in flags[3]
    assert stage.FLAG_MULTI_AUDIO not in flags[28]


@pytest.mark.slow
def test_an_unreadable_file_does_not_stop_the_stage(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)
    place_fake_media(roots.data, WINTER_FOLDER, [29])  # a placeholder, not real media

    result = stage.run(default_config, roots, workers=1)

    assert len(result.frame) == 2
    assert len(result.report.failed) == 1
    assert result.report.failed[0].session_id == 29
    readable = dict(zip(result.frame["session_id"], result.frame["readable"], strict=True))
    assert readable[28]
    assert not readable[29]


@pytest.mark.slow
def test_running_a_subset_keeps_the_other_rows(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    """Piloting on three sessions must not discard the rest of the table."""
    make_real_media(28)
    make_real_media(3, folder="January 17 2026")
    stage.run(default_config, roots, workers=1)

    result = stage.run(default_config, roots, session_ids=[28], workers=1)

    assert sorted(result.frame["session_id"]) == [3, 28]
    assert [o.session_id for o in result.report.outcomes] == [28]


@pytest.mark.slow
def test_force_discards_the_previous_table(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)
    make_real_media(3, folder="January 17 2026")
    stage.run(default_config, roots, workers=1)

    result = stage.run(default_config, roots, session_ids=[28], workers=1, force=True)
    assert list(result.frame["session_id"]) == [28]


@pytest.mark.slow
def test_a_requested_but_absent_session_is_noted(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)
    result = stage.run(default_config, roots, session_ids=[28, 999], workers=1)
    assert any("999" in note for note in result.report.notes)


@pytest.mark.slow
def test_duration_outliers_are_flagged_across_a_cohort(roots: DataRoots, make_real_media: Any):
    """The window is narrowed so short synthetic clips exercise the real rule."""
    config = load_config(
        DEFAULT,
        overrides={"dataset.duration": {"min_seconds": 7.0, "max_seconds": 20.0, "mad_k": 2.0}},
    )
    make_real_media(28, duration=10.0)
    make_real_media(3, folder="January 17 2026", duration=10.0)
    make_real_media(210, folder=SUMMER_FOLDER, duration=4.0)

    result = stage.run(config, roots, workers=1)
    flags = dict(zip(result.frame["session_id"], result.frame["flags"], strict=True))
    assert stage.FLAG_SHORT in flags[210]
    assert stage.FLAG_KNOWN_SHORT in flags[210]
    assert flags[28] == ""
