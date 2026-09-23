"""The turns stage.

The arithmetic itself is covered in test_turn_math.py. What is tested here is
the stage around it: that it refuses to run without a role mapping, that the
tables it writes are what the facial stages will read, and that the features it
records match the hand-worked values for a session built with known timing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.conftest import WINTER_FOLDER, place_fake_media
from vc_multimodal.config import AppConfig, load_config
from vc_multimodal.contracts import TURN_SCHEMA, feature_schema, validate
from vc_multimodal.features.spans import Span
from vc_multimodal.features.turn_math import FEATURE_NAMES, ROLE_PARTICIPANT, ROLE_PSYCHIATRIST
from vc_multimodal.io_utils import read_parquet, write_parquet
from vc_multimodal.paths import DataRoots, RawSession
from vc_multimodal.roles import RoleMapping, write_role_mapping
from vc_multimodal.stages import turns as stage
from vc_multimodal.stages import vad as vad_stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"

ROLES = {"SPEAKER_00": ROLE_PSYCHIATRIST, "SPEAKER_01": ROLE_PARTICIPANT}


def _session(session_id: int = 28) -> RawSession:
    return RawSession(
        session_id=session_id,
        wave="winter",
        date_folder=WINTER_FOLDER,
        path=Path(f"/nowhere/{session_id}.mp4"),
    )


def write_speech(roots: DataRoots, session_id: int, rows: list[tuple[str, float, float]]) -> Path:
    """Write a speech-span table as `vc vad` would."""
    frame = pd.DataFrame(rows, columns=["speaker", "start_s", "end_s"])
    frame.insert(0, "session_id", session_id)
    frame["session_id"] = frame["session_id"].astype("int64")
    frame["speaker"] = frame["speaker"].astype("string")
    for column in ("start_s", "end_s"):
        frame[column] = frame[column].astype("float64")
    return write_parquet(vad_stage.speech_path(roots, session_id), frame)


def clean_exchange_rows() -> list[tuple[str, float, float]]:
    """Two exchanges with exactly 1 s between speakers, worked out by hand."""
    return [
        ("SPEAKER_00", 0.0, 10.0),
        ("SPEAKER_01", 11.0, 19.0),
        ("SPEAKER_00", 20.0, 30.0),
        ("SPEAKER_01", 32.0, 40.0),
    ]


# ---------------------------------------------------------------------------
# refusing to run without a role mapping
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_without_a_role_mapping_the_session_fails_with_instructions(
    roots: DataRoots, raw_tree: Path, default_config: AppConfig
):
    """Guessing which speaker is the participant would silently invert features."""
    place_fake_media(roots.data, WINTER_FOLDER, [28])
    write_speech(roots, 28, clean_exchange_rows())

    result = stage.run(default_config, roots, workers=1)

    assert [o.session_id for o in result.report.failed] == [28]
    message = result.report.failed[0].message
    assert "vc assign-speakers" in message
    assert "roles.csv" in message


@pytest.mark.slow
def test_a_hand_written_mapping_lets_a_session_be_piloted(
    roots: DataRoots, raw_tree: Path, default_config: AppConfig
):
    place_fake_media(roots.data, WINTER_FOLDER, [28])
    write_speech(roots, 28, clean_exchange_rows())
    pd.DataFrame(
        [(28, speaker, role) for speaker, role in ROLES.items()],
        columns=["session_id", "speaker", "role"],
    ).to_csv(roots.work / "roles.csv", index=False)

    result = stage.run(default_config, roots, workers=1)

    assert result.report.ok
    row = result.frame.iloc[0]
    assert row["qc__role_source"] == "manual"
    # Flagged, so a manual mapping is never mistaken for recorded evidence.
    assert stage.FLAG_MANUAL_ROLES in row["qc__flags"]


@pytest.mark.slow
def test_missing_speech_spans_point_at_the_vad_stage(
    roots: DataRoots, raw_tree: Path, default_config: AppConfig
):
    place_fake_media(roots.data, WINTER_FOLDER, [28])
    write_role_mapping(roots.work, 28, ROLES)

    result = stage.run(default_config, roots, workers=1)

    assert "vc vad" in result.report.failed[0].message


# ---------------------------------------------------------------------------
# the tables it writes
# ---------------------------------------------------------------------------
@pytest.fixture
def prepared(roots: DataRoots, raw_tree: Path) -> Any:
    """A session with speech spans and a recorded role assignment."""

    def factory(
        session_id: int = 28, rows: list[tuple[str, float, float]] | None = None
    ) -> AppConfig:
        place_fake_media(roots.data, WINTER_FOLDER, [session_id])
        write_speech(roots, session_id, rows or clean_exchange_rows())
        write_role_mapping(roots.work, session_id, ROLES)
        return load_config(DEFAULT)

    return factory


@pytest.mark.slow
def test_the_turn_table_records_each_turn_and_its_latency(roots: DataRoots, prepared: Any):
    config = prepared()

    stage.run(config, roots, workers=1)

    turns = read_parquet(stage.turns_path(roots, 28))
    validate(turns, TURN_SCHEMA)
    assert list(turns["role"]) == [
        ROLE_PSYCHIATRIST,
        ROLE_PARTICIPANT,
        ROLE_PSYCHIATRIST,
        ROLE_PARTICIPANT,
    ]
    # Latency belongs to the responding turn, and only to it.
    latencies = dict(zip(turns["turn_index"], turns["latency_s"], strict=True))
    assert latencies[1] == pytest.approx(1.0)
    assert latencies[3] == pytest.approx(2.0)
    assert pd.isna(latencies[0])
    assert pd.isna(latencies[2])


@pytest.mark.slow
def test_the_timeline_is_written_for_the_facial_stages(roots: DataRoots, prepared: Any):
    config = prepared()

    stage.run(config, roots, workers=1)

    timeline = read_parquet(stage.timeline_path(roots, 28))
    assert set(timeline["state"]) == {"speaking", "listening"}
    speaking = timeline.loc[timeline["state"] == "speaking"]
    listening = timeline.loc[timeline["state"] == "listening"]
    # Participant speaks 8 + 8 s; the psychiatrist's 10 + 10 s is listening.
    assert (speaking["end_s"] - speaking["start_s"]).sum() == pytest.approx(16.0)
    assert (listening["end_s"] - listening["start_s"]).sum() == pytest.approx(20.0)


@pytest.mark.slow
def test_the_work_tables_stay_in_the_work_tree(roots: DataRoots, prepared: Any):
    config = prepared()
    stage.run(config, roots, workers=1)
    assert stage.turns_path(roots, 28).is_relative_to(roots.work)
    assert stage.timeline_path(roots, 28).is_relative_to(roots.work)


@pytest.mark.slow
def test_the_feature_table_follows_the_naming_convention(roots: DataRoots, prepared: Any):
    config = prepared()

    result = stage.run(config, roots, workers=1)

    validate(result.frame, feature_schema([*FEATURE_NAMES, *stage.QC_COLUMNS]))
    assert set(FEATURE_NAMES) <= set(result.frame.columns)


@pytest.mark.slow
def test_the_recorded_features_match_the_hand_worked_values(roots: DataRoots, prepared: Any):
    """The same session as test_turn_math's worked example, through the stage."""
    config = prepared()

    row = stage.run(config, roots, workers=1).frame.iloc[0]

    assert row["turns__latency_mean"] == pytest.approx(1.5)
    assert row["turns__latency_median"] == pytest.approx(1.5)
    assert row["turns__participant_speaking_ratio"] == pytest.approx(16.0 / 36.0)
    assert row["turns__overlap_ratio"] == pytest.approx(0.0)
    assert row["turns__participant_turn_duration_mean"] == pytest.approx(8.0)
    assert row["turns__psychiatrist_turn_duration_mean"] == pytest.approx(10.0)
    assert row["qc__n_turns"] == 4
    assert row["qc__n_latencies"] == 2
    assert row["qc__n_interruptions"] == 0


@pytest.mark.slow
def test_an_interruption_is_counted_and_kept_out_of_the_latency_mean(
    roots: DataRoots, prepared: Any
):
    config = prepared(
        28,
        [
            ("SPEAKER_00", 0.0, 10.0),
            ("SPEAKER_01", 9.0, 18.0),  # begins before the psychiatrist stops
            ("SPEAKER_00", 20.0, 25.0),
            ("SPEAKER_01", 26.0, 30.0),
        ],
    )

    row = stage.run(config, roots, workers=1).frame.iloc[0]

    assert row["qc__n_interruptions"] == 1
    assert row["qc__n_latencies"] == 1
    assert row["turns__latency_mean"] == pytest.approx(1.0)
    assert row["turns__overlap_ratio"] > 0.0


@pytest.mark.slow
def test_a_session_with_one_speaker_is_flagged_not_silently_zeroed(roots: DataRoots, prepared: Any):
    config = prepared(28, [("SPEAKER_00", 0.0, 30.0)])

    row = stage.run(config, roots, workers=1).frame.iloc[0]

    assert stage.FLAG_NO_PARTICIPANT in row["qc__flags"]
    assert pd.isna(row["turns__latency_mean"])
    assert pd.isna(row["turns__participant_turn_duration_mean"])


@pytest.mark.slow
def test_too_few_latencies_is_flagged(roots: DataRoots, prepared: Any):
    """Latency statistics over one or two transitions mean very little."""
    config = prepared(28, [("SPEAKER_00", 0.0, 5.0), ("SPEAKER_01", 6.0, 10.0)])
    row = stage.run(config, roots, workers=1).frame.iloc[0]
    assert stage.FLAG_NO_LATENCIES in row["qc__flags"]


@pytest.mark.slow
def test_an_unassigned_speaker_is_flagged(roots: DataRoots, prepared: Any):
    config = prepared(28, [*clean_exchange_rows(), ("SPEAKER_07", 45.0, 50.0)])
    row = stage.run(config, roots, workers=1).frame.iloc[0]
    assert stage.FLAG_UNKNOWN_SPEAKER in row["qc__flags"]


@pytest.mark.slow
def test_completed_sessions_are_skipped_but_stay_in_the_feature_table(
    roots: DataRoots, prepared: Any
):
    config = prepared()
    stage.run(config, roots, workers=1)

    second = stage.run(config, roots, workers=1)

    assert len(second.report.skipped) == 1
    assert len(second.frame) == 1
    assert second.frame.iloc[0]["turns__latency_mean"] == pytest.approx(1.5)


@pytest.mark.slow
def test_force_recomputes(roots: DataRoots, prepared: Any):
    config = prepared()
    stage.run(config, roots, workers=1)
    assert len(stage.run(config, roots, workers=1, force=True).report.succeeded) == 1


# ---------------------------------------------------------------------------
# rates use the decoded duration
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_rates_are_based_on_the_decoded_audio_duration(roots: DataRoots, prepared: Any):
    """One recording's file is truncated, so its stated duration would lie."""
    config = prepared()
    pd.DataFrame(
        {
            "session_id": [28],
            "duration_s": [120.0],
        }
    ).to_csv(roots.out / "audio_qc.csv", index=False)

    row = stage.run(config, roots, workers=1).frame.iloc[0]

    # Four turns in two minutes.
    assert row["turns__n_per_minute"] == pytest.approx(2.0)


@pytest.mark.slow
def test_without_any_duration_the_last_speech_offset_is_used(roots: DataRoots, prepared: Any):
    config = prepared()
    row = stage.run(config, roots, workers=1).frame.iloc[0]
    # The exchange ends at 40 s, so four turns over 40 s is six per minute.
    assert row["turns__n_per_minute"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# tables and summary
# ---------------------------------------------------------------------------
def test_the_feature_row_is_built_without_touching_the_filesystem():
    by_role = {
        ROLE_PSYCHIATRIST: (Span(0.0, 10.0), Span(20.0, 30.0)),
        ROLE_PARTICIPANT: (Span(11.0, 19.0), Span(32.0, 40.0)),
    }
    row = stage.feature_row(
        _session(),
        by_role,
        RoleMapping(28, ROLES, "assigned"),
        duration_s=60.0,
        config=load_config(DEFAULT),
    )
    assert row["turns__latency_mean"] == pytest.approx(1.5)
    # Two transitions is below the minimum for latency statistics to mean much,
    # so that flag is expected here and is the only one.
    assert row["qc__flags"] == stage.FLAG_NO_LATENCIES


def test_an_empty_feature_table_is_typed_and_valid():
    frame = stage.build_frame([])
    validate(frame, feature_schema([*FEATURE_NAMES, *stage.QC_COLUMNS]))
    assert str(frame["session_id"].dtype) == "int64"


def test_the_summary_reports_the_headline_measures():
    row = stage.feature_row(
        _session(),
        {
            ROLE_PSYCHIATRIST: (Span(0.0, 10.0),),
            ROLE_PARTICIPANT: (Span(11.0, 19.0),),
        },
        RoleMapping(28, ROLES, "assigned"),
        duration_s=60.0,
        config=load_config(DEFAULT),
    )
    text = "\n".join(stage.summarise(stage.build_frame([row])))
    assert "turns per minute" in text
    assert "median response latency" in text
    assert "role mapping source: assigned -> 1 session(s)" in text


def test_the_summary_of_nothing():
    assert stage.summarise(pd.DataFrame()) == ["no sessions were processed"]
