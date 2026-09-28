"""The aggregate stage: the table that ships.

Tested hardest are the properties a reader of `features.csv` depends on: that a
session missing a stage still appears and says so, that a window with too
little measured time contributes nothing rather than noise, that the two
facial backends are never pooled, and that the twelve confirmatory features
named in ADR 13 are actually produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import WINTER_FOLDER, place_fake_media
from vc_multimodal.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from vc_multimodal.contracts import ContractError, feature_schema, validate
from vc_multimodal.faces import FaceError, MediaPipeBackend
from vc_multimodal.features.spans import Span
from vc_multimodal.io_utils import write_parquet
from vc_multimodal.paths import DataRoots, RawSession
from vc_multimodal.qc_notes import QcNote, QcNotes
from vc_multimodal.runner import StageReport
from vc_multimodal.stages import aggregate as stage
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import prosody as prosody_stage
from vc_multimodal.stages import turns as turns_stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"


def _session(session_id: int = 28) -> RawSession:
    return RawSession(
        session_id=session_id,
        wave="winter",
        date_folder=WINTER_FOLDER,
        path=Path(f"/nowhere/{session_id}.mp4"),
    )


def face_frames(
    config: AppConfig,
    *,
    n: int = 600,
    step_s: float = 0.2,
    value: float = 0.4,
    detected: bool = True,
) -> pd.DataFrame:
    """A per-frame facial table as `vc face` writes it."""
    data: dict[str, object] = {
        "session_id": [28] * n,
        "frame_index": [i * 5 for i in range(n)],
        "timestamp_s": [i * step_s for i in range(n)],
        "detected": [detected] * n,
        "confidence": [1.0 if detected else 0.0] * n,
    }
    for key in config.face.unit_keys:
        data[key] = [value if detected else None] * n
    data["jaw"] = [0.2] * n
    data["blink"] = [0.1] * n
    for axis in ("head_pitch", "head_yaw", "head_roll"):
        data[axis] = [float(i % 7) for i in range(n)]
    frame = pd.DataFrame(data)
    frame["detected"] = frame["detected"].astype(bool)
    return frame


def timeline_frame(speaking: list[Span], listening: list[Span]) -> pd.DataFrame:
    rows = [(28, "speaking", s.start, s.end) for s in speaking]
    rows += [(28, "listening", s.start, s.end) for s in listening]
    frame = pd.DataFrame(rows, columns=["session_id", "state", "start_s", "end_s"])
    frame["session_id"] = frame["session_id"].astype("int64")
    frame["state"] = frame["state"].astype("string")
    for column in ("start_s", "end_s"):
        frame[column] = frame[column].astype("float64")
    return frame


# ---------------------------------------------------------------------------
# the timeline
# ---------------------------------------------------------------------------
def test_the_timeline_splits_into_two_states():
    frame = timeline_frame([Span(0.0, 10.0)], [Span(20.0, 30.0)])
    timeline = stage.Timeline.from_frame(frame)
    assert timeline.speaking == (Span(0.0, 10.0),)
    assert timeline.listening == (Span(20.0, 30.0),)


def test_an_unknown_state_is_ignored():
    frame = timeline_frame([Span(0.0, 1.0)], [])
    frame.loc[len(frame)] = [28, "daydreaming", 5.0, 6.0]
    timeline = stage.Timeline.from_frame(frame)
    assert timeline.speaking == (Span(0.0, 1.0),)
    assert timeline.listening == ()


def test_an_empty_timeline():
    timeline = stage.Timeline.from_frame(timeline_frame([], []))
    assert timeline.speaking == ()
    assert timeline.listening == ()


# ---------------------------------------------------------------------------
# summarising the two windows
# ---------------------------------------------------------------------------
def test_the_same_units_are_summarised_over_both_windows(default_config: AppConfig):
    """The contrast this project adds: same measure, two windows."""
    frames = face_frames(default_config)
    timeline = stage.Timeline(speaking=(Span(0.0, 60.0),), listening=(Span(60.0, 119.0),))

    features, windows = stage.summarise_face(frames, timeline, default_config)

    assert features["face_speaking__au12_mean"] == pytest.approx(0.4)
    assert features["face_listening__au12_mean"] == pytest.approx(0.4)
    assert windows.speaking_seconds == pytest.approx(60.0)
    assert windows.listening_seconds == pytest.approx(59.0)


def test_each_window_sees_only_its_own_frames(default_config: AppConfig):
    frames = face_frames(default_config)
    # A distinct value in the second half, to prove the windows separate.
    frames.loc[frames["timestamp_s"] >= 60.0, "au12"] = 0.9
    timeline = stage.Timeline(speaking=(Span(0.0, 60.0),), listening=(Span(60.0, 119.0),))

    features, _ = stage.summarise_face(frames, timeline, default_config)

    assert features["face_speaking__au12_mean"] == pytest.approx(0.4)
    assert features["face_listening__au12_mean"] == pytest.approx(0.9)


def test_a_window_with_too_little_measured_time_yields_nothing(
    default_config: AppConfig,
):
    """Ten frames of listening can produce a mean, and it would be noise."""
    frames = face_frames(default_config)
    timeline = stage.Timeline(speaking=(Span(0.0, 60.0),), listening=(Span(60.0, 62.0),))

    features, windows = stage.summarise_face(frames, timeline, default_config)

    assert features["face_speaking__au12_mean"] is not None
    assert features["face_listening__au12_mean"] is None
    assert windows.listening_seconds == pytest.approx(2.0)


def test_undetected_frames_do_not_contribute(default_config: AppConfig):
    frames = face_frames(default_config, detected=False)
    timeline = stage.Timeline(speaking=(Span(0.0, 119.0),), listening=())

    features, windows = stage.summarise_face(frames, timeline, default_config)

    assert features["face_speaking__au12_mean"] is None
    assert windows.measured_speaking == pytest.approx(0.0)


def test_partially_measured_windows_report_their_coverage(default_config: AppConfig):
    frames = face_frames(default_config)
    frames.loc[frames.index[:300], "detected"] = False
    timeline = stage.Timeline(speaking=(Span(0.0, 119.0),), listening=())

    _, windows = stage.summarise_face(frames, timeline, default_config)

    assert windows.measured_speaking == pytest.approx(0.5, abs=0.02)


def test_head_pose_is_summarised_as_movement_only(default_config: AppConfig):
    frames = face_frames(default_config)
    timeline = stage.Timeline(speaking=(Span(0.0, 119.0),), listening=())

    features, _ = stage.summarise_face(frames, timeline, default_config)

    assert features["face_speaking__head_yaw_sd"] is not None
    assert "face_speaking__head_yaw_mean" not in features


# ---------------------------------------------------------------------------
# the feature columns
# ---------------------------------------------------------------------------
def test_the_face_columns_cover_both_windows(default_config: AppConfig):
    names = stage.face_feature_names(default_config)
    assert sum(1 for name in names if name.startswith("face_speaking__")) == 15
    assert sum(1 for name in names if name.startswith("face_listening__")) == 15


def test_the_confirmatory_features_are_produced(default_config: AppConfig):
    """ADR 13 named these before any label existed; the stage must emit them."""
    produced = set(stage.face_feature_names(default_config))
    tiers = default_config.model.tiers
    for family in ("face_speaking", "face_listening"):
        for name in tiers.primary_features[family]:
            assert name in produced, name


# ---------------------------------------------------------------------------
# upstream tables
# ---------------------------------------------------------------------------
def write_upstream(roots: DataRoots, *, turns: bool = True, prosody: bool = True) -> None:
    """Write feature tables as the turn and prosody stages would."""
    if turns:
        turns_stage.build_frame(
            [
                {
                    "session_id": 28,
                    "wave": "winter",
                    "turns__latency_median": 1.2,
                    "turns__participant_speaking_ratio": 0.45,
                    "turns__overlap_ratio": 0.02,
                    "qc__role_source": "assigned",
                    "qc__flags": "",
                }
            ]
        ).to_csv(roots.out / turns_stage.TURN_FEATURES_FILENAME, index=False)
    if prosody:
        prosody_stage.build_frame(
            [
                {
                    "session_id": 28,
                    "wave": "winter",
                    "prosody__f0_semitone_sd": 2.5,
                    "qc__role_source": "assigned",
                    "qc__flags": "prosody_manual_role_mapping",
                }
            ]
        ).to_csv(roots.out / prosody_stage.PROSODY_FEATURES_FILENAME, index=False)


def test_both_upstream_tables_are_read(roots: DataRoots):
    write_upstream(roots)
    upstream = stage.load_upstream(roots)
    assert upstream.missing == ()
    columns = upstream.feature_columns()
    assert "turns__latency_median" in columns
    assert "prosody__f0_semitone_sd" in columns


def test_a_missing_upstream_table_is_named(roots: DataRoots):
    write_upstream(roots, prosody=False)
    assert stage.load_upstream(roots).missing == ("prosody",)


def test_upstream_flags_are_carried_forward(roots: DataRoots):
    write_upstream(roots)
    _, flags = stage.load_upstream(roots).row_for(28)
    assert "prosody_manual_role_mapping" in flags


def test_only_feature_and_selected_qc_columns_are_carried(roots: DataRoots):
    write_upstream(roots)
    values, _ = stage.load_upstream(roots).row_for(28)
    assert "turns__latency_median" in values
    assert values["qc__role_source"] == "assigned"
    # Upstream QC that belongs to its own table does not travel.
    assert "qc__n_turns" not in values


def test_a_session_absent_from_the_upstream_table_carries_nothing(roots: DataRoots):
    write_upstream(roots)
    values, flags = stage.load_upstream(roots).row_for(999)
    assert values == {}
    assert flags == []


# ---------------------------------------------------------------------------
# rows, including the incomplete ones
# ---------------------------------------------------------------------------
def prepare(
    roots: DataRoots, config: AppConfig, *, face: bool = True, timeline: bool = True
) -> None:
    """Put a session's inputs in place."""
    write_upstream(roots)
    if face:
        write_parquet(face_stage.face_path(roots, 28), face_frames(config))
        face_stage.write_backend_record(
            roots, 28, MediaPipeBackend(config.face.mediapipe, model_dir=Path("/nowhere"))
        )
    if timeline:
        write_parquet(
            turns_stage.timeline_path(roots, 28),
            timeline_frame([Span(0.0, 60.0)], [Span(60.0, 119.0)]),
        )


def test_a_complete_row_carries_features_from_every_stage(
    roots: DataRoots, default_config: AppConfig
):
    prepare(roots, default_config)
    row = stage._session_row(
        _session(),
        config=default_config,
        roots=roots,
        upstream=stage.load_upstream(roots),
        backends=face_stage.stored_backends(roots),
    )

    assert row["turns__latency_median"] == pytest.approx(1.2)
    assert row["prosody__f0_semitone_sd"] == pytest.approx(2.5)
    assert row["face_speaking__au12_mean"] == pytest.approx(0.4)
    assert row["qc__face_backend"] == "mediapipe"
    assert row["qc__stages_missing"] == ""


def test_a_session_with_no_facial_data_still_gets_a_row(
    roots: DataRoots, default_config: AppConfig
):
    """A quietly short table would be worse than an honest one with gaps."""
    prepare(roots, default_config, face=False)

    row = stage._session_row(
        _session(),
        config=default_config,
        roots=roots,
        upstream=stage.load_upstream(roots),
        backends={},
    )

    assert row["turns__latency_median"] == pytest.approx(1.2)
    assert row["face_speaking__au12_mean"] is None
    assert "face" in str(row["qc__stages_missing"])
    assert stage.FLAG_NO_FACE_DATA in str(row["qc__flags"])
    assert stage.FLAG_MISSING_STAGE in str(row["qc__flags"])


def test_a_missing_timeline_is_reported_as_the_turns_stage(
    roots: DataRoots, default_config: AppConfig
):
    prepare(roots, default_config, timeline=False)
    row = stage._session_row(
        _session(),
        config=default_config,
        roots=roots,
        upstream=stage.load_upstream(roots),
        backends=face_stage.stored_backends(roots),
    )
    assert "turns" in str(row["qc__stages_missing"])


def test_a_short_window_is_flagged(roots: DataRoots, default_config: AppConfig):
    prepare(roots, default_config, timeline=False)
    write_parquet(
        turns_stage.timeline_path(roots, 28),
        timeline_frame([Span(0.0, 60.0)], [Span(60.0, 61.0)]),
    )

    row = stage._session_row(
        _session(),
        config=default_config,
        roots=roots,
        upstream=stage.load_upstream(roots),
        backends=face_stage.stored_backends(roots),
    )

    assert stage.FLAG_SHORT_LISTENING in str(row["qc__flags"])
    assert stage.FLAG_SHORT_SPEAKING not in str(row["qc__flags"])


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def test_the_table_satisfies_the_feature_contract(default_config: AppConfig):
    columns = stage.all_feature_names(default_config, upstream=("turns__latency_median",))
    row: dict[str, object] = {"session_id": 28, "wave": "winter"}
    row.update(dict.fromkeys(columns, 0.5))
    row.update(dict.fromkeys(stage.QC_COLUMNS, ""))
    row["qc__face_frames_speaking"] = 10
    row["qc__face_frames_listening"] = 10

    frame = stage.build_frame([row], columns)

    validate(frame, feature_schema([*columns, *stage.QC_COLUMNS]))
    assert str(frame["session_id"].dtype) == "int64"


def test_an_empty_table_is_typed(default_config: AppConfig):
    columns = stage.all_feature_names(default_config, upstream=())
    frame = stage.build_frame([], columns)
    validate(frame, feature_schema([*columns, *stage.QC_COLUMNS]))
    assert frame.empty


def test_the_feature_budget_is_a_ceiling_not_a_limit(default_config: AppConfig):
    """It catches a bug generating columns; the tiering does the real work."""
    stage._check_budget(["turns__a"] * 10, default_config)

    with pytest.raises(ContractError, match="above the ceiling"):
        stage._check_budget(["turns__a"] * 500, default_config)


def test_the_budget_error_points_at_the_decision(default_config: AppConfig):
    with pytest.raises(ContractError) as caught:
        stage._check_budget(["turns__a"] * 500, default_config)
    assert "not a scientific limit" in str(caught.value)
    assert "0012" in str(caught.value)


# ---------------------------------------------------------------------------
# running the stage
# ---------------------------------------------------------------------------
@pytest.fixture
def one_session(roots: DataRoots, raw_tree: Path, default_config: AppConfig) -> AppConfig:
    place_fake_media(roots.data, WINTER_FOLDER, [28])
    prepare(roots, default_config)
    return default_config


def test_the_stage_writes_the_feature_table(roots: DataRoots, one_session: AppConfig):
    result = stage.run(one_session, roots)

    assert result.report.ok
    assert result.path == roots.out / "features.csv"
    assert len(result.frame) == 1
    assert result.frame.iloc[0]["face_speaking__au12_mean"] == pytest.approx(0.4)


def test_the_written_table_carries_every_family(roots: DataRoots, one_session: AppConfig):
    result = stage.run(one_session, roots)
    families = {name.split("__")[0] for name in result.feature_columns}
    assert families == {"turns", "prosody", "face_speaking", "face_listening"}


def test_the_stage_reports_the_analysis_tiers(roots: DataRoots, one_session: AppConfig):
    """The count of confirmatory tests is the thing a reviewer looks for."""
    result = stage.run(one_session, roots)
    text = "\n".join(result.tier_lines)
    assert "confirmatory" in text
    assert "4 test(s)" in text
    assert "holm correction" in text


def test_the_stage_refuses_facial_measures_from_two_backends(
    roots: DataRoots, one_session: AppConfig
):
    write_parquet(face_stage.face_path(roots, 3), face_frames(one_session))
    record = face_stage.backend_sidecar_path(roots, 3)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        '{"session_id": 3, "backend": "openface", "backend_version": "x"}',
        encoding="utf-8",
    )

    with pytest.raises(FaceError, match="more than one backend"):
        stage.run(one_session, roots)


def test_a_session_with_nothing_at_all_still_appears(
    roots: DataRoots, raw_tree: Path, default_config: AppConfig
):
    place_fake_media(roots.data, WINTER_FOLDER, [28])

    result = stage.run(default_config, roots)

    assert len(result.frame) == 1
    row = result.frame.iloc[0]
    assert row["session_id"] == 28
    assert stage.FLAG_MISSING_STAGE in str(row["qc__flags"])
    assert "turns" in str(row["qc__stages_missing"])


def test_the_run_notes_which_upstream_tables_were_absent(
    roots: DataRoots, raw_tree: Path, default_config: AppConfig
):
    place_fake_media(roots.data, WINTER_FOLDER, [28])
    result = stage.run(default_config, roots)
    assert any("vc turns" in note for note in result.report.notes)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_the_summary_reports_counts_by_family(roots: DataRoots, one_session: AppConfig):
    result = stage.run(one_session, roots)
    text = "\n".join(stage.summarise(result, one_session))
    assert "by family:" in text
    assert "face_speaking=15" in text
    assert "feature(s)" in text


def test_the_summary_reports_completeness(roots: DataRoots, one_session: AppConfig):
    result = stage.run(one_session, roots)
    text = "\n".join(stage.summarise(result, one_session))
    assert "per-session completeness" in text


def test_the_summary_reports_the_backend(roots: DataRoots, one_session: AppConfig):
    result = stage.run(one_session, roots)
    assert "facial backend: mediapipe" in "\n".join(stage.summarise(result, one_session))


def test_the_summary_of_nothing(default_config: AppConfig):
    empty = stage.AggregateResult(
        report=stage.StageReport(stage="aggregate"),
        frame=pd.DataFrame(),
        path=Path("/nowhere"),
    )
    assert stage.summarise(empty, default_config) == ["no sessions were aggregated"]


# ---------------------------------------------------------------------------
# features that cannot contribute
#
# The diarization output partitions time, so simultaneous speech is absent from
# it by construction and the overlap features are exactly zero in all 62
# sessions. One of them held a confirmatory slot. Noticing that by eye is not a
# process; this is.
# ---------------------------------------------------------------------------
def test_a_constant_feature_is_named(default_config: AppConfig):
    frame = pd.DataFrame(
        {
            "session_id": [1, 2, 3],
            "turns__overlap_ratio": [0.0, 0.0, 0.0],
            "turns__latency_median": [1.0, 2.0, 3.0],
        }
    )
    found = stage.constant_features(frame, ["turns__overlap_ratio", "turns__latency_median"])
    assert found == ("turns__overlap_ratio",)


def test_a_feature_varying_only_in_the_last_bit_counts_as_constant(
    default_config: AppConfig,
):
    frame = pd.DataFrame({"session_id": [1, 2], "turns__a": [1.0, 1.0 + 1e-15]})
    assert stage.constant_features(frame, ["turns__a"]) == ("turns__a",)


def test_an_entirely_missing_feature_is_not_called_constant(default_config: AppConfig):
    """No values is a different problem from one value, and reads differently."""
    frame = pd.DataFrame({"session_id": [1, 2], "turns__a": [np.nan, np.nan]})
    assert stage.constant_features(frame, ["turns__a"]) == ()


def test_a_single_session_is_not_evidence_of_constancy(default_config: AppConfig):
    """With one row every feature looks constant, so the report would be noise."""
    frame = pd.DataFrame({"session_id": [1], "turns__a": [1.0]})
    # It is reported, but the summary is only meaningful across a cohort; the
    # behaviour is documented rather than special-cased.
    assert stage.constant_features(frame, ["turns__a"]) == ("turns__a",)


def test_the_summary_calls_out_a_constant_confirmatory_feature(
    default_config: AppConfig,
):
    """A wasted pre-registered slot is a design problem, not a curiosity."""
    primary = default_config.model.tiers.primary_features["turns"][0]
    result = stage.AggregateResult(
        report=stage.StageReport(stage="aggregate"),
        frame=pd.DataFrame(
            {
                "session_id": [1, 2],
                "wave": ["winter", "winter"],
                primary: [0.0, 0.0],
                "qc__flags": ["", ""],
                "qc__speaking_seconds": [60.0, 60.0],
                "qc__listening_seconds": [60.0, 60.0],
                "qc__face_backend": ["mediapipe", "mediapipe"],
            }
        ),
        path=Path("/nowhere"),
        feature_columns=(primary,),
        constant_features=(primary,),
    )

    text = "\n".join(stage.summarise(result, default_config))

    assert "NO VARIANCE" in text
    assert "CONFIRMATORY" in text
    assert "needs replacing" in text


def test_the_summary_is_quiet_when_everything_varies(default_config: AppConfig):
    result = stage.AggregateResult(
        report=stage.StageReport(stage="aggregate"),
        frame=pd.DataFrame(
            {
                "session_id": [1, 2],
                "wave": ["winter", "winter"],
                "turns__a": [1.0, 2.0],
                "qc__flags": ["", ""],
                "qc__speaking_seconds": [60.0, 60.0],
                "qc__listening_seconds": [60.0, 60.0],
                "qc__face_backend": ["mediapipe", "mediapipe"],
            }
        ),
        path=Path("/nowhere"),
        feature_columns=("turns__a",),
    )
    assert "NO VARIANCE" not in "\n".join(stage.summarise(result, default_config))


def test_the_overlap_features_are_no_longer_confirmatory(default_config: AppConfig):
    """They are exactly zero in every session with this diarization source."""
    primary = set(default_config.model.tiers.primary_columns)
    assert "turns__overlap_ratio" not in primary
    assert "turns__interruption_rate" not in primary
    # And the slot went to a feature that does vary.
    assert "turns__n_per_minute" in primary


# ---------------------------------------------------------------------------
# human-confirmed QC notes
# ---------------------------------------------------------------------------
BLUR = (
    "participant's camera is too out of focus for face tracking; confirmed by "
    "watching the recording"
)


def annotated_frame() -> pd.DataFrame:
    """A table with both modalities present for two sessions.

    Built with every QC column the real table carries, so `summarise` can read
    it as it would a real one.
    """
    frame = pd.DataFrame(
        {
            "session_id": [43, 44],
            "wave": ["winter", "winter"],
            "turns__latency_median": [1.5, 2.0],
            "prosody__f0_semitone_sd": [2.5, 3.0],
            "face_speaking__au12_mean": [0.1, 0.2],
            "face_listening__au06_mean": [0.3, 0.4],
        }
    )
    for column in stage.QC_COLUMNS:
        if column in frame.columns:
            continue
        frame[column] = ["", ""] if column in stage._STRING_QC else [1.0, 1.0]
    frame["qc__flags"] = ["face_too_many_frames_dropped", ""]
    return frame


def face_unavailable(session_id: int = 43) -> QcNotes:
    return QcNotes(notes=(QcNote(session_id, "face", "unavailable", BLUR, "tester", "2026-09-28"),))


FEATURE_COLUMNS = [
    "turns__latency_median",
    "prosody__f0_semitone_sd",
    "face_speaking__au12_mean",
    "face_listening__au06_mean",
]


class TestConfirmedNotesAreApplied:
    def test_the_named_modality_is_withheld(self) -> None:
        updated, blanked = stage.apply_qc_notes(
            annotated_frame(), face_unavailable(), FEATURE_COLUMNS
        )
        row = updated.set_index("session_id").loc[43]
        assert pd.isna(row["face_speaking__au12_mean"])
        assert pd.isna(row["face_listening__au06_mean"])
        assert set(blanked[43]) == {"face_speaking__au12_mean", "face_listening__au06_mean"}

    def test_the_other_modalities_are_untouched(self) -> None:
        # The whole point of scoping a note: a blurry camera says nothing about
        # the audio, so session 43 keeps its turn-taking and prosodic features.
        updated, _ = stage.apply_qc_notes(annotated_frame(), face_unavailable(), FEATURE_COLUMNS)
        row = updated.set_index("session_id").loc[43]
        assert row["turns__latency_median"] == 1.5
        assert row["prosody__f0_semitone_sd"] == 2.5

    def test_other_sessions_are_untouched(self) -> None:
        updated, _ = stage.apply_qc_notes(annotated_frame(), face_unavailable(), FEATURE_COLUMNS)
        row = updated.set_index("session_id").loc[44]
        assert row["face_speaking__au12_mean"] == 0.2
        assert row["qc__annotations"] == ""

    def test_the_finding_is_recorded_on_the_row(self) -> None:
        updated, _ = stage.apply_qc_notes(annotated_frame(), face_unavailable(), FEATURE_COLUMNS)
        row = updated.set_index("session_id").loc[43]
        assert row["qc__annotations"] == "face=unavailable"
        assert BLUR in str(row["qc__annotation_reason"])

    def test_a_flag_is_added_without_losing_the_existing_ones(self) -> None:
        updated, _ = stage.apply_qc_notes(annotated_frame(), face_unavailable(), FEATURE_COLUMNS)
        flags = str(updated.set_index("session_id").loc[43]["qc__flags"]).split(";")
        assert "face_too_many_frames_dropped" in flags
        assert "annotated_face_speaking_unavailable" in flags

    def test_a_degraded_note_records_without_withholding(self) -> None:
        notes = QcNotes(notes=(QcNote(43, "face", "degraded", BLUR, "tester", "2026-09-28"),))
        updated, blanked = stage.apply_qc_notes(annotated_frame(), notes, FEATURE_COLUMNS)
        row = updated.set_index("session_id").loc[43]
        assert row["face_speaking__au12_mean"] == 0.1
        assert row["qc__annotations"] == "face=degraded"
        assert blanked == {}

    def test_an_audio_note_withholds_audio_and_keeps_face(self) -> None:
        notes = QcNotes(
            notes=(
                QcNote(
                    43,
                    "audio",
                    "unavailable",
                    "hum throughout, confirmed by listening",
                    "tester",
                    "2026-09-28",
                ),
            )
        )
        updated, _ = stage.apply_qc_notes(annotated_frame(), notes, FEATURE_COLUMNS)
        row = updated.set_index("session_id").loc[43]
        assert pd.isna(row["turns__latency_median"])
        assert pd.isna(row["prosody__f0_semitone_sd"])
        assert row["face_speaking__au12_mean"] == 0.1

    def test_no_notes_leaves_the_table_alone(self) -> None:
        original = annotated_frame()
        updated, blanked = stage.apply_qc_notes(original, QcNotes(), FEATURE_COLUMNS)
        assert blanked == {}
        pd.testing.assert_frame_equal(updated, original)

    def test_a_note_for_an_absent_session_is_reported_not_fatal(
        self, package_logs: pytest.LogCaptureFixture
    ) -> None:
        stage.apply_qc_notes(annotated_frame(), face_unavailable(999), FEATURE_COLUMNS)
        assert "999" in package_logs.text

    def test_the_summary_says_what_was_withheld_and_why(self) -> None:
        updated, blanked = stage.apply_qc_notes(
            annotated_frame(), face_unavailable(), FEATURE_COLUMNS
        )
        result = stage.AggregateResult(
            report=StageReport(stage=stage.STAGE),
            frame=updated,
            path=Path("features.csv"),
            feature_columns=tuple(FEATURE_COLUMNS),
            qc_notes=face_unavailable(),
            blanked_by_note=blanked,
        )
        text = "\n".join(stage.summarise(result, load_config(DEFAULT_CONFIG_PATH)))
        assert "session 43" in text
        assert BLUR in text
        assert "2 feature(s) withheld" in text
        assert "other modalities are unaffected" in text
