"""The face stage, its backends, and the rule against mixing them.

The backend is expected to change: MediaPipe is the default only because an
OpenFace run has not been confirmed. So the things tested hardest here are the
ones that make that switch safe - the backend recorded on every row, the
refusal to pool two backends, and the note that says a switch is a full rerun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import REAL_FACE_MODEL, WINTER_FOLDER, place_fake_media
from tests.synth import generators as gen
from vc_multimodal.config import AppConfig, load_config
from vc_multimodal.faces import (
    FaceError,
    MediaPipeBackend,
    OpenFaceBackend,
    get_backend,
    require_single_backend,
)
from vc_multimodal.features.face_math import FrameMeasure
from vc_multimodal.features.sampling import resolve_sampling
from vc_multimodal.handoff_text import FACE_BACKEND_NOTE, GAZE_ABSENCE_NOTE, notes
from vc_multimodal.io_utils import read_parquet
from vc_multimodal.paths import DataRoots, RawSession
from vc_multimodal.stages import face as stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"


def _session(session_id: int = 28) -> RawSession:
    return RawSession(
        session_id=session_id,
        wave="winter",
        date_folder=WINTER_FOLDER,
        path=Path(f"/nowhere/{session_id}.mp4"),
    )


# ---------------------------------------------------------------------------
# the rule against mixing backends
# ---------------------------------------------------------------------------
def test_one_backend_is_accepted():
    assert require_single_backend(["mediapipe", "mediapipe"]) == "mediapipe"


def test_no_sessions_is_not_an_error():
    assert require_single_backend([]) == ""
    assert require_single_backend(["", ""]) == ""


def test_two_backends_are_refused():
    """Different scales for the same constructs: pooling them describes nothing."""
    with pytest.raises(FaceError, match="more than one backend"):
        require_single_backend(["mediapipe", "openface"])


def test_the_refusal_says_a_switch_is_a_full_rerun():
    with pytest.raises(FaceError) as caught:
        require_single_backend(["mediapipe", "openface"], context="the face QC table")
    message = str(caught.value)
    assert "never a top-up" in message
    assert "vc face --force" in message
    assert "the face QC table" in message
    assert "mediapipe" in message and "openface" in message


def test_the_stage_refuses_a_mixed_qc_table(roots: DataRoots, default_config: AppConfig):
    """The check runs where sessions are combined, which is where it matters."""
    rows = [
        _qc_row(28, backend="mediapipe"),
        _qc_row(3, backend="openface"),
    ]
    frame = stage.build_frame(rows)
    with pytest.raises(FaceError, match="more than one backend"):
        require_single_backend([str(name) for name in frame["backend"]])


# ---------------------------------------------------------------------------
# the handoff note
# ---------------------------------------------------------------------------
def test_the_handoff_note_explains_the_rerun_requirement():
    assert "not topping up" in FACE_BACKEND_NOTE
    assert "qc__face_backend" in FACE_BACKEND_NOTE
    assert "are not OpenFace action unit intensities" in FACE_BACKEND_NOTE


def test_the_handoff_note_tells_the_reader_what_to_do_about_a_mixed_bundle():
    """They will not read the code, so the instruction has to be in the README."""
    assert "ask for it to be rebuilt" in FACE_BACKEND_NOTE


def test_the_handoff_notes_record_the_absent_gaze_channel():
    assert "no gaze features" in GAZE_ABSENCE_NOTE.lower()
    assert "Tobii" in GAZE_ABSENCE_NOTE
    assert "absent from this bundle, not negative" in GAZE_ABSENCE_NOTE


def test_every_note_is_carried():
    assert FACE_BACKEND_NOTE in notes()
    assert GAZE_ABSENCE_NOTE in notes()


# ---------------------------------------------------------------------------
# the crop
# ---------------------------------------------------------------------------
def letterboxed_frame(bar: int = 180) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[bar : 720 - bar, :] = 190
    return frame


def test_the_crop_is_the_participant_tile_inside_the_content_area(
    default_config: AppConfig,
):
    crop, letterboxed = stage.participant_crop(default_config, letterboxed_frame())

    assert letterboxed
    assert crop.to_pixels(1280, 720) == (640, 180, 640, 360)


def test_the_crop_follows_the_configured_participant_side():
    config = load_config(
        DEFAULT,
        overrides={"video.participant_tile": "left", "video.psychiatrist_tile": "right"},
    )
    crop, _ = stage.participant_crop(config, letterboxed_frame())
    assert crop.to_pixels(1280, 720) == (0, 180, 640, 360)


def test_without_letterboxing_the_crop_is_half_the_frame(default_config: AppConfig):
    plain = np.full((720, 1280, 3), 190, dtype=np.uint8)
    crop, letterboxed = stage.participant_crop(default_config, plain)
    assert not letterboxed
    assert crop.to_pixels(1280, 720) == (640, 0, 640, 720)


def test_letterbox_correction_can_be_disabled():
    config = load_config(DEFAULT, overrides={"video.letterbox_detection": "off"})
    crop, letterboxed = stage.participant_crop(config, letterboxed_frame())
    assert not letterboxed
    # Half the whole frame, bars included, which is the wrong crop but asked for.
    assert crop.to_pixels(1280, 720) == (640, 0, 640, 720)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def frame_measures(n: int = 4, *, detected: bool = True) -> list[FrameMeasure]:
    return [
        FrameMeasure(
            frame_index=index * 5,
            timestamp_s=index * 0.2,
            detected=detected,
            confidence=1.0 if detected else 0.0,
            units={"au01": 0.1, "au02": 0.2, "au04": 0.3, "au06": 0.4, "au12": 0.5}
            if detected
            else {},
            jaw=0.2 if detected else None,
            blink=0.1 if detected else None,
            head=(1.0, 2.0, 3.0) if detected else None,
        )
        for index in range(n)
    ]


def test_the_per_frame_table_has_a_column_per_action_unit(default_config: AppConfig):
    table = stage.frame_table(28, frame_measures(), default_config.face.unit_keys)
    for key in default_config.face.unit_keys:
        assert key in table.columns
    assert list(table.columns)[:5] == [
        "session_id",
        "frame_index",
        "timestamp_s",
        "detected",
        "confidence",
    ]


def test_undetected_frames_are_kept_as_rows_with_no_values(default_config: AppConfig):
    """The dropped fraction is only meaningful if every frame is accounted for."""
    table = stage.frame_table(28, frame_measures(detected=False), default_config.face.unit_keys)
    assert len(table) == 4
    assert not table["detected"].any()
    assert table["au12"].isna().all()


def test_head_pose_is_split_into_three_columns(default_config: AppConfig):
    table = stage.frame_table(28, frame_measures(), default_config.face.unit_keys)
    assert table.iloc[0]["head_pitch"] == pytest.approx(1.0)
    assert table.iloc[0]["head_yaw"] == pytest.approx(2.0)
    assert table.iloc[0]["head_roll"] == pytest.approx(3.0)


def test_nothing_in_the_table_is_called_gaze(default_config: AppConfig):
    table = stage.frame_table(28, frame_measures(), default_config.face.unit_keys)
    assert not any("gaze" in column or "look" in column for column in table.columns)


def _qc_row(session_id: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_id": session_id,
        "wave": "winter",
        "backend": "mediapipe",
        "backend_version": "mediapipe/0.10.35+model:abc",
        "native_fps": 25.0,
        "sample_fps": 5.0,
        "frame_step": 5,
        "n_frames_sampled": 3480,
        "n_frames_measured": 3400,
        "dropped_fraction": 0.023,
        "mean_confidence": 1.0,
        "letterbox_detected": True,
        "crop_x": 0.5,
        "crop_y": 0.25,
        "crop_width": 0.5,
        "crop_height": 0.5,
        "flags": "",
    }
    row.update(overrides)
    return row


def test_the_qc_table_records_the_backend_on_every_row():
    frame = stage.build_frame([_qc_row(1), _qc_row(2)])
    assert set(frame["backend"]) == {"mediapipe"}
    assert frame["backend_version"].notna().all()


def test_an_empty_qc_table_is_typed():
    frame = stage.build_frame([])
    assert str(frame["session_id"].dtype) == "int64"
    assert frame.empty


def test_the_qc_record_flags_too_many_dropped_frames(default_config: AppConfig):
    record = stage.qc_record(
        _session(),
        frame_measures(10, detected=False),
        backend=MediaPipeBackend(default_config.face.mediapipe, model_dir=Path("/nowhere")),
        sampling=resolve_sampling(25.0, 5.0),
        crop=default_config.video.tiles["right"],
        letterboxed=True,
        config=default_config,
    )
    assert stage.FLAG_NO_FACE_FOUND in record["flags"]
    assert record["dropped_fraction"] == pytest.approx(1.0)


def test_the_qc_record_flags_an_undetected_letterbox(default_config: AppConfig):
    """Every real recording is letterboxed, so failing to find it is notable."""
    record = stage.qc_record(
        _session(),
        frame_measures(),
        backend=MediaPipeBackend(default_config.face.mediapipe, model_dir=Path("/nowhere")),
        sampling=resolve_sampling(25.0, 5.0),
        crop=default_config.video.tiles["right"],
        letterboxed=False,
        config=default_config,
    )
    assert stage.FLAG_NO_LETTERBOX in record["flags"]


def test_the_qc_record_notes_absent_head_pose(default_config: AppConfig):
    measures = [
        FrameMeasure(frame_index=0, timestamp_s=0.0, detected=True, confidence=1.0, head=None)
    ]
    record = stage.qc_record(
        _session(),
        measures,
        backend=MediaPipeBackend(default_config.face.mediapipe, model_dir=Path("/nowhere")),
        sampling=resolve_sampling(25.0, 5.0),
        crop=default_config.video.tiles["right"],
        letterboxed=True,
        config=default_config,
    )
    assert stage.FLAG_NO_HEAD_POSE in record["flags"]


def test_the_qc_record_reports_the_sampling_actually_used(default_config: AppConfig):
    record = stage.qc_record(
        _session(),
        frame_measures(),
        backend=MediaPipeBackend(default_config.face.mediapipe, model_dir=Path("/nowhere")),
        sampling=resolve_sampling(25.0, 5.0),
        crop=default_config.video.tiles["right"],
        letterboxed=True,
        config=default_config,
    )
    assert record["native_fps"] == pytest.approx(25.0)
    assert record["sample_fps"] == pytest.approx(5.0)
    assert record["frame_step"] == 5


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
def test_the_default_backend_is_mediapipe(roots: DataRoots, default_config: AppConfig):
    assert isinstance(get_backend(default_config, roots), MediaPipeBackend)


def test_the_openface_backend_can_be_selected(roots: DataRoots):
    config = load_config(
        DEFAULT,
        overrides={"face.backend": "openface", "face.openface.csv_dir": "openface"},
    )
    (roots.work / "openface").mkdir(parents=True, exist_ok=True)
    backend = get_backend(config, roots)
    assert isinstance(backend, OpenFaceBackend)
    assert backend.available()


def test_mediapipe_without_its_model_explains_where_to_put_it(
    roots: DataRoots, default_config: AppConfig
):
    backend = get_backend(default_config, roots)
    assert not backend.available()
    reason = backend.unavailable_reason()
    assert "face_landmarker.task" in reason
    assert "pinned by hash" in reason


def test_openface_without_a_directory_explains_itself(roots: DataRoots):
    config = load_config(DEFAULT, overrides={"face.backend": "openface"})
    backend = get_backend(config, roots)
    assert not backend.available()
    assert "csv_dir is not set" in backend.unavailable_reason()


def test_a_wrong_model_hash_is_refused(roots: DataRoots, default_config: AppConfig):
    """Features from a different model are not comparable with existing ones."""
    model_dir = roots.work / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "face_landmarker.task").write_bytes(b"not the real model")

    backend = MediaPipeBackend(default_config.face.mediapipe, model_dir=model_dir)

    assert backend.available()
    with pytest.raises(FaceError, match="but config pins"):
        backend.verify_model()


def test_an_unpinned_model_warns_rather_than_refusing(roots: DataRoots):
    config = load_config(DEFAULT, overrides={"face.mediapipe.model_sha256": None})
    model_dir = roots.work / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "face_landmarker.task").write_bytes(b"anything")
    MediaPipeBackend(config.face.mediapipe, model_dir=model_dir).verify_model()


# ---------------------------------------------------------------------------
# the OpenFace importer
# ---------------------------------------------------------------------------
def openface_csv(path: Path, *, n_frames: int = 10, success: int = 1) -> Path:
    """Write a CSV shaped like OpenFace output, headers and all."""
    frame = pd.DataFrame(
        {
            "frame": range(1, n_frames + 1),
            "timestamp": [i / 25.0 for i in range(n_frames)],
            "confidence": [0.95] * n_frames,
            "success": [success] * n_frames,
            "AU01_r": [0.5] * n_frames,
            "AU02_r": [0.2] * n_frames,
            "AU04_r": [0.1] * n_frames,
            "AU06_r": [1.2] * n_frames,
            "AU12_r": [2.4] * n_frames,
            "pose_Rx": [0.1] * n_frames,
            "pose_Ry": [0.2] * n_frames,
            "pose_Rz": [0.0] * n_frames,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_openface_output_is_imported(roots: DataRoots, default_config: AppConfig):
    csv_dir = roots.work / "openface"
    openface_csv(csv_dir / "28.csv", n_frames=25)
    backend = OpenFaceBackend(default_config.face.openface, csv_dir=csv_dir)

    measures = backend.measure_session(
        _session(),
        config=default_config,
        crop=default_config.video.tiles["right"],
        sampling=resolve_sampling(25.0, 5.0),
    )

    assert len(measures) == 5  # every fifth frame of 25
    assert measures[0].units["au12"] == pytest.approx(2.4)
    # OpenFace reports a graded confidence, unlike MediaPipe.
    assert measures[0].confidence == pytest.approx(0.95)
    assert measures[0].head is not None


def test_openface_failure_rows_become_undetected_frames(
    roots: DataRoots, default_config: AppConfig
):
    csv_dir = roots.work / "openface"
    openface_csv(csv_dir / "28.csv", success=0)
    backend = OpenFaceBackend(default_config.face.openface, csv_dir=csv_dir)

    measures = backend.measure_session(
        _session(),
        config=default_config,
        crop=default_config.video.tiles["right"],
        sampling=resolve_sampling(25.0, 5.0),
    )

    assert measures
    assert not any(measure.detected for measure in measures)


def test_a_missing_openface_csv_names_what_it_looked_for(
    roots: DataRoots, default_config: AppConfig
):
    csv_dir = roots.work / "openface"
    csv_dir.mkdir(parents=True, exist_ok=True)
    backend = OpenFaceBackend(default_config.face.openface, csv_dir=csv_dir)
    with pytest.raises(FaceError, match="no OpenFace output for session 28"):
        backend.measure_session(
            _session(),
            config=default_config,
            crop=default_config.video.tiles["right"],
            sampling=resolve_sampling(25.0, 5.0),
        )


def test_a_csv_without_action_units_says_how_to_produce_them(
    roots: DataRoots, default_config: AppConfig
):
    csv_dir = roots.work / "openface"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": [1], "confidence": [0.9], "success": [1]}).to_csv(
        csv_dir / "28.csv", index=False
    )
    backend = OpenFaceBackend(default_config.face.openface, csv_dir=csv_dir)
    with pytest.raises(FaceError, match="-aus"):
        backend.measure_session(
            _session(),
            config=default_config,
            crop=default_config.video.tiles["right"],
            sampling=resolve_sampling(25.0, 5.0),
        )


def test_openface_headers_with_leading_spaces_are_handled(
    roots: DataRoots, default_config: AppConfig
):
    """OpenFace writes ` confidence`, ` AU12_r` and so on."""
    csv_dir = roots.work / "openface"
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / "28.csv"
    openface_csv(path, n_frames=5)
    text = path.read_text(encoding="utf-8").splitlines()
    text[0] = ", ".join(f" {column}" for column in text[0].split(","))
    path.write_text("\n".join(text), encoding="utf-8")

    backend = OpenFaceBackend(default_config.face.openface, csv_dir=csv_dir)
    measures = backend.measure_session(
        _session(),
        config=default_config,
        crop=default_config.video.tiles["right"],
        sampling=resolve_sampling(25.0, 5.0),
    )
    assert measures[0].units["au12"] == pytest.approx(2.4)


def test_the_openface_version_records_where_it_was_imported_from(
    roots: DataRoots, default_config: AppConfig
):
    csv_dir = roots.work / "openface"
    csv_dir.mkdir(parents=True, exist_ok=True)
    version = OpenFaceBackend(default_config.face.openface, csv_dir=csv_dir).version()
    assert version.startswith("openface/imported:")
    assert "openface" in version


# ---------------------------------------------------------------------------
# running the stage against real video
# ---------------------------------------------------------------------------
@pytest.fixture
def model_available(roots: DataRoots) -> Path:
    """The real landmarker model, copied into the test work tree, or skip."""
    if REAL_FACE_MODEL is None:
        pytest.skip("the face landmarker model has not been downloaded")
    target = roots.work / "models" / "face_landmarker.task"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(REAL_FACE_MODEL.read_bytes())
    return target


@pytest.fixture
def face_session(raw_tree: Path, tmp_path: Path, ffmpeg_bin: str) -> Any:
    """A letterboxed 1280x720 session with face-like tiles, as the real ones are."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def factory(session_id: int = 28, *, detectable: bool = True) -> None:
        session = gen.alternating_session(
            session_id, n_turns=4, turn_s=2.0, gap_s=0.5, duration=10.0, fps=25.0
        )
        video = gen.write_letterboxed_face_video(
            scratch / f"{session_id}_v.mp4", session, detectable=detectable
        )
        audio = gen.write_session_wav(scratch / f"{session_id}.wav", session)
        gen.mux(
            raw_tree / WINTER_FOLDER / f"{session_id}.mp4",
            video,
            [audio],
            ffmpeg=ffmpeg_bin,
        )

    return factory


@pytest.mark.slow
def test_the_stage_measures_a_real_recording(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    face_session(28)

    result = stage.run(default_config, roots, workers=1)

    assert result.report.ok, result.report.summary_lines()
    row = result.frame.iloc[0]
    assert row["backend"] == "mediapipe"
    assert row["frame_step"] == 5
    assert row["n_frames_measured"] > 0
    assert row["letterbox_detected"]


@pytest.mark.slow
def test_the_sampled_frames_are_evenly_spaced(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    """25 fps native at 5 fps means exactly every fifth frame."""
    face_session(28)
    stage.run(default_config, roots, workers=1)

    table = read_parquet(stage.face_path(roots, 28))
    steps = set(table["frame_index"].diff().dropna().astype(int))
    assert steps == {5}


@pytest.mark.slow
def test_every_configured_action_unit_is_measured(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    face_session(28)
    stage.run(default_config, roots, workers=1)

    table = read_parquet(stage.face_path(roots, 28))
    measured = table.loc[table["detected"]]
    assert not measured.empty
    for key in default_config.face.unit_keys:
        assert measured[key].notna().all(), key


@pytest.mark.slow
def test_no_frame_is_left_on_disk(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    face_session(28)
    stage.run(default_config, roots, workers=1)
    leftovers = [
        path
        for path in roots.work.rglob("*")
        if path.is_file() and path.suffix in {".png", ".jpg", ".jpeg"}
    ]
    assert leftovers == []


@pytest.mark.slow
def test_a_recording_with_no_face_is_flagged_rather_than_failed(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    face_session(28, detectable=False)

    result = stage.run(default_config, roots, workers=1)

    assert result.report.ok
    row = result.frame.iloc[0]
    assert row["dropped_fraction"] == pytest.approx(1.0)
    assert stage.FLAG_NO_FACE_FOUND in row["flags"]


@pytest.mark.slow
def test_an_undecodable_recording_fails_only_its_own_session(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    face_session(28)
    place_fake_media(roots.data, WINTER_FOLDER, [29])

    result = stage.run(default_config, roots, workers=1)

    assert [o.session_id for o in result.report.failed] == [29]
    assert stage.face_path(roots, 28).exists()


@pytest.mark.slow
def test_completed_sessions_are_skipped_but_stay_in_the_table(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    face_session(28)
    stage.run(default_config, roots, workers=1)

    second = stage.run(default_config, roots, workers=1)

    assert len(second.report.skipped) == 1
    assert len(second.frame) == 1
    assert second.frame.iloc[0]["backend"] == "mediapipe"


@pytest.mark.slow
def test_a_sample_rate_that_does_not_divide_the_frame_rate_is_refused(
    roots: DataRoots, face_session: Any, model_available: Path
):
    """10 fps at 25 fps native would alternate 2- and 3-frame steps."""
    face_session(28)
    config = load_config(DEFAULT, overrides={"face.sample_fps": 10.0})

    result = stage.run(config, roots, workers=1)

    assert not result.report.ok
    assert "does not divide" in result.report.failed[0].message


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_the_summary_reports_the_backend_and_the_units(default_config: AppConfig):
    text = "\n".join(stage.summarise(stage.build_frame([_qc_row(1)]), default_config))
    assert "mediapipe" in text
    assert "au12" in text
    assert "every 5th frame" in text


def test_the_summary_says_head_pose_is_not_gaze(default_config: AppConfig):
    text = "\n".join(stage.summarise(stage.build_frame([_qc_row(1)]), default_config))
    assert "not gaze" in text


def test_the_summary_names_the_worst_session(default_config: AppConfig):
    frame = stage.build_frame([_qc_row(1, dropped_fraction=0.01), _qc_row(2, dropped_fraction=0.5)])
    text = "\n".join(stage.summarise(frame, default_config))
    assert "worst session(s): [2]" in text


def test_the_summary_of_nothing(default_config: AppConfig):
    assert stage.summarise(pd.DataFrame(), default_config) == ["no sessions were measured"]
