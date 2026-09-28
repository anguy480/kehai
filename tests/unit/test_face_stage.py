"""The face stage, its backends, and the rule against mixing them.

The backend is expected to change: MediaPipe is the default only because an
OpenFace run has not been confirmed. So the things tested hardest here are the
ones that make that switch safe - the backend recorded on every row, the
refusal to pool two backends, and the note that says a switch is a full rerun.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pytest
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from tests.conftest import REAL_FACE_MODEL, WINTER_FOLDER, place_fake_media
from tests.synth import generators as gen
from vc_multimodal.config import AppConfig, CropBox, load_config
from vc_multimodal.faces import (
    FaceError,
    MediaPipeBackend,
    OpenFaceBackend,
    get_backend,
    require_single_backend,
)
from vc_multimodal.features.face_math import FrameMeasure, head_pose_from_matrix
from vc_multimodal.features.sampling import resolve_sampling
from vc_multimodal.handoff_text import (
    FACE_BACKEND_NOTE,
    GAZE_ABSENCE_NOTE,
    REMOTE_RECORDING_NOTE,
    notes,
)
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


# ---------------------------------------------------------------------------
# head pose, verified against the model rather than against our own convention
#
# The mapping from a rotation matrix to named angles is easy to get wrong and
# impossible to catch by round-tripping: an earlier version returned the three
# in the order (yaw, roll, pitch) while labelling them (pitch, yaw, roll), and
# every internal test passed. Rotating an image in its own plane is a rotation
# about the camera's optical axis, so it must appear as roll and nothing else.
# ---------------------------------------------------------------------------
def frontal_face_image(size: int = 480) -> np.ndarray:
    """A crude frontal face the landmarker can find, drawn upright."""
    image = np.full((size, size, 3), 205, np.uint8)
    cx = cy = size // 2
    cv2.ellipse(image, (cx, cy), (95, 125), 0, 0, 360, (212, 184, 164), -1)
    for sign in (-1, 1):
        eye_x = cx + sign * 38
        cv2.ellipse(image, (eye_x, cy - 30), (17, 10), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(image, (eye_x, cy - 30), 7, (35, 35, 45), -1)
        cv2.ellipse(image, (eye_x, cy - 52), (20, 7), 0, 180, 360, (70, 50, 40), 3)
    cv2.line(image, (cx, cy - 20), (cx, cy + 18), (180, 150, 135), 3)
    cv2.ellipse(image, (cx, cy + 52), (30, 12), 0, 0, 360, (90, 55, 55), -1)
    return image


def _model_pose(image: np.ndarray, model: Path) -> tuple[Any, Any]:
    """Run the real landmarker on one image, returning its matrix and angles."""
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )
        )
    if not result.facial_transformation_matrixes:
        return None, None
    matrix = np.asarray(result.facial_transformation_matrixes[0])
    return matrix, head_pose_from_matrix(matrix)


@pytest.mark.slow
def test_the_transform_really_is_a_rigid_rotation(model_available: Path):
    """The extraction assumes an orthonormal 3x3 in the top-left, row-major."""
    matrix, _ = _model_pose(frontal_face_image(), model_available)
    assert matrix is not None, "the landmarker found no face to measure"
    assert matrix.shape == (4, 4)

    rotation = matrix[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-3)
    # Translation in the last column, not the last row.
    assert matrix[3, :] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("degrees", [-20.0, -10.0, 10.0, 20.0])
def test_an_in_plane_rotation_appears_as_roll(model_available: Path, degrees: float):
    """Rotation about the optical axis is roll, by definition of the axes."""
    upright = frontal_face_image()
    _, baseline = _model_pose(upright, model_available)
    assert baseline is not None

    spin = cv2.getRotationMatrix2D((240, 240), degrees, 1.0)
    rotated = cv2.warpAffine(upright, spin, (480, 480), borderValue=(205, 205, 205))
    _, measured = _model_pose(rotated, model_available)
    assert measured is not None

    # Roll tracks the applied rotation, with its sign.
    assert measured[2] - baseline[2] == pytest.approx(degrees, abs=2.0)
    # Yaw does not: an in-plane spin is not a turn of the head.
    assert measured[1] - baseline[1] == pytest.approx(0.0, abs=3.0)


@pytest.mark.slow
def test_an_upright_face_is_not_reported_as_rolled(model_available: Path):
    """The channel that in-plane rotation moves must read near zero upright."""
    _, angles = _model_pose(frontal_face_image(), model_available)
    assert angles is not None
    assert abs(angles[2]) < 5.0


def test_both_backends_name_the_axes_the_same_way(roots: DataRoots, default_config: AppConfig):
    """The backend is expected to change, so head_pitch must not mean one axis
    under MediaPipe and another under OpenFace."""
    csv_dir = roots.work / "openface"
    path = openface_csv(csv_dir / "28.csv", n_frames=5)
    # Rewrite the pose columns so each axis carries a distinct angle.
    frame = pd.read_csv(path)
    frame["pose_Rx"] = np.radians(10.0)  # about X: nodding
    frame["pose_Ry"] = np.radians(20.0)  # about Y: turning
    frame["pose_Rz"] = np.radians(30.0)  # about Z: tilting
    frame.to_csv(path, index=False)

    measures = OpenFaceBackend(default_config.face.openface, csv_dir=csv_dir).measure_session(
        _session(),
        config=default_config,
        crop=default_config.video.tiles["right"],
        sampling=resolve_sampling(25.0, 5.0),
    )

    pitch, yaw, roll = measures[0].head
    assert pitch == pytest.approx(10.0, abs=1e-6)
    assert yaw == pytest.approx(20.0, abs=1e-6)
    assert roll == pytest.approx(30.0, abs=1e-6)

    # And the same angles through the MediaPipe path give the same names.
    matrix = np.eye(4)
    matrix[:3, :3] = _compose(10.0, 20.0, 30.0)
    assert head_pose_from_matrix(matrix) == pytest.approx((10.0, 20.0, 30.0), abs=1e-6)


def _compose(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """`Rz(roll) @ Ry(yaw) @ Rx(pitch)`, built from the axes themselves."""
    a, b, c = (math.radians(angle) for angle in (pitch, yaw, roll))
    about_x = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    about_y = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    about_z = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return about_z @ about_y @ about_x


# ---------------------------------------------------------------------------
# geometry stability
#
# The crop was measured from a single frame one second in. That frame can be a
# fade-in or a title card, and a layout that changes mid-session - active
# speaker view, which has not been ruled out for every recording - would go
# unnoticed.
# ---------------------------------------------------------------------------
def test_identical_boxes_agree(default_config: AppConfig):
    box = default_config.video.tiles["right"]
    assert stage._boxes_agree([box, box, box])


def test_a_single_box_cannot_disagree(default_config: AppConfig):
    assert stage._boxes_agree([default_config.video.tiles["right"]])
    assert stage._boxes_agree([])


def test_boxes_differing_within_tolerance_agree():
    """Sub-pixel jitter between frames is the same box, not a moved one."""
    assert stage._boxes_agree(
        [
            CropBox(x=0.5, y=0.25, width=0.495, height=0.5),
            CropBox(x=0.505, y=0.25, width=0.495, height=0.5),
        ]
    )


def test_a_moved_tile_is_a_disagreement():
    """Which is what a switch to active-speaker view would look like."""
    assert not stage._boxes_agree(
        [
            CropBox(x=0.5, y=0.25, width=0.5, height=0.5),
            CropBox(x=0.0, y=0.0, width=1.0, height=1.0),
        ]
    )


def test_unstable_geometry_is_flagged(default_config: AppConfig):
    record = stage.qc_record(
        _session(),
        frame_measures(),
        backend=MediaPipeBackend(default_config.face.mediapipe, model_dir=Path("/nowhere")),
        sampling=resolve_sampling(25.0, 5.0),
        crop=default_config.video.tiles["right"],
        letterboxed=True,
        crop_stable=False,
        config=default_config,
    )
    assert stage.FLAG_CROP_UNSTABLE in record["flags"]


@pytest.mark.slow
def test_the_crop_is_checked_at_several_points(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    face_session(28)
    result = stage.run(default_config, roots, workers=1)
    row = result.frame.iloc[0]
    # A statically laid out recording is stable, so no flag.
    assert stage.FLAG_CROP_UNSTABLE not in str(row["flags"])
    assert row["letterbox_detected"]


# ---------------------------------------------------------------------------
# the backend record survives the QC table
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_the_backend_is_recorded_beside_the_measurements(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    face_session(28)
    stage.run(default_config, roots, workers=1)

    record = stage.read_backend_record(roots, 28)
    assert record is not None
    assert record["backend"] == "mediapipe"
    assert "mediapipe/" in record["backend_version"]


@pytest.mark.slow
def test_deleting_the_qc_table_remeasures_rather_than_writing_a_stub(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    """A session with no QC row is not done, whatever else is on disk.

    This used to skip the session and assemble a row from the backend sidecar
    alone: an identifier and a backend name with every measurement empty. That
    row was indistinguishable from a failed measurement and made every rate
    computed over the table wrong. A session whose row is missing is now
    re-measured, which is the only way to get a complete one.
    """
    face_session(28)
    stage.run(default_config, roots, workers=1)
    (roots.out / stage.FACE_QC_FILENAME).unlink()

    result = stage.run(default_config, roots, workers=1)

    assert len(result.report.skipped) == 0
    assert len(result.report.succeeded) == 1
    row = result.frame.iloc[0]
    assert row["backend"] == "mediapipe"
    # The measurement columns are what a stub left empty.
    assert pd.notna(row["n_frames_sampled"])
    assert pd.notna(row["n_frames_measured"])
    assert pd.notna(row["dropped_fraction"])


@pytest.mark.slow
def test_a_backend_change_on_a_subset_is_refused(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    """The case the design is for: re-extract some sessions with the other tool.

    Checked across everything on disk, not only the sessions in this run, so a
    subset rerun cannot leave the work tree quietly inconsistent.
    """
    face_session(28)
    stage.run(default_config, roots, workers=1)

    # A second session already measured with the other backend.
    stage.face_path(roots, 3).parent.mkdir(parents=True, exist_ok=True)
    stage.frame_table(3, frame_measures(), default_config.face.unit_keys).to_parquet(
        stage.face_path(roots, 3), index=False
    )
    stage.write_backend_record(
        roots, 3, OpenFaceBackend(default_config.face.openface, csv_dir=roots.work / "of")
    )

    with pytest.raises(FaceError, match="more than one backend"):
        stage.run(default_config, roots, workers=1)


def test_a_missing_or_corrupt_record_reads_as_absent(roots: DataRoots):
    assert stage.read_backend_record(roots, 28) is None
    path = stage.backend_sidecar_path(roots, 28)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert stage.read_backend_record(roots, 28) is None
    path.write_text('{"session_id": 28}', encoding="utf-8")
    assert stage.read_backend_record(roots, 28) is None


def test_stored_backends_ignores_unrelated_files(roots: DataRoots, default_config: AppConfig):
    stage.face_dir(roots)
    stage.write_backend_record(
        roots, 28, MediaPipeBackend(default_config.face.mediapipe, model_dir=Path("/nowhere"))
    )
    (stage.face_dir(roots) / "notes.backend.json").write_text("{}", encoding="utf-8")

    stored = stage.stored_backends(roots)

    assert set(stored) == {28}
    assert stored[28]["backend"] == "mediapipe"


# ---------------------------------------------------------------------------
# partial runs and skipped sessions
#
# Both of these were shipped broken and found by piloting on three sessions:
# `vc --force --sessions 130 face` left face_qc.csv holding only session 130,
# and before that, session 130 had a row with every column empty.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_a_partial_rerun_keeps_the_rows_it_did_not_measure(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    """The bug: writing only this run's rows deleted every other session's.

    With 62 sessions and individual reruns, this silently destroyed coverage on
    every run, and nothing failed while it happened.
    """
    face_session(28)
    face_session(3)
    first = stage.run(default_config, roots, workers=1)
    assert sorted(first.frame["session_id"]) == [3, 28]

    rerun = stage.run(default_config, roots, session_ids=[3], workers=1, force=True)

    assert sorted(rerun.frame["session_id"]) == [3, 28]
    assert len(rerun.report.succeeded) == 1


@pytest.mark.slow
def test_a_skipped_session_keeps_a_populated_row(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    """The other bug: a skipped session was written as a row of empty columns.

    Empty is indistinguishable from a failed measurement, so it made the summary
    report rates over sessions that had never been measured.
    """
    face_session(28)
    stage.run(default_config, roots, workers=1)

    again = stage.run(default_config, roots, workers=1)

    assert len(again.report.skipped) == 1
    row = again.frame.iloc[0]
    for column in (
        "backend",
        "n_frames_sampled",
        "n_frames_measured",
        "dropped_fraction",
        "letterbox_detected",
    ):
        assert pd.notna(row[column]), f"{column} is empty for a skipped session"


@pytest.mark.slow
def test_a_partial_rerun_leaves_the_untouched_rows_unchanged(
    roots: DataRoots, default_config: AppConfig, face_session: Any, model_available: Path
):
    """A kept row is the row that was written, not a reconstruction of it."""
    face_session(28)
    face_session(3)
    first = stage.run(default_config, roots, workers=1)
    before = first.frame.set_index("session_id").loc[28].to_dict()

    rerun = stage.run(default_config, roots, session_ids=[3], workers=1, force=True)

    after = rerun.frame.set_index("session_id").loc[28].to_dict()
    for column, value in before.items():
        if pd.isna(value):
            assert pd.isna(after[column]), column
        else:
            assert after[column] == value, column


def test_the_summary_counts_rates_over_measured_sessions_only(default_config: AppConfig):
    """A row with no value is not a session where letterboxing was absent.

    This is what made the pilot report "letterbox corrected in 2 of 3 sessions"
    when the answer was 2 of 2 measured.
    """
    frame = pd.DataFrame(
        {
            "session_id": [1, 2, 3],
            "wave": ["winter"] * 3,
            "backend": ["mediapipe"] * 3,
            "n_frames_sampled": [50, 50, None],
            "n_frames_measured": [50, 50, None],
            "frame_step": [5, 5, None],
            "dropped_fraction": [0.0, 0.0, None],
            "letterbox_detected": [True, True, None],
            "flags": ["", "", ""],
        }
    )
    lines = "\n".join(stage.summarise(frame, default_config))
    assert "2 of 2 measured session(s)" in lines
    assert "1 session(s) not measured" in lines


# ---------------------------------------------------------------------------
# the remote-recording note
# ---------------------------------------------------------------------------
def flowed(text: str) -> str:
    """Prose with its line breaks collapsed.

    A note is wrapped for reading, and rewrapping it when a sentence is edited
    must not break a test about what it says.
    """
    return " ".join(text.split())


def test_the_remote_recording_note_blames_the_recording_not_the_participant():
    assert "nothing to do with the participants themselves" in flowed(REMOTE_RECORDING_NOTE)


def test_the_remote_recording_note_names_the_column_to_check():
    assert "qc__face_dropped_fraction" in flowed(REMOTE_RECORDING_NOTE)


def test_the_remote_recording_note_names_the_lab_study_and_what_it_had():
    """The comparison a reviewer reaches for, and why it does not hold."""
    note = flowed(REMOTE_RECORDING_NOTE)
    assert "Miyamoto et al. 2025" in note
    assert "Acta Psychologica 254:104782" in note
    assert "controlled lighting" in note
    assert "eye tracker" in note


def test_the_remote_recording_note_keeps_the_alternative_explanation_live():
    assert "cannot be ruled out" in flowed(REMOTE_RECORDING_NOTE)


def test_the_remote_recording_note_forbids_modelling_tracking_quality():
    assert "must not be modelled as one" in flowed(REMOTE_RECORDING_NOTE)


def test_every_note_is_carried_into_the_bundle():
    assert notes() == (FACE_BACKEND_NOTE, GAZE_ABSENCE_NOTE, REMOTE_RECORDING_NOTE)
