"""OpenFace run, smoothing and the face table. Synthetic data only."""

from __future__ import annotations

import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import WINTER_FOLDER, place_fake_media
from vc_multimodal.config import AppConfig, CropBox, load_config
from vc_multimodal.exploratory.per_modality import face_table
from vc_multimodal.exploratory.per_modality.plan import load_plan
from vc_multimodal.faces.base import FaceError
from vc_multimodal.faces.openface_backend import OpenFaceBackend, crop_filter, extract
from vc_multimodal.features.face_math import FrameMeasure, smooth_within_runs, smoothing_window
from vc_multimodal.features.sampling import resolve_sampling
from vc_multimodal.io_utils import write_parquet
from vc_multimodal.paths import DataRoots, RawSession
from vc_multimodal.qc_notes import record as record_qc_note
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import turns as turns_stage

DEFAULT = Path("config/default.yaml")


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("fps", "frames"), [(25.0, 25), (30.0, 31), (24.0, 25), (29.97, 31), (5.0, 5)]
)
def test_one_second_is_the_nearest_odd_number_of_frames(fps: float, frames: int) -> None:
    assert smoothing_window(fps, 1.0) == frames


def test_a_window_needs_positive_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        smoothing_window(0.0, 1.0)


# ---------------------------------------------------------------------------
# Smoothing within runs
# ---------------------------------------------------------------------------
def test_inside_a_run_it_is_a_centred_mean_with_partial_edges() -> None:
    values = np.arange(10, dtype=float)
    smoothed = smooth_within_runs(np.arange(10), np.ones(10, bool), values, 3)
    expected = pd.Series(values).rolling(3, center=True, min_periods=1).mean().to_numpy()
    assert smoothed == pytest.approx(expected)


def test_it_never_reaches_across_a_dropped_frame() -> None:
    values = np.array([0.0, 0.0, 0.0, 9.0, 5.0, 5.0, 5.0])
    valid = np.array([True, True, True, False, True, True, True])
    smoothed = smooth_within_runs(np.arange(7), valid, values, 5)
    # The invalid frame's 9.0 leaks into neither neighbour, and is itself NaN.
    assert smoothed[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert np.isnan(smoothed[3])
    assert smoothed[4:] == pytest.approx([5.0, 5.0, 5.0])


def test_it_never_reaches_across_a_gap_in_the_frame_numbers() -> None:
    frames = np.array([1, 2, 3, 10, 11, 12])
    values = np.array([1.0, 1.0, 1.0, 7.0, 7.0, 7.0])
    smoothed = smooth_within_runs(frames, np.ones(6, bool), values, 25)
    assert smoothed == pytest.approx(values)


def test_a_non_finite_value_breaks_a_run_like_a_dropped_frame() -> None:
    values = np.array([2.0, 2.0, np.nan, 4.0, 4.0])
    smoothed = smooth_within_runs(np.arange(5), np.ones(5, bool), values, 5)
    assert smoothed[[0, 1, 3, 4]] == pytest.approx([2.0, 2.0, 4.0, 4.0])
    assert np.isnan(smoothed[2])


def test_no_valid_frame_gives_all_nan() -> None:
    smoothed = smooth_within_runs(np.arange(4), np.zeros(4, bool), np.ones(4), 3)
    assert np.isnan(smoothed).all()


def test_an_even_window_is_refused() -> None:
    with pytest.raises(ValueError, match="odd"):
        smooth_within_runs(np.arange(4), np.ones(4, bool), np.ones(4), 4)


# ---------------------------------------------------------------------------
# The backend smooths before it samples
# ---------------------------------------------------------------------------
def openface_frame(
    n: int, *, au12: np.ndarray, confidence: np.ndarray | None = None
) -> pd.DataFrame:
    """An OpenFace CSV's columns, one row per frame from frame 1."""
    frames = np.arange(1, n + 1)
    data: dict[str, object] = {
        "frame": frames,
        "face_id": 0,
        "timestamp": (frames - 1) / 25.0,
        "confidence": np.full(n, 0.98) if confidence is None else confidence,
        "success": 1,
        "pose_Rx": 0.0,
        "pose_Ry": 0.0,
        "pose_Rz": 0.0,
    }
    for column in ("AU01_r", "AU02_r", "AU04_r", "AU06_r"):
        data[column] = 0.0
    data["AU12_r"] = au12
    return pd.DataFrame(data)


def measure(frame: pd.DataFrame, tmp_path: Path, config: AppConfig) -> list[FrameMeasure]:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir(exist_ok=True)
    frame.to_csv(csv_dir / "28.csv", index=False)
    backend = OpenFaceBackend(config.face.openface, csv_dir=csv_dir)
    session = RawSession(28, "winter", WINTER_FOLDER, tmp_path / "28.mp4")
    return list(
        backend.measure_session(
            session,
            config=config,
            crop=CropBox(x=0.0, y=0.0, width=1.0, height=1.0),
            sampling=resolve_sampling(25.0, config.face.sample_fps),
        )
    )


def test_sampled_values_are_the_one_second_mean_at_the_native_rate(tmp_path: Path) -> None:
    config = load_config(DEFAULT)
    au12 = np.zeros(100)
    au12[50] = 2.5  # one frame of expression
    by_frame = {
        m.frame_index: m.units["au12"]
        for m in measure(openface_frame(100, au12=au12), tmp_path, config)
    }
    # A 25-frame window reaches 12 frames either side of frame 50, so the
    # sampled frames 40 to 60 each carry 2.5 / 25, and 35 and 65 nothing.
    for index in (40, 45, 50, 55, 60):
        assert by_frame[index] == pytest.approx(0.1)
    assert by_frame[35] == pytest.approx(0.0)
    assert by_frame[65] == pytest.approx(0.0)


def test_unsmoothed_a_spike_between_sampled_frames_is_not_seen(tmp_path: Path) -> None:
    config = load_config(DEFAULT, overrides={"face.openface.smoothing_s": None})
    au12 = np.zeros(100)
    au12[51] = 2.5
    measures = measure(openface_frame(100, au12=au12), tmp_path, config)
    assert all(m.units["au12"] == 0.0 for m in measures)


def test_a_low_confidence_frame_is_dropped_before_smoothing(tmp_path: Path) -> None:
    config = load_config(DEFAULT)
    au12 = np.zeros(100)
    au12[52] = 5.0
    confidence = np.full(100, 0.98)
    confidence[52] = 0.2  # success, but below face.min_confidence
    measures = measure(openface_frame(100, au12=au12, confidence=confidence), tmp_path, config)
    assert all(m.units["au12"] == pytest.approx(0.0) for m in measures if m.detected)


def test_a_subsampled_csv_cannot_be_smoothed(tmp_path: Path) -> None:
    config = load_config(DEFAULT)
    frame = openface_frame(100, au12=np.zeros(100)).iloc[::5]
    with pytest.raises(FaceError, match="one row per frame"):
        measure(frame, tmp_path, config)


# ---------------------------------------------------------------------------
# Running OpenFace
# ---------------------------------------------------------------------------
def script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def fake_ffmpeg(tmp_path: Path) -> Path:
    """Writes its last argument, as ffmpeg writes its output file."""
    return Path(script(tmp_path / "ffmpeg", 'for last; do :; done\necho video > "$last"\n'))


def fake_openface(tmp_path: Path, *, fail: bool = False) -> str:
    """Writes <out_dir>/<stem of -f>.csv and an extra file, as OpenFace does."""
    if fail:
        return script(tmp_path / "FeatureExtraction", "echo 'no model' >&2\nexit 3\n")
    body = (
        'while [ $# -gt 0 ]; do case "$1" in -f) f="$2"; shift;; -out_dir) o="$2"; shift;; esac; '
        "shift; done\n"
        'mkdir -p "$o"\n'
        's=$(basename "$f" .mp4)\n'
        'echo "frame,success" > "$o/$s.csv"\n'
        'echo details > "$o/${s}_of_details.txt"\n'
    )
    return script(tmp_path / "FeatureExtraction", body)


def test_only_the_csv_is_kept(tmp_path: Path) -> None:
    target = tmp_path / "csv" / "28.csv"
    scratch = tmp_path / "scratch" / "28"
    extract(
        tmp_path / "28.mp4",
        CropBox(x=0.5, y=0.0, width=0.5, height=1.0),
        executable=fake_openface(tmp_path),
        ffmpeg=fake_ffmpeg(tmp_path),
        target=target,
        scratch=scratch,
    )
    assert target.read_text().startswith("frame,success")
    assert not scratch.exists()  # no tile video, no details file


def test_a_failed_run_leaves_nothing_behind_and_says_why(tmp_path: Path) -> None:
    target = tmp_path / "csv" / "28.csv"
    scratch = tmp_path / "scratch" / "28"
    with pytest.raises(FaceError, match="no model"):
        extract(
            tmp_path / "28.mp4",
            CropBox(x=0.5, y=0.0, width=0.5, height=1.0),
            executable=fake_openface(tmp_path, fail=True),
            ffmpeg=fake_ffmpeg(tmp_path),
            target=target,
            scratch=scratch,
        )
    assert not target.exists()
    assert not scratch.exists()


def test_the_crop_is_in_even_pixels_from_fractions() -> None:
    text = crop_filter(CropBox(x=0.5, y=0.1, width=0.5, height=0.8))
    assert text == (
        "crop=trunc(iw*0.500000/2)*2:trunc(ih*0.800000/2)*2:trunc(iw*0.500000):trunc(ih*0.100000)"
    )


# ---------------------------------------------------------------------------
# Each backend has its own folder
# ---------------------------------------------------------------------------
def test_openface_measures_never_share_mediapipes_folder(roots: DataRoots) -> None:
    assert face_stage.face_path(roots, 3) == roots.work / "face" / "3.parquet"
    assert face_stage.face_path(roots, 3, "openface") == roots.work / "face_openface" / "3.parquet"
    assert face_stage.face_qc_filename() == "face_qc.csv"
    assert face_stage.face_qc_filename("mediapipe") == "face_qc.csv"
    assert face_stage.face_qc_filename("openface") == "face_openface_qc.csv"


# ---------------------------------------------------------------------------
# The face table
# ---------------------------------------------------------------------------
def per_frame(session_id: int, *, n: int = 3000) -> pd.DataFrame:
    """A per-frame table as `vc face --backend openface` writes it, at 5 fps."""
    data: dict[str, object] = {
        "session_id": session_id,
        "frame_index": np.arange(n) * 5,
        "timestamp_s": np.arange(n) * 0.2,
        "detected": True,
        "confidence": 0.98,
    }
    for key in ("au01", "au02", "au04", "au06", "au12"):
        data[key] = np.linspace(0.0, 1.0, n)
    for key in ("jaw", "blink", "head_pitch", "head_yaw", "head_roll"):
        data[key] = np.nan
    return pd.DataFrame(data)


def timeline(session_id: int) -> pd.DataFrame:
    rows = [(session_id, "speaking", 0.0, 300.0), (session_id, "listening", 300.0, 600.0)]
    frame = pd.DataFrame(rows, columns=["session_id", "state", "start_s", "end_s"])
    frame["state"] = frame["state"].astype("string")
    return frame


def openface_session(roots: DataRoots, session_id: int, backend: str = "openface") -> None:
    write_parquet(face_stage.face_path(roots, session_id, "openface"), per_frame(session_id))
    face_stage.backend_sidecar_path(roots, session_id, "openface").write_text(
        f'{{"session_id": {session_id}, "backend": "{backend}", "backend_version": "test"}}'
    )
    write_parquet(turns_stage.timeline_path(roots, session_id), timeline(session_id))


@pytest.mark.usefixtures("raw_tree")
def test_the_face_table_has_the_planned_columns_and_withholds_session_43(
    roots: DataRoots, default_config: AppConfig
) -> None:
    sessions = [3, 28, 43]
    place_fake_media(roots.data, WINTER_FOLDER, sessions)
    for sid in sessions:
        openface_session(roots, sid)
    record_qc_note(
        roots.work / default_config.qc.notes_path,
        session_id=43,
        modality="face",
        status="unavailable",
        reason="camera out of focus in this synthetic test",
        recorded_by="test",
    )

    plan = load_plan()
    table = face_table.build(default_config, roots, plan)
    frame = pd.read_csv(table.path)
    columns = list(face_table.planned_columns(plan))
    assert len(columns) == 18
    assert [c for c in frame.columns if c.startswith("face_")] == columns
    assert frame.loc[frame["session_id"] == 43, columns].isna().all(axis=None)
    assert frame.loc[frame["session_id"] != 43, columns].notna().all(axis=None)
    assert set(frame["qc__face_backend"]) == {"openface"}
    assert set(table.blanked) == {43}
    # The ramp rises through the session, so speaking (first) sits below listening.
    row = frame.loc[frame["session_id"] == 3].iloc[0]
    assert row["face_speaking__au12_mean"] < row["face_listening__au12_mean"]
    assert set(table.distributions["unit"]) == {"au01", "au06", "au12"}


@pytest.mark.usefixtures("raw_tree")
def test_the_face_table_refuses_mediapipe_measures(
    roots: DataRoots, default_config: AppConfig
) -> None:
    place_fake_media(roots.data, WINTER_FOLDER, [3])
    openface_session(roots, 3, backend="mediapipe")
    with pytest.raises(face_table.FaceTableError, match="mediapipe"):
        face_table.build(default_config, roots, load_plan())
