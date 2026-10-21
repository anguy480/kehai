"""Facial measurement with OpenFace 2.x: run here, or import its CSV output.

First-class rather than a fallback. OpenFace is the lab's house pipeline: the
action units extracted here are the set used in its published work, and the
final run may happen on a lab machine that has it (docs/decisions/0013). It is
also the more robust of the two, since it does not depend on this project's
MediaPipe build working.

Two ways in, one result:

* **Run.** When `FeatureExtraction` is available, the participant's tile is
  cropped out of the recording by the same rule `vc face` uses for MediaPipe,
  written to a scratch video, and measured. Only OpenFace's CSV is kept; the
  scratch video and everything else OpenFace writes are deleted.
* **Import.** Otherwise a CSV already in `csv_dir` is read, as produced on
  another machine.

OpenFace reports a graded confidence and a success flag per frame, so unlike
MediaPipe the confidence threshold has something real to act on.

Smoothing, when configured, happens here at the video's own frame rate and
before the frames are sampled: a centred rolling mean over each run of valid
frames, never across a dropped one (`features.face_math.smooth_within_runs`).

Intensities are on OpenFace's own scale (roughly 0-5 for `AU*_r`), which is not
the scale of a MediaPipe blendshape score. The backend is recorded on every row
so a table mixing them is refused rather than pooled.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from vc_multimodal.faces.base import FaceBackend, FaceError
from vc_multimodal.features.face_math import FrameMeasure, smooth_within_runs, smoothing_window
from vc_multimodal.ffmpeg import FfmpegError, FfmpegTools
from vc_multimodal.logging_setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vc_multimodal.config import AppConfig, CropBox, OpenFaceConfig
    from vc_multimodal.features.sampling import FrameSampling
    from vc_multimodal.paths import RawSession

logger = get_logger(__name__)

# Filename patterns tried for one session's CSV, in order.
CSV_PATTERNS: Final = ("{session_id}.csv", "{session_id}/{session_id}.csv")

# OpenFace's own column names.
FRAME_COLUMN: Final = "frame"
TIMESTAMP_COLUMN: Final = "timestamp"
POSE_COLUMNS: Final = ("pose_Rx", "pose_Ry", "pose_Rz")

#: The OpenFace binary, when not configured and not on PATH.
OPENFACE_BIN_ENV: Final = "VC_OPENFACE_BIN"
EXECUTABLE_NAME: Final = "FeatureExtraction"

# The scratch tile is re-encoded close to losslessly: the recording is already
# lossy, and this keeps the second generation from adding visible loss.
_TILE_CRF: Final = "12"
# Generous: an 11-minute session measured in tens of minutes on a laptop.
_EXTRACTION_TIMEOUT_S: Final = 6 * 60 * 60
_STDERR_TAIL: Final = 600
_MIN_ROWS_FOR_STEP: Final = 2


def resolve_executable(configured: str | None) -> str | None:
    """FeatureExtraction: configured, else $VC_OPENFACE_BIN, else on PATH."""
    candidate = configured or os.environ.get(OPENFACE_BIN_ENV) or shutil.which(EXECUTABLE_NAME)
    if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


class OpenFaceBackend(FaceBackend):
    """Per-frame action unit intensities from OpenFace.

    Args:
        config: The openface section, giving the directory and column names.
        csv_dir: Where the CSVs live, resolved under the work root.
        executable: FeatureExtraction, to run OpenFace for a session whose CSV
            is not there yet. None imports only.
        scratch_dir: Where the scratch tile video goes while OpenFace runs.
    """

    name = "openface"

    def __init__(
        self,
        config: OpenFaceConfig,
        *,
        csv_dir: Path | None,
        executable: str | None = None,
        scratch_dir: Path | None = None,
    ) -> None:
        """Store where to read from, and how to run OpenFace."""
        self.config = config
        self.csv_dir = Path(csv_dir) if csv_dir is not None else None
        self.executable = executable
        self.scratch_dir = Path(scratch_dir) if scratch_dir is not None else None

    @property
    def runs_openface(self) -> bool:
        """Whether a missing CSV is produced here rather than refused."""
        return self.executable is not None and self.csv_dir is not None

    def available(self) -> bool:
        """Whether OpenFace can run here, or there is output to import."""
        if self.csv_dir is None:
            return False
        return self.runs_openface or self.csv_dir.is_dir()

    def unavailable_reason(self) -> str:
        """Why the backend cannot run."""
        if self.csv_dir is None:
            return (
                "face.openface.csv_dir is not set. Point it at the directory of "
                "OpenFace CSV output, relative to $VC_WORK_ROOT."
            )
        if not self.available():
            return (
                f"OpenFace's {EXECUTABLE_NAME} was not found (face.openface.executable, "
                f"${OPENFACE_BIN_ENV}, or on PATH), and {self.csv_dir} holds no output to "
                f"import. See the README for setting OpenFace up."
            )
        return ""

    def version(self) -> str:
        """Identifier recorded in the manifest.

        The binary's path when OpenFace runs here, where it was imported from
        otherwise, and the smoothing applied either way. OpenFace's own version
        is not recoverable from its CSV and has to be noted by hand.
        """
        source = (
            f"binary:{self.executable}"
            if self.executable
            else f"imported:{self.csv_dir.name if self.csv_dir else 'unset'}"
        )
        smoothing = self.config.smoothing_s
        smoothed = f"+smoothed:{smoothing:g}s" if smoothing else ""
        return f"openface/{source}{smoothed}"

    def find_csv(self, session_id: int) -> Path | None:
        """Locate one session's CSV."""
        if self.csv_dir is None:
            return None
        for pattern in CSV_PATTERNS:
            candidate = self.csv_dir / pattern.format(session_id=session_id)
            if candidate.is_file():
                return candidate
        return None

    def measure_session(
        self,
        session: RawSession,
        *,
        config: AppConfig,
        crop: CropBox,
        sampling: FrameSampling,
    ) -> tuple[FrameMeasure, ...]:
        """Measure one session, running OpenFace first if its CSV is missing.

        When OpenFace runs here it sees the participant's tile only, cropped by
        `crop`. An imported CSV was produced elsewhere, on whatever video it
        was given, and which region that was is recorded by hand.

        Raises:
            FaceError: if OpenFace fails, or the CSV is missing, unreadable or
                lacks the columns the configured action units need.
        """
        if not self.available():
            raise FaceError(self.unavailable_reason())

        path = self.find_csv(session.session_id)
        if path is None and self.runs_openface:
            path = self._extract(session, crop)
        if path is None:
            tried = ", ".join(p.format(session_id=session.session_id) for p in CSV_PATTERNS)
            msg = (
                f"no OpenFace output for session {session.session_id} in {self.csv_dir}; "
                f"tried: {tried}"
            )
            raise FaceError(msg)

        frame = _read_csv(path)
        _require_columns(frame, config, path)
        return _to_measures(frame, config=config, sampling=sampling)

    def _extract(self, session: RawSession, crop: CropBox) -> Path:
        """Run OpenFace on the participant's tile and keep only its CSV."""
        assert self.executable is not None
        assert self.csv_dir is not None
        try:
            ffmpeg = FfmpegTools.discover().ffmpeg
        except FfmpegError as exc:
            raise FaceError(str(exc)) from exc
        scratch = self.scratch_dir or self.csv_dir.parent / "tmp"
        logger.info("openface: measuring session %s", session.session_id)
        return extract(
            session.path,
            crop,
            executable=self.executable,
            ffmpeg=ffmpeg,
            target=self.csv_dir / f"{session.session_id}.csv",
            scratch=scratch / str(session.session_id),
        )


def crop_filter(crop: CropBox) -> str:
    """An ffmpeg crop to the tile, in even pixel sizes as H.264 needs."""
    return (
        f"crop=trunc(iw*{crop.width:.6f}/2)*2:trunc(ih*{crop.height:.6f}/2)*2:"
        f"trunc(iw*{crop.x:.6f}):trunc(ih*{crop.y:.6f})"
    )


def _run(command: Sequence[str], what: str) -> None:
    """Run one subprocess, raising with the tail of its stderr."""
    try:
        result = subprocess.run(
            list(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_EXTRACTION_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"{what} timed out"
        raise FaceError(msg) from exc
    if result.returncode != 0:
        msg = f"{what} failed ({result.returncode}): {result.stderr.strip()[-_STDERR_TAIL:]}"
        raise FaceError(msg)


def extract(
    video: Path,
    crop: CropBox,
    *,
    executable: str,
    ffmpeg: Path,
    target: Path,
    scratch: Path,
) -> Path:
    """Crop the tile, run FeatureExtraction on it, and keep its CSV at `target`.

    Frames pass through one to one (`-vsync passthrough`), so OpenFace's frame
    numbers and timestamps are the recording's own. Only action units and head
    pose are requested, so OpenFace writes no images, aligned faces or tracked
    video, and the scratch folder is removed whatever happens.

    Raises:
        FaceError: if ffmpeg or OpenFace fails, or OpenFace writes no CSV.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    tile = scratch / "tile.mp4"
    output = scratch / "openface"
    try:
        _run(
            [
                str(ffmpeg),
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-an",
                "-vsync",
                "passthrough",
                "-vf",
                crop_filter(crop),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                _TILE_CRF,
                "-pix_fmt",
                "yuv420p",
                str(tile),
            ],
            f"cropping the tile from {video.name}",
        )
        _run(
            [executable, "-f", str(tile), "-out_dir", str(output), "-aus", "-pose"],
            f"OpenFace on {video.name}",
        )
        produced = output / f"{tile.stem}.csv"
        if not produced.is_file():
            msg = f"OpenFace wrote no CSV for {video.name}"
            raise FaceError(msg)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".csv.partial")
        shutil.move(str(produced), partial)
        partial.replace(target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return target


def _read_csv(path: Path) -> pd.DataFrame:
    """Read an OpenFace CSV, whose headers carry leading spaces."""
    try:
        frame = pd.read_csv(path, skipinitialspace=True)
    except (OSError, ValueError) as exc:
        msg = f"could not read {path.name}: {exc}"
        raise FaceError(msg) from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _require_columns(frame: pd.DataFrame, config: AppConfig, path: Path) -> None:
    """Check the CSV has what the configured action units need."""
    face = config.face
    needed = [
        face.openface.success_column,
        face.openface.confidence_column,
        *[unit.openface_column for unit in face.action_units],
    ]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        msg = (
            f"{path.name} is missing column(s) {missing}. Run OpenFace with the "
            f"action unit output enabled (-aus), or adjust "
            f"face.action_units[*].openface_column to match its output."
        )
        raise FaceError(msg)


def _frame_step(frame: pd.DataFrame) -> int:
    """The spacing of the CSV's frame numbers: 1 unless already subsampled."""
    if FRAME_COLUMN not in frame.columns or len(frame) < _MIN_ROWS_FOR_STEP:
        return 1
    indices = frame[FRAME_COLUMN].astype("int64")
    return int(indices.diff().dropna().median() or 1)


def _sampled_rows(frame: pd.DataFrame, sampling: FrameSampling) -> pd.DataFrame:
    """Take the rows on our sampling grid.

    OpenFace numbers frames from one and processes every frame, so the grid is
    applied here rather than assumed. A CSV that has already been subsampled is
    used as-is, with a warning, since re-subsampling it would silently thin the
    data further.
    """
    if FRAME_COLUMN not in frame.columns:
        return frame
    step = _frame_step(frame)
    if step > 1:
        logger.warning(
            "the OpenFace output is already subsampled (every %d frames); using "
            "it as-is rather than thinning it further",
            step,
        )
        return frame
    indices = frame[FRAME_COLUMN].astype("int64")
    # OpenFace counts from one; our sampling counts from zero.
    return frame.loc[(indices - 1) % sampling.step == 0]


def valid_frames(frame: pd.DataFrame, config: AppConfig) -> np.ndarray:
    """Frames with a usable face: success == 1 and the confidence threshold met.

    The threshold is the same `face.min_confidence` the face stage applies to
    every backend, so this adds no filter; it applies it early enough for the
    smoothing to respect it.
    """
    face = config.face
    success = pd.to_numeric(frame[face.openface.success_column], errors="coerce")
    confidence = pd.to_numeric(frame[face.openface.confidence_column], errors="coerce")
    return ((success == 1) & (confidence >= face.min_confidence)).to_numpy(dtype=bool)


def smooth_units(frame: pd.DataFrame, config: AppConfig, native_fps: float) -> pd.DataFrame:
    """Smooth each action unit within runs of valid frames, at the native rate.

    Raises:
        FaceError: if the CSV is not one row per frame, which smoothing at the
            native rate needs.
    """
    seconds = config.face.openface.smoothing_s
    if not seconds:
        return frame
    if FRAME_COLUMN not in frame.columns or _frame_step(frame) != 1:
        msg = (
            "smoothing needs OpenFace output with one row per frame and a frame "
            "column; this CSV is subsampled or unnumbered"
        )
        raise FaceError(msg)
    ordered = frame.sort_values(FRAME_COLUMN, kind="stable").reset_index(drop=True)
    window = smoothing_window(native_fps, seconds)
    valid = valid_frames(ordered, config)
    index = ordered[FRAME_COLUMN].to_numpy(dtype=np.int64)
    for unit in config.face.action_units:
        values = pd.to_numeric(ordered[unit.openface_column], errors="coerce").to_numpy()
        ordered[unit.openface_column] = smooth_within_runs(index, valid, values, window)
    return ordered


def _to_measures(
    frame: pd.DataFrame, *, config: AppConfig, sampling: FrameSampling
) -> tuple[FrameMeasure, ...]:
    """Convert OpenFace rows into per-frame measures, smoothed if configured."""
    face = config.face
    frame = smooth_units(frame, config, sampling.native_fps)
    rows = _sampled_rows(frame, sampling)
    units: Sequence[tuple[str, str]] = [
        (unit.key, unit.openface_column) for unit in face.action_units
    ]
    has_pose = all(column in rows.columns for column in POSE_COLUMNS)

    measures: list[FrameMeasure] = []
    for _, row in rows.iterrows():
        frame_index = int(row.get(FRAME_COLUMN, len(measures) + 1)) - 1
        timestamp = (
            float(row[TIMESTAMP_COLUMN])
            if TIMESTAMP_COLUMN in rows.columns and pd.notna(row[TIMESTAMP_COLUMN])
            else sampling.timestamp(frame_index)
        )
        success = bool(row[face.openface.success_column])
        confidence = float(row[face.openface.confidence_column])
        if not success:
            measures.append(FrameMeasure.missing(frame_index, timestamp))
            continue

        head = None
        if has_pose:
            head = _pose_from_radians(
                float(row[POSE_COLUMNS[0]]),
                float(row[POSE_COLUMNS[1]]),
                float(row[POSE_COLUMNS[2]]),
            )

        measures.append(
            FrameMeasure(
                frame_index=frame_index,
                timestamp_s=timestamp,
                detected=True,
                confidence=confidence,
                units={
                    key: (float(row[column]) if pd.notna(row[column]) else None)
                    for key, column in units
                },
                jaw=None,
                blink=None,
                head=head,
            )
        )
    return tuple(measures)


def _pose_from_radians(rx: float, ry: float, rz: float) -> tuple[float, float, float] | None:
    """Convert OpenFace's pose angles, which are already Euler, to degrees.

    OpenFace reports rotation directly rather than as a matrix, so this does
    not go through `head_pose_from_matrix`; the conversion is kept next to it so
    the two conventions are visible together. pose_Rx is pitch, Ry is yaw, Rz
    is roll, all in radians in the camera's frame - head orientation, not gaze.
    """
    values = (rx, ry, rz)
    if not all(math.isfinite(value) for value in values):
        return None
    return tuple(math.degrees(value) for value in values)  # type: ignore[return-value]
