"""Facial measurement by importing OpenFace 2.0 output.

First-class rather than a fallback. OpenFace is the lab's house pipeline: the
action units extracted here are the set used in its published work, and the
final run may happen on a lab machine that has it (docs/decisions/0013). It is
also the more robust of the two, since it does not depend on this project's
MediaPipe build working.

OpenFace reports a graded confidence and a success flag per frame, so unlike
MediaPipe the confidence threshold has something real to act on.

Intensities are on OpenFace's own scale (roughly 0-5 for `AU*_r`), which is not
the scale of a MediaPipe blendshape score. The backend is recorded on every row
so a table mixing them is refused rather than pooled.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from vc_multimodal.faces.base import FaceBackend, FaceError
from vc_multimodal.features.face_math import FrameMeasure
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


class OpenFaceBackend(FaceBackend):
    """Reads per-frame action unit intensities from OpenFace CSV output.

    Args:
        config: The openface section, giving the directory and column names.
        csv_dir: Where the CSVs live, resolved under the work root.
    """

    name = "openface"

    def __init__(self, config: OpenFaceConfig, *, csv_dir: Path | None) -> None:
        """Store where to read from."""
        self.config = config
        self.csv_dir = Path(csv_dir) if csv_dir is not None else None

    def available(self) -> bool:
        """Whether a directory of OpenFace output is configured and present."""
        return self.csv_dir is not None and self.csv_dir.is_dir()

    def unavailable_reason(self) -> str:
        """Why the backend cannot run."""
        if self.csv_dir is None:
            return (
                "face.openface.csv_dir is not set. Point it at the directory of "
                "OpenFace CSV output, relative to $VC_WORK_ROOT."
            )
        if not self.csv_dir.is_dir():
            return f"face.openface.csv_dir points at {self.csv_dir}, which is not a directory"
        return ""

    def version(self) -> str:
        """Identifier recorded in the manifest.

        The extraction happened elsewhere, so this records where the output was
        read from. OpenFace's own version is not recoverable from its CSV and
        has to be noted by hand.
        """
        binary = shutil.which("FeatureExtraction")
        found = f"+binary:{binary}" if binary else ""
        location = self.csv_dir.name if self.csv_dir else "unset"
        return f"openface/imported:{location}{found}"

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
        crop: CropBox,  # noqa: ARG002 - OpenFace ran on whatever it was given
        sampling: FrameSampling,
    ) -> tuple[FrameMeasure, ...]:
        """Read one session's CSV and resample it onto our frame grid.

        The crop is not used: OpenFace was run elsewhere, on whatever video it
        was given, so which region it looked at is a property of that run and
        is recorded by hand rather than imposed here.

        Raises:
            FaceError: if the CSV is missing, unreadable or lacks the columns
                the configured action units need.
        """
        if not self.available():
            raise FaceError(self.unavailable_reason())

        path = self.find_csv(session.session_id)
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


def _sampled_rows(frame: pd.DataFrame, sampling: FrameSampling) -> pd.DataFrame:
    """Take the rows on our sampling grid.

    OpenFace numbers frames from one and processes every frame, so the grid is
    applied here rather than assumed. A CSV that has already been subsampled is
    used as-is, with a warning, since re-subsampling it would silently thin the
    data further.
    """
    if FRAME_COLUMN not in frame.columns:
        return frame
    indices = frame[FRAME_COLUMN].astype("int64")
    if len(indices) > 1:
        step = int(indices.diff().dropna().median() or 1)
        if step > 1:
            logger.warning(
                "the OpenFace output is already subsampled (every %d frames); using "
                "it as-is rather than thinning it further",
                step,
            )
            return frame
    # OpenFace counts from one; our sampling counts from zero.
    return frame.loc[(indices - 1) % sampling.step == 0]


def _to_measures(
    frame: pd.DataFrame, *, config: AppConfig, sampling: FrameSampling
) -> tuple[FrameMeasure, ...]:
    """Convert OpenFace rows into per-frame measures."""
    face = config.face
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
