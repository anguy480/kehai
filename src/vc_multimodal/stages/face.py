"""Stage: facial action units from the participant's video tile.

Frames are sampled at a rate that divides the recording's own frame rate
exactly, cropped to the participant's tile after correcting for the letterbox
bars these recordings carry, and measured. Frames are decoded, measured and
discarded: nothing is written and nothing is displayed.

Which action units, and why those, is settled by the lab's own published work
rather than here (docs/decisions/0013). The speaking/listening split happens in
`vc aggregate`, which joins these per-frame measures with the timeline `vc
turns` produced; this stage does not need to know which role is speaking.

The backend is recorded on every session. MediaPipe blendshape scores and
OpenFace action unit intensities are different scales for the same constructs,
so a table mixing them is refused rather than pooled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import pandas as pd

from vc_multimodal.config import AppConfig, CropBox
from vc_multimodal.faces import FaceBackend, FaceError, get_backend, require_single_backend
from vc_multimodal.features.face_math import (
    FrameMeasure,
    apply_confidence_threshold,
    dropped_fraction,
)
from vc_multimodal.features.geometry import detect_content_box, resolve_regions
from vc_multimodal.features.sampling import FrameSampling, SamplingError, resolve_sampling
from vc_multimodal.ffmpeg import FfmpegError, FfmpegTools, parse_media_info
from vc_multimodal.io_utils import read_csv, write_csv, write_parquet
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.runner import StageReport, run_sessions

logger = get_logger(__name__)

STAGE: Final = "face"
FACE_DIRNAME: Final = "face"
FACE_QC_FILENAME: Final = "face_qc.csv"

FLAG_TOO_MANY_DROPPED: Final = "face_too_many_frames_dropped"
FLAG_NO_FACE_FOUND: Final = "face_no_frames_measured"
FLAG_NO_LETTERBOX: Final = "face_letterbox_not_detected"
FLAG_NO_HEAD_POSE: Final = "face_head_pose_unavailable"

QC_COLUMN_ORDER: Final = (
    "session_id",
    "wave",
    "backend",
    "backend_version",
    "native_fps",
    "sample_fps",
    "frame_step",
    "n_frames_sampled",
    "n_frames_measured",
    "dropped_fraction",
    "mean_confidence",
    "letterbox_detected",
    "crop_x",
    "crop_y",
    "crop_width",
    "crop_height",
    "flags",
)


def face_dir(roots: DataRoots) -> Path:
    """Directory holding per-frame facial measures, under the work root."""
    return roots.work_path(FACE_DIRNAME, create_parent=True)


def face_path(roots: DataRoots, session_id: int) -> Path:
    """Where one session's per-frame measures are written."""
    return roots.work_path(FACE_DIRNAME, f"{session_id}.parquet")


def frame_table(
    session_id: int, measures: Sequence[FrameMeasure], unit_keys: Sequence[str]
) -> pd.DataFrame:
    """Build the per-frame table for one session.

    One row per sampled frame, including the frames with no usable face: the
    dropped fraction only means something if every frame looked at is present.
    """
    data: dict[str, object] = {
        "session_id": [session_id] * len(measures),
        "frame_index": [m.frame_index for m in measures],
        "timestamp_s": [m.timestamp_s for m in measures],
        "detected": [m.detected for m in measures],
        "confidence": [m.confidence for m in measures],
    }
    for key in unit_keys:
        data[key] = [m.units.get(key) if m.detected else None for m in measures]
    data["jaw"] = [m.jaw for m in measures]
    data["blink"] = [m.blink for m in measures]
    for index, axis in enumerate(("head_pitch", "head_yaw", "head_roll")):
        data[axis] = [m.head[index] if m.head is not None else None for m in measures]

    frame = pd.DataFrame(data)
    frame["session_id"] = frame["session_id"].astype("int64")
    frame["frame_index"] = frame["frame_index"].astype("int64")
    frame["detected"] = frame["detected"].astype(bool)
    for column in frame.columns:
        if column not in {"session_id", "frame_index", "detected"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def participant_crop(config: AppConfig, frame: cv2.typing.MatLike) -> tuple[CropBox, bool]:
    """The participant's tile, corrected for letterboxing.

    Returns:
        The crop in fractional frame coordinates, and whether bars were found.
    """
    content = detect_content_box(frame) if config.video.letterbox_detection == "auto" else None
    height, width = frame.shape[:2]
    regions = resolve_regions(config, width, height, content=content)
    participant = next((region for region in regions if region.role == "participant"), regions[-1])
    return participant.tile_box, bool(content and content.detected)


def qc_record(
    session: RawSession,
    measures: Sequence[FrameMeasure],
    *,
    backend: FaceBackend,
    sampling: FrameSampling,
    crop: CropBox,
    letterboxed: bool,
    config: AppConfig,
) -> dict[str, object]:
    """Summarise one session's facial measurement."""
    detected = [m for m in measures if m.detected]
    dropped = dropped_fraction(measures)
    confidences = [m.confidence for m in detected]

    flags: list[str] = []
    if not detected:
        flags.append(FLAG_NO_FACE_FOUND)
    elif dropped is not None and dropped > config.face.max_dropped_fraction:
        flags.append(FLAG_TOO_MANY_DROPPED)
    if not letterboxed and config.video.letterbox_detection == "auto":
        flags.append(FLAG_NO_LETTERBOX)
    if detected and all(m.head is None for m in detected):
        flags.append(FLAG_NO_HEAD_POSE)

    return {
        "session_id": session.session_id,
        "wave": session.wave,
        "backend": backend.name,
        "backend_version": backend.version(),
        "native_fps": round(sampling.native_fps, 4),
        "sample_fps": round(sampling.effective_fps, 4),
        "frame_step": sampling.step,
        "n_frames_sampled": len(measures),
        "n_frames_measured": len(detected),
        "dropped_fraction": None if dropped is None else round(dropped, 4),
        "mean_confidence": (round(sum(confidences) / len(confidences), 4) if confidences else None),
        "letterbox_detected": letterboxed,
        "crop_x": round(crop.x, 6),
        "crop_y": round(crop.y, 6),
        "crop_width": round(crop.width, 6),
        "crop_height": round(crop.height, 6),
        "flags": ";".join(dict.fromkeys(flags)),
    }


def build_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Assemble face QC rows into a correctly typed table."""
    frame = pd.DataFrame(list(rows), columns=list(QC_COLUMN_ORDER))
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce").astype("int64")
    for column in ("wave", "backend", "backend_version", "flags"):
        frame[column] = frame[column].fillna("").astype("string")
    for column in ("frame_step", "n_frames_sampled", "n_frames_measured"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in (
        "native_fps",
        "sample_fps",
        "dropped_fraction",
        "mean_confidence",
        "crop_x",
        "crop_y",
        "crop_width",
        "crop_height",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["letterbox_detected"] = frame["letterbox_detected"].astype("boolean")
    return frame.sort_values("session_id", ignore_index=True)


@dataclass(frozen=True, slots=True)
class FaceResult:
    """What the face stage produced."""

    report: StageReport
    frame: pd.DataFrame
    path: Path


def _native_fps(roots: DataRoots, session: RawSession, tools: FfmpegTools) -> float | None:
    """The recording's frame rate, from the inventory or by probing."""
    path = roots.out_path("inventory.csv", create_parent=False)
    if path.exists():
        try:
            inventory = read_csv(path)
        except (OSError, ValueError):  # pragma: no cover - defensive
            inventory = None
        if inventory is not None and {"session_id", "fps"} <= set(inventory.columns):
            match = inventory.loc[inventory["session_id"] == session.session_id, "fps"]
            if not match.empty and pd.notna(match.iloc[0]):
                return float(match.iloc[0])
    try:
        return (
            parse_media_info(tools.probe(session.path)).duration_s
            and parse_media_info(tools.probe(session.path)).fps
        )
    except FfmpegError:
        return None


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    session_ids: Sequence[int] | None = None,
    workers: int | None = None,
    force: bool = False,
    backend: FaceBackend | None = None,
    tools: FfmpegTools | None = None,
) -> FaceResult:
    """Measure facial action units for every requested session.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to measure, or None for all.
        workers: Parallel workers, or None to choose automatically.
        force: Re-measure sessions whose per-frame table already exists.
        backend: Face backend; built from config if omitted.
        tools: Located binaries; discovered if omitted.

    Returns:
        The stage report and the written QC table.

    Raises:
        FaceError: if the backend cannot run, or if the assembled table would
            mix backends.
    """
    engine = backend or get_backend(config, roots)
    if not engine.available():
        raise FaceError(engine.unavailable_reason())
    logger.info("%s: backend %s (%s)", STAGE, engine.name, engine.version())

    binaries = tools or FfmpegTools.discover()
    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []
    face_dir(roots)
    records: dict[int, Mapping[str, object]] = {}

    def is_done(session: RawSession) -> bool:
        return face_path(roots, session.session_id).exists()

    def measure_one(session: RawSession) -> str:
        native = _native_fps(roots, session, binaries)
        try:
            sampling = resolve_sampling(native, config.face.sample_fps)
        except SamplingError as exc:
            raise FaceError(str(exc)) from exc

        crop, letterboxed = _crop_for(session, config, binaries, roots)
        measures = engine.measure_session(session, config=config, crop=crop, sampling=sampling)
        measures = apply_confidence_threshold(measures, config.face.min_confidence)

        table = frame_table(session.session_id, measures, config.face.unit_keys)
        write_parquet(face_path(roots, session.session_id), table)

        record = qc_record(
            session,
            measures,
            backend=engine,
            sampling=sampling,
            crop=crop,
            letterboxed=letterboxed,
            config=config,
        )
        records[session.session_id] = record
        dropped = record["dropped_fraction"]
        shown = f"{dropped:.1%}" if isinstance(dropped, float) else "n/a"
        return (
            f"{record['n_frames_measured']}/{record['n_frames_sampled']} frame(s) "
            f"measured, {shown} dropped"
        )

    report = run_sessions(
        STAGE,
        selected,
        measure_one,
        workers=workers,
        force=force,
        is_done=is_done,
        backend="threads",
        notes=notes,
    )

    # A skipped session keeps its recorded QC row, read back from the previous
    # table so the backend it used is still checked against the others.
    previous = _previous_qc(roots)
    for outcome in report.skipped:
        if outcome.session_id in previous:
            records[outcome.session_id] = previous[outcome.session_id]

    frame = build_frame(list(records.values()))
    require_single_backend(
        [str(name) for name in frame["backend"].tolist()], context="the face QC table"
    )
    target = roots.out_path(FACE_QC_FILENAME)
    write_csv(target, frame)
    logger.info("wrote %s with %d row(s)", target, len(frame))

    return FaceResult(report=report, frame=frame, path=target)


def _crop_for(
    session: RawSession, config: AppConfig, tools: FfmpegTools, roots: DataRoots
) -> tuple[CropBox, bool]:
    """Work out the participant crop from one frame of the recording."""
    scratch = roots.work_path("tmp", STAGE, f"{session.session_id}.png")
    try:
        tools.extract_frame(session.path, 1.0, scratch)
        frame = cv2.imread(str(scratch), cv2.IMREAD_COLOR)
    finally:
        scratch.unlink(missing_ok=True)
    if frame is None:
        msg = f"could not read a frame from {session.path.name} to find the crop"
        raise FaceError(msg)
    return participant_crop(config, frame)


def _previous_qc(roots: DataRoots) -> dict[int, Mapping[str, object]]:
    """The previous QC rows, keyed by session, or empty."""
    path = roots.out_path(FACE_QC_FILENAME, create_parent=False)
    if not path.exists():
        return {}
    try:
        frame = read_csv(path)
    except (OSError, ValueError):  # pragma: no cover - defensive
        return {}
    if "session_id" not in frame.columns:
        return {}
    return {
        int(row["session_id"]): {k: row[k] for k in frame.columns} for _, row in frame.iterrows()
    }


def summarise(frame: pd.DataFrame, config: AppConfig) -> list[str]:
    """Summarise facial measurement. Counts and rates only."""
    if frame.empty:
        return ["no sessions were measured"]

    backends = sorted({name for name in frame["backend"].dropna() if name})
    lines = [
        f"facial measures for {len(frame)} session(s) using {', '.join(backends)}",
        f"action units: {', '.join(config.face.unit_keys)}",
    ]

    sampled = frame["n_frames_sampled"].dropna()
    if not sampled.empty:
        lines.append(
            f"frames sampled per session: min {sampled.min()}, "
            f"median {sampled.median():.0f}, max {sampled.max()} "
            f"(every {int(frame['frame_step'].dropna().median())}th frame)"
        )

    dropped = frame["dropped_fraction"].dropna()
    if not dropped.empty:
        lines.append(
            f"frames with no usable face: min {dropped.min():.1%}, "
            f"median {dropped.median():.1%}, max {dropped.max():.1%}"
        )
        worst_id = frame.loc[frame["dropped_fraction"] == dropped.max(), "session_id"]
        lines.append(f"  worst session(s): {sorted(int(i) for i in worst_id)}")

    letterboxed = int(frame["letterbox_detected"].fillna(False).sum())
    lines.append(f"letterbox corrected in {letterboxed} of {len(frame)} session(s)")
    lines.append(
        "  head pose is recorded as head pose. It is not gaze: these recordings "
        "have no eye tracker (docs/decisions/0013)."
    )

    flagged = frame.loc[frame["flags"].astype(str) != ""]
    lines.append("")
    if flagged.empty:
        lines.append("flags: none")
    else:
        by_flag: dict[str, list[int]] = {}
        for session_id, raw in zip(flagged["session_id"], flagged["flags"], strict=True):
            for flag in str(raw).split(";"):
                if flag:
                    by_flag.setdefault(flag, []).append(int(session_id))
        lines.append("flags:")
        lines.extend(
            f"  {name}: {len(ids)} session(s) {sorted(ids)}"
            for name, ids in sorted(by_flag.items())
        )
    return lines
