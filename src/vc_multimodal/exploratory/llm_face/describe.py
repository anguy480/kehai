"""Describe each session's facial behaviour in words, from the per-frame table.

The windows are exactly the ones `vc aggregate` summarises: the participant's
speaking and listening spans from `vc turns`, with the same minimum of measured
time below which a window is not described. A session whose face is marked
unavailable by a confirmed QC note gets no description; one marked degraded is
described and flagged.

Descriptions are written to `$VC_WORK_ROOT/llm_face/` and are never printed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.exploratory.llm_face import template
from vc_multimodal.features.aggregate_math import mask_in_spans, window_coverage
from vc_multimodal.io_utils import read_parquet, write_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots
from vc_multimodal.qc_notes import STATUS_DEGRADED, STATUS_UNAVAILABLE, QcNotes
from vc_multimodal.qc_notes import load as load_qc_notes
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import turns as turns_stage
from vc_multimodal.stages.aggregate import Timeline

logger = get_logger(__name__)

STAGE: Final = "llm-face-describe"
DIRNAME: Final = "llm_face"
DESCRIPTIONS_FILE: Final = "descriptions.csv"
FACE_MODALITY: Final = "face"
PARAGRAPH_BREAK: Final = "\n\n"


def output_dir(roots: DataRoots) -> Path:
    """Where this analysis keeps its descriptions and ratings."""
    path = roots.work / DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(path: Path) -> str:
    """Digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    """Digest of a string, UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def template_sha256() -> str:
    """Digest of the template source, carried by every description."""
    return file_sha256(Path(template.__file__))


def window_stats(
    window: str,
    frames: pd.DataFrame,
    inside: np.ndarray,
    *,
    units: Sequence[str],
    step_s: float,
    measured_seconds: float,
    measured_fraction: float | None,
) -> template.WindowStats:
    """Reduce one window's measured frames to what the template describes."""
    keep = inside & np.asarray(frames["detected"], dtype=bool)
    rows = frames.loc[keep].sort_values("timestamp_s")
    times = np.asarray(rows["timestamp_s"], dtype=np.float64)

    stats: list[template.UnitStats] = []
    for unit in units:
        if unit not in rows.columns:
            continue
        values = np.asarray(rows[unit], dtype=np.float64)
        usable = ~np.isnan(values)
        stats.append(template.unit_stats(unit, times[usable], values[usable], step_s))

    head_sd: float | None = None
    sharp: float | None = None
    if {"head_pitch", "head_yaw"} <= set(rows.columns):
        head_sd, sharp = template.head_stats(
            times,
            np.asarray(rows["head_pitch"], dtype=np.float64),
            np.asarray(rows["head_yaw"], dtype=np.float64),
            step_s,
            measured_seconds,
        )
    return template.WindowStats(
        window=window,
        measured_seconds=measured_seconds,
        measured_fraction=measured_fraction,
        units=tuple(stats),
        head_sd_deg=head_sd,
        sharp_moves_per_min=sharp,
    )


def describe_session(frames: pd.DataFrame, timeline: Timeline, config: AppConfig) -> str:
    """The full description: a speaking paragraph, then a listening paragraph."""
    times = np.asarray(frames["timestamp_s"], dtype=np.float64)
    detected = np.asarray(frames["detected"], dtype=bool)
    step_s = 1.0 / config.face.sample_fps
    windows = (
        ("speaking", timeline.speaking, config.aggregate.min_speaking_s),
        ("listening", timeline.listening, config.aggregate.min_listening_s),
    )
    paragraphs: list[str] = []
    for window, spans, minimum in windows:
        coverage = window_coverage(times, detected, spans)
        if coverage.measured_seconds < minimum:
            paragraphs.append(template.render_insufficient(window))
            continue
        stats = window_stats(
            window,
            frames,
            mask_in_spans(times, spans),
            units=config.face.unit_keys,
            step_s=step_s,
            measured_seconds=coverage.measured_seconds,
            measured_fraction=coverage.measured_fraction,
        )
        paragraphs.append(template.render_window(stats))
    return PARAGRAPH_BREAK.join(paragraphs)


def face_status(notes: QcNotes, session_id: int) -> str:
    """The confirmed face status for a session, or '' if none was recorded."""
    statuses = {n.status for n in notes.for_session(session_id) if n.modality == FACE_MODALITY}
    if STATUS_UNAVAILABLE in statuses:
        return STATUS_UNAVAILABLE
    if STATUS_DEGRADED in statuses:
        return STATUS_DEGRADED
    return ""


@dataclass(frozen=True, slots=True)
class DescribeResult:
    """What `run` wrote, as counts only."""

    path: Path
    n_sessions: int
    n_described: int
    skipped: tuple[tuple[int, str], ...]
    flagged: tuple[tuple[int, str], ...]
    template_sha256: str
    file_sha256: str
    words_min: int
    words_max: int

    def report_lines(self) -> list[str]:
        """A summary that contains no description."""
        return [
            f"descriptions: {self.n_described} of {self.n_sessions} session(s) -> {self.path}",
            f"  skipped: {list(self.skipped) or 'none'}",
            f"  flagged: {list(self.flagged) or 'none'}",
            f"  length: {self.words_min}-{self.words_max} words",
            f"  template sha256: {self.template_sha256}",
            f"  file sha256: {self.file_sha256}",
        ]


def run(config: AppConfig, roots: DataRoots) -> DescribeResult:
    """Describe every session that has face frames."""
    notes = load_qc_notes(roots.work / config.qc.notes_path)
    digest = template_sha256()
    sessions = sorted(int(p.stem) for p in face_stage.face_dir(roots).glob("*.parquet"))

    rows: list[dict[str, object]] = []
    skipped: list[tuple[int, str]] = []
    flagged: list[tuple[int, str]] = []
    for session_id in sessions:
        status = face_status(notes, session_id)
        flag = f"{FACE_MODALITY}={status}" if status else ""
        timeline_file = turns_stage.timeline_path(roots, session_id)
        if status == STATUS_UNAVAILABLE or not timeline_file.exists():
            reason = flag or "no timeline"
            skipped.append((session_id, reason))
            rows.append({"session_id": session_id, "qc_flag": reason, "description": ""})
            continue
        if flag:
            flagged.append((session_id, flag))
        frames = read_parquet(face_stage.face_path(roots, session_id))
        timeline = Timeline.from_frame(read_parquet(timeline_file))
        text = describe_session(frames, timeline, config)
        rows.append({"session_id": session_id, "qc_flag": flag, "description": text})
        logger.info("%s: session %d described (%d words)", STAGE, session_id, len(text.split()))

    table = pd.DataFrame(rows, columns=["session_id", "qc_flag", "description"])
    table["description_sha256"] = [text_sha256(t) if t else "" for t in table["description"]]
    table["template_sha256"] = digest
    path = output_dir(roots) / DESCRIPTIONS_FILE
    write_csv(path, table)

    lengths = [len(str(t).split()) for t in table["description"] if t]
    return DescribeResult(
        path=path,
        n_sessions=len(sessions),
        n_described=len(lengths),
        skipped=tuple(skipped),
        flagged=tuple(flagged),
        template_sha256=digest,
        file_sha256=file_sha256(path),
        words_min=min(lengths, default=0),
        words_max=max(lengths, default=0),
    )
