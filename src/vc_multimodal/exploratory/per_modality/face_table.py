"""The OpenFace face table: 18 features, built with aggregate's own functions.

Reads the per-frame output of `vc face --backend openface` (smoothed in the
backend) and the speaking/listening timeline `vc turns` produced, and computes
the pre-registered face features: AU01, AU06 and AU12, each with a mean, an SD
and a 90th percentile, in each window. The windowing, the 30 s minimum of
measured time, the statistics and the QC-note blanking are `vc aggregate`'s own
functions, run on a configuration narrowed to those three units, so the only
differences from the existing face features are the backend and the smoothing.

The table is written under the output root, never to `features.csv`, and no
label is read. The summary it prints is counts and distributions only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.exploratory.per_modality.analyze import FACE_BACKEND, file_sha256
from vc_multimodal.exploratory.per_modality.plan import Plan
from vc_multimodal.faces import require_single_backend
from vc_multimodal.features.aggregate_math import mask_in_spans
from vc_multimodal.io_utils import read_parquet, write_csv
from vc_multimodal.paths import DataRoots, discover_sessions
from vc_multimodal.qc_notes import load as load_qc_notes
from vc_multimodal.stages import aggregate as aggregate_stage
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import turns as turns_stage

STAGE: Final = "per_modality_face_table"
FACE_TABLE_FILENAME: Final = "face_openface_features.csv"
WINDOWS: Final = (aggregate_stage.SPEAKING, aggregate_stage.LISTENING)

#: The units the pre-registration uses, and the only ones summarised.
UNITS: Final = ("au01", "au06", "au12")

FLAG_NO_OPENFACE: Final = "no_openface_measures"
FLAG_NO_TIMELINE: Final = "no_timeline"

_PERCENTILES: Final = (1, 10, 25, 50, 75, 90, 99)
_NEAR_ZERO: Final = 0.01


class FaceTableError(RuntimeError):
    """Raised when the face table cannot be built as planned."""


def face_table_path(roots: DataRoots) -> Path:
    """Where the table is written, under the output root."""
    return roots.out_path("exploratory", "per_modality", FACE_TABLE_FILENAME)


def narrowed_config(config: AppConfig) -> AppConfig:
    """The configuration with only AU01, AU06 and AU12, each with a p90, no pose.

    Raises:
        FaceTableError: if any of the three units is not configured.
    """
    units = tuple(unit for unit in config.face.action_units if unit.key in UNITS)
    if tuple(unit.key for unit in units) != UNITS:
        msg = f"face.action_units must include {list(UNITS)}"
        raise FaceTableError(msg)
    face = config.face.model_copy(update={"action_units": units})
    aggregate = config.aggregate.model_copy(
        update={"peak_action_units": UNITS, "pose_measures": ()}
    )
    return config.model_copy(update={"face": face, "aggregate": aggregate})


def planned_columns(plan: Plan) -> tuple[str, ...]:
    """The face columns the plan names, speaking then listening."""
    return (*plan.modalities[WINDOWS[0]].features, *plan.modalities[WINDOWS[1]].features)


@dataclass(frozen=True, slots=True)
class FaceTable:
    """The table, where it went, and what describes it."""

    frame: pd.DataFrame
    path: Path
    sha256: str
    columns: tuple[str, ...]
    valid_fraction: Mapping[int, float | None]
    distributions: pd.DataFrame
    blanked: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    without_measures: tuple[int, ...] = ()

    def report_lines(self) -> list[str]:
        """Counts and digests only: no label is involved."""
        frame = self.frame
        measured = frame[list(self.columns)].notna().any(axis=1)
        lines = [
            f"wrote {self.path}",
            f"SHA-256 {self.sha256}",
            f"{len(frame)} session(s), {int(measured.sum())} with face features, "
            f"{len(self.columns)} feature(s)",
        ]
        lines.extend(
            f"  session {sid}: {len(cols)} feature(s) withheld by a QC note"
            for sid, cols in sorted(self.blanked.items())
        )
        if self.without_measures:
            lines.append(f"NOTE: no OpenFace measures for {list(self.without_measures)}")
        return lines

    def valid_fraction_lines(self) -> list[str]:
        """Valid frames per session, on the sampled grid."""
        values = [v for v in self.valid_fraction.values() if v is not None]
        lines = ["valid-frame fraction per session (5 fps grid, success and confidence):"]
        lines.extend(
            f"  {sid}: {'n/a' if v is None else f'{v:.3f}'}"
            for sid, v in sorted(self.valid_fraction.items())
        )
        if values:
            lines.append(
                f"  min {min(values):.3f}, median {float(np.median(values)):.3f}, "
                f"max {max(values):.3f}"
            )
        return lines

    def distribution_lines(self) -> list[str]:
        """Each unit's smoothed intensity over valid frames, and the features."""
        lines = ["smoothed AU intensity over valid frames, by window:"]
        lines.append(self.distributions.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        values = self.frame[list(self.columns)]
        described = values.describe(percentiles=[0.5]).T
        described["missing"] = values.isna().sum()
        lines.extend(["", "session-level features:"])
        lines.append(
            described[["count", "missing", "min", "50%", "max"]].to_string(
                float_format=lambda v: f"{v:.3f}"
            )
        )
        return lines


def _frame_distribution(pooled: Mapping[tuple[str, str], list[np.ndarray]]) -> pd.DataFrame:
    """Percentiles of each unit's valid smoothed values, per window."""
    rows: list[dict[str, Any]] = []
    for (unit, window), chunks in pooled.items():
        values = np.concatenate(chunks) if chunks else np.zeros(0)
        values = values[np.isfinite(values)]
        row: dict[str, Any] = {"unit": unit, "window": window, "frames": int(values.size)}
        if values.size:
            row["mean"] = float(values.mean())
            row["sd"] = float(values.std(ddof=1)) if values.size > 1 else np.nan
            for q in _PERCENTILES:
                row[f"p{q}"] = float(np.percentile(values, q))
            row["max"] = float(values.max())
            row[f"share<{_NEAR_ZERO}"] = float((values < _NEAR_ZERO).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _pool(
    pooled: dict[tuple[str, str], list[np.ndarray]],
    frames: pd.DataFrame,
    timeline: aggregate_stage.Timeline,
) -> None:
    """Add one session's valid smoothed values to the pooled distributions."""
    detected = np.asarray(frames["detected"], dtype=bool)
    times = np.asarray(frames["timestamp_s"], dtype=np.float64)
    masks = {
        "all": detected,
        aggregate_stage.SPEAKING: detected & mask_in_spans(times, timeline.speaking),
        aggregate_stage.LISTENING: detected & mask_in_spans(times, timeline.listening),
    }
    for unit in UNITS:
        if unit not in frames.columns:
            continue
        values = np.asarray(frames[unit], dtype=np.float64)
        for window, mask in masks.items():
            pooled[(unit, window)].append(values[mask])


def build(config: AppConfig, roots: DataRoots, plan: Plan) -> FaceTable:
    """Build and write the face table from the OpenFace per-frame output.

    Raises:
        FaceTableError: if the columns would not be the planned ones, or the
            stored measures are not OpenFace's.
    """
    narrowed = narrowed_config(config)
    columns = planned_columns(plan)
    named = aggregate_stage.face_feature_names(narrowed)
    if named != columns:
        msg = f"aggregate would name the features {named}, not the planned {columns}"
        raise FaceTableError(msg)

    stored = face_stage.stored_backends(roots, FACE_BACKEND)
    backend = require_single_backend(
        [record["backend"] for record in stored.values()], context="the OpenFace measures"
    )
    if stored and backend != FACE_BACKEND:
        msg = f"the OpenFace folder holds {backend} measures"
        raise FaceTableError(msg)

    sessions = discover_sessions(roots.data, config.dataset).sessions
    rows: list[dict[str, object]] = []
    valid_fraction: dict[int, float | None] = {}
    without: list[int] = []
    pooled: dict[tuple[str, str], list[np.ndarray]] = {
        (unit, window): [] for unit in UNITS for window in ("all", *WINDOWS)
    }
    for session in sessions:
        sid = session.session_id
        row: dict[str, object] = {"session_id": sid, "wave": session.wave}
        row.update(dict.fromkeys(columns))
        flags: list[str] = []
        face_file = face_stage.face_path(roots, sid, FACE_BACKEND)
        timeline_file = turns_stage.timeline_path(roots, sid)
        frames = read_parquet(face_file) if face_file.exists() else None
        dropped = aggregate_stage.session_dropped_fraction(frames) if frames is not None else None
        valid_fraction[sid] = None if dropped is None else 1.0 - dropped
        windows = aggregate_stage.WindowSummary()
        if frames is None:
            flags.append(FLAG_NO_OPENFACE)
            without.append(sid)
        elif not timeline_file.exists():
            flags.append(FLAG_NO_TIMELINE)
        else:
            timeline = aggregate_stage.Timeline.from_frame(read_parquet(timeline_file))
            features, windows = aggregate_stage.summarise_face(frames, timeline, narrowed)
            row.update({name: features.get(name) for name in columns})
            _pool(pooled, frames, timeline)
        row.update(
            {
                "qc__face_backend": stored.get(sid, {}).get("backend", ""),
                "qc__speaking_seconds": windows.speaking_seconds,
                "qc__listening_seconds": windows.listening_seconds,
                "qc__face_measured_speaking": windows.measured_speaking,
                "qc__face_measured_listening": windows.measured_listening,
                "qc__face_valid_fraction": valid_fraction[sid],
                "qc__annotations": "",
                "qc__annotation_reason": "",
                "qc__flags": ";".join(flags),
            }
        )
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("session_id", ignore_index=True)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    notes = load_qc_notes(roots.work / config.qc.notes_path)
    frame, blanked = aggregate_stage.apply_qc_notes(frame, notes, columns)

    path = face_table_path(roots)
    write_csv(path, frame)
    return FaceTable(
        frame=frame,
        path=path,
        sha256=file_sha256(path),
        columns=columns,
        valid_fraction=valid_fraction,
        distributions=_frame_distribution(pooled),
        blanked=blanked,
        without_measures=tuple(without),
    )
