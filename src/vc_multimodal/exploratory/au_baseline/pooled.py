"""Whole-session facial features: the speaking and listening windows pooled.

The feature table summarises each action unit over the speaking window and the
listening window separately. This module adds a third window, `face_pooled`,
covering the participant's speaking and listening time together, by exactly
aggregate's rules - it calls aggregate's own functions rather than restating
them:

* a frame counts if its timestamp falls in either window and a face was found
  in it; mutual silence belongs to neither window and so not to this one;
* the statistics are aggregate's plan: mean and sd for every unit, p90 for the
  peak units, sd for head pose;
* a window with less measured time than aggregate's per-window minimum yields
  no features rather than noise;
* a confirmed QC note that makes either face window unavailable blanks this
  one too, since it contains both.

As a check that the per-frame output is still the one behind the frozen table,
the speaking and listening windows are recomputed alongside and compared with
the bundle's `features.csv`. Only counts and the largest difference are
reported; no value is printed.
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
from vc_multimodal.features.aggregate_math import (
    feature_names,
    mask_in_spans,
    stats_plan,
    summarise_window,
    window_coverage,
)
from vc_multimodal.io_utils import read_csv, read_parquet, write_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots
from vc_multimodal.qc_notes import QcNotes, columns_for_families
from vc_multimodal.qc_notes import load as load_qc_notes
from vc_multimodal.stages import aggregate as aggregate_stage
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import turns as turns_stage

logger = get_logger(__name__)

STAGE: Final = "au-baseline-pooled"
WINDOW: Final = "face_pooled"
DIRNAME: Final = "au_baseline"
POOLED_FILE: Final = "face_pooled.csv"

#: The QC families whose unavailability blanks the pooled window.
_FACE_FAMILIES: Final = (aggregate_stage.SPEAKING, aggregate_stage.LISTENING)

#: Largest difference tolerated between a recomputed and a frozen feature: the
#: frozen table went through a CSV round trip.
REPRODUCTION_TOLERANCE: Final = 1e-9


class PooledError(RuntimeError):
    """Raised when the pooled features cannot be computed or trusted."""


def file_sha256(path: Path) -> str:
    """Digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pooled_path(roots: DataRoots) -> Path:
    """Where the pooled table is written, under the work root."""
    return roots.work_path(DIRNAME, POOLED_FILE)


def pooled_feature_names(config: AppConfig) -> tuple[str, ...]:
    """The pooled window's columns, in aggregate's order."""
    return feature_names(
        WINDOW,
        config.face.unit_keys,
        config.aggregate.peak_action_units,
        config.aggregate.pose_measures,
    )


def minimum_seconds(config: AppConfig) -> float:
    """Measured time the pooled window needs: the stricter per-window minimum."""
    return max(config.aggregate.min_speaking_s, config.aggregate.min_listening_s)


def summarise_pooled(
    frames: pd.DataFrame, timeline: aggregate_stage.Timeline, config: AppConfig
) -> dict[str, float | None]:
    """Summarise per-frame measures over speaking and listening together."""
    units = list(config.face.unit_keys)
    pose = list(config.aggregate.pose_measures)
    plan = stats_plan(units, config.aggregate.peak_action_units, pose)
    names = pooled_feature_names(config)

    timestamps = np.asarray(frames["timestamp_s"], dtype=np.float64)
    detected = np.asarray(frames["detected"], dtype=bool)
    spans = (*timeline.speaking, *timeline.listening)

    if window_coverage(timestamps, detected, spans).measured_seconds < minimum_seconds(config):
        return dict.fromkeys(names)

    values = {
        measure: np.asarray(frames[measure], dtype=np.float64)
        for measure in [*units, *pose]
        if measure in frames.columns
    }
    inside = mask_in_spans(timestamps, spans)
    summary = summarise_window(values, inside, detected, stats_by_measure=plan)
    features: dict[str, float | None] = {f"{WINDOW}__{k}": v for k, v in summary.items()}
    for name in names:
        features.setdefault(name, None)
    return features


@dataclass(frozen=True, slots=True)
class SessionResult:
    """One session's pooled features, and its recomputed windows for the check."""

    pooled: dict[str, float | None]
    windows: dict[str, float | None]


def session_features(
    session_id: int, config: AppConfig, roots: DataRoots, notes: QcNotes
) -> SessionResult:
    """Pooled and recomputed windowed features for one session, notes applied."""
    face_file = face_stage.face_path(roots, session_id)
    timeline_file = turns_stage.timeline_path(roots, session_id)
    windowed = aggregate_stage.face_feature_names(config)
    if not face_file.exists() or not timeline_file.exists():
        return SessionResult(dict.fromkeys(pooled_feature_names(config)), dict.fromkeys(windowed))

    frames = read_parquet(face_file)
    timeline = aggregate_stage.Timeline.from_frame(read_parquet(timeline_file))
    pooled = summarise_pooled(frames, timeline, config)
    windows, _ = aggregate_stage.summarise_face(frames, timeline, config)

    unavailable = notes.unavailable_families(session_id)
    for name in columns_for_families(list(windows), unavailable):
        windows[name] = None
    if any(family in unavailable for family in _FACE_FAMILIES):
        pooled = dict.fromkeys(pooled)
    return SessionResult(pooled, windows)


@dataclass(frozen=True, slots=True)
class Reproduction:
    """How the recomputed windows compare with the frozen table."""

    n_sessions: int
    n_values: int
    n_missing_mismatches: int
    max_abs_difference: float

    @property
    def matches(self) -> bool:
        """Same missingness everywhere and every value within tolerance."""
        return self.n_missing_mismatches == 0 and self.max_abs_difference <= REPRODUCTION_TOLERANCE


def compare(recomputed: pd.DataFrame, frozen: pd.DataFrame, columns: Sequence[str]) -> Reproduction:
    """Compare two tables on `columns`, aligned by session.

    Raises:
        PooledError: if the tables do not cover the same sessions.
    """
    left = recomputed.set_index("session_id").sort_index()
    right = frozen.set_index("session_id").sort_index()
    if list(left.index) != list(right.index):
        msg = "the recomputed and frozen tables cover different sessions"
        raise PooledError(msg)
    a = left[list(columns)].to_numpy(dtype=np.float64)
    b = right[list(columns)].to_numpy(dtype=np.float64)
    missing_mismatch = np.isnan(a) != np.isnan(b)
    both = ~(np.isnan(a) | np.isnan(b))
    diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    return Reproduction(
        n_sessions=len(left),
        n_values=int(both.sum()),
        n_missing_mismatches=int(missing_mismatch.sum()),
        max_abs_difference=diff,
    )


@dataclass(frozen=True, slots=True)
class PooledResult:
    """What the pooled step wrote, and what it checked."""

    path: Path
    sha256: str
    n_sessions: int
    n_with_features: int
    n_blanked_by_notes: int
    reproduction: Reproduction

    def report_lines(self) -> list[str]:
        """Counts, a digest and the check: no feature value."""
        r = self.reproduction
        return [
            f"pooled AU features for {self.n_sessions} session(s) -> {self.path}",
            f"  sessions with pooled features: {self.n_with_features}",
            f"  sessions blanked by a QC note: {self.n_blanked_by_notes}",
            f"  sha256: {self.sha256}",
            f"  check against the frozen features.csv: {r.n_sessions} session(s), "
            f"{r.n_values} value(s), {r.n_missing_mismatches} missingness mismatch(es), "
            f"max |difference| {r.max_abs_difference:.3g} "
            f"({'MATCHES' if r.matches else 'DOES NOT MATCH'})",
        ]


def run(config: AppConfig, roots: DataRoots, *, bundle_features: Path) -> PooledResult:
    """Compute and write the pooled table for every session in the frozen table.

    Raises:
        PooledError: if the recomputed windows do not reproduce the frozen table,
            in which case nothing is written.
    """
    frozen = read_csv(bundle_features)
    session_ids = [int(v) for v in frozen["session_id"]]
    notes = load_qc_notes(roots.work / config.qc.notes_path)

    pooled_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    blanked = 0
    for session_id in session_ids:
        result = session_features(session_id, config, roots, notes)
        if any(f in notes.unavailable_families(session_id) for f in _FACE_FAMILIES):
            blanked += 1
        pooled_rows.append({"session_id": session_id, **result.pooled})
        window_rows.append({"session_id": session_id, **result.windows})

    names = pooled_feature_names(config)
    pooled = pd.DataFrame(pooled_rows, columns=["session_id", *names])
    windows = pd.DataFrame(window_rows)
    reproduction = compare(windows, frozen, aggregate_stage.face_feature_names(config))
    if not reproduction.matches:
        msg = (
            "the per-frame output does not reproduce the frozen features.csv "
            f"({reproduction.n_missing_mismatches} missingness mismatch(es), max |difference| "
            f"{reproduction.max_abs_difference:.3g}); the pooled table was not written"
        )
        raise PooledError(msg)

    path = pooled_path(roots)
    write_csv(path, pooled)
    logger.info("%s: wrote %s", STAGE, path)
    return PooledResult(
        path=path,
        sha256=file_sha256(path),
        n_sessions=len(pooled),
        n_with_features=int(pooled[list(names)].notna().any(axis=1).sum()),
        n_blanked_by_notes=blanked,
        reproduction=reproduction,
    )
