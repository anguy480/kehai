"""Stage: one row per session, which is the table that ships.

Joins what the earlier stages produced into the feature table the handoff
bundle carries: the turn features, the prosodic features, and the facial action
units summarised separately over the time the participant was speaking and the
time they were listening.

That split is this project's addition to the lab's prior work
(docs/decisions/0013), and it is the reason this stage exists rather than each
stage writing its own features straight out: the windows come from `vc turns`
and the measures from `vc face`, and neither knows about the other.

Three rules govern the output:

* **A session missing an upstream stage still gets a row**, with those features
  absent and the stage named in `qc__stages_missing`. A silently short table
  would be worse than an honest one with gaps.
* **A window with too little measured time yields no features for that
  window**, rather than statistics resting on a handful of frames.
* **Facial features from two different backends are never pooled.** The
  backend is carried on every row and a mixed table is refused
  (docs/decisions/0013).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.contracts import ContractError, family_of, feature_schema, validate
from vc_multimodal.faces import require_single_backend
from vc_multimodal.features.aggregate_math import (
    WindowCoverage,
    feature_names,
    mask_in_spans,
    stats_plan,
    summarise_window,
    window_coverage,
)
from vc_multimodal.features.spans import Span
from vc_multimodal.io_utils import read_csv, read_parquet, write_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.modeling.tiers import describe_plan, resolve_tiers
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.qc_notes import QcNotes, columns_for_families
from vc_multimodal.qc_notes import load as load_qc_notes
from vc_multimodal.runner import StageReport, run_sessions
from vc_multimodal.session_tables import carry_forward, combine
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import prosody as prosody_stage
from vc_multimodal.stages import turns as turns_stage

logger = get_logger(__name__)

STAGE: Final = "aggregate"
FEATURES_FILENAME: Final = "features.csv"

SPEAKING: Final = "face_speaking"
LISTENING: Final = "face_listening"

FLAG_MISSING_STAGE: Final = "aggregate_upstream_stage_missing"
FLAG_SHORT_SPEAKING: Final = "aggregate_too_little_speaking"
FLAG_SHORT_LISTENING: Final = "aggregate_too_little_listening"
FLAG_NO_FACE_DATA: Final = "aggregate_no_facial_measures"
FLAG_TOO_MANY_FEATURES: Final = "aggregate_feature_budget_exceeded"

# A feature varying by less than this across sessions cannot distinguish them.
_CONSTANT_TOLERANCE: Final = 1e-12

# Beyond this many sessions per flag, print the count without the IDs.
_MAX_LISTED_SESSIONS: Final = 12

QC_COLUMNS: Final = (
    "qc__stages_missing",
    "qc__role_source",
    "qc__face_backend",
    "qc__speaking_seconds",
    "qc__listening_seconds",
    "qc__face_frames_speaking",
    "qc__face_frames_listening",
    "qc__face_measured_speaking",
    "qc__face_measured_listening",
    "qc__annotations",
    "qc__annotation_reason",
    "qc__flags",
)

_STRING_QC: Final = (
    "qc__stages_missing",
    "qc__role_source",
    "qc__face_backend",
    "qc__annotations",
    "qc__annotation_reason",
    "qc__flags",
)
_INT_QC: Final = ("qc__face_frames_speaking", "qc__face_frames_listening")


def features_path(roots: DataRoots) -> Path:
    """Where the feature table is written."""
    return roots.out_path(FEATURES_FILENAME)


def face_feature_names(config: AppConfig) -> tuple[str, ...]:
    """Every facial feature column, both windows, in order."""
    units = config.face.unit_keys
    peaks = config.aggregate.peak_action_units
    pose = config.aggregate.pose_measures
    return (
        *feature_names(SPEAKING, units, peaks, pose),
        *feature_names(LISTENING, units, peaks, pose),
    )


def all_feature_names(config: AppConfig, *, upstream: Sequence[str]) -> tuple[str, ...]:
    """Every feature column the table will carry.

    Args:
        config: Resolved configuration.
        upstream: Feature columns found in the upstream tables, in their order.
    """
    return (*upstream, *face_feature_names(config))


@dataclass(frozen=True, slots=True)
class Timeline:
    """The participant's speaking and listening spans for one session."""

    speaking: tuple[Span, ...] = ()
    listening: tuple[Span, ...] = ()

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Timeline:
        """Read a timeline table as written by `vc turns`."""
        by_state: dict[str, list[Span]] = {"speaking": [], "listening": []}
        for state, start, end in zip(frame["state"], frame["start_s"], frame["end_s"], strict=True):
            key = str(state)
            if key in by_state:
                by_state[key].append(Span(float(start), float(end)))
        return cls(speaking=tuple(by_state["speaking"]), listening=tuple(by_state["listening"]))


@dataclass(frozen=True, slots=True)
class WindowSummary:
    """How much of each window was measured, for the QC columns."""

    speaking_seconds: float = 0.0
    listening_seconds: float = 0.0
    frames_speaking: int = 0
    frames_listening: int = 0
    measured_speaking: float | None = None
    measured_listening: float | None = None


def summarise_face(
    frames: pd.DataFrame, timeline: Timeline, config: AppConfig
) -> tuple[dict[str, float | None], WindowSummary]:
    """Summarise per-frame facial measures over both windows.

    Returns:
        The facial features, and how much of each window was measured.
    """
    units = list(config.face.unit_keys)
    pose = list(config.aggregate.pose_measures)
    plan = stats_plan(units, config.aggregate.peak_action_units, pose)

    timestamps = np.asarray(frames["timestamp_s"], dtype=np.float64)
    detected = np.asarray(frames["detected"], dtype=bool)
    values = {
        measure: np.asarray(frames[measure], dtype=np.float64)
        for measure in [*units, *pose]
        if measure in frames.columns
    }

    features: dict[str, float | None] = {}
    coverage: dict[str, WindowCoverage] = {}
    minimums = {
        SPEAKING: config.aggregate.min_speaking_s,
        LISTENING: config.aggregate.min_listening_s,
    }

    for window, spans in ((SPEAKING, timeline.speaking), (LISTENING, timeline.listening)):
        measured = window_coverage(timestamps, detected, spans)
        coverage[window] = measured
        names = feature_names(window, units, config.aggregate.peak_action_units, pose)

        if measured.measured_seconds < minimums[window]:
            # Not enough measured time for a statistic to mean anything, so the
            # window contributes nothing rather than noise.
            features.update(dict.fromkeys(names))
            continue

        inside = mask_in_spans(timestamps, spans)
        summary = summarise_window(values, inside, detected, stats_by_measure=plan)
        features.update({f"{window}__{key}": value for key, value in summary.items()})
        # Anything the plan expects but the frame table did not carry.
        for name in names:
            features.setdefault(name, None)

    speaking, listening = coverage[SPEAKING], coverage[LISTENING]

    def fraction(item: WindowCoverage) -> float | None:
        return None if item.measured_fraction is None else round(item.measured_fraction, 4)

    return features, WindowSummary(
        speaking_seconds=round(speaking.seconds, 3),
        listening_seconds=round(listening.seconds, 3),
        frames_speaking=speaking.n_frames,
        frames_listening=listening.n_frames,
        measured_speaking=fraction(speaking),
        measured_listening=fraction(listening),
    )


@dataclass(frozen=True, slots=True)
class UpstreamTables:
    """The feature tables the earlier stages wrote."""

    turns: pd.DataFrame | None = None
    prosody: pd.DataFrame | None = None

    @property
    def missing(self) -> tuple[str, ...]:
        """Which upstream stages produced nothing."""
        absent = []
        if self.turns is None:
            absent.append("turns")
        if self.prosody is None:
            absent.append("prosody")
        return tuple(absent)

    def feature_columns(self) -> tuple[str, ...]:
        """Feature columns across both tables, in table order."""
        columns: list[str] = []
        for table in (self.turns, self.prosody):
            if table is None:
                continue
            columns.extend(name for name in table.columns if family_of(name) is not None)
        return tuple(dict.fromkeys(columns))

    def row_for(self, session_id: int) -> tuple[dict[str, object], list[str]]:
        """The upstream features for one session, and the flags they carry."""
        values: dict[str, object] = {}
        flags: list[str] = []
        for table in (self.turns, self.prosody):
            if table is None:
                continue
            match = table.loc[table["session_id"] == session_id]
            if match.empty:
                continue
            row = match.iloc[0]
            for name in table.columns:
                if family_of(name) is not None:
                    values[name] = row[name]
            if "qc__flags" in table.columns and pd.notna(row["qc__flags"]):
                flags.extend(flag for flag in str(row["qc__flags"]).split(";") if flag)
            if "qc__role_source" in table.columns and pd.notna(row["qc__role_source"]):
                values.setdefault("qc__role_source", str(row["qc__role_source"]))
        return values, flags


def load_upstream(roots: DataRoots) -> UpstreamTables:
    """Read the turn and prosody feature tables, where they exist."""

    def read(name: str) -> pd.DataFrame | None:
        path = roots.out_path(name, create_parent=False)
        if not path.exists():
            return None
        try:
            return read_csv(path)
        except (OSError, ValueError):  # pragma: no cover - defensive
            logger.warning("could not read %s; treating it as absent", name)
            return None

    return UpstreamTables(
        turns=read(turns_stage.TURN_FEATURES_FILENAME),
        prosody=read(prosody_stage.PROSODY_FEATURES_FILENAME),
    )


def build_frame(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    """Assemble the feature table with declared dtypes."""
    ordered = ["session_id", "wave", *columns, *QC_COLUMNS]
    frame = pd.DataFrame(list(rows), columns=ordered)
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce").astype("int64")
    frame["wave"] = frame["wave"].fillna("").astype("string")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in _STRING_QC:
        frame[column] = frame[column].fillna("").astype("string")
    for column in _INT_QC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in (
        "qc__speaking_seconds",
        "qc__listening_seconds",
        "qc__face_measured_speaking",
        "qc__face_measured_listening",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame.sort_values("session_id", ignore_index=True)


@dataclass(frozen=True, slots=True)
class AggregateResult:
    """What the aggregate stage produced."""

    report: StageReport
    frame: pd.DataFrame
    path: Path
    feature_columns: tuple[str, ...] = ()
    tier_lines: tuple[str, ...] = field(default=())
    #: Features taking one value across every session. They cannot contribute
    #: to any model, and a constant confirmatory feature is a design problem
    #: rather than a curiosity.
    constant_features: tuple[str, ...] = ()
    #: Confirmed findings that were applied, and the columns each blanked.
    qc_notes: QcNotes = field(default_factory=QcNotes)
    blanked_by_note: Mapping[int, tuple[str, ...]] = field(default_factory=dict)


def _n_present(row: Mapping[str, object], columns: Sequence[str]) -> int:
    """How many of `columns` this row actually carries a number for."""
    found = 0
    for name in columns:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        found += 1
    return found


def _session_row(
    session: RawSession,
    *,
    config: AppConfig,
    roots: DataRoots,
    upstream: UpstreamTables,
    backends: Mapping[int, Mapping[str, str]],
) -> dict[str, object]:
    """Assemble one session's row, features and QC together.

    A session missing an upstream stage still gets a row: the features it
    cannot have are absent and the stage is named, which is more useful than a
    table that is quietly short.
    """
    values, flags = upstream.row_for(session.session_id)
    row: dict[str, object] = {"session_id": session.session_id, "wave": session.wave}
    row.update(values)

    stages_missing = list(upstream.missing)
    if not values:
        stages_missing = sorted({*stages_missing, "turns", "prosody"})

    face_file = face_stage.face_path(roots, session.session_id)
    timeline_file = turns_stage.timeline_path(roots, session.session_id)
    windows = WindowSummary()

    if not face_file.exists() or not timeline_file.exists():
        if not face_file.exists():
            stages_missing.append("face")
        if not timeline_file.exists():
            stages_missing.append("turns")
        row.update(dict.fromkeys(face_feature_names(config)))
        flags.append(FLAG_NO_FACE_DATA)
    else:
        timeline = Timeline.from_frame(read_parquet(timeline_file))
        features, windows = summarise_face(read_parquet(face_file), timeline, config)
        row.update(features)
        if windows.speaking_seconds < config.aggregate.min_speaking_s:
            flags.append(FLAG_SHORT_SPEAKING)
        if windows.listening_seconds < config.aggregate.min_listening_s:
            flags.append(FLAG_SHORT_LISTENING)

    if stages_missing:
        flags.append(FLAG_MISSING_STAGE)

    record = backends.get(session.session_id, {})
    row.update(
        {
            "qc__stages_missing": ";".join(sorted(set(stages_missing))),
            "qc__role_source": values.get("qc__role_source", ""),
            "qc__face_backend": record.get("backend", ""),
            "qc__speaking_seconds": windows.speaking_seconds,
            "qc__listening_seconds": windows.listening_seconds,
            "qc__face_frames_speaking": windows.frames_speaking,
            "qc__face_frames_listening": windows.frames_listening,
            "qc__face_measured_speaking": windows.measured_speaking,
            "qc__face_measured_listening": windows.measured_listening,
            "qc__flags": ";".join(dict.fromkeys(flags)),
        }
    )
    return row


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    session_ids: Sequence[int] | None = None,
    workers: int | None = None,
    force: bool = False,
) -> AggregateResult:
    """Assemble the feature table.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to include, or None for every discovered session.
        workers: Unused; the work is a join and runs in one pass.
        force: Unused; the table is always rebuilt from its inputs.

    Returns:
        The stage report and the written feature table.

    Raises:
        ContractError: if the table breaks its schema or the feature budget.
        FaceError: if the facial measures mix backends.
    """
    del workers  # the join is cheap and always recomputed

    upstream = load_upstream(roots)
    for stage_name in upstream.missing:
        logger.warning(
            "%s: no feature table from `vc %s`; those columns will be absent",
            STAGE,
            stage_name,
        )

    backends = face_stage.stored_backends(roots)
    require_single_backend(
        [record["backend"] for record in backends.values()],
        context="the stored facial measurements",
    )

    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []
    notes.extend(f"no feature table from `vc {name}`" for name in upstream.missing)

    columns = all_feature_names(config, upstream=upstream.feature_columns())
    rows: dict[int, Mapping[str, object]] = {}

    def aggregate_one(session: RawSession) -> str:
        row = _session_row(
            session, config=config, roots=roots, upstream=upstream, backends=backends
        )
        rows[session.session_id] = row
        present = _n_present(row, columns)
        return f"{present}/{len(columns)} feature(s)"

    report = run_sessions(STAGE, selected, aggregate_one, workers=1, backend="threads", notes=notes)

    # Rows for sessions outside this run are kept rather than deleted, so a
    # subset rerun tops the table up instead of shrinking it to the subset.
    target = features_path(roots)
    carried = carry_forward(
        target,
        computed=set(rows),
        columns=["session_id", "wave", *columns, *QC_COLUMNS],
        stage=STAGE,
        force=force,
    )
    frame = build_frame(combine(rows, carried), columns)
    confirmed = load_qc_notes(roots.work / config.qc.notes_path)
    frame, blanked = apply_qc_notes(frame, confirmed, columns)
    validate(frame, feature_schema([*columns, *QC_COLUMNS]), context=STAGE)
    _check_budget(columns, config)

    write_csv(target, frame)
    logger.info("wrote %s with %d row(s) and %d feature(s)", target, len(frame), len(columns))

    plan = resolve_tiers([str(name) for name in frame.columns], config.model)
    tier_lines = tuple(describe_plan(plan, config.model))
    for line in tier_lines:
        logger.info("%s: %s", STAGE, line)

    constant = constant_features(frame, columns)
    if constant:
        logger.warning(
            "%s: %d feature(s) take one value across every session and cannot "
            "contribute to any model: %s",
            STAGE,
            len(constant),
            list(constant),
        )

    return AggregateResult(
        report=report,
        frame=frame,
        path=target,
        feature_columns=tuple(columns),
        tier_lines=tier_lines,
        constant_features=constant,
        qc_notes=confirmed,
        blanked_by_note=blanked,
    )


#: Flag prefix for a session whose features a person marked unusable.
FLAG_ANNOTATED: Final = "annotated"


def apply_qc_notes(
    frame: pd.DataFrame, notes: QcNotes, columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[int, tuple[str, ...]]]:
    """Record every confirmed finding, and blank what it says not to use.

    Blanking rather than dropping the session is the whole point of a note being
    scoped to a modality: a camera too blurry to track faces says nothing about
    the audio, so session 43 keeps its turn-taking and prosodic features and
    loses only the facial ones.

    Blanking rather than leaving the values in place matters just as much. A
    measurement nobody should use is more dangerous than an absent one, because
    whoever did not read the note will model it.

    Returns:
        The table, and the columns blanked per session.
    """
    if not notes or frame.empty:
        return frame, {}

    updated = frame.copy()
    blanked: dict[int, tuple[str, ...]] = {}
    for session_id in notes.sessions:
        mask = updated["session_id"] == session_id
        if not mask.any():
            logger.warning(
                "%s: a QC note names session %s, which is not in the feature table",
                STAGE,
                session_id,
            )
            continue

        updated.loc[mask, "qc__annotations"] = notes.labels(session_id)
        updated.loc[mask, "qc__annotation_reason"] = notes.reasons(session_id)

        families = notes.unavailable_families(session_id)
        affected = columns_for_families(columns, families)
        if affected:
            updated.loc[mask, list(affected)] = pd.NA
            blanked[session_id] = affected
            flags = [
                *(str(v) for v in updated.loc[mask, "qc__flags"] if str(v)),
                *(f"{FLAG_ANNOTATED}_{family}_unavailable" for family in families),
            ]
            updated.loc[mask, "qc__flags"] = ";".join(dict.fromkeys(";".join(flags).split(";")))
            logger.info(
                "%s: session %s: %d feature(s) blanked by a confirmed note (%s)",
                STAGE,
                session_id,
                len(affected),
                notes.labels(session_id),
            )
    return updated, blanked


def constant_features(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[str, ...]:
    """Features that take a single value across every session.

    Worth naming rather than leaving to be discovered during modelling: a
    constant column cannot distinguish sessions, standardising it divides by
    zero, and one occupying a confirmatory slot wastes it. This is how the
    overlap features were found to be structurally zero - whisper-diarization
    partitions time, so simultaneous speech is absent from its output by
    construction rather than by chance.
    """
    found: list[str] = []
    for name in columns:
        values = frame[name].dropna()
        if values.empty:
            continue
        if float(values.max() - values.min()) <= _CONSTANT_TOLERANCE:
            found.append(name)
    return tuple(found)


def _check_budget(columns: Sequence[str], config: AppConfig) -> None:
    """Refuse a table far wider than the budget allows.

    A sanity ceiling, not the scientific argument: that is the
    confirmatory/exploratory split (docs/decisions/0012). This exists to catch
    a bug that generates hundreds of columns.

    Raises:
        ContractError: if the ceiling is exceeded.
    """
    ceiling = config.aggregate.max_features
    if len(columns) > ceiling:
        msg = (
            f"the feature table has {len(columns)} features, above the ceiling of "
            f"{ceiling} (aggregate.max_features). That ceiling is a guard against a "
            f"bug generating columns, not a scientific limit; if the growth is "
            f"intended, raise it deliberately and revisit docs/decisions/0012."
        )
        raise ContractError(msg)


def _variance_lines(result: AggregateResult, config: AppConfig) -> list[str]:
    """Report features that cannot contribute, and say when one is confirmatory."""
    if not result.constant_features:
        return []

    lines = [
        "",
        f"NO VARIANCE: {len(result.constant_features)} feature(s) take one value "
        f"across every session, so they cannot contribute to any model:",
    ]
    lines.extend(f"  {name}" for name in result.constant_features)

    primary = set(config.model.tiers.primary_columns)
    constant_primary = [name for name in result.constant_features if name in primary]
    if constant_primary:
        lines.append(
            f"  {constant_primary} are CONFIRMATORY features. A constant confirmatory "
            f"feature wastes a pre-registered slot and needs replacing before the "
            f"analysis runs."
        )
    return lines


def _annotation_lines(result: AggregateResult) -> list[str]:
    """What the confirmed findings did to this table.

    Reported every run, not only when something changes: a reader who does not
    know a modality was withheld will read its absence as a bug.
    """
    if not result.qc_notes:
        return []
    lines = list(result.qc_notes.report_lines())
    for session_id, columns in sorted(result.blanked_by_note.items()):
        families = sorted({column.split("__", 1)[0] for column in columns})
        lines.append(
            f"  session {session_id}: {len(columns)} feature(s) withheld "
            f"({', '.join(families)}); its other modalities are unaffected"
        )
    return lines


def summarise(result: AggregateResult, config: AppConfig) -> list[str]:
    """Summarise the feature table. Counts and coverage only."""
    frame = result.frame
    if frame.empty:
        return ["no sessions were aggregated"]

    columns = list(result.feature_columns)
    lines = [
        f"{len(frame)} session(s) x {len(columns)} feature(s) "
        f"(ceiling {config.aggregate.max_features})"
    ]

    by_family: dict[str, int] = {}
    for name in columns:
        family = family_of(name)
        if family:
            by_family[family] = by_family.get(family, 0) + 1
    lines.append(
        "by family: "
        + ", ".join(f"{family}={count}" for family, count in sorted(by_family.items()))
    )

    complete = int(frame[columns].notna().all(axis=1).sum())
    lines.append(f"sessions with every feature present: {complete} of {len(frame)}")
    coverage = frame[columns].notna().mean(axis=1)
    lines.append(
        f"per-session completeness: min {coverage.min():.0%}, "
        f"median {coverage.median():.0%}, max {coverage.max():.0%}"
    )

    for column, label in (
        ("qc__speaking_seconds", "participant speaking"),
        ("qc__listening_seconds", "participant listening"),
    ):
        values = frame[column].dropna()
        if not values.empty:
            lines.append(
                f"{label}: min {values.min() / 60:.1f} min, "
                f"median {values.median() / 60:.1f} min, max {values.max() / 60:.1f} min"
            )

    backends = sorted({name for name in frame["qc__face_backend"].dropna() if name})
    if backends:
        lines.append(f"facial backend: {', '.join(backends)}")

    lines.extend(_variance_lines(result, config))

    lines.append("")
    lines.extend(result.tier_lines)

    flagged = frame.loc[frame["qc__flags"].astype(str) != ""]
    lines.append("")
    if flagged.empty:
        lines.append("flags: none")
    else:
        by_flag: dict[str, list[int]] = {}
        for session_id, raw in zip(flagged["session_id"], flagged["qc__flags"], strict=True):
            for flag in str(raw).split(";"):
                if flag:
                    by_flag.setdefault(flag, []).append(int(session_id))
        lines.append("flags:")
        lines.extend(
            f"  {name}: {len(ids)} session(s)"
            + (f" {sorted(ids)}" if len(ids) <= _MAX_LISTED_SESSIONS else "")
            for name, ids in sorted(by_flag.items())
        )

    annotations = _annotation_lines(result)
    if annotations:
        lines.append("")
        lines.extend(annotations)
    return lines
