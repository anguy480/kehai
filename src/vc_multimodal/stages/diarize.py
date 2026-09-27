"""Stage 3: who spoke when.

Every recording carries one mixed audio stream, confirmed across all 62, so
there is no per-speaker audio and diarization is unavoidable rather than
optional: every later speaker attribution rests on it.

The backend is pluggable (see docs/decisions/0002). Segments are normalised to
one form, validated, and written under `$VC_WORK_ROOT` as Parquet so dtypes
survive the trip to the next stage.

Transcript text is the most sensitive artifact in the project. It is written
only into the work tree, only when `diarization.keep_text` is on, and it is
never printed, never logged and never included in a handoff bundle. The QC
table this stage writes into `$VC_OUT_ROOT` has no text column.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.contracts import DIARIZATION_QC_SCHEMA, SEGMENT_SCHEMA, validate
from vc_multimodal.diarization import (
    DiarizationBackend,
    DiarizationError,
    ImportBackend,
    Segment,
    covered_time,
    get_backend,
    overlap_time,
    speakers_in,
    strip_text,
    total_speech,
)
from vc_multimodal.ffmpeg import FfmpegError, FfmpegTools, parse_media_info
from vc_multimodal.io_utils import read_csv, write_csv, write_parquet
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.runner import StageReport, run_sessions
from vc_multimodal.session_tables import carry_forward, combine, has_row

logger = get_logger(__name__)

STAGE: Final = "diarize"
SEGMENTS_DIRNAME: Final = "segments"
DIARIZATION_QC_FILENAME: Final = "diarization_qc.csv"

FLAG_SPEAKER_COUNT: Final = "diarization_unexpected_speaker_count"
FLAG_LOW_COVERAGE: Final = "diarization_low_coverage"
FLAG_OVERLAP_HEAVY: Final = "diarization_heavy_overlap"
FLAG_NO_TEXT: Final = "diarization_no_text"

# Above this fraction of segment time spent overlapping, attribution is
# suspect rather than merely lively.
_HEAVY_OVERLAP_FRACTION: Final = 0.25

# Beyond this many sessions per flag, print the count without the IDs.
_MAX_LISTED_SESSIONS: Final = 12

COLUMN_ORDER: Final = (
    "session_id",
    "wave",
    "backend",
    "n_segments",
    "n_speakers",
    "speakers",
    "segment_seconds",
    "covered_seconds",
    "overlap_seconds",
    "coverage_fraction",
    "has_text",
    "flags",
)


def segments_dir(roots: DataRoots) -> Path:
    """Directory holding normalised segments, under the work root."""
    return roots.work_path(SEGMENTS_DIRNAME, create_parent=True)


def segments_path(roots: DataRoots, session_id: int) -> Path:
    """Where one session's segments are written."""
    return roots.work_path(SEGMENTS_DIRNAME, f"{session_id}.parquet")


def segments_frame(session_id: int, segments: Sequence[Segment]) -> pd.DataFrame:
    """Build the segment table for one session.

    The `text` column is present only when at least one segment carries text,
    so a table written with text off has no column to leak.
    """
    data: dict[str, object] = {
        "session_id": [session_id] * len(segments),
        "speaker": [segment.speaker for segment in segments],
        "start_s": [float(segment.start) for segment in segments],
        "end_s": [float(segment.end) for segment in segments],
    }
    if any(segment.text is not None for segment in segments):
        data["text"] = [segment.text for segment in segments]

    frame = pd.DataFrame(data)
    frame["session_id"] = frame["session_id"].astype("int64")
    frame["speaker"] = frame["speaker"].astype("string")
    for column in ("start_s", "end_s"):
        frame[column] = frame[column].astype("float64")
    return frame


def qc_record(
    session: RawSession,
    segments: Sequence[Segment],
    *,
    backend_name: str,
    duration_s: float | None,
    config: AppConfig,
) -> dict[str, object]:
    """Summarise one session's diarization, without any transcript text."""
    diarization = config.diarization
    speakers = speakers_in(segments)
    segment_seconds = total_speech(segments)
    covered = covered_time(segments)
    overlap = overlap_time(segments)
    coverage = covered / duration_s if duration_s and duration_s > 0 else None

    flags: list[str] = []
    if len(speakers) != diarization.expected_speakers:
        flags.append(FLAG_SPEAKER_COUNT)
    if coverage is not None and coverage < diarization.min_coverage_fraction:
        flags.append(FLAG_LOW_COVERAGE)
    if segment_seconds > 0 and overlap / segment_seconds > _HEAVY_OVERLAP_FRACTION:
        flags.append(FLAG_OVERLAP_HEAVY)
    has_text = any(segment.text for segment in segments)
    if not has_text:
        flags.append(FLAG_NO_TEXT)

    return {
        "session_id": session.session_id,
        "wave": session.wave,
        "backend": backend_name,
        "n_segments": len(segments),
        "n_speakers": len(speakers),
        "speakers": ";".join(speakers),
        "segment_seconds": round(segment_seconds, 3),
        "covered_seconds": round(covered, 3),
        "overlap_seconds": round(overlap, 3),
        "coverage_fraction": None if coverage is None else round(coverage, 4),
        "has_text": has_text,
        "flags": ";".join(dict.fromkeys(flags)),
    }


def build_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Assemble diarization QC rows into a correctly typed table."""
    frame = pd.DataFrame(list(rows), columns=list(COLUMN_ORDER))
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce").astype("int64")
    for column in ("wave", "backend", "speakers", "flags"):
        frame[column] = frame[column].fillna("").astype("string")
    for column in ("n_segments", "n_speakers"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in (
        "segment_seconds",
        "covered_seconds",
        "overlap_seconds",
        "coverage_fraction",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["has_text"] = frame["has_text"].astype("boolean")
    return frame.sort_values("session_id", ignore_index=True)


@dataclass(frozen=True, slots=True)
class DiarizeResult:
    """What the diarize stage produced."""

    report: StageReport
    frame: pd.DataFrame
    path: Path


def _durations(roots: DataRoots) -> Mapping[int, float]:
    """Session durations from the inventory, for coverage. Empty if absent."""
    path = roots.out_path("inventory.csv", create_parent=False)
    if not path.exists():
        return {}
    try:
        frame = read_csv(path)
    except (OSError, ValueError):  # pragma: no cover - defensive
        return {}
    if "session_id" not in frame or "duration_s" not in frame:
        return {}
    return {
        int(session_id): float(duration)
        for session_id, duration in zip(frame["session_id"], frame["duration_s"], strict=True)
        if pd.notna(duration)
    }


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    session_ids: Sequence[int] | None = None,
    workers: int | None = None,
    force: bool = False,
    backend: DiarizationBackend | None = None,
    tools: FfmpegTools | None = None,
) -> DiarizeResult:
    """Diarize every requested session and write segments plus a QC table.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to diarize, or None for all.
        workers: Parallel workers, or None to choose automatically.
        force: Re-diarize sessions whose segments already exist.
        backend: Backend to use; built from config if omitted.
        tools: Located binaries; discovered only if a duration is needed.

    Returns:
        The stage report and the written QC table.

    Raises:
        DiarizationError: if the chosen backend cannot run at all. A backend
            that works but fails on one session is a per-session failure.
    """
    engine = backend or get_backend(config, roots)
    if not engine.available():
        raise DiarizationError(engine.unavailable_reason())
    logger.info("%s: backend %s (%s)", STAGE, engine.name, engine.version())

    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []

    # An import backend can say up front which sessions it has files for,
    # which is far more useful than 62 identical per-session failures.
    if isinstance(engine, ImportBackend):
        scan = engine.scan(
            [session.session_id for session in selected],
            # Every discovered session, so a --sessions subset does not report
            # the other sessions' files as unplaceable.
            known_ids=[session.session_id for session in discovery.sessions],
        )
        for line in scan.report_lines():
            logger.info("%s: %s", STAGE, line)
        notes.extend(scan.report_lines()[1:])

    durations = dict(_durations(roots))
    segments_dir(roots)
    records: dict[int, Mapping[str, object]] = {}

    qc_target = roots.out_path(DIARIZATION_QC_FILENAME)

    def is_done(session: RawSession) -> bool:
        """Done means every output exists, including this session's QC row.

        A session whose artifacts are on disk but whose row is not is not done:
        skipping it would leave the table permanently short of a row, because
        nothing else ever writes one. This is how a row lost to an earlier
        partial run heals itself.
        """
        return segments_path(roots, session.session_id).exists() and has_row(
            qc_target, session.session_id
        )

    def diarize_one(session: RawSession) -> str:
        segments = engine.segments(session)
        if not config.diarization.keep_text:
            segments = strip_text(segments)

        frame = segments_frame(session.session_id, segments)
        validate(frame, SEGMENT_SCHEMA, context=f"session {session.session_id}")
        write_parquet(segments_path(roots, session.session_id), frame)

        duration = durations.get(session.session_id)
        if duration is None:
            # Only needed for the coverage figure. Diarization itself reads a
            # file, not the media, so a recording that will not probe must not
            # cost us the diarization we already have.
            try:
                binaries = tools or FfmpegTools.discover()
                duration = parse_media_info(binaries.probe(session.path)).duration_s
            except FfmpegError as exc:
                logger.warning(
                    "session %s: could not read a duration for the coverage figure (%s)",
                    session.session_id,
                    type(exc).__name__,
                )

        records[session.session_id] = qc_record(
            session,
            segments,
            backend_name=engine.name,
            duration_s=duration,
            config=config,
        )
        speakers = records[session.session_id]["n_speakers"]
        return f"{len(segments)} segment(s), {speakers} speaker(s)"

    report = run_sessions(
        STAGE,
        selected,
        diarize_one,
        workers=workers,
        force=force,
        is_done=is_done,
        backend="threads",
        notes=notes,
    )

    # A skipped session still belongs in the QC table, so its segments are
    # re-read rather than re-diarized.
    for outcome in report.skipped:
        session = next(s for s in selected if s.session_id == outcome.session_id)
        stored = pd.read_parquet(segments_path(roots, outcome.session_id))
        segments = tuple(
            Segment(
                speaker=str(speaker),
                start=float(start),
                end=float(end),
                text=None,
            )
            for speaker, start, end in zip(
                stored["speaker"], stored["start_s"], stored["end_s"], strict=True
            )
        )
        record = dict(
            qc_record(
                session,
                segments,
                backend_name=engine.name,
                duration_s=durations.get(outcome.session_id),
                config=config,
            )
        )
        # Whether text was stored is a property of the file, not of the
        # segments just re-read without it.
        record["has_text"] = "text" in stored.columns
        if record["has_text"]:
            record["flags"] = ";".join(
                flag for flag in str(record["flags"]).split(";") if flag and flag != FLAG_NO_TEXT
            )
        records[outcome.session_id] = record

    # Rows for sessions this run did not compute are kept, whether they were
    # left out by --sessions or skipped as already done. Writing only this
    # run's rows would delete every other session's.
    carried = carry_forward(
        qc_target,
        computed=set(records),
        columns=list(COLUMN_ORDER),
        stage=STAGE,
        force=force,
    )
    frame = build_frame(combine(records, carried))
    validate(frame, DIARIZATION_QC_SCHEMA, context=STAGE)

    write_csv(qc_target, frame)
    logger.info(
        "wrote %s with %d row(s) (%d from this run, %d kept)",
        qc_target,
        len(frame),
        len(records),
        len(carried.rows),
    )

    return DiarizeResult(
        report=report.with_notes(carried.notes(STAGE)), frame=frame, path=qc_target
    )


def summarise(frame: pd.DataFrame, config: AppConfig) -> list[str]:
    """Summarise diarization: counts and times only, never transcript text."""
    if frame.empty:
        return ["no sessions were diarized"]

    expected = config.diarization.expected_speakers
    lines = [
        f"diarized {len(frame)} session(s) with backend "
        f"{', '.join(sorted(set(frame['backend'].dropna())))}"
    ]

    segments = frame["n_segments"].dropna()
    if not segments.empty:
        lines.append(
            f"segments per session: min {segments.min()}, "
            f"median {segments.median():.0f}, max {segments.max()}"
        )

    speaker_counts = frame["n_speakers"].value_counts().sort_index()
    lines.append(
        "speakers per session: "
        + ", ".join(f"{n} -> {count} session(s)" for n, count in speaker_counts.items())
    )
    wrong = sorted(int(i) for i in frame.loc[frame["n_speakers"] != expected, "session_id"])
    if wrong:
        lines.append(f"  {len(wrong)} session(s) do not have exactly {expected} speakers: {wrong}")
        lines.append("  flagged, not corrected: a split or merged voice changes every later stage")

    coverage = frame["coverage_fraction"].dropna()
    if not coverage.empty:
        lines.append(
            f"segment coverage of the recording: min {coverage.min():.2f}, "
            f"median {coverage.median():.2f}, max {coverage.max():.2f}"
        )
        lines.append(
            "  high coverage is expected: whisper-diarization segments can span "
            "silences, which is why `vc vad` refines them."
        )

    overlap = frame["overlap_seconds"].dropna()
    if not overlap.empty:
        lines.append(
            f"overlapping speech: median {overlap.median():.1f}s, max {overlap.max():.1f}s"
        )

    with_text = int(frame["has_text"].fillna(False).sum())
    lines.append(
        f"transcript text stored for {with_text} of {len(frame)} session(s) "
        f"(work tree only, never in a handoff bundle)"
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
            f"  {name}: {len(ids)} session(s)"
            + (f" {sorted(ids)}" if len(ids) <= _MAX_LISTED_SESSIONS else "")
            for name, ids in sorted(by_flag.items())
        )

    return lines
