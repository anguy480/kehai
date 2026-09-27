"""Stage: turns, response latency, pauses, and the speaking/listening timeline.

Built from the VAD-refined speech spans rather than from diarized segment
boundaries, because those follow transcription units and span silence
(docs/decisions/0004). Requires a role mapping: which diarized speaker is the
participant decides what every number here means, so the stage refuses to run
without one rather than assuming.

Writes three things: the turn structure and the speaking/listening timeline
into `$VC_WORK_ROOT` for the facial stages to consume, and the session-level
turn features into `$VC_OUT_ROOT`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.contracts import TURN_SCHEMA, feature_schema, validate
from vc_multimodal.features.spans import Span, covered_duration
from vc_multimodal.features.turn_math import (
    FEATURE_NAMES,
    ROLE_PARTICIPANT,
    ROLE_PSYCHIATRIST,
    Turn,
    build_turns,
    count_interruptions,
    response_latencies,
    speaking_timeline,
    turn_features,
    usable_latencies,
)
from vc_multimodal.io_utils import read_csv, read_parquet, write_csv, write_parquet
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.roles import RoleMapping, RolesUnavailableError, load_role_mapping, spans_by_role
from vc_multimodal.runner import StageReport, run_sessions
from vc_multimodal.session_tables import carry_forward, combine, has_row
from vc_multimodal.stages.vad import speech_path

logger = get_logger(__name__)

STAGE: Final = "turns"
TURNS_DIRNAME: Final = "turns"
TIMELINE_DIRNAME: Final = "timeline"
TURN_FEATURES_FILENAME: Final = "turn_features.csv"

FLAG_NO_PARTICIPANT: Final = "turns_no_participant_speech"
FLAG_NO_PSYCHIATRIST: Final = "turns_no_psychiatrist_speech"
FLAG_NO_LATENCIES: Final = "turns_no_usable_latencies"
FLAG_UNKNOWN_SPEAKER: Final = "turns_unassigned_speaker"
FLAG_MANUAL_ROLES: Final = "turns_manual_role_mapping"

# Below this many usable response latencies, the latency statistics rest on too
# little to mean much.
_MIN_LATENCIES: Final = 3

QC_COLUMNS: Final = (
    "qc__role_source",
    "qc__n_turns",
    "qc__n_latencies",
    "qc__n_interruptions",
    "qc__participant_speech_s",
    "qc__listening_s",
    "qc__flags",
)


def turns_path(roots: DataRoots, session_id: int) -> Path:
    """Where one session's turn structure is written."""
    return roots.work_path(TURNS_DIRNAME, f"{session_id}.parquet")


def timeline_path(roots: DataRoots, session_id: int) -> Path:
    """Where one session's speaking/listening timeline is written."""
    return roots.work_path(TIMELINE_DIRNAME, f"{session_id}.parquet")


def turns_frame(
    session_id: int, turns: Sequence[Turn], latencies: Mapping[int, float]
) -> pd.DataFrame:
    """Build the turn table, carrying each turn's response latency where it has one."""
    frame = pd.DataFrame(
        {
            "session_id": [session_id] * len(turns),
            "turn_index": [turn.index for turn in turns],
            "role": [turn.role for turn in turns],
            "start_s": [turn.start for turn in turns],
            "end_s": [turn.end for turn in turns],
            "n_spans": [turn.n_spans for turn in turns],
            "latency_s": [latencies.get(turn.index) for turn in turns],
        }
    )
    frame["session_id"] = frame["session_id"].astype("int64")
    frame["turn_index"] = frame["turn_index"].astype("int64")
    frame["role"] = frame["role"].astype("string")
    frame["n_spans"] = frame["n_spans"].astype("int64")
    for column in ("start_s", "end_s", "latency_s"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def timeline_frame(
    session_id: int, speaking: Sequence[Span], listening: Sequence[Span]
) -> pd.DataFrame:
    """Build the speaking/listening timeline table for the facial stages."""
    rows = [(session_id, "speaking", span.start, span.end) for span in speaking]
    rows += [(session_id, "listening", span.start, span.end) for span in listening]
    frame = pd.DataFrame(rows, columns=["session_id", "state", "start_s", "end_s"])
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce").astype("int64")
    frame["state"] = frame["state"].astype("string")
    for column in ("start_s", "end_s"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame.sort_values(["start_s", "state"], ignore_index=True)


def _as_spans(pairs: Sequence[tuple[float, float]]) -> tuple[Span, ...]:
    """Convert (start, end) pairs to spans."""
    return tuple(Span(start, end) for start, end in pairs)


def feature_row(
    session: RawSession,
    by_role: Mapping[str, Sequence[Span]],
    mapping: RoleMapping,
    *,
    duration_s: float,
    config: AppConfig,
) -> dict[str, object]:
    """Compute one session's turn features and QC columns."""
    turns_config = config.turns
    turns = build_turns(by_role, merge_gap=turns_config.merge_same_speaker_gap_s)
    latencies = response_latencies(turns)
    usable = usable_latencies(latencies, max_latency_s=turns_config.max_latency_s)
    timeline = speaking_timeline(by_role)

    flags: list[str] = []
    if not by_role.get(ROLE_PARTICIPANT):
        flags.append(FLAG_NO_PARTICIPANT)
    if not by_role.get(ROLE_PSYCHIATRIST):
        flags.append(FLAG_NO_PSYCHIATRIST)
    if len(usable) < _MIN_LATENCIES:
        flags.append(FLAG_NO_LATENCIES)
    if by_role.get("unknown"):
        flags.append(FLAG_UNKNOWN_SPEAKER)
    if mapping.source != "assigned":
        flags.append(FLAG_MANUAL_ROLES)

    features = turn_features(
        by_role,
        duration_s=duration_s,
        merge_gap_s=turns_config.merge_same_speaker_gap_s,
        min_pause_s=turns_config.min_pause_s,
        max_latency_s=turns_config.max_latency_s,
    )

    row: dict[str, object] = {"session_id": session.session_id, "wave": session.wave}
    row.update(features)
    row.update(
        {
            "qc__role_source": mapping.source,
            "qc__n_turns": len(turns),
            "qc__n_latencies": len(usable),
            "qc__n_interruptions": count_interruptions(latencies),
            "qc__participant_speech_s": round(
                covered_duration(by_role.get(ROLE_PARTICIPANT, ())), 3
            ),
            "qc__listening_s": round(timeline.listening_seconds, 3),
            "qc__flags": ";".join(dict.fromkeys(flags)),
        }
    )
    return row


def build_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Assemble the turn feature table with declared dtypes."""
    columns = ["session_id", "wave", *FEATURE_NAMES, *QC_COLUMNS]
    frame = pd.DataFrame(list(rows), columns=columns)
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce").astype("int64")
    for column in ("wave", "qc__role_source", "qc__flags"):
        frame[column] = frame[column].fillna("").astype("string")
    for column in FEATURE_NAMES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in ("qc__n_turns", "qc__n_latencies", "qc__n_interruptions"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ("qc__participant_speech_s", "qc__listening_s"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame.sort_values("session_id", ignore_index=True)


@dataclass(frozen=True, slots=True)
class TurnsResult:
    """What the turns stage produced."""

    report: StageReport
    frame: pd.DataFrame
    path: Path


def _durations(roots: DataRoots) -> Mapping[int, float]:
    """Session durations, preferring the decoded audio over the container.

    `audio_qc.csv` records what was actually decoded, which is the right
    denominator for a rate: one recording's file is truncated, and its stated
    duration would understate every per-minute figure.
    """
    for filename, column in (("audio_qc.csv", "duration_s"), ("inventory.csv", "duration_s")):
        path = roots.out_path(filename, create_parent=False)
        if not path.exists():
            continue
        try:
            frame = read_csv(path)
        except (OSError, ValueError):  # pragma: no cover - defensive
            continue
        if "session_id" in frame and column in frame:
            return {
                int(session_id): float(value)
                for session_id, value in zip(frame["session_id"], frame[column], strict=True)
                if pd.notna(value)
            }
    return {}


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    session_ids: Sequence[int] | None = None,
    workers: int | None = None,
    force: bool = False,
) -> TurnsResult:
    """Derive turns and turn features for every requested session.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to process, or None for all.
        workers: Parallel workers, or None to choose automatically.
        force: Recompute sessions whose turn tables already exist.

    Returns:
        The stage report and the written feature table.
    """
    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []
    durations = dict(_durations(roots))
    rows: dict[int, Mapping[str, object]] = {}

    qc_target = roots.out_path(TURN_FEATURES_FILENAME)

    def is_done(session: RawSession) -> bool:
        """Done means every output exists, including this session's QC row.

        A session whose artifacts are on disk but whose row is not is not done:
        skipping it would leave the table permanently short of a row, because
        nothing else ever writes one. This is how a row lost to an earlier
        partial run heals itself.
        """
        return (
            turns_path(roots, session.session_id).exists()
            and timeline_path(roots, session.session_id).exists()
            and has_row(qc_target, session.session_id)
        )

    def process_one(session: RawSession) -> str:
        speech_file = speech_path(roots, session.session_id)
        if not speech_file.exists():
            msg = f"no speech spans for session {session.session_id}; run `vc vad` first"
            raise FileNotFoundError(msg)

        mapping = load_role_mapping(roots.work, session.session_id)
        speech = read_parquet(speech_file)
        by_role = {role: _as_spans(pairs) for role, pairs in spans_by_role(speech, mapping).items()}

        duration = durations.get(session.session_id)
        if duration is None or duration <= 0:
            # Fall back to the end of the last speech span: a rate needs a
            # denominator, and this is the only one available here.
            duration = max((span.end for spans in by_role.values() for span in spans), default=0.0)

        turns = build_turns(by_role, merge_gap=config.turns.merge_same_speaker_gap_s)
        latencies = {latency.turn_index: latency.seconds for latency in response_latencies(turns)}
        turn_table = turns_frame(session.session_id, turns, latencies)
        validate(turn_table, TURN_SCHEMA, context=f"session {session.session_id}")
        write_parquet(turns_path(roots, session.session_id), turn_table)

        timeline = speaking_timeline(by_role)
        write_parquet(
            timeline_path(roots, session.session_id),
            timeline_frame(session.session_id, timeline.speaking, timeline.listening),
        )

        rows[session.session_id] = feature_row(
            session, by_role, mapping, duration_s=duration, config=config
        )
        return (
            f"{len(turns)} turn(s), {len(latencies)} transition(s), "
            f"{timeline.speaking_seconds / 60:.1f} min speaking"
        )

    report = run_sessions(
        STAGE,
        selected,
        process_one,
        workers=workers,
        force=force,
        is_done=is_done,
        backend="threads",
        notes=notes,
    )

    # A skipped session is recomputed from its stored speech spans, which is
    # cheap, rather than left out of the feature table.
    for outcome in report.skipped:
        session = next(s for s in selected if s.session_id == outcome.session_id)
        try:
            mapping = load_role_mapping(roots.work, outcome.session_id)
        except RolesUnavailableError:  # pragma: no cover - would have failed above
            continue
        speech = read_parquet(speech_path(roots, outcome.session_id))
        by_role = {role: _as_spans(pairs) for role, pairs in spans_by_role(speech, mapping).items()}
        duration = durations.get(outcome.session_id) or max(
            (span.end for spans in by_role.values() for span in spans), default=0.0
        )
        rows[outcome.session_id] = feature_row(
            session, by_role, mapping, duration_s=duration, config=config
        )

    # Rows for sessions this run did not compute are kept, whether they were
    # left out by --sessions or skipped as already done. Writing only this
    # run's rows would delete every other session's.
    carried = carry_forward(
        qc_target,
        computed=set(rows),
        columns=["session_id", "wave", *FEATURE_NAMES, *QC_COLUMNS],
        stage=STAGE,
        force=force,
    )
    frame = build_frame(combine(rows, carried))
    validate(
        frame,
        feature_schema([*FEATURE_NAMES, *QC_COLUMNS]),
        context=STAGE,
    )

    write_csv(qc_target, frame)
    logger.info(
        "wrote %s with %d row(s) (%d from this run, %d kept)",
        qc_target,
        len(frame),
        len(rows),
        len(carried.rows),
    )

    return TurnsResult(report=report.with_notes(carried.notes(STAGE)), frame=frame, path=qc_target)


def summarise(frame: pd.DataFrame) -> list[str]:
    """Summarise the turn features. Aggregate statistics only."""
    if frame.empty:
        return ["no sessions were processed"]

    lines = [f"turn features for {len(frame)} session(s)"]

    for column, label in (
        ("turns__n_per_minute", "turns per minute"),
        ("turns__participant_speaking_ratio", "participant share of speech"),
        ("turns__latency_median", "median response latency (s)"),
        ("turns__overlap_ratio", "overlapping speech share"),
    ):
        values = frame[column].dropna()
        if values.empty:
            lines.append(f"{label}: not measurable")
        else:
            lines.append(
                f"{label}: min {values.min():.3f}, median {values.median():.3f}, "
                f"max {values.max():.3f} (n={len(values)})"
            )

    interruptions = frame["qc__n_interruptions"].dropna()
    if not interruptions.empty:
        lines.append(
            f"interruptions per session: median {interruptions.median():.0f}, "
            f"max {interruptions.max()}"
        )

    sources = frame["qc__role_source"].value_counts()
    lines.append(
        "role mapping source: "
        + ", ".join(f"{source} -> {count} session(s)" for source, count in sources.items())
    )

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
            f"  {name}: {len(ids)} session(s) {sorted(ids)}"
            for name, ids in sorted(by_flag.items())
        )
    return lines
