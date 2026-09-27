"""Stage: prosody of the participant's speech.

Participant speech only, with overlapping speech excluded. Both restrictions
are the point of the stage rather than details of it: a pitch measurement taken
while two people are talking describes neither of them, and a measurement taken
on the wrong speaker is worse than a missing one.

Measures come from Praat via parselmouth, so they are the same quantities the
literature reports. F0 is expressed in semitones relative to the participant's
own median, so that absolute pitch differences between speakers - roughly an
octave along sex - cannot dominate (docs/decisions/0005).

Requires a role mapping, for the same reason `vc turns` does, and refuses to
run without one rather than guessing which voice to measure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.contracts import feature_schema, validate
from vc_multimodal.features.prosody_math import (
    FEATURE_NAMES,
    SpanMeasures,
    median_or_none,
    pool,
    prosody_features,
)
from vc_multimodal.features.spans import Span, clip, overlap_duration, subtract
from vc_multimodal.features.turn_math import ROLE_PARTICIPANT
from vc_multimodal.io_utils import read_parquet, write_csv, write_parquet
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.prosody import ProsodyBackend, ProsodyError, get_backend
from vc_multimodal.roles import RoleMapping, load_role_mapping, spans_by_role
from vc_multimodal.runner import StageReport, run_sessions
from vc_multimodal.session_tables import carry_forward, combine, has_row
from vc_multimodal.stages.extract_audio import audio_path
from vc_multimodal.stages.vad import read_mono_wav, speech_path

logger = get_logger(__name__)

STAGE: Final = "prosody"
PROSODY_DIRNAME: Final = "prosody"
PROSODY_FEATURES_FILENAME: Final = "prosody_features.csv"

FLAG_NO_PARTICIPANT: Final = "prosody_no_participant_speech"
FLAG_TOO_LITTLE_SPEECH: Final = "prosody_too_little_speech"
FLAG_NO_VOICED_FRAMES: Final = "prosody_no_voiced_frames"
FLAG_SPANS_FAILED: Final = "prosody_some_spans_unmeasurable"
FLAG_MANUAL_ROLES: Final = "prosody_manual_role_mapping"

# Below this much analysed speech, the session statistics rest on too little.
_MIN_ANALYSED_S: Final = 20.0

QC_COLUMNS: Final = (
    "qc__role_source",
    "qc__n_spans_analysed",
    "qc__n_spans_failed",
    "qc__analysed_seconds",
    "qc__overlap_excluded_seconds",
    "qc__f0_median_hz",
    "qc__voiced_frames",
    "qc__backend",
    "qc__flags",
)


def prosody_path(roots: DataRoots, session_id: int) -> Path:
    """Where one session's per-span measures are written."""
    return roots.work_path(PROSODY_DIRNAME, f"{session_id}.parquet")


def prosody_dir(roots: DataRoots) -> Path:
    """Directory holding per-span prosodic measures, under the work root."""
    return roots.work_path(PROSODY_DIRNAME, create_parent=True)


def analysis_spans(
    by_role: Mapping[str, Sequence[Span]],
    *,
    exclude_overlap: bool,
    min_analysis_s: float,
    extent: Span | None = None,
) -> tuple[tuple[Span, ...], float]:
    """Work out which stretches of participant speech to measure.

    Args:
        by_role: Role to that role's speech spans.
        exclude_overlap: Remove any stretch where another role is also
            speaking. A measurement taken across two voices describes neither.
        min_analysis_s: Discard stretches too short to measure. A fragment of a
            few hundred milliseconds yields a pitch value, and it is noise.
        extent: Clip everything to this span, so nothing can reach past the
            audio that actually exists.

    Returns:
        The stretches to analyse, and how many seconds were dropped as overlap.
    """
    participant = list(by_role.get(ROLE_PARTICIPANT, ()))
    if not participant:
        return (), 0.0

    others = [span for role, spans in by_role.items() if role != ROLE_PARTICIPANT for span in spans]
    excluded = overlap_duration(participant, others) if others else 0.0

    usable = subtract(participant, others) if exclude_overlap and others else tuple(participant)
    if extent is not None:
        usable = clip(usable, extent)
    kept = tuple(span for span in usable if span.duration >= min_analysis_s)
    return kept, excluded


def measure_spans(
    samples: np.ndarray,
    sample_rate: int,
    spans: Sequence[Span],
    *,
    backend: ProsodyBackend,
    config: AppConfig,
) -> tuple[list[SpanMeasures], int]:
    """Measure every analysis span, tolerating individual failures.

    A span Praat cannot analyse is counted rather than fatal: one unusable
    fragment should not cost a session its other twenty utterances.

    Returns:
        The measures obtained, and how many spans could not be measured.
    """
    measures: list[SpanMeasures] = []
    failed = 0
    for span in spans:
        first = max(0, round(span.start * sample_rate))
        last = min(samples.size, round(span.end * sample_rate))
        if last <= first:
            failed += 1
            continue
        try:
            measures.append(
                backend.measure(samples[first:last], sample_rate, config=config.prosody)
            )
        except ProsodyError:
            failed += 1
    return measures, failed


def span_frame(
    session_id: int, spans: Sequence[Span], measures: Sequence[SpanMeasures]
) -> pd.DataFrame:
    """Build the per-span measure table kept for QC.

    Frame-level contours are not stored: they are large, and everything the
    later stages need is already pooled into the features.
    """
    rows = [
        {
            "session_id": session_id,
            "span_index": index,
            "start_s": span.start,
            "end_s": span.end,
            "duration_s": round(measure.duration_s, 3),
            "n_pitch_frames": measure.n_pitch_frames,
            "n_voiced_frames": measure.n_voiced,
            "jitter_local": measure.jitter_local,
            "shimmer_local": measure.shimmer_local,
        }
        for index, (span, measure) in enumerate(zip(spans, measures, strict=True))
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "session_id",
            "span_index",
            "start_s",
            "end_s",
            "duration_s",
            "n_pitch_frames",
            "n_voiced_frames",
            "jitter_local",
            "shimmer_local",
        ],
    )
    for column in ("session_id", "span_index", "n_pitch_frames", "n_voiced_frames"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("int64")
    for column in ("start_s", "end_s", "duration_s", "jitter_local", "shimmer_local"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def feature_row(
    session: RawSession,
    measures: Sequence[SpanMeasures],
    mapping: RoleMapping,
    *,
    n_failed: int,
    overlap_excluded_s: float,
    had_participant_speech: bool,
    backend_name: str,
) -> dict[str, object]:
    """Compute one session's prosodic features and QC columns."""
    features = prosody_features(measures)
    analysed_s = sum(measure.duration_s for measure in measures)
    voiced_frames = sum(measure.n_voiced for measure in measures)
    # The semitone reference: the median of every voiced frame. Taken through
    # the same helper the features use, which returns None for an unvoiced
    # session rather than a warning and a NaN.
    f0_median = median_or_none(pool(measures)["f0_hz"])

    flags: list[str] = []
    if not had_participant_speech:
        flags.append(FLAG_NO_PARTICIPANT)
    if analysed_s < _MIN_ANALYSED_S:
        flags.append(FLAG_TOO_LITTLE_SPEECH)
    if voiced_frames == 0:
        flags.append(FLAG_NO_VOICED_FRAMES)
    if n_failed:
        flags.append(FLAG_SPANS_FAILED)
    if mapping.source != "assigned":
        flags.append(FLAG_MANUAL_ROLES)

    row: dict[str, object] = {"session_id": session.session_id, "wave": session.wave}
    row.update(features)
    row.update(
        {
            "qc__role_source": mapping.source,
            "qc__n_spans_analysed": len(measures),
            "qc__n_spans_failed": n_failed,
            "qc__analysed_seconds": round(analysed_s, 3),
            "qc__overlap_excluded_seconds": round(overlap_excluded_s, 3),
            # The semitone reference, recorded so a questionable pitch feature
            # can be traced back to it. Absolute pitch is QC, not a feature:
            # as a feature it would reintroduce the sex difference that
            # semitone normalisation exists to remove.
            "qc__f0_median_hz": None
            if f0_median is None or f0_median <= 0.0
            else round(f0_median, 3),
            "qc__voiced_frames": voiced_frames,
            "qc__backend": backend_name,
            "qc__flags": ";".join(dict.fromkeys(flags)),
        }
    )
    return row


def build_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Assemble the prosodic feature table with declared dtypes."""
    columns = ["session_id", "wave", *FEATURE_NAMES, *QC_COLUMNS]
    frame = pd.DataFrame(list(rows), columns=columns)
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce").astype("int64")
    for column in ("wave", "qc__role_source", "qc__backend", "qc__flags"):
        frame[column] = frame[column].fillna("").astype("string")
    for column in FEATURE_NAMES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in ("qc__n_spans_analysed", "qc__n_spans_failed", "qc__voiced_frames"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ("qc__analysed_seconds", "qc__overlap_excluded_seconds", "qc__f0_median_hz"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame.sort_values("session_id", ignore_index=True)


@dataclass(frozen=True, slots=True)
class ProsodyResult:
    """What the prosody stage produced."""

    report: StageReport
    frame: pd.DataFrame
    path: Path


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    session_ids: Sequence[int] | None = None,
    workers: int | None = None,
    force: bool = False,
    backend: ProsodyBackend | None = None,
) -> ProsodyResult:
    """Measure participant prosody for every requested session.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to measure, or None for all.
        workers: Parallel workers, or None to choose automatically.
        force: Recompute sessions whose measures already exist.
        backend: Measurement backend; built from config if omitted.

    Returns:
        The stage report and the written feature table.

    Raises:
        ProsodyError: if the chosen backend cannot run at all.
    """
    engine = backend or get_backend(config)
    if not engine.available():
        raise ProsodyError(engine.unavailable_reason())
    logger.info("%s: backend %s (%s)", STAGE, engine.name, engine.version())

    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []
    prosody_dir(roots)
    rows: dict[int, Mapping[str, object]] = {}

    qc_target = roots.out_path(PROSODY_FEATURES_FILENAME)

    def is_done(session: RawSession) -> bool:
        """Done means every output exists, including this session's QC row.

        A session whose artifacts are on disk but whose row is not is not done:
        skipping it would leave the table permanently short of a row, because
        nothing else ever writes one. This is how a row lost to an earlier
        partial run heals itself.
        """
        return prosody_path(roots, session.session_id).exists() and has_row(
            qc_target, session.session_id
        )

    def measure_one(session: RawSession) -> str:
        speech_file = speech_path(roots, session.session_id)
        if not speech_file.exists():
            msg = f"no speech spans for session {session.session_id}; run `vc vad` first"
            raise FileNotFoundError(msg)
        audio_file = audio_path(roots, session.session_id)
        if not audio_file.exists():
            msg = (
                f"no extracted audio for session {session.session_id}; run `vc extract-audio` first"
            )
            raise FileNotFoundError(msg)

        mapping = load_role_mapping(roots.work, session.session_id)
        speech = read_parquet(speech_file)
        by_role = {
            role: tuple(Span(start, end) for start, end in pairs)
            for role, pairs in spans_by_role(speech, mapping).items()
        }

        samples, sample_rate = read_mono_wav(audio_file)
        extent = Span(0.0, samples.size / sample_rate if sample_rate else 0.0)
        spans, overlap_excluded = analysis_spans(
            by_role,
            exclude_overlap=config.prosody.exclude_overlap,
            min_analysis_s=config.prosody.min_analysis_s,
            extent=extent,
        )

        measures, failed = measure_spans(samples, sample_rate, spans, backend=engine, config=config)
        write_parquet(
            prosody_path(roots, session.session_id),
            span_frame(session.session_id, spans[: len(measures)], measures),
        )

        rows[session.session_id] = feature_row(
            session,
            measures,
            mapping,
            n_failed=failed,
            overlap_excluded_s=overlap_excluded,
            had_participant_speech=bool(by_role.get(ROLE_PARTICIPANT)),
            backend_name=engine.name,
        )
        analysed = sum(measure.duration_s for measure in measures)
        return (
            f"{len(measures)} span(s), {analysed / 60:.1f} min analysed, "
            f"{overlap_excluded:.1f}s overlap excluded"
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

    # A skipped session is recomputed from its stored per-span table, which
    # holds the durations and voiced counts but not the contours, so the
    # feature row cannot be rebuilt: it is re-measured only if asked.
    for outcome in report.skipped:
        logger.info(
            "session %s: measures already exist; rerun with --force to refresh its features",
            outcome.session_id,
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
    validate(frame, feature_schema([*FEATURE_NAMES, *QC_COLUMNS]), context=STAGE)

    write_csv(qc_target, frame)
    logger.info(
        "wrote %s with %d row(s) (%d from this run, %d kept)",
        qc_target,
        len(frame),
        len(rows),
        len(carried.rows),
    )

    return ProsodyResult(
        report=report.with_notes(carried.notes(STAGE)), frame=frame, path=qc_target
    )


def summarise(frame: pd.DataFrame) -> list[str]:
    """Summarise the prosodic features. Aggregate statistics only."""
    if frame.empty:
        return ["no sessions were measured"]

    lines = [f"prosodic features for {len(frame)} session(s)"]

    analysed = frame["qc__analysed_seconds"].dropna()
    if not analysed.empty:
        lines.append(
            f"participant speech analysed: min {analysed.min() / 60:.1f} min, "
            f"median {analysed.median() / 60:.1f} min, max {analysed.max() / 60:.1f} min"
        )
    excluded = frame["qc__overlap_excluded_seconds"].dropna()
    if not excluded.empty:
        lines.append(
            f"overlapping speech excluded: median {excluded.median():.1f}s, "
            f"max {excluded.max():.1f}s"
        )

    for column, label in (
        ("prosody__f0_semitone_sd", "F0 variability (semitones)"),
        ("prosody__f0_semitone_range", "F0 range, 5th-95th pct (semitones)"),
        ("prosody__intensity_mean_db", "mean intensity (dB)"),
        ("prosody__jitter_local", "jitter (local)"),
        ("prosody__shimmer_local", "shimmer (local)"),
        ("prosody__hnr_db", "harmonics-to-noise (dB)"),
        ("prosody__speech_rate_proxy", "speech rate proxy (peaks/s)"),
    ):
        values = frame[column].dropna()
        if values.empty:
            lines.append(f"{label}: not measurable")
        else:
            lines.append(
                f"{label}: min {values.min():.3f}, median {values.median():.3f}, "
                f"max {values.max():.3f} (n={len(values)})"
            )

    reference = frame["qc__f0_median_hz"].dropna()
    if not reference.empty:
        lines.append(
            f"semitone reference (each speaker's own median F0): "
            f"{reference.min():.0f}-{reference.max():.0f} Hz across sessions"
        )
        lines.append(
            "  pitch features are relative to each speaker's own median, so this "
            "spread does not enter them."
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
