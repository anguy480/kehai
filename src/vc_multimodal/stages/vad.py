"""Stage: recover true speech boundaries inside the diarized segments.

Diarization says who speaks and roughly when. Its boundaries follow
transcription units, and whisper-diarization segments are known to span long
silences, so taking them as speech boundaries would systematically understate
pauses and distort every latency measure. Silero voice activity detection
therefore decides where speech actually starts and stops, and all timing
features are computed from that (docs/decisions/0004).

Two modes, because the sensible implementation and the literal reading of that
decision differ:

* `intersect` (default) runs the detector once over the whole recording and
  intersects the detected speech with each diarized segment. Silero is a
  recurrent model whose accuracy depends on surrounding context, so one
  continuous pass is both more reliable at boundaries and much faster than
  hundreds of short ones.
* `per_segment` runs the detector separately on each segment's audio, which is
  the literal reading. Available for comparison.

Either way the output is speech spans attributed to diarization's anonymous
speaker labels: roles are assigned later.
"""

from __future__ import annotations

import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.contracts import SPEECH_SCHEMA, validate
from vc_multimodal.features.spans import Span, clip, covered_duration, intersect, merge
from vc_multimodal.io_utils import read_parquet, write_csv, write_parquet
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.runner import StageReport, run_sessions
from vc_multimodal.session_tables import carry_forward, combine, has_row
from vc_multimodal.stages.diarize import segments_path
from vc_multimodal.stages.extract_audio import audio_path

logger = get_logger(__name__)

STAGE: Final = "vad"
SPEECH_DIRNAME: Final = "speech"
VAD_QC_FILENAME: Final = "vad_qc.csv"

FLAG_NO_SPEECH: Final = "vad_no_speech_detected"
FLAG_MOSTLY_SILENT: Final = "vad_segments_mostly_silent"
FLAG_MISSING_AUDIO: Final = "vad_audio_missing"

# Below this fraction of diarized segment time surviving as detected speech,
# the segments were mostly silence, which is exactly what this stage exists to
# catch but is also worth flagging when it is extreme.
_MOSTLY_SILENT_BELOW: Final = 0.35

_INT16_FULL_SCALE: Final = 32768.0
_MONO: Final = 1
_SAMPLE_WIDTH_BYTES: Final = 2

COLUMN_ORDER: Final = (
    "session_id",
    "wave",
    "mode",
    "n_segments",
    "n_speech_spans",
    "segment_seconds",
    "speech_seconds",
    "retained_fraction",
    "n_speakers",
    "flags",
)


class SpeechDetector(Protocol):
    """Finds speech in a waveform.

    A protocol so the stage can be tested against a deterministic stub instead
    of loading a neural model.
    """

    def detect(self, samples: np.ndarray, sample_rate: int) -> tuple[Span, ...]:
        """Return the spans of `samples` that contain speech."""


class SileroDetector:
    """Silero VAD, loaded once and reused.

    Args:
        config: Thresholds and durations from `vad` configuration.
    """

    def __init__(self, config: Any) -> None:
        """Store configuration; the model is loaded on first use."""
        self.config = config
        self._model: Any | None = None

    def version(self) -> str:
        """Identifier recorded in the run manifest."""
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            return f"silero-vad/{version('silero-vad')}"
        except PackageNotFoundError:  # pragma: no cover - defensive
            return "silero-vad/unknown"

    def model(self) -> Any:
        """Load the model once."""
        if self._model is None:
            from silero_vad import load_silero_vad  # noqa: PLC0415 - heavy import

            self._model = load_silero_vad()
        return self._model

    def detect(self, samples: np.ndarray, sample_rate: int) -> tuple[Span, ...]:
        """Detect speech in a mono waveform."""
        import torch  # noqa: PLC0415 - heavy import
        from silero_vad import get_speech_timestamps  # noqa: PLC0415 - heavy import

        if samples.size == 0:
            return ()

        timestamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)),
            self.model(),
            sampling_rate=sample_rate,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_ms,
            min_silence_duration_ms=self.config.min_silence_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            return_seconds=True,
        )
        return tuple(Span(float(t["start"]), float(t["end"])) for t in timestamps)


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit WAV into floats in [-1, 1].

    Raises:
        OSError: if the file cannot be read.
        ValueError: if it is not the mono 16-bit audio this pipeline writes.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if channels != _MONO or width != _SAMPLE_WIDTH_BYTES:
        msg = f"{path.name} is {channels}-channel {width * 8}-bit; expected mono 16-bit"
        raise ValueError(msg)

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / _INT16_FULL_SCALE
    return samples, sample_rate


@dataclass(frozen=True, slots=True)
class SpeakerSpeech:
    """Detected speech for one speaker in one session."""

    speaker: str
    spans: tuple[Span, ...]


def segments_by_speaker(frame: pd.DataFrame) -> dict[str, tuple[Span, ...]]:
    """Group diarized segments into spans per speaker label."""
    grouped: dict[str, list[Span]] = {}
    for speaker, start, end in zip(frame["speaker"], frame["start_s"], frame["end_s"], strict=True):
        grouped.setdefault(str(speaker), []).append(Span(float(start), float(end)))
    return {speaker: merge(spans) for speaker, spans in grouped.items()}


def refine_by_intersection(
    speech: Sequence[Span], by_speaker: Mapping[str, Sequence[Span]]
) -> tuple[SpeakerSpeech, ...]:
    """Attribute globally detected speech to each speaker's segments."""
    return tuple(
        SpeakerSpeech(speaker=speaker, spans=intersect(segments, speech))
        for speaker, segments in by_speaker.items()
    )


def refine_per_segment(
    samples: np.ndarray,
    sample_rate: int,
    by_speaker: Mapping[str, Sequence[Span]],
    detector: SpeechDetector,
) -> tuple[SpeakerSpeech, ...]:
    """Run the detector separately on each segment's audio.

    The literal reading of "VAD within each diarized segment". Each slice is
    detected independently and its results shifted back into recording time.
    """
    results: list[SpeakerSpeech] = []
    for speaker, segments in by_speaker.items():
        found: list[Span] = []
        for segment in merge(segments):
            first = round(segment.start * sample_rate)
            last = round(segment.end * sample_rate)
            slice_ = samples[max(0, first) : min(len(samples), last)]
            if slice_.size == 0:
                continue
            offset = max(0, first) / sample_rate
            found.extend(span.shifted(offset) for span in detector.detect(slice_, sample_rate))
        results.append(SpeakerSpeech(speaker=speaker, spans=merge(found)))
    return tuple(results)


def speech_frame(session_id: int, speech: Sequence[SpeakerSpeech]) -> pd.DataFrame:
    """Build the speech-span table for one session.

    Carries speaker labels and no transcript text: this table describes timing
    only, so nothing sensitive travels with it.
    """
    rows = [
        (session_id, item.speaker, span.start, span.end) for item in speech for span in item.spans
    ]
    frame = pd.DataFrame(rows, columns=["session_id", "speaker", "start_s", "end_s"])
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce").astype("int64")
    frame["speaker"] = frame["speaker"].astype("string")
    for column in ("start_s", "end_s"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame.sort_values(["start_s", "speaker"], ignore_index=True)


def speech_path(roots: DataRoots, session_id: int) -> Path:
    """Where one session's refined speech spans are written."""
    return roots.work_path(SPEECH_DIRNAME, f"{session_id}.parquet")


def speech_dir(roots: DataRoots) -> Path:
    """Directory holding refined speech spans, under the work root."""
    return roots.work_path(SPEECH_DIRNAME, create_parent=True)


def qc_record(
    session: RawSession,
    *,
    mode: str,
    segments: Mapping[str, Sequence[Span]],
    speech: Sequence[SpeakerSpeech],
) -> dict[str, object]:
    """Summarise how much diarized time survived as detected speech."""
    segment_seconds = covered_duration([s for spans in segments.values() for s in spans])
    speech_seconds = covered_duration([s for item in speech for s in item.spans])
    retained = speech_seconds / segment_seconds if segment_seconds > 0 else None

    flags: list[str] = []
    if speech_seconds <= 0:
        flags.append(FLAG_NO_SPEECH)
    elif retained is not None and retained < _MOSTLY_SILENT_BELOW:
        flags.append(FLAG_MOSTLY_SILENT)

    return {
        "session_id": session.session_id,
        "wave": session.wave,
        "mode": mode,
        "n_segments": sum(len(spans) for spans in segments.values()),
        "n_speech_spans": sum(len(item.spans) for item in speech),
        "segment_seconds": round(segment_seconds, 3),
        "speech_seconds": round(speech_seconds, 3),
        "retained_fraction": None if retained is None else round(retained, 4),
        "n_speakers": len([item for item in speech if item.spans]),
        "flags": ";".join(dict.fromkeys(flags)),
    }


def build_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Assemble VAD QC rows into a correctly typed table."""
    frame = pd.DataFrame(list(rows), columns=list(COLUMN_ORDER))
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce").astype("int64")
    for column in ("wave", "mode", "flags"):
        frame[column] = frame[column].fillna("").astype("string")
    for column in ("n_segments", "n_speech_spans", "n_speakers"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ("segment_seconds", "speech_seconds", "retained_fraction"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame.sort_values("session_id", ignore_index=True)


@dataclass(frozen=True, slots=True)
class VadResult:
    """What the VAD stage produced."""

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
    detector: SpeechDetector | None = None,
) -> VadResult:
    """Refine every requested session's diarized segments into speech spans.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to refine, or None for all.
        workers: Parallel workers, or None to choose automatically.
        force: Recompute sessions whose speech spans already exist.
        detector: Speech detector; Silero is loaded if omitted.

    Returns:
        The stage report and the written QC table.
    """
    engine = detector if detector is not None else SileroDetector(config.vad)
    mode = config.vad.mode
    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []
    speech_dir(roots)
    records: dict[int, Mapping[str, object]] = {}

    qc_target = roots.out_path(VAD_QC_FILENAME)

    def is_done(session: RawSession) -> bool:
        """Done means every output exists, including this session's QC row.

        A session whose artifacts are on disk but whose row is not is not done:
        skipping it would leave the table permanently short of a row, because
        nothing else ever writes one. This is how a row lost to an earlier
        partial run heals itself.
        """
        return speech_path(roots, session.session_id).exists() and has_row(
            qc_target, session.session_id
        )

    def refine_one(session: RawSession) -> str:
        segments_file = segments_path(roots, session.session_id)
        if not segments_file.exists():
            msg = f"no diarized segments for session {session.session_id}; run `vc diarize` first"
            raise FileNotFoundError(msg)
        audio_file = audio_path(roots, session.session_id)
        if not audio_file.exists():
            msg = (
                f"no extracted audio for session {session.session_id}; run `vc extract-audio` first"
            )
            raise FileNotFoundError(msg)

        by_speaker = segments_by_speaker(read_parquet(segments_file))
        samples, sample_rate = read_mono_wav(audio_file)
        extent = Span(0.0, len(samples) / sample_rate if sample_rate else 0.0)

        if mode == "per_segment":
            speech = refine_per_segment(samples, sample_rate, by_speaker, engine)
        else:
            detected = engine.detect(samples, sample_rate)
            speech = refine_by_intersection(detected, by_speaker)

        # Nothing may extend past the audio actually decoded, which matters for
        # the recording whose file is truncated.
        speech = tuple(
            SpeakerSpeech(speaker=item.speaker, spans=clip(item.spans, extent)) for item in speech
        )

        frame = speech_frame(session.session_id, speech)
        validate(frame, SPEECH_SCHEMA, context=f"session {session.session_id}")
        write_parquet(speech_path(roots, session.session_id), frame)

        record = qc_record(session, mode=mode, segments=by_speaker, speech=speech)
        records[session.session_id] = record
        retained = record["retained_fraction"]
        shown = f"{retained:.2f}" if isinstance(retained, float) else "n/a"
        return f"{record['n_speech_spans']} span(s), {shown} of segment time retained"

    report = run_sessions(
        STAGE,
        selected,
        refine_one,
        workers=workers,
        force=force,
        is_done=is_done,
        backend="threads",
        notes=notes,
    )

    # A skipped session still belongs in the QC table.
    for outcome in report.skipped:
        session = next(s for s in selected if s.session_id == outcome.session_id)
        stored = read_parquet(speech_path(roots, outcome.session_id))
        speech = tuple(
            SpeakerSpeech(speaker=speaker, spans=spans)
            for speaker, spans in segments_by_speaker(stored).items()
        )
        segments_file = segments_path(roots, outcome.session_id)
        by_speaker = (
            segments_by_speaker(read_parquet(segments_file)) if segments_file.exists() else {}
        )
        records[outcome.session_id] = qc_record(
            session, mode=mode, segments=by_speaker, speech=speech
        )

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

    write_csv(qc_target, frame)
    logger.info(
        "wrote %s with %d row(s) (%d from this run, %d kept)",
        qc_target,
        len(frame),
        len(records),
        len(carried.rows),
    )

    return VadResult(report=report.with_notes(carried.notes(STAGE)), frame=frame, path=qc_target)


def summarise(frame: pd.DataFrame) -> list[str]:
    """Summarise how much diarized time survived as speech."""
    if frame.empty:
        return ["no sessions were processed"]

    modes = ", ".join(sorted(set(frame["mode"].dropna())))
    lines = [f"refined {len(frame)} session(s) in {modes} mode"]

    retained = frame["retained_fraction"].dropna()
    if not retained.empty:
        lines.append(
            f"share of diarized segment time that is actually speech: "
            f"min {retained.min():.2f}, median {retained.median():.2f}, "
            f"max {retained.max():.2f}"
        )
        lines.append(
            "  a median well below 1.0 is the expected result and the reason this "
            "stage exists: diarized segments span silence."
        )

    speech = frame["speech_seconds"].dropna()
    if not speech.empty:
        lines.append(
            f"detected speech per session: min {speech.min() / 60:.1f} min, "
            f"median {speech.median() / 60:.1f} min, max {speech.max() / 60:.1f} min"
        )

    spans = frame["n_speech_spans"].dropna()
    if not spans.empty:
        lines.append(
            f"speech spans per session: min {spans.min()}, "
            f"median {spans.median():.0f}, max {spans.max()}"
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
