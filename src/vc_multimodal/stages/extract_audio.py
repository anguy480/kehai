"""Stage 2: extract mono 16 kHz audio, and compare the stereo channels.

Every recording carries one mixed AAC stream in stereo at 48 kHz, confirmed by
`vc inventory` across all 62. Later stages want mono 16 kHz, which is what
Silero VAD and speaker embeddings expect.

While the audio is being decoded anyway, the two channels are compared. Zoom
sometimes pans speakers across the stereo field, and if the channels genuinely
differ then that partial separation is a cheap signal about who is speaking that
owes nothing to diarization. If they are effectively duplicates, that is worth
knowing too, so nothing downstream is built on a separation that is not there.

Decoding streams through a pipe in chunks: nothing holds a whole recording in
memory, and the resulting statistics are exact rather than sampled.
"""

from __future__ import annotations

import json
import subprocess
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.contracts import AUDIO_QC_SCHEMA, validate
from vc_multimodal.features.channels import (
    FLAG_NOT_STEREO,
    FLAG_PARTIAL_SEPARATION,
    StereoAccumulator,
    StereoStats,
    downmix_to_mono,
    stereo_flags,
)
from vc_multimodal.ffmpeg import FfmpegError, FfmpegTools, MediaInfo, parse_media_info
from vc_multimodal.io_utils import atomic_path, write_csv, write_json
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.runner import StageReport, run_sessions

logger = get_logger(__name__)

STAGE: Final = "extract-audio"
AUDIO_DIRNAME: Final = "audio"
AUDIO_QC_FILENAME: Final = "audio_qc.csv"

FLAG_UNEXPECTED_LAYOUT: Final = "audio_stream_layout_unexpected"

_INT16_FULL_SCALE: Final = 32768.0
_STEREO: Final = 2

# Beyond this many sessions per flag, list the count instead of every ID.
_MAX_LISTED_SESSIONS: Final = 12

COLUMN_ORDER: Final = (
    "session_id",
    "wave",
    "sample_rate",
    "source_channels",
    "duration_s",
    "n_samples",
    "active_fraction",
    "lr_correlation",
    "ild_db",
    "rms_left",
    "rms_right",
    "peak_left",
    "peak_right",
    "bit_identical",
    "flags",
)


def audio_dir(roots: DataRoots) -> Path:
    """Directory holding extracted audio, under the work root."""
    return roots.work_path(AUDIO_DIRNAME, create_parent=True)


def audio_path(roots: DataRoots, session_id: int, *, stream: int = 0) -> Path:
    """Where one session's mono WAV is written.

    A stream index is included only for the second and later streams, so the
    common single-stream case keeps a plain `<session_id>.wav`.
    """
    suffix = "" if stream == 0 else f"_a{stream}"
    return roots.work_path(AUDIO_DIRNAME, f"{session_id}{suffix}.wav")


def stats_path(roots: DataRoots, session_id: int) -> Path:
    """Where one session's channel statistics sidecar is written.

    Numbers only, no audio and no text. Kept per session so that a rerun which
    skips completed work can still assemble the full QC table.
    """
    return roots.work_path(AUDIO_DIRNAME, f"{session_id}.stats.json")


def _pcm_command(
    tools: FfmpegTools, media: Path, *, stream: int, channels: int, sample_rate: int
) -> list[str]:
    """Build the ffmpeg arguments that decode one stream to raw PCM on stdout."""
    return [
        str(tools.ffmpeg),
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(media),
        "-map",
        f"0:a:{stream}",
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-",
    ]


def _decode_to_mono_wav(
    command: Sequence[str],
    target: Path,
    *,
    sample_rate: int,
    decode_channels: int,
    bytes_per_frame: int,
    chunk_bytes: int,
    accumulator: StereoAccumulator | None,
    source_name: str,
) -> int:
    """Stream ffmpeg's PCM output into a mono WAV, feeding the accumulator.

    Reads in chunks so no whole recording is ever held in memory, and carries a
    partial frame across a chunk boundary rather than dropping it.

    Returns:
        The number of mono samples written.

    Raises:
        FfmpegError: if ffmpeg fails or decodes nothing.
    """
    n_samples = 0
    # argv is built from located binaries and config, never a shell string.
    process = subprocess.Popen(list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:  # pragma: no cover - defensive
        process.kill()
        msg = "ffmpeg produced no stdout pipe"
        raise FfmpegError(msg)

    try:
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)

            leftover = b""
            while True:
                block = process.stdout.read(chunk_bytes)
                if not block:
                    break
                block = leftover + block
                usable = len(block) - (len(block) % bytes_per_frame)
                leftover = block[usable:]
                if not usable:
                    continue

                scaled = np.frombuffer(block[:usable], dtype="<i2").astype(np.float32)
                scaled /= _INT16_FULL_SCALE
                if decode_channels == _STEREO:
                    stereo = scaled.reshape(-1, _STEREO)
                    if accumulator is not None:
                        accumulator.update(stereo)
                    mono = downmix_to_mono(stereo)
                else:
                    mono = scaled

                n_samples += int(mono.size)
                writer.writeframes(
                    np.clip(mono * _INT16_FULL_SCALE, -32768, 32767).astype("<i2").tobytes()
                )
    finally:
        stderr = process.stderr.read() if process.stderr else b""
        process.stdout.close()
        returncode = process.wait()

    if returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:300]
        msg = f"ffmpeg failed decoding audio from {source_name}: {detail}"
        raise FfmpegError(msg)
    if n_samples == 0:
        msg = f"no audio samples were decoded from {source_name}"
        raise FfmpegError(msg)
    return n_samples


@dataclass(frozen=True, slots=True)
class ExtractedAudio:
    """The result of extracting one audio stream."""

    session_id: int
    path: Path
    sample_rate: int
    source_channels: int
    n_samples: int
    stats: StereoStats | None

    @property
    def duration_s(self) -> float:
        """Length of the written audio in seconds."""
        return self.n_samples / self.sample_rate if self.sample_rate else 0.0


def extract_stream(
    session: RawSession,
    *,
    tools: FfmpegTools,
    info: MediaInfo,
    config: AppConfig,
    target: Path,
    stream: int = 0,
) -> ExtractedAudio:
    """Decode one audio stream to a mono WAV, measuring the channels en route.

    A stereo source is decoded as stereo so the channels can be compared, then
    averaged to mono, which is what ffmpeg's own `-ac 1` downmix does. A mono
    source is decoded directly and no comparison is possible.

    Args:
        session: The recording to read.
        tools: Located binaries.
        info: Container metadata, for the source channel count.
        config: Resolved configuration.
        target: Where to write the mono WAV.
        stream: Which audio stream to decode.

    Returns:
        What was written, and the channel statistics where available.

    Raises:
        FfmpegError: if decoding fails or produces nothing.
    """
    audio_config = config.audio
    probe = audio_config.stereo_probe
    source_channels = 0
    if info.audio_streams and stream < len(info.audio_streams):
        source_channels = info.audio_streams[stream].channels or 0

    read_stereo = probe.enabled and source_channels >= _STEREO
    decode_channels = _STEREO if read_stereo else 1
    accumulator = StereoAccumulator(activity_floor=probe.activity_floor) if read_stereo else None

    bytes_per_frame = 2 * decode_channels
    chunk_frames = max(1, int(probe.chunk_seconds * audio_config.sample_rate))
    chunk_bytes = chunk_frames * bytes_per_frame

    command = _pcm_command(
        tools,
        session.path,
        stream=stream,
        channels=decode_channels,
        sample_rate=audio_config.sample_rate,
    )

    with atomic_path(target, suffix=".wav") as tmp:
        n_samples = _decode_to_mono_wav(
            command,
            tmp,
            sample_rate=audio_config.sample_rate,
            decode_channels=decode_channels,
            bytes_per_frame=bytes_per_frame,
            chunk_bytes=chunk_bytes,
            accumulator=accumulator,
            source_name=session.path.name,
        )

    return ExtractedAudio(
        session_id=session.session_id,
        path=target,
        sample_rate=audio_config.sample_rate,
        source_channels=source_channels,
        n_samples=n_samples,
        stats=accumulator.result() if accumulator is not None else None,
    )


def stats_record(
    extracted: ExtractedAudio, *, wave_name: str, config: AppConfig, extra_flags: Sequence[str] = ()
) -> dict[str, object]:
    """Build the per-session statistics record written beside the audio."""
    probe = config.audio.stereo_probe
    stats = extracted.stats
    flags = list(extra_flags)

    if stats is None:
        flags.append(FLAG_NOT_STEREO)
    else:
        flags.extend(
            stereo_flags(
                stats,
                correlated_above=probe.correlated_above,
                strong_separation_below=probe.strong_separation_below,
                imbalance_db=probe.imbalance_db,
            )
        )

    return {
        "session_id": extracted.session_id,
        "wave": wave_name,
        "sample_rate": extracted.sample_rate,
        "source_channels": extracted.source_channels,
        "duration_s": round(extracted.duration_s, 3),
        "n_samples": extracted.n_samples,
        "active_fraction": None if stats is None else round(stats.active_fraction, 4),
        "lr_correlation": None
        if stats is None or stats.correlation is None
        else round(stats.correlation, 6),
        "ild_db": None if stats is None or stats.ild_db is None else round(stats.ild_db, 3),
        "rms_left": None if stats is None else round(stats.rms_left, 6),
        "rms_right": None if stats is None else round(stats.rms_right, 6),
        "peak_left": None if stats is None else round(stats.peak_left, 6),
        "peak_right": None if stats is None else round(stats.peak_right, 6),
        "bit_identical": None if stats is None else stats.bit_identical,
        "flags": ";".join(dict.fromkeys(flags)),
    }


def build_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Assemble audio QC rows into a correctly typed table."""
    frame = pd.DataFrame(list(rows), columns=list(COLUMN_ORDER))
    for column in ("sample_rate", "source_channels", "n_samples"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in (
        "duration_s",
        "active_fraction",
        "lr_correlation",
        "ild_db",
        "rms_left",
        "rms_right",
        "peak_left",
        "peak_right",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["bit_identical"] = frame["bit_identical"].astype("boolean")
    frame["flags"] = frame["flags"].fillna("").astype(str)
    return frame.sort_values("session_id", ignore_index=True)


@dataclass(frozen=True, slots=True)
class ExtractAudioResult:
    """What the extract-audio stage produced."""

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
    tools: FfmpegTools | None = None,
) -> ExtractAudioResult:
    """Extract mono audio for every requested session and write the QC table.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to extract, or None for all.
        workers: Parallel workers, or None to choose automatically.
        force: Re-extract sessions whose audio already exists.
        tools: Located binaries; discovered if omitted.

    Returns:
        The stage report and the written QC table.
    """
    binaries = tools or FfmpegTools.discover()
    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []
    audio_dir(roots)

    def is_done(session: RawSession) -> bool:
        return (
            audio_path(roots, session.session_id).exists()
            and stats_path(roots, session.session_id).exists()
        )

    def extract_one(session: RawSession) -> str:
        info = parse_media_info(binaries.probe(session.path))
        extra: list[str] = []
        expected_mixed = config.audio.stream_layout in {"auto", "mixed"}
        if expected_mixed and info.n_audio_streams > 1:
            extra.append(FLAG_UNEXPECTED_LAYOUT)

        # A file with several streams gets one WAV per stream; only the first is
        # measured, since the probe describes the mixed stream's stereo field.
        written: list[ExtractedAudio] = []
        for stream in range(max(info.n_audio_streams, 1)):
            written.append(
                extract_stream(
                    session,
                    tools=binaries,
                    info=info,
                    config=config,
                    target=audio_path(roots, session.session_id, stream=stream),
                    stream=stream,
                )
            )

        record = stats_record(written[0], wave_name=session.wave, config=config, extra_flags=extra)
        write_json(stats_path(roots, session.session_id), record)

        correlation = record["lr_correlation"]
        shown = f"{correlation:.3f}" if isinstance(correlation, float) else "n/a"
        return f"{len(written)} stream(s), {written[0].duration_s:.1f}s, L/R r={shown}"

    report = run_sessions(
        STAGE,
        selected,
        extract_one,
        workers=workers,
        force=force,
        is_done=is_done,
        backend="threads",
        notes=notes,
    )

    rows: list[Mapping[str, object]] = []
    for sidecar in sorted(audio_dir(roots).glob("*.stats.json")):
        try:
            rows.append(json.loads(sidecar.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            logger.warning("could not read %s; it will be rebuilt on the next run", sidecar.name)

    frame = build_frame(rows)
    validate(frame, AUDIO_QC_SCHEMA, context=STAGE)
    target = roots.out_path(AUDIO_QC_FILENAME)
    write_csv(target, frame)
    logger.info("wrote %s with %d row(s)", target, len(frame))

    return ExtractAudioResult(report=report, frame=frame, path=target)


def summarise(frame: pd.DataFrame) -> list[str]:
    """Summarise extracted audio and the stereo comparison, metadata only."""
    if frame.empty:
        return ["no audio has been extracted"]

    lines = [f"audio extracted for {len(frame)} session(s)"]

    durations = frame["duration_s"].dropna()
    if not durations.empty:
        lines.append(
            f"duration: min {durations.min() / 60:.1f} min, "
            f"median {durations.median() / 60:.1f} min, "
            f"max {durations.max() / 60:.1f} min"
        )

    rates = frame["sample_rate"].dropna().unique()
    channels = frame["source_channels"].dropna().unique()
    lines.append(
        f"written as mono {', '.join(str(int(r)) for r in rates)} Hz "
        f"from {', '.join(str(int(c)) for c in channels)}-channel source(s)"
    )

    correlation = frame["lr_correlation"].dropna()
    lines.append("")
    if correlation.empty:
        lines.append("left/right comparison: not available for any session")
    else:
        lines.append(
            f"left/right correlation over {len(correlation)} session(s): "
            f"min {correlation.min():.4f}, median {correlation.median():.4f}, "
            f"max {correlation.max():.4f}"
        )
        # Read the recorded flags rather than re-applying a threshold here, so
        # the configured value remains the single source of truth.
        has_separation = (
            frame["flags"].astype(str).str.contains(FLAG_PARTIAL_SEPARATION, regex=False)
        )
        separated = frame.loc[has_separation, "session_id"]
        if separated.empty:
            lines.append(
                "  every session's channels carry the same signal: there is no stereo "
                "separation to exploit, and nothing downstream should assume any."
            )
        else:
            ids = sorted(int(i) for i in separated)
            lines.append(f"  {len(ids)} session(s) show some channel separation: {ids}")
            lines.append(
                "  worth following up: partial separation would give a "
                "diarization-independent signal about who is speaking."
            )

    ild = frame["ild_db"].dropna()
    if not ild.empty:
        lines.append(
            f"channel balance: median {ild.median():+.2f} dB, "
            f"largest imbalance {ild.abs().max():.2f} dB"
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
