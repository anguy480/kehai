"""ffmpeg and ffprobe: discovery, metadata probing and frame extraction.

The binaries are never installed by this project. On the development machine
they come from conda-forge and are only on `PATH` while conda is active, so
absolute paths can be pinned in `.env` as `VC_FFMPEG` / `VC_FFPROBE`. Discovery
prefers those and falls back to `PATH`, failing with an explanation rather than
attempting an install.

Probing reads metadata only: container and stream properties, never pixels or
samples. `extract_frame` is the one function that decodes, and it exists for
`vc preview`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

FFMPEG_ENV: Final = "VC_FFMPEG"
FFPROBE_ENV: Final = "VC_FFPROBE"

# Above this relative gap between the base and average frame rates, a file is
# reported as variable frame rate, which makes frame timestamps unreliable.
VFR_TOLERANCE: Final = 0.01

_PROBE_TIMEOUT_S: Final = 120.0
_FRAME_TIMEOUT_S: Final = 120.0


class FfmpegError(RuntimeError):
    """Raised when a binary is missing or an ffmpeg/ffprobe call fails."""


def _resolve_binary(name: str, env_var: str) -> Path:
    """Find one binary, preferring `$env_var` over `PATH`."""
    configured = os.environ.get(env_var, "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        msg = (
            f"{env_var} is set to {candidate}, which is not an executable file. "
            f"Fix it in .env, or unset it to fall back to PATH."
        )
        raise FfmpegError(msg)

    found = shutil.which(name)
    if found:
        return Path(found)

    msg = (
        f"{name} was not found. This project does not install it.\n"
        f"  - If it is provided by conda, either activate that environment or "
        f"set {env_var} in .env to its absolute path (`make env` does this).\n"
        f"  - Homebrew cannot install ffmpeg on macOS 14; use conda-forge."
    )
    raise FfmpegError(msg)


@dataclass(frozen=True, slots=True)
class FfmpegTools:
    """Located ffmpeg and ffprobe binaries."""

    ffmpeg: Path
    ffprobe: Path

    @classmethod
    def discover(cls) -> FfmpegTools:
        """Locate both binaries.

        Raises:
            FfmpegError: if either is missing or not executable.
        """
        tools = cls(
            ffmpeg=_resolve_binary("ffmpeg", FFMPEG_ENV),
            ffprobe=_resolve_binary("ffprobe", FFPROBE_ENV),
        )
        logger.debug("using ffmpeg=%s ffprobe=%s", tools.ffmpeg, tools.ffprobe)
        return tools

    def versions(self) -> dict[str, str]:
        """First version line of each binary, for the run manifest."""
        return {
            "ffmpeg": _first_line(self.ffmpeg),
            "ffprobe": _first_line(self.ffprobe),
        }

    def probe(self, media: Path) -> dict[str, Any]:
        """Return ffprobe's JSON for `media` (format and streams).

        Raises:
            FfmpegError: if ffprobe fails or emits unparseable output.
        """
        command = [
            str(self.ffprobe),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(media),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=_PROBE_TIMEOUT_S
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"ffprobe timed out after {_PROBE_TIMEOUT_S:.0f}s on {media.name}"
            raise FfmpegError(msg) from exc
        if result.returncode != 0:
            msg = f"ffprobe failed on {media.name}: {result.stderr.strip()[:300]}"
            raise FfmpegError(msg)
        try:
            parsed: dict[str, Any] = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            msg = f"ffprobe returned unparseable output for {media.name}"
            raise FfmpegError(msg) from exc
        return parsed

    def extract_frame(self, media: Path, timestamp_s: float, target: Path) -> Path:
        """Decode a single frame at `timestamp_s` into `target`.

        Seeks before the input for speed, then decodes one frame. Used only by
        `vc preview`; no other stage writes an image.

        Raises:
            FfmpegError: if ffmpeg fails or writes nothing.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{max(timestamp_s, 0.0):.3f}",
            "-i",
            str(media),
            "-frames:v",
            "1",
            "-f",
            "image2",
            str(target),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=_FRAME_TIMEOUT_S
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"ffmpeg timed out extracting a frame from {media.name}"
            raise FfmpegError(msg) from exc
        if result.returncode != 0 or not target.exists():
            msg = (
                f"ffmpeg could not extract a frame at {timestamp_s:.1f}s from "
                f"{media.name}: {result.stderr.strip()[:300]}"
            )
            raise FfmpegError(msg)
        return target


def _first_line(binary: Path) -> str:
    """Return the first line of `binary -version`, or a placeholder."""
    try:
        result = subprocess.run(
            [str(binary), "-version"], capture_output=True, text=True, check=False, timeout=30.0
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - defensive
        return "unknown"
    return result.stdout.splitlines()[0].strip() if result.stdout else "unknown"


def parse_rational(value: object) -> float | None:
    """Parse an ffprobe rational such as `"30000/1001"` into a float.

    Returns None for missing, malformed or zero-denominator values, which
    ffprobe emits as `"0/0"` for streams with no meaningful rate.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    numerator_text, separator, denominator_text = text.partition("/")
    numerator = _as_float(numerator_text)
    if numerator is None:
        return None
    if not separator:
        return numerator

    denominator = _as_float(denominator_text)
    if denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _as_float(value: object) -> float | None:
    """Coerce an ffprobe field to float, returning None when absent or invalid.

    ffprobe reports every field as a string, so values are coerced via `str`
    rather than assuming a numeric type.
    """
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    """Coerce an ffprobe field to int, returning None when absent or invalid.

    Parsed as a float first, so a field reported as `"1080.0"` still yields an
    integer instead of being discarded.
    """
    number = _as_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        return None


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """One audio stream's properties."""

    index: int
    codec: str | None
    channels: int | None
    sample_rate: int | None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """The metadata the inventory stage records for one recording."""

    duration_s: float | None
    size_bytes: int | None
    video_codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    fps_variable: bool
    audio_streams: tuple[AudioStreamInfo, ...]

    @property
    def n_audio_streams(self) -> int:
        """How many audio streams the container holds."""
        return len(self.audio_streams)

    @property
    def primary_audio(self) -> AudioStreamInfo | None:
        """The first audio stream, or None if the file has no audio."""
        return self.audio_streams[0] if self.audio_streams else None


def parse_media_info(probe: dict[str, Any]) -> MediaInfo:
    """Turn ffprobe JSON into a `MediaInfo`.

    Missing fields become None rather than raising: a recording with unusual or
    incomplete metadata should still appear in the inventory, flagged, instead
    of aborting the stage.

    Variable frame rate is inferred by comparing the base frame rate
    (`r_frame_rate`) with the average (`avg_frame_rate`). Detecting it properly
    would mean decoding every frame's timestamp; this is the cheap signal, and
    it is reported as a flag for a human to follow up rather than acted on.
    """
    container = probe.get("format", {}) or {}
    streams = probe.get("streams", []) or []

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    base_fps = parse_rational(video.get("r_frame_rate")) if video else None
    average_fps = parse_rational(video.get("avg_frame_rate")) if video else None
    fps = average_fps or base_fps
    variable = False
    if base_fps and average_fps and average_fps > 0:
        variable = abs(base_fps - average_fps) / average_fps > VFR_TOLERANCE

    duration = _as_float(container.get("duration"))
    if duration is None and video is not None:
        duration = _as_float(video.get("duration"))

    return MediaInfo(
        duration_s=duration,
        size_bytes=_as_int(container.get("size")),
        video_codec=video.get("codec_name") if video else None,
        width=_as_int(video.get("width")) if video else None,
        height=_as_int(video.get("height")) if video else None,
        fps=fps,
        fps_variable=variable,
        audio_streams=tuple(
            AudioStreamInfo(
                index=_as_int(stream.get("index")) or position,
                codec=stream.get("codec_name"),
                channels=_as_int(stream.get("channels")),
                sample_rate=_as_int(stream.get("sample_rate")),
            )
            for position, stream in enumerate(audio)
        ),
    )
