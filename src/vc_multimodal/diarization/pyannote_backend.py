"""Diarize locally with pyannote, as a fallback.

Used only if the original whisper-diarization output cannot be supplied. It is
a fallback rather than the default because re-diarizing would confound "new
modality" with "new transcripts" in the comparison against the lab manuscript
(see docs/decisions/0002).

pyannote lives behind the optional `pyannote` extra, so neither CI nor the
label holder's machine needs a Hugging Face token or a model download. The
pipeline is injectable, which keeps this module testable without either.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Final

from vc_multimodal.diarization.base import (
    DiarizationBackend,
    DiarizationError,
    Segment,
    canonical_speaker,
    sort_segments,
)
from vc_multimodal.logging_setup import get_logger

if TYPE_CHECKING:
    from vc_multimodal.config import PyannoteConfig
    from vc_multimodal.paths import RawSession

logger = get_logger(__name__)

_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - import success depends on the optional extra
    from pyannote.audio import Pipeline as _Pipeline
# Any import failure at all means the backend is simply unavailable.
except Exception as exc:
    _Pipeline = None
    _IMPORT_ERROR = exc

_MIN_SEGMENT_S: Final = 0.01


def segments_from_annotation(annotation: Any) -> tuple[Segment, ...]:
    """Convert a pyannote `Annotation` into normalised segments.

    Isolated from the pipeline so the conversion can be tested without pyannote
    installed: anything that iterates as `(turn, _, label)` works here, which is
    what `Annotation.itertracks(yield_label=True)` yields.

    Raises:
        DiarizationError: if the annotation yields no usable segments.
    """
    segments: list[Segment] = []
    for turn, _track, label in annotation.itertracks(yield_label=True):
        start, end = float(turn.start), float(turn.end)
        if end - start < _MIN_SEGMENT_S:
            continue
        # pyannote carries no transcript, which makes this the safer source:
        # there is nothing sensitive in its output to begin with.
        segments.append(
            Segment(speaker=canonical_speaker(str(label)), start=start, end=end, text=None)
        )

    if not segments:
        msg = "pyannote returned no speech segments"
        raise DiarizationError(msg)
    return sort_segments(segments)


class PyannoteBackend(DiarizationBackend):
    """Runs a pyannote speaker-diarization pipeline locally.

    Args:
        config: Model, revision, speaker count and the token's variable name.
        pipeline: A ready pipeline. Supplying one skips loading entirely, which
            is how this is tested.
    """

    name = "pyannote"

    def __init__(self, config: PyannoteConfig, *, pipeline: Any | None = None) -> None:
        """Store configuration; the model is loaded lazily on first use."""
        self.config = config
        self._pipeline = pipeline

    def _token(self) -> str | None:
        """The Hugging Face token from the configured environment variable."""
        return os.environ.get(self.config.token_env, "").strip() or None

    def available(self) -> bool:
        """Whether pyannote is installed and a token is present."""
        if self._pipeline is not None:
            return True
        return _IMPORT_ERROR is None and self._token() is not None

    def unavailable_reason(self) -> str:
        """Why the backend cannot run."""
        if self._pipeline is not None:
            return ""
        if _IMPORT_ERROR is not None:
            return (
                f"pyannote.audio is not installed ({type(_IMPORT_ERROR).__name__}); "
                f"install the optional extra with `uv sync --extra pyannote`"
            )
        if self._token() is None:
            return (
                f"{self.config.token_env} is not set. pyannote's models are gated, so "
                f"a Hugging Face token is required; put it in .env."
            )
        return ""

    def version(self) -> str:
        """Model and revision, recorded in the manifest.

        The revision is what makes a run reproducible, so an unpinned model is
        recorded as such rather than silently passing for a pinned one.
        """
        revision = self.config.revision or "unpinned"
        return f"pyannote/{self.config.model}@{revision}"

    def pipeline(self) -> Any:
        """Load the pipeline once, on first use.

        Raises:
            DiarizationError: if the backend is unavailable or loading fails.
        """
        if self._pipeline is not None:
            return self._pipeline
        if not self.available():
            raise DiarizationError(self.unavailable_reason())
        if _Pipeline is None:  # pragma: no cover - guarded by available()
            raise DiarizationError(self.unavailable_reason())

        if self.config.revision is None:
            logger.warning(
                "pyannote model %s is not pinned to a revision; results may change "
                "when the upstream model does. Set diarization.pyannote.revision.",
                self.config.model,
            )
        try:  # pragma: no cover - requires a token and a model download
            self._pipeline = _Pipeline.from_pretrained(
                self.config.model,
                use_auth_token=self._token(),
                revision=self.config.revision,
            )
        except Exception as exc:  # pragma: no cover - network and auth failures
            msg = f"could not load pyannote pipeline {self.config.model}: {exc}"
            raise DiarizationError(msg) from exc
        return self._pipeline

    def segments(self, session: RawSession) -> tuple[Segment, ...]:
        """Diarize one session's audio.

        Reads the mono WAV written by `vc extract-audio` when it exists, since
        pyannote expects 16 kHz and decoding the mp4 again would be wasteful.

        Raises:
            DiarizationError: if the pipeline is unavailable or fails.
        """
        pipeline = self.pipeline()
        try:
            annotation = pipeline(str(session.path), num_speakers=self.config.num_speakers)
        except Exception as exc:
            msg = f"pyannote failed on session {session.session_id}: {exc}"
            raise DiarizationError(msg) from exc
        return segments_from_annotation(annotation)
