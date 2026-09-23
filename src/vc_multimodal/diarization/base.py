"""The diarization backend interface.

Every backend answers one question — who spoke when — and returns it in one
normalised form, so no later stage knows or cares where the answer came from.
That matters here because the preferred source is a file produced elsewhere by
whisper-diarization, while the fallback runs pyannote locally, and the two must
be interchangeable for the comparison against the lab manuscript to mean
anything.

Segment text is carried optionally and is the most sensitive artifact in the
project: it is what participants said. It is written only under
`$VC_WORK_ROOT`, never printed, and never included in a handoff bundle.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from vc_multimodal.paths import RawSession

# Canonical speaker label. Backends emit wildly different spellings
# ("Speaker 0", "SPEAKER_00", "spk1"), and they are normalised to this so that
# role assignment and every QC table downstream can rely on one form.
SPEAKER_LABEL: Final = "SPEAKER_{index:02d}"
_SPEAKER_DIGITS: Final = re.compile(r"(\d+)")


class DiarizationError(RuntimeError):
    """Raised when diarization output is missing, unusable or unparseable."""


def canonical_speaker(label: str, *, fallback_index: int = 0) -> str:
    """Normalise a backend's speaker label to `SPEAKER_NN`.

    Args:
        label: Whatever the backend called the speaker.
        fallback_index: Index to use when the label carries no number.

    Returns:
        The canonical label.
    """
    match = _SPEAKER_DIGITS.search(label)
    index = int(match.group(1)) if match else fallback_index
    return SPEAKER_LABEL.format(index=index)


@dataclass(frozen=True, slots=True)
class Segment:
    """One stretch of speech attributed to one speaker.

    Attributes:
        speaker: Canonical speaker label.
        start: Start time in seconds from the beginning of the recording.
        end: End time in seconds.
        text: What was said, where the backend provides it. Treated as the most
            sensitive artifact in the project.
    """

    speaker: str
    start: float
    end: float
    text: str | None = None

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.end - self.start

    def without_text(self) -> Segment:
        """A copy carrying no transcript text."""
        return replace(self, text=None)


def sort_segments(segments: Iterable[Segment]) -> tuple[Segment, ...]:
    """Order segments by start time, then end time, then speaker."""
    return tuple(sorted(segments, key=lambda s: (s.start, s.end, s.speaker)))


def strip_text(segments: Iterable[Segment]) -> tuple[Segment, ...]:
    """Drop transcript text from every segment."""
    return tuple(segment.without_text() for segment in segments)


def speakers_in(segments: Sequence[Segment]) -> tuple[str, ...]:
    """Distinct speaker labels, in order of first appearance."""
    return tuple(dict.fromkeys(segment.speaker for segment in segments))


def total_speech(segments: Sequence[Segment]) -> float:
    """Summed segment duration, counting overlaps twice."""
    return sum(segment.duration for segment in segments)


def covered_time(segments: Sequence[Segment]) -> float:
    """Wall-clock time covered by at least one segment, counting overlaps once."""
    if not segments:
        return 0.0
    covered = 0.0
    current_start, current_end = None, None
    for segment in sort_segments(segments):
        if current_end is None or segment.start > current_end:
            if current_end is not None and current_start is not None:
                covered += current_end - current_start
            current_start, current_end = segment.start, segment.end
        else:
            current_end = max(current_end, segment.end)
    if current_start is not None and current_end is not None:
        covered += current_end - current_start
    return covered


def overlap_time(segments: Sequence[Segment]) -> float:
    """Time covered by more than one speaker at once."""
    return max(0.0, total_speech(segments) - covered_time(segments))


class DiarizationBackend(ABC):
    """Produces normalised speech segments for one session."""

    #: Short name, as used in configuration and recorded in the manifest.
    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can run right now.

        Never raises: an unavailable backend is a reportable state, since the
        choice of source is a project decision still being resolved.
        """

    @abstractmethod
    def unavailable_reason(self) -> str:
        """Why the backend cannot run, empty when it can."""

    @abstractmethod
    def version(self) -> str:
        """Identifier recorded in the run manifest."""

    @abstractmethod
    def segments(self, session: RawSession) -> tuple[Segment, ...]:
        """Return the session's speech segments, ordered by start time.

        Raises:
            DiarizationError: if this session cannot be diarized.
        """
