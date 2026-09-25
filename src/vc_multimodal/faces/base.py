"""The face backend interface.

Session-level rather than frame-level, because the two backends work
differently and pretending otherwise would distort one of them. MediaPipe
decodes the video and measures sampled frames; OpenFace was run elsewhere and
leaves a CSV to be read. Both return the same per-frame measures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vc_multimodal.config import AppConfig, CropBox
    from vc_multimodal.features.face_math import FrameMeasure
    from vc_multimodal.features.sampling import FrameSampling
    from vc_multimodal.paths import RawSession


class FaceError(RuntimeError):
    """Raised when a backend is unavailable or cannot measure a session."""


class FaceBackend(ABC):
    """Measures facial action units across one session."""

    #: Short name, recorded on every row and in the manifest.
    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can run right now. Never raises."""

    @abstractmethod
    def unavailable_reason(self) -> str:
        """Why the backend cannot run, empty when it can."""

    @abstractmethod
    def version(self) -> str:
        """Identifier recorded in the run manifest."""

    def close(self) -> None:  # noqa: B027 - the no-op default is the point
        """Release whatever the backend holds open.

        Not abstract: an importer that reads a CSV has nothing to release, and
        making every backend write an empty method would say nothing. Backends
        holding a model override this.
        """

    @abstractmethod
    def measure_session(
        self,
        session: RawSession,
        *,
        config: AppConfig,
        crop: CropBox,
        sampling: FrameSampling,
    ) -> tuple[FrameMeasure, ...]:
        """Measure the sampled frames of one session.

        Args:
            session: The recording.
            config: Resolved configuration, for the action units and thresholds.
            crop: The participant's tile, in fractional frame coordinates and
                already corrected for letterboxing.
            sampling: Which frames to take.

        Returns:
            One measure per sampled frame, including the frames where no face
            was found: the dropped fraction is only meaningful if every frame
            looked at is accounted for.

        Raises:
            FaceError: if the session cannot be measured at all.
        """
