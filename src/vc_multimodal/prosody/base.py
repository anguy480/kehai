"""The prosody backend interface.

One method: measure a stretch of audio and return frame-level contours plus
the per-span voice-quality measures. Everything that turns those into session
features lives in `features/prosody_math.py`, so a second backend only has to
produce measurements, not statistics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from vc_multimodal.config import ProsodyConfig
    from vc_multimodal.features.prosody_math import SpanMeasures


class ProsodyError(RuntimeError):
    """Raised when a backend is unavailable or cannot measure a span."""


class ProsodyBackend(ABC):
    """Measures prosody in a stretch of mono audio."""

    #: Short name, as used in configuration and recorded in the manifest.
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

    @abstractmethod
    def measure(
        self, samples: np.ndarray, sample_rate: int, *, config: ProsodyConfig
    ) -> SpanMeasures:
        """Measure one stretch of speech.

        Args:
            samples: Mono audio in [-1, 1].
            sample_rate: Samples per second.
            config: Pitch range and analysis settings.

        Returns:
            The frame-level contours and per-span measures for this stretch.

        Raises:
            ProsodyError: if the span cannot be measured.
        """
