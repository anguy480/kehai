"""The OCR backend interface.

One method, `read`, taking an image and returning recognised lines. Backends are
told nothing about sessions or tiles, so they stay trivially substitutable and
the tests can inject a fake instead of depending on a platform OCR engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class OcrError(RuntimeError):
    """Raised when a backend is unavailable or fails on an image."""


@dataclass(frozen=True, slots=True)
class OcrLine:
    """One recognised line of text.

    Attributes:
        text: The recognised string. Treated as personal data: compared, never
            printed, logged or written.
        confidence: Backend confidence in [0, 1].
    """

    text: str
    confidence: float


class OcrBackend(ABC):
    """Reads text from images."""

    #: Short name, as used in configuration and recorded in the manifest.
    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can run on this machine right now.

        Never raises: an unavailable backend is a normal, reportable state, not
        an error, because the pipeline falls back to the assumed side.
        """

    @abstractmethod
    def version(self) -> str:
        """Identifier recorded in the run manifest."""

    @abstractmethod
    def read(self, image: np.ndarray, *, languages: tuple[str, ...]) -> tuple[OcrLine, ...]:
        """Recognise text in a BGR image.

        Args:
            image: Image as an OpenCV BGR array.
            languages: Preferred recognition languages, most preferred first.

        Returns:
            Recognised lines, in no guaranteed order.

        Raises:
            OcrError: if recognition fails.
        """
