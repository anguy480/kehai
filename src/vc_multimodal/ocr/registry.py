"""Backend lookup by configured name."""

from __future__ import annotations

from typing import Final

from vc_multimodal.ocr.apple_vision import AppleVisionOcr
from vc_multimodal.ocr.base import OcrBackend, OcrError, OcrLine
from vc_multimodal.ocr.tesseract import TesseractOcr


class NullOcr(OcrBackend):
    """A backend that reads nothing, for `backend: none`.

    Selecting it makes every session inconclusive, which is a supported state:
    `vc verify-layout` then falls back to the assumed side and flags it.
    """

    name = "none"

    def available(self) -> bool:
        """Always False: this backend deliberately does nothing."""
        return False

    def unavailable_reason(self) -> str:
        """Why nothing will be recognised."""
        return "OCR is disabled (speakers.label_ocr.backend is 'none')"

    def version(self) -> str:
        """Fixed identifier."""
        return "none"

    def read(
        self,
        image: object,  # noqa: ARG002 - present to satisfy the interface
        *,
        languages: tuple[str, ...],  # noqa: ARG002 - present to satisfy the interface
    ) -> tuple[OcrLine, ...]:
        """Always raises, because this backend recognises nothing."""
        raise OcrError(self.unavailable_reason())


_BACKENDS: Final[dict[str, type[OcrBackend]]] = {
    AppleVisionOcr.name: AppleVisionOcr,
    TesseractOcr.name: TesseractOcr,
    NullOcr.name: NullOcr,
}


def get_backend(name: str) -> OcrBackend:
    """Instantiate the backend called `name`.

    Raises:
        OcrError: if no backend has that name.
    """
    try:
        return _BACKENDS[name]()
    except KeyError as exc:
        msg = f"unknown OCR backend {name!r}; available: {sorted(_BACKENDS)}"
        raise OcrError(msg) from exc


def available_backends() -> tuple[str, ...]:
    """Names of the backends that can actually run on this machine."""
    return tuple(name for name, cls in _BACKENDS.items() if cls().available())
