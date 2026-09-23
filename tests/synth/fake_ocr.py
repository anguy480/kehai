"""A scriptable OCR backend, so tests never depend on a platform OCR engine.

Text is matched to the image it should be read from by image size, which is
enough for the label patches this pipeline crops and keeps the fake free of any
knowledge of sessions or tiles.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vc_multimodal.ocr.base import OcrBackend, OcrError, OcrLine


class FakeOcr(OcrBackend):
    """Returns pre-scripted lines, optionally varying by image width.

    Args:
        lines_by_width: Image width in pixels to the lines to return for it.
            Widths are how the stage distinguishes a left patch from a right
            patch in tests.
        default: Lines returned for any width not listed.
        is_available: What `available()` reports.
        fail_after: Raise `OcrError` once this many reads have happened, to
            exercise the per-read error path.
    """

    name = "fake"

    def __init__(
        self,
        lines_by_width: dict[int, Sequence[tuple[str, float]]] | None = None,
        *,
        default: Sequence[tuple[str, float]] = (),
        is_available: bool = True,
        fail_after: int | None = None,
    ) -> None:
        self.lines_by_width = lines_by_width or {}
        self.default = default
        self.is_available = is_available
        self.fail_after = fail_after
        self.calls = 0
        self.widths_seen: list[int] = []

    def available(self) -> bool:
        return self.is_available

    def unavailable_reason(self) -> str:
        return "the fake backend was configured as unavailable"

    def version(self) -> str:
        return "fake/1.0"

    def read(self, image: np.ndarray, *, languages: tuple[str, ...]) -> tuple[OcrLine, ...]:
        if not self.is_available:
            raise OcrError(self.unavailable_reason())
        self.calls += 1
        width = int(image.shape[1])
        self.widths_seen.append(width)
        if self.fail_after is not None and self.calls > self.fail_after:
            msg = "scripted OCR failure"
            raise OcrError(msg)
        scripted = self.lines_by_width.get(width, self.default)
        return tuple(OcrLine(text=text, confidence=confidence) for text, confidence in scripted)


class SideScriptedOcr(OcrBackend):
    """Returns one label for the left half of a frame and another for the right.

    The stage crops each tile from the full frame, so the two patches are the
    same size; this backend distinguishes them by remembering the order in which
    they are requested, which is left then right for every sampled timestamp.
    """

    name = "fake-sides"

    def __init__(
        self, left: str, right: str, *, confidence: float = 0.9, is_available: bool = True
    ) -> None:
        self.left = left
        self.right = right
        self.confidence = confidence
        self.is_available = is_available
        self.calls = 0

    def available(self) -> bool:
        return self.is_available

    def unavailable_reason(self) -> str:
        return "the fake backend was configured as unavailable"

    def version(self) -> str:
        return "fake-sides/1.0"

    def read(self, image: np.ndarray, *, languages: tuple[str, ...]) -> tuple[OcrLine, ...]:
        if not self.is_available:
            raise OcrError(self.unavailable_reason())
        text = self.left if self.calls % 2 == 0 else self.right
        self.calls += 1
        if not text:
            return ()
        return (OcrLine(text=text, confidence=self.confidence),)
