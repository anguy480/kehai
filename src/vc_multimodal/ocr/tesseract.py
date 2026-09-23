"""OCR through an external tesseract binary, if one is installed.

A fallback for machines without the Vision framework. tesseract is not a Python
dependency of this project and is not installed by it: the binary is either
present or the backend reports itself unavailable.

Recognition writes to stdout, so no image and no text file is created.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING, Final

import cv2

from vc_multimodal.ocr.base import OcrBackend, OcrError, OcrLine

if TYPE_CHECKING:
    import numpy as np

_TIMEOUT_S: Final = 60.0

# tesseract names its language data differently from BCP-47 tags.
_LANGUAGE_CODES: Final = {
    "ja-JP": "jpn",
    "ja": "jpn",
    "en-US": "eng",
    "en": "eng",
    "zh-Hans": "chi_sim",
    "ko-KR": "kor",
}


class TesseractOcr(OcrBackend):
    """Text recognition by piping a PNG to `tesseract stdin stdout`."""

    name = "tesseract"

    def available(self) -> bool:
        """Whether a tesseract binary is on PATH."""
        return shutil.which("tesseract") is not None

    def unavailable_reason(self) -> str:
        """Why the backend cannot run."""
        if not self.available():
            return (
                "tesseract is not on PATH. This project does not install it; "
                "install it yourself or use the apple_vision backend."
            )
        return ""

    def version(self) -> str:
        """First line of `tesseract --version`."""
        binary = shutil.which("tesseract")
        if binary is None:
            return "unavailable"
        result = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, check=False, timeout=_TIMEOUT_S
        )
        first = result.stdout.splitlines()
        return first[0].strip() if first else "unknown"

    def read(self, image: np.ndarray, *, languages: tuple[str, ...]) -> tuple[OcrLine, ...]:
        """Recognise text in a BGR image.

        tesseract reports no per-line confidence in this mode, so every line is
        returned with a confidence of 1.0 and the caller's threshold has no
        effect. That is recorded in the QC output as the backend's limitation
        rather than hidden.

        Raises:
            OcrError: if the backend is unavailable or tesseract fails.
        """
        if not self.available():
            raise OcrError(self.unavailable_reason())
        if image.size == 0:
            return ()

        binary = shutil.which("tesseract")
        if binary is None:  # pragma: no cover - checked above
            raise OcrError(self.unavailable_reason())

        codes = [_LANGUAGE_CODES[tag] for tag in languages if tag in _LANGUAGE_CODES]
        command = [binary, "stdin", "stdout"]
        if codes:
            command += ["-l", "+".join(dict.fromkeys(codes))]

        try:
            result = subprocess.run(
                command,
                input=_encode_png(image),
                capture_output=True,
                check=False,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            msg = "tesseract timed out"
            raise OcrError(msg) from exc
        if result.returncode != 0:
            msg = f"tesseract exited {result.returncode}"
            raise OcrError(msg)

        text = result.stdout.decode("utf-8", errors="replace")
        return tuple(
            OcrLine(text=line.strip(), confidence=1.0) for line in text.splitlines() if line.strip()
        )


def _encode_png(image: np.ndarray) -> bytes:
    """Encode a BGR array as PNG bytes, in memory.

    Isolated in its own function so the "no frame is ever written" guard has a
    single, obvious place to point at: this encodes to memory and hands the
    bytes to a subprocess on stdin. Nothing reaches the filesystem.
    """
    ok, buffer = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover - defensive
        msg = "could not encode the image for tesseract"
        raise OcrError(msg)
    return bytes(buffer)
