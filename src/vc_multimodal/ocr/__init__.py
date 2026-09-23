"""On-device OCR backends, used only to read Zoom name labels.

Recognised text is a person's name. It is compared in memory and never printed,
logged or written to any file. Nothing in this package returns text to a caller
that persists it; `vc verify-layout` reduces text to a side and a count before
anything leaves the process.
"""

from __future__ import annotations

from vc_multimodal.ocr.base import OcrBackend, OcrError, OcrLine
from vc_multimodal.ocr.registry import available_backends, get_backend

__all__ = [
    "OcrBackend",
    "OcrError",
    "OcrLine",
    "available_backends",
    "get_backend",
]
