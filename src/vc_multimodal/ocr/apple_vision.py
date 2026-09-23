"""OCR through the macOS Vision framework.

Chosen as the default because it is on-device, needs no model download and no
external binary, handles Japanese, and runs natively on Apple silicon. It is
macOS-only, so it lives behind the optional `ocr` extra and reports itself
unavailable elsewhere; `vc verify-layout` then falls back to the assumed side.

Frames are converted to a CGImage from raw bytes in memory. No image is written
to disk and no image is even encoded, which keeps the "never save a frame" rule
intact for a stage that has to look at pixels.
"""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING, Any

from vc_multimodal.ocr.base import OcrBackend, OcrError, OcrLine

if TYPE_CHECKING:
    import numpy as np

_IMPORT_ERROR: Exception | None = None

# pyobjc resolves framework attributes lazily, on first access, and that
# resolution is NOT thread-safe: concurrent workers raced and raised
# `KeyError: 'CGColorSpaceCreateDeviceRGB'`. Everything needed is therefore
# bound once here, at import, while the process is still single-threaded.
try:  # pragma: no cover - import success depends on the platform
    import cv2
    import Quartz
    import Vision
    from Foundation import NSData

    _cg_data_provider = Quartz.CGDataProviderCreateWithCFData
    _cg_image_create = Quartz.CGImageCreate
    _cg_device_rgb = Quartz.CGColorSpaceCreateDeviceRGB
    _cg_bitmap_info = Quartz.kCGImageAlphaPremultipliedLast | Quartz.kCGBitmapByteOrderDefault
    _cg_rendering_intent = Quartz.kCGRenderingIntentDefault
    _ns_data = NSData
    _image_request_handler = Vision.VNImageRequestHandler
    _recognize_text_request = Vision.VNRecognizeTextRequest
    _recognition_level_accurate = Vision.VNRequestTextRecognitionLevelAccurate
# Any import failure at all means the backend is simply unavailable.
except Exception as exc:
    _IMPORT_ERROR = exc


_BITS_PER_COMPONENT = 8
_BITS_PER_PIXEL = 32
_BYTES_PER_PIXEL = 4


class AppleVisionOcr(OcrBackend):
    """Text recognition through `VNRecognizeTextRequest`."""

    name = "apple_vision"

    def available(self) -> bool:
        """Whether this is macOS and the Vision bindings imported."""
        return sys.platform == "darwin" and _IMPORT_ERROR is None

    def unavailable_reason(self) -> str:
        """Why the backend cannot run, for a report rather than an exception."""
        if sys.platform != "darwin":
            return f"the Vision framework is macOS-only (this is {sys.platform})"
        if _IMPORT_ERROR is not None:
            return (
                f"the Vision bindings are not installed ({type(_IMPORT_ERROR).__name__}); "
                f"install the optional extra with `uv sync --extra ocr`"
            )
        return ""

    def version(self) -> str:
        """Platform identifier, since Vision ships with the OS."""
        return f"apple-vision/macos-{platform.mac_ver()[0] or 'unknown'}"

    def _cg_image(self, image: np.ndarray) -> Any:
        """Wrap a BGR array as a CGImage without touching the filesystem."""
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        height, width = rgba.shape[:2]
        data = _ns_data.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes)
        provider = _cg_data_provider(data)
        return _cg_image_create(
            width,
            height,
            _BITS_PER_COMPONENT,
            _BITS_PER_PIXEL,
            width * _BYTES_PER_PIXEL,
            _cg_device_rgb(),
            _cg_bitmap_info,
            provider,
            None,
            False,
            _cg_rendering_intent,
        )

    def read(self, image: np.ndarray, *, languages: tuple[str, ...]) -> tuple[OcrLine, ...]:
        """Recognise text in a BGR image.

        Raises:
            OcrError: if the backend is unavailable or Vision reports a failure.
        """
        if not self.available():
            raise OcrError(self.unavailable_reason())
        if image.size == 0:
            return ()

        cg_image = self._cg_image(image)
        if cg_image is None:  # pragma: no cover - defensive
            msg = "could not build a CGImage from the frame"
            raise OcrError(msg)

        handler = _image_request_handler.alloc().initWithCGImage_options_(cg_image, None)
        request = _recognize_text_request.alloc().init()
        request.setRecognitionLevel_(_recognition_level_accurate)
        if languages:
            request.setRecognitionLanguages_(list(languages))

        ok, error = handler.performRequests_error_([request], None)
        if not ok:  # pragma: no cover - depends on the platform
            msg = f"Vision text recognition failed: {error}"
            raise OcrError(msg)

        lines: list[OcrLine] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:  # pragma: no cover - defensive
                continue
            best = candidates[0]
            lines.append(OcrLine(text=str(best.string()), confidence=float(best.confidence())))
        return tuple(lines)
