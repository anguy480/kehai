"""The OCR backends and their registry.

The recognition engines themselves are not this project's code, so what matters
here is the contract around them: an unavailable backend must be a reportable
state rather than a crash, because the layout check falls back to the assumed
side when OCR cannot run.
"""

from __future__ import annotations

import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.synth.fake_ocr import FakeOcr
from vc_multimodal.ocr import OcrError, available_backends, get_backend
from vc_multimodal.ocr.apple_vision import AppleVisionOcr
from vc_multimodal.ocr.base import OcrBackend, OcrLine
from vc_multimodal.ocr.registry import NullOcr
from vc_multimodal.ocr.tesseract import TesseractOcr, _encode_png


def _label_image(text: str = "DR SATO", width: int = 360) -> np.ndarray:
    image = np.full((70, width, 3), 25, dtype=np.uint8)
    cv2.putText(image, text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
    return image


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["apple_vision", "tesseract", "none"])
def test_every_configurable_backend_can_be_built(name: str):
    backend = get_backend(name)
    assert isinstance(backend, OcrBackend)
    assert backend.name == name


def test_an_unknown_backend_names_the_alternatives():
    with pytest.raises(OcrError, match="unknown OCR backend"):
        get_backend("ocropus")


def test_available_backends_is_a_subset_of_the_configurable_ones():
    assert set(available_backends()) <= {"apple_vision", "tesseract", "none"}


def test_availability_never_raises():
    """Callers branch on availability, so it must not be an error path."""
    for name in ("apple_vision", "tesseract", "none"):
        assert isinstance(get_backend(name).available(), bool)


# ---------------------------------------------------------------------------
# the null backend
# ---------------------------------------------------------------------------
def test_the_null_backend_is_never_available():
    backend = NullOcr()
    assert not backend.available()
    assert "disabled" in backend.unavailable_reason()
    assert backend.version() == "none"


def test_the_null_backend_refuses_to_read():
    with pytest.raises(OcrError, match="disabled"):
        NullOcr().read(_label_image(), languages=("en-US",))


# ---------------------------------------------------------------------------
# tesseract
# ---------------------------------------------------------------------------
def test_tesseract_availability_follows_the_binary():
    assert TesseractOcr().available() == (shutil.which("tesseract") is not None)


@pytest.mark.skipif(shutil.which("tesseract") is not None, reason="tesseract is installed")
def test_tesseract_explains_itself_when_absent():
    backend = TesseractOcr()
    assert "not on PATH" in backend.unavailable_reason()
    assert "does not install it" in backend.unavailable_reason()
    assert backend.version() == "unavailable"
    with pytest.raises(OcrError, match="not on PATH"):
        backend.read(_label_image(), languages=("en-US",))


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed")
def test_tesseract_reads_a_label():  # pragma: no cover - depends on the machine
    lines = TesseractOcr().read(_label_image(), languages=("en-US",))
    assert any("SATO" in line.text.upper() for line in lines)


def test_png_encoding_happens_in_memory(tmp_path: Path):
    """The frame is piped to the binary on stdin; nothing is written to disk."""
    before = set(tmp_path.iterdir())
    encoded = _encode_png(_label_image())
    assert encoded.startswith(b"\x89PNG")
    assert set(tmp_path.iterdir()) == before


# ---------------------------------------------------------------------------
# apple vision
# ---------------------------------------------------------------------------
def test_apple_vision_availability_matches_the_platform():
    assert AppleVisionOcr().available() == (
        sys.platform == "darwin" and AppleVisionOcr().unavailable_reason() == ""
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="this is macOS")
def test_apple_vision_explains_itself_off_macos():  # pragma: no cover - CI only
    backend = AppleVisionOcr()
    assert "macOS-only" in backend.unavailable_reason()
    with pytest.raises(OcrError, match="macOS-only"):
        backend.read(_label_image(), languages=("en-US",))


@pytest.mark.skipif(not AppleVisionOcr().available(), reason="the Vision framework is unavailable")
class TestAppleVisionOnThisMachine:
    """Only runs where the framework is actually present."""

    def test_it_reads_a_drawn_label(self):
        lines = AppleVisionOcr().read(_label_image(), languages=("en-US",))
        assert any("SATO" in line.text.upper() for line in lines)
        assert all(0.0 <= line.confidence <= 1.0 for line in lines)

    def test_it_reports_a_version_for_the_manifest(self):
        assert AppleVisionOcr().version().startswith("apple-vision/macos-")

    def test_an_empty_image_reads_as_nothing(self):
        assert AppleVisionOcr().read(np.zeros((0, 0, 3), np.uint8), languages=()) == ()

    def test_a_blank_image_reads_as_nothing(self):
        blank = np.full((60, 200, 3), 30, np.uint8)
        assert AppleVisionOcr().read(blank, languages=("en-US",)) == ()

    def test_no_languages_is_accepted(self):
        assert isinstance(AppleVisionOcr().read(_label_image(), languages=()), tuple)

    def test_concurrent_reads_do_not_race(self):
        """pyobjc resolves framework attributes lazily, and not thread-safely."""
        backend = AppleVisionOcr()

        def read(index: int) -> tuple[OcrLine, ...]:
            return backend.read(_label_image(f"DR SATO {index}"), languages=("en-US",))

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read, range(16)))

        assert all(results), "every concurrent read should return at least one line"


# ---------------------------------------------------------------------------
# the fake, which the layout tests depend on
# ---------------------------------------------------------------------------
def test_the_fake_backend_honours_its_script():
    backend = FakeOcr({360: [("DR SATO", 0.9)]}, default=[("OTHER", 0.4)])
    assert backend.read(_label_image(width=360), languages=())[0].text == "DR SATO"
    assert backend.read(_label_image(width=200), languages=())[0].text == "OTHER"
    assert backend.calls == 2


def test_the_fake_backend_can_be_unavailable():
    backend = FakeOcr(is_available=False)
    with pytest.raises(OcrError):
        backend.read(_label_image(), languages=())
    assert backend.calls == 0


def test_the_fake_backend_can_fail_partway():
    backend = FakeOcr(default=[("X", 1.0)], fail_after=1)
    assert backend.read(_label_image(), languages=())
    with pytest.raises(OcrError, match="scripted"):
        backend.read(_label_image(), languages=())
