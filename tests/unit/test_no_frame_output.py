"""Guard the 'never save frames or annotated video' rule with a static check.

`cv2` in this environment comes from `opencv-contrib-python` (a MediaPipe
requirement), so it is GUI-capable and `imshow` exists. Packaging cannot
enforce the rule for us, so it is enforced here: no module under `src/` may
display a frame or write image/video output, with one deliberate exception.

`vc preview` exists precisely to write one cropped frame per session to
`$VC_OUT_ROOT/previews` for visual confirmation of the participant tile, so
`stages/preview.py` is allowed to write images. It must still never display one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "vc_multimodal"

# Calls that would put a frame on screen. Forbidden everywhere.
DISPLAY_CALLS = ("imshow", "namedWindow", "waitKey", "startWindowThread")

# Calls that would persist a frame or a video. Forbidden everywhere except the
# specific call each module below is allowed, so widening one exception cannot
# quietly widen the others.
WRITE_CALLS = ("imwrite", "VideoWriter", "imencode")
WRITE_ALLOWLIST: dict[str, frozenset[str]] = {
    # `vc preview` exists to write one still frame per session for a human to
    # look at. It may write an image; it may not encode one for anything else.
    "stages/preview.py": frozenset({"imwrite"}),
    # The tesseract backend encodes a frame to PNG bytes IN MEMORY and hands
    # them to the binary on stdin. Nothing reaches the filesystem. It may not
    # call imwrite.
    "ocr/tesseract.py": frozenset({"imencode"}),
}


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _relpath(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _offenders(path: Path, calls: tuple[str, ...]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # Strip comments so that prose about these calls (including this rule being
    # documented in code) does not trip the check.
    code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return [call for call in calls if re.search(rf"\b{call}\s*\(", code)]


@pytest.mark.parametrize("module", _modules(), ids=_relpath)
def test_module_never_displays_a_frame(module: Path) -> None:
    assert _offenders(module, DISPLAY_CALLS) == []


@pytest.mark.parametrize("module", _modules(), ids=_relpath)
def test_module_only_makes_the_image_output_call_it_is_allowed(module: Path) -> None:
    allowed = WRITE_ALLOWLIST.get(_relpath(module), frozenset())
    forbidden = tuple(call for call in WRITE_CALLS if call not in allowed)
    assert _offenders(module, forbidden) == []


def test_the_allowlist_only_covers_modules_that_exist() -> None:
    """A stale entry would silently permit image output in a renamed module."""
    existing = {_relpath(path) for path in _modules()}
    assert set(WRITE_ALLOWLIST) <= existing


@pytest.mark.parametrize("module", sorted(WRITE_ALLOWLIST), ids=lambda name: name)
def test_each_allowlisted_module_actually_makes_its_allowed_call(module: str) -> None:
    """Otherwise the exception outlives the reason for it."""
    path = SRC / module
    assert set(_offenders(path, WRITE_CALLS)) == set(WRITE_ALLOWLIST[module])


def test_the_check_itself_can_fail(tmp_path: Path) -> None:
    """A guard that cannot fail is not a guard."""
    bad = tmp_path / "bad.py"
    bad.write_text("import cv2\ncv2.imshow('x', frame)\n", encoding="utf-8")
    assert _offenders(bad, DISPLAY_CALLS) == ["imshow"]
