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

# Calls that would persist a frame or a video. Forbidden outside the preview
# stage, which is the one command whose whole purpose is a single still frame.
WRITE_CALLS = ("imwrite", "VideoWriter", "imencode")
WRITE_ALLOWLIST = {"stages/preview.py"}


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
def test_module_never_writes_images_outside_preview(module: Path) -> None:
    if _relpath(module) in WRITE_ALLOWLIST:
        pytest.skip("preview stage is the one place a still frame may be written")
    assert _offenders(module, WRITE_CALLS) == []


def test_the_check_itself_can_fail(tmp_path: Path) -> None:
    """A guard that cannot fail is not a guard."""
    bad = tmp_path / "bad.py"
    bad.write_text("import cv2\ncv2.imshow('x', frame)\n", encoding="utf-8")
    assert _offenders(bad, DISPLAY_CALLS) == ["imshow"]
