"""Face backend selection, and the rule against mixing them."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from vc_multimodal.faces.base import FaceBackend, FaceError
from vc_multimodal.faces.mediapipe_backend import MediaPipeBackend
from vc_multimodal.faces.openface_backend import OpenFaceBackend

if TYPE_CHECKING:
    from vc_multimodal.config import AppConfig
    from vc_multimodal.paths import DataRoots

BACKEND_NAMES: Final = ("mediapipe", "openface")

MODELS_DIRNAME: Final = "models"


def get_backend(config: AppConfig, roots: DataRoots) -> FaceBackend:
    """Build the configured face backend.

    Raises:
        FaceError: if the configured name is unknown.
    """
    name = config.face.backend
    if name == "mediapipe":
        return MediaPipeBackend(
            config.face.mediapipe,
            model_dir=roots.work_path(MODELS_DIRNAME, create_parent=False),
        )
    if name == "openface":
        csv_dir = config.face.openface.csv_dir
        return OpenFaceBackend(
            config.face.openface,
            csv_dir=_under_work(roots, csv_dir) if csv_dir else None,
        )
    msg = f"unknown face backend {name!r}; available: {list(BACKEND_NAMES)}"  # type: ignore[unreachable]
    raise FaceError(msg)


def _under_work(roots: DataRoots, relative: str) -> Path:
    """Resolve a configured directory beneath the work root."""
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    return roots.work_path(relative, create_parent=False)


def require_single_backend(backends: Sequence[str], *, context: str = "") -> str:
    """Check that every session was measured with the same backend.

    MediaPipe blendshape scores and OpenFace action unit intensities are
    different scales for the same constructs, so a table holding both describes
    nothing (docs/decisions/0013). Switching backend means a full rerun, not a
    top-up, and this is what makes the difference between the two impossible to
    miss.

    Args:
        backends: The backend recorded for each session.
        context: Added to the error message.

    Returns:
        The single backend name, or an empty string for no sessions.

    Raises:
        FaceError: if more than one backend is present.
    """
    present = sorted({name for name in backends if name})
    if not present:
        return ""
    if len(present) > 1:
        where = f" in {context}" if context else ""
        msg = (
            f"facial measurements{where} come from more than one backend: {present}. "
            f"MediaPipe blendshape scores and OpenFace action unit intensities are "
            f"different scales for the same constructs, so the two cannot be pooled. "
            f"Re-extract every session with one backend: `vc face --force`. Switching "
            f"backend is a full rerun, never a top-up."
        )
        raise FaceError(msg)
    return present[0]
