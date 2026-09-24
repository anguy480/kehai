"""Face-landmark backends: MediaPipe by default, OpenFace CSV importer."""

from __future__ import annotations

from vc_multimodal.faces.base import FaceBackend, FaceError
from vc_multimodal.faces.mediapipe_backend import MediaPipeBackend
from vc_multimodal.faces.openface_backend import OpenFaceBackend
from vc_multimodal.faces.registry import (
    BACKEND_NAMES,
    MODELS_DIRNAME,
    get_backend,
    require_single_backend,
)

__all__ = [
    "BACKEND_NAMES",
    "MODELS_DIRNAME",
    "FaceBackend",
    "FaceError",
    "MediaPipeBackend",
    "OpenFaceBackend",
    "get_backend",
    "require_single_backend",
]
