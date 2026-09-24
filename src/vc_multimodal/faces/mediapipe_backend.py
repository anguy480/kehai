"""Facial measurement with the MediaPipe Face Landmarker.

The default backend, because it runs on the development machine with no
external tool. It is the fragile half of the pair: version 1.0.1 aborts the
process on macOS arm64 inside the detector subgraph, so the dependency is
pinned below 1.0 (docs/decisions/0003).

Frames are decoded, measured and discarded. `grab()` is used to step over the
frames that are not sampled, which skips their decode entirely, and only a
sampled frame is retrieved. No frame is written anywhere
(tests/unit/test_no_frame_output.py enforces that statically).

Blendshape scores are not action unit intensities. They are recorded under AU
names because they measure the same constructs, with the backend recorded
alongside so the two are never pooled.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np

from vc_multimodal.faces.base import FaceBackend, FaceError
from vc_multimodal.features.face_math import (
    PRESENT,
    FrameMeasure,
    combine_blendshapes,
    head_pose_from_matrix,
)
from vc_multimodal.logging_setup import get_logger

if TYPE_CHECKING:
    from vc_multimodal.config import AppConfig, CropBox, MediaPipeConfig
    from vc_multimodal.features.sampling import FrameSampling
    from vc_multimodal.paths import RawSession

logger = get_logger(__name__)

_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - import success depends on the platform
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
# Any import failure at all means the backend is unavailable.
except Exception as exc:
    _IMPORT_ERROR = exc

# A crop smaller than this in either direction is not a face to measure.
_MIN_CROP_PIXELS: Final = 16


class MediaPipeBackend(FaceBackend):
    """Measures action units from MediaPipe blendshapes.

    Args:
        config: The mediapipe section, giving the model asset and its hash.
        model_dir: Where the model asset lives, under the work root.
    """

    name = "mediapipe"

    def __init__(self, config: MediaPipeConfig, *, model_dir: Path) -> None:
        """Store configuration; the model is loaded on first use."""
        self.config = config
        self.model_dir = Path(model_dir)
        self._landmarker: Any | None = None

    @property
    def model_path(self) -> Path:
        """Where the model asset is expected."""
        return self.model_dir / self.config.model_asset

    def available(self) -> bool:
        """Whether the bindings imported and the model asset is present."""
        return _IMPORT_ERROR is None and self.model_path.is_file()

    def unavailable_reason(self) -> str:
        """Why the backend cannot run."""
        if _IMPORT_ERROR is not None:
            return f"mediapipe is not importable ({type(_IMPORT_ERROR).__name__})"
        if not self.model_path.is_file():
            return (
                f"the face landmarker model is not at {self.model_path}. Download it "
                f"from {self.config.model_url} into that directory; it is pinned by "
                f"hash in config and is not fetched automatically, so a run cannot "
                f"silently pick up a different model."
            )
        return ""

    def version(self) -> str:
        """Library version and the model's hash, both pinned in the manifest."""
        library = "unknown" if _IMPORT_ERROR is not None else mp.__version__
        return f"mediapipe/{library}+model:{self.model_digest()[:16]}"

    def model_digest(self) -> str:
        """SHA-256 of the model asset on disk."""
        if not self.model_path.is_file():
            return "absent"
        return hashlib.sha256(self.model_path.read_bytes()).hexdigest()

    def verify_model(self) -> None:
        """Check the model against its pinned hash.

        Raises:
            FaceError: if the model is missing or does not match.
        """
        if not self.available():
            raise FaceError(self.unavailable_reason())
        expected = self.config.model_sha256
        if expected is None:
            logger.warning(
                "the face landmarker model is not pinned by hash; set "
                "face.mediapipe.model_sha256 so a changed model cannot pass unnoticed"
            )
            return
        actual = self.model_digest()
        if actual != expected:
            msg = (
                f"the face landmarker model at {self.model_path} has hash {actual}, "
                f"but config pins {expected}. Features from a different model are not "
                f"comparable with features already extracted."
            )
            raise FaceError(msg)

    def landmarker(self, config: AppConfig) -> Any:
        """Create the landmarker once, on first use."""
        if self._landmarker is not None:
            return self._landmarker
        self.verify_model()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=self.config.head_pose,
            num_faces=1,
            # The detector applies these itself and returns nothing below them,
            # so the configured threshold is enforced here rather than by
            # filtering a score afterwards.
            min_face_detection_confidence=config.face.min_confidence,
            min_face_presence_confidence=config.face.min_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        return self._landmarker

    def close(self) -> None:
        """Release the landmarker."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def measure_session(
        self,
        session: RawSession,
        *,
        config: AppConfig,
        crop: CropBox,
        sampling: FrameSampling,
    ) -> tuple[FrameMeasure, ...]:
        """Decode, sample, crop and measure one session.

        Raises:
            FaceError: if the video cannot be opened or the model is wrong.
        """
        landmarker = self.landmarker(config)
        capture = cv2.VideoCapture(str(session.path))
        if not capture.isOpened():
            msg = f"OpenCV could not open {session.path.name}"
            raise FaceError(msg)

        measures: list[FrameMeasure] = []
        try:
            frame_index = 0
            while True:
                # grab() advances without fully decoding, so the frames between
                # samples cost almost nothing.
                if not capture.grab():
                    break
                if frame_index % sampling.step == 0:
                    ok, frame = capture.retrieve()
                    if ok and frame is not None:
                        measures.append(
                            self._measure_frame(
                                frame,
                                landmarker,
                                config=config,
                                crop=crop,
                                frame_index=frame_index,
                                timestamp_s=sampling.timestamp(frame_index),
                            )
                        )
                    else:
                        measures.append(
                            FrameMeasure.missing(frame_index, sampling.timestamp(frame_index))
                        )
                frame_index += 1
        finally:
            capture.release()

        if not measures:
            msg = f"no frames were decoded from {session.path.name}"
            raise FaceError(msg)
        return tuple(measures)

    def _measure_frame(
        self,
        frame: cv2.typing.MatLike,
        landmarker: Any,
        *,
        config: AppConfig,
        crop: CropBox,
        frame_index: int,
        timestamp_s: float,
    ) -> FrameMeasure:
        """Measure one decoded frame, cropped to the participant's tile."""
        height, width = frame.shape[:2]
        left, top, box_w, box_h = crop.to_pixels(width, height)
        if box_w < _MIN_CROP_PIXELS or box_h < _MIN_CROP_PIXELS:
            return FrameMeasure.missing(frame_index, timestamp_s)

        tile = frame[top : top + box_h, left : left + box_w]
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        )
        result = landmarker.detect(image)
        if not result.face_blendshapes:
            return FrameMeasure.missing(frame_index, timestamp_s)

        scores = {
            category.category_name: float(category.score) for category in result.face_blendshapes[0]
        }
        face = config.face
        units = {
            unit.key: combine_blendshapes(scores, unit.blendshapes) for unit in face.action_units
        }
        head = None
        if self.config.head_pose and result.facial_transformation_matrixes:
            head = head_pose_from_matrix(np.asarray(result.facial_transformation_matrixes[0]))

        return FrameMeasure(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            detected=True,
            # MediaPipe filters by its own thresholds and reports no score, so
            # a returned face is presence rather than a graded confidence.
            confidence=PRESENT,
            units=units,
            jaw=combine_blendshapes(scores, [face.jaw_blendshape]),
            blink=combine_blendshapes(scores, face.blink_blendshapes),
            head=head,
        )
