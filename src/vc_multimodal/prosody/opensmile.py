"""openSMILE eGeMAPS: the extension point, not an implementation.

eGeMAPSv02 is the standard acoustic set in affective computing, so being able
to report it would make this work directly comparable with that literature.
It is deliberately not implemented yet, for two reasons worth recording rather
than discovering later:

* openSMILE is an external binary that this project does not install, in the
  same way as ffmpeg and whisper-diarization. It has to be present before
  anything here can call it.
* eGeMAPSv02 is 88 features. With 62 sessions, adding it wholesale would blow
  the feature budget several times over (docs/decisions/0006) and guarantee
  overfitting. Using it means choosing a subset, or treating it as a separate
  pre-registered comparison, and that is a research decision rather than a
  coding one.

So this backend reports itself unavailable with that explanation, and the
interface it implements is the seam where the work would go.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np

from vc_multimodal.prosody.base import ProsodyBackend, ProsodyError

if TYPE_CHECKING:
    from vc_multimodal.config import OpenSmileConfig, ProsodyConfig
    from vc_multimodal.features.prosody_math import SpanMeasures


class OpenSmileBackend(ProsodyBackend):
    """Placeholder for an openSMILE feature extractor.

    Args:
        config: The openSMILE section, giving the feature set and executable.
    """

    name = "opensmile"

    def __init__(self, config: OpenSmileConfig) -> None:
        """Store the configuration; nothing is loaded."""
        self.config = config

    def _executable(self) -> str | None:
        """The configured executable, or one found on PATH."""
        if self.config.executable:
            return self.config.executable
        return shutil.which("SMILExtract")

    def available(self) -> bool:
        """Always False: the extractor is not implemented yet."""
        return False

    def unavailable_reason(self) -> str:
        """Why this backend cannot run, and what it would take."""
        found = self._executable()
        location = f"found at {found}" if found else "not found on PATH"
        return (
            f"the openSMILE backend is an extension point, not an implementation "
            f"(binary {location}). {self.config.feature_set} is 88 features, which "
            f"would overrun the feature budget for 62 sessions several times over, "
            f"so adopting it means choosing a subset first. See "
            f"src/vc_multimodal/prosody/opensmile.py."
        )

    def version(self) -> str:
        """Identifier recorded in the manifest."""
        return f"opensmile/not-implemented:{self.config.feature_set}"

    def measure(
        self,
        samples: np.ndarray,  # noqa: ARG002 - part of the interface
        sample_rate: int,  # noqa: ARG002 - part of the interface
        *,
        config: ProsodyConfig,  # noqa: ARG002 - part of the interface
    ) -> SpanMeasures:
        """Always raises: see `unavailable_reason`."""
        raise ProsodyError(self.unavailable_reason())
