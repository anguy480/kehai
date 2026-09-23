"""Backend selection from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vc_multimodal.prosody.base import ProsodyBackend, ProsodyError
from vc_multimodal.prosody.opensmile import OpenSmileBackend
from vc_multimodal.prosody.parselmouth_backend import ParselmouthBackend

if TYPE_CHECKING:
    from vc_multimodal.config import AppConfig

BACKEND_NAMES = ("parselmouth", "opensmile")


def get_backend(config: AppConfig) -> ProsodyBackend:
    """Build the prosody backend the configuration asks for.

    openSMILE is selected by enabling it explicitly, since it is an extension
    point rather than a working alternative.

    Raises:
        ProsodyError: if the chosen backend cannot run.
    """
    if config.prosody.opensmile.enabled:
        backend = OpenSmileBackend(config.prosody.opensmile)
        if not backend.available():
            raise ProsodyError(backend.unavailable_reason())
        return backend  # pragma: no cover - unreachable until implemented
    return ParselmouthBackend()
