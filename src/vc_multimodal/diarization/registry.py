"""Backend selection from configuration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vc_multimodal.diarization.base import DiarizationBackend, DiarizationError
from vc_multimodal.diarization.import_backend import ImportBackend
from vc_multimodal.diarization.pyannote_backend import PyannoteBackend
from vc_multimodal.diarization.whisper_diarization import WhisperDiarizationBackend

if TYPE_CHECKING:
    from vc_multimodal.config import AppConfig
    from vc_multimodal.paths import DataRoots

BACKEND_NAMES = ("import", "pyannote", "whisper_diarization")


def get_backend(config: AppConfig, roots: DataRoots) -> DiarizationBackend:
    """Build the configured diarization backend.

    Paths in configuration are relative to `$VC_WORK_ROOT`, so that no absolute
    path to clinical data is ever committed.

    Raises:
        DiarizationError: if the configured backend name is unknown.
        ConfigError: if the chosen backend is missing required configuration.
    """
    name = config.diarization.backend

    if name == "import":
        import_dir = config.diarization.require_import_dir()
        return ImportBackend(
            _under_work(roots, import_dir),
            config.diarization.import_patterns,
            keep_text=config.diarization.keep_text,
        )

    if name == "pyannote":
        return PyannoteBackend(config.diarization.pyannote)

    if name == "whisper_diarization":
        output_dir = config.diarization.whisper_diarization.output_dir
        return WhisperDiarizationBackend(
            config.diarization.whisper_diarization,
            output_dir=_under_work(roots, output_dir) if output_dir else None,
            patterns=config.diarization.import_patterns,
            keep_text=config.diarization.keep_text,
        )

    msg = f"unknown diarization backend {name!r}; available: {list(BACKEND_NAMES)}"  # type: ignore[unreachable]
    raise DiarizationError(msg)


def _under_work(roots: DataRoots, relative: str) -> Path:
    """Resolve a configured directory beneath the work root."""
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    return roots.work_path(relative, create_parent=False)
