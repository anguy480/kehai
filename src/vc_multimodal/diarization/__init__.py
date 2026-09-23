"""Diarization backends behind a single normalized interface."""

from __future__ import annotations

from vc_multimodal.diarization.base import (
    DiarizationBackend,
    DiarizationError,
    Segment,
    canonical_speaker,
    covered_time,
    overlap_time,
    sort_segments,
    speakers_in,
    strip_text,
    total_speech,
)
from vc_multimodal.diarization.import_backend import ImportBackend, ImportScan, scan_import_dir
from vc_multimodal.diarization.pyannote_backend import PyannoteBackend
from vc_multimodal.diarization.registry import BACKEND_NAMES, get_backend
from vc_multimodal.diarization.whisper_diarization import WhisperDiarizationBackend

__all__ = [
    "BACKEND_NAMES",
    "DiarizationBackend",
    "DiarizationError",
    "ImportBackend",
    "ImportScan",
    "PyannoteBackend",
    "Segment",
    "WhisperDiarizationBackend",
    "canonical_speaker",
    "covered_time",
    "get_backend",
    "overlap_time",
    "scan_import_dir",
    "sort_segments",
    "speakers_in",
    "strip_text",
    "total_speech",
]
