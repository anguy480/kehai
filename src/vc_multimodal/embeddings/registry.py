"""Choosing a speaker-embedding backend."""

from __future__ import annotations

from pathlib import Path

from vc_multimodal.config import AppConfig
from vc_multimodal.embeddings.base import SpeakerEmbedder
from vc_multimodal.embeddings.ecapa import EcapaEmbedder


def get_embedder(config: AppConfig, work_root: Path) -> SpeakerEmbedder:
    """The configured embedding backend.

    There is one backend, and `speakers.embedding.backend` is a `Literal`, so
    the configuration cannot name another. A second backend adds a branch here
    and a member there; nothing validates the name at runtime because pydantic
    already has.
    """
    return EcapaEmbedder(config.speakers.embedding, work_root)
