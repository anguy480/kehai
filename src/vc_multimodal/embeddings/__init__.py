"""Speaker embeddings for role assignment."""

from __future__ import annotations

from vc_multimodal.embeddings.base import (
    REQUIRED_SAMPLE_RATE,
    EmbeddingError,
    SpeakerEmbedder,
    cosine_similarity,
)
from vc_multimodal.embeddings.registry import get_embedder

__all__ = [
    "REQUIRED_SAMPLE_RATE",
    "EmbeddingError",
    "SpeakerEmbedder",
    "cosine_similarity",
    "get_embedder",
]
