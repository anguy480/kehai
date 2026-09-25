"""ECAPA-TDNN speaker embeddings, via SpeechBrain.

Chosen because it is the standard open speaker-verification model, runs on CPU
at a speed that does not matter next to video decoding, needs no access token,
and is pinned to an exact model revision so the numbers are reproducible.

Validated on this cohort before being adopted, with controls in both
directions: two halves of one reference clip score 0.72-0.79 against each
other, while a psychiatrist and a participant from the same session score
0.18-0.68. The overlap at the top of that second range is why only the ranking
is used, never an absolute threshold.

The model is cached under `$VC_WORK_ROOT`, never in the repository and never in
a user-wide cache, so that the project's data roots hold everything the run
depends on.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Final

import numpy as np

from vc_multimodal.config import SpeakerEmbeddingConfig
from vc_multimodal.embeddings.base import EmbeddingError, SpeakerEmbedder
from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

NAME: Final = "ecapa"

#: Shortest stretch the model is given. Below about a second an embedding is
#: dominated by whatever phonemes happen to be in it rather than by the voice.
MIN_DURATION_S: Final = 0.5


class EcapaEmbedder(SpeakerEmbedder):
    """SpeechBrain's ECAPA-TDNN, loaded once and reused."""

    name = NAME

    def __init__(self, config: SpeakerEmbeddingConfig, work_root: Path) -> None:
        """Prepare the backend; the model itself loads on first use."""
        self._config = config
        self._work_root = work_root
        self._model: Any | None = None
        self._reason = ""

    # -- availability ----------------------------------------------------
    def available(self) -> bool:
        """Whether SpeechBrain and torch can be imported."""
        if self._model is not None:
            return True
        try:
            import speechbrain  # noqa: F401, PLC0415
            import torch  # noqa: F401, PLC0415
        except ImportError as exc:
            self._reason = (
                f"speaker embeddings need the optional `speaker` extra "
                f"(`uv sync --extra speaker`): {exc}"
            )
            return False
        return True

    def unavailable_reason(self) -> str:
        """Why the backend cannot run, empty when it can."""
        if self.available():
            return ""
        return self._reason

    def version(self) -> str:
        """The model source and the exact revision it is pinned to."""
        revision = self._config.revision or "unpinned"
        return f"{NAME}:{self._config.source}@{revision}"

    def model_digest(self) -> str | None:
        """SHA-256 of the model weights, once they are on disk.

        Recorded in the manifest so a bundle names the weights that produced
        its numbers, not just the name of a model that may have moved.
        """
        weights = self._savedir() / "embedding_model.ckpt"
        if not weights.exists():
            return None
        digest = hashlib.sha256()
        with weights.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    # -- loading ---------------------------------------------------------
    def _savedir(self) -> Path:
        return self._work_root / self._config.cache_dir

    def _load(self) -> Any:
        """Load the model, downloading it under the work root if needed."""
        if self._model is not None:
            return self._model
        self.require_available()

        from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415

        savedir = self._savedir()
        savedir.parent.mkdir(parents=True, exist_ok=True)
        # Keep the Hugging Face cache inside the project's work root: the run
        # should depend on nothing outside the three configured data roots.
        os.environ.setdefault("HF_HOME", str(self._work_root / self._config.hf_home))

        kwargs: dict[str, Any] = {
            "source": self._config.source,
            "savedir": str(savedir),
            "run_opts": {"device": "cpu"},
        }
        if self._config.revision:
            kwargs["revision"] = self._config.revision

        logger.info("loading %s", self.version())
        try:
            self._model = EncoderClassifier.from_hparams(**kwargs)
        except Exception as exc:  # torch and the model hub raise broadly
            msg = (
                f"could not load {self._config.source}: {exc}. The model is downloaded "
                f"once into $VC_WORK_ROOT/{self._config.cache_dir}; check the network, "
                f"or point `speakers.embedding.cache_dir` at an existing copy."
            )
            raise EmbeddingError(msg) from exc
        return self._model

    # -- embedding -------------------------------------------------------
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed one stretch of mono speech.

        Raises:
            EmbeddingError: if the backend is unavailable, the rate is wrong,
                or the audio is shorter than `MIN_DURATION_S`.
        """
        self.check_rate(sample_rate)
        duration = len(audio) / float(sample_rate)
        if duration < MIN_DURATION_S:
            msg = (
                f"{duration:.2f}s of audio is too short to embed; at least "
                f"{MIN_DURATION_S}s is needed for the vector to describe a voice "
                f"rather than a phoneme"
            )
            raise EmbeddingError(msg)

        import torch  # noqa: PLC0415

        model = self._load()
        samples = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        with torch.no_grad():
            embedded = model.encode_batch(samples.unsqueeze(0))
        return np.asarray(embedded.squeeze().cpu().numpy(), dtype=np.float64)
