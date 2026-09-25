"""Speaker embeddings: the primary evidence for who is who.

Role assignment is the most consequential step in this pipeline. If the two
speakers are swapped in one session, that session's prosody describes the
psychiatrist, its speaking/listening windows invert, and every feature for that
participant is wrong while looking entirely normal. So the evidence has to be
strong, and where it is weak the session has to say so rather than guess.

An embedding maps a stretch of speech to a vector in which the same voice lands
near itself. Comparing each diarized speaker against a reference recording of
the psychiatrist then identifies the psychiatrist directly, rather than by
proxy.

Two properties of this problem shape the interface:

* **Only the ranking is meaningful.** Absolute cosine similarity varies with
  recording conditions - measured on this cohort, participants score 0.18-0.27
  against the psychiatrist reference in the summer wave and 0.46-0.60 in the
  winter wave, because the channels differ. A fixed threshold would therefore
  behave differently per wave. The decision is which speaker scores highest,
  and how far ahead of the next.
* **Availability is a first-class state.** The backend needs a model download
  and an optional dependency, so a machine without them must get a clear
  message rather than an import error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

#: Sample rate every backend here expects. Audio is resampled once, in
#: `vc extract-audio`, rather than silently per call.
REQUIRED_SAMPLE_RATE = 16_000


class EmbeddingError(RuntimeError):
    """Raised when speaker embeddings cannot be produced."""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings.

    Returns 0.0 for a zero vector rather than raising: a silent stretch of
    audio can produce one, and "no evidence" is the honest reading.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerEmbedder(ABC):
    """Turns speech into a vector in which the same voice lands near itself."""

    #: Short name, as used in configuration and recorded in QC.
    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can run right now. Never raises."""

    @abstractmethod
    def unavailable_reason(self) -> str:
        """Why the backend cannot run, empty when it can."""

    @abstractmethod
    def version(self) -> str:
        """Identifier recorded in the run manifest."""

    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed one stretch of mono speech.

        Args:
            audio: Mono samples in [-1, 1].
            sample_rate: Must be `REQUIRED_SAMPLE_RATE`.

        Raises:
            EmbeddingError: if the backend is unavailable, the rate is wrong,
                or the audio is too short to embed.
        """

    def require_available(self) -> None:
        """Raise unless the backend can run.

        Raises:
            EmbeddingError: with the reason and what to do about it.
        """
        if not self.available():
            raise EmbeddingError(self.unavailable_reason())

    def check_rate(self, sample_rate: int) -> None:
        """Refuse a rate the backend was not trained for.

        Raises:
            EmbeddingError: if the rate is wrong. Resampling here would hide a
                pipeline mistake and change the numbers.
        """
        if sample_rate != REQUIRED_SAMPLE_RATE:
            msg = (
                f"{self.name} expects {REQUIRED_SAMPLE_RATE} Hz audio, got {sample_rate} Hz. "
                f"`vc extract-audio` writes 16 kHz; resampling here would quietly change "
                f"the similarities."
            )
            raise EmbeddingError(msg)
