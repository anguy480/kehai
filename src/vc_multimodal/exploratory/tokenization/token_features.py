# Vendored from analysis.py in https://github.com/hirokitan1155/virtual_client_SRS2_analysis
# at commit ec07010, under the following licence:
#
# MIT License
#
# Copyright (c) 2026 Hiroki Tanaka
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The manuscript's eight lemma-based text features, ported from analysis.py.

Only the token block of `analysis.py` (its lines 160-293) and the eight features
computed from it (lines 297-393) are kept: Jaccard, Cosine, Word_count,
Word_count_doc, Word_ratio, Mean_words_per_turn, MTLD_patient and MTLD_doctor.
Everything that reads `demo_2.csv`, computes a correlation, calls OpenAI or
encodes a sentence is left out, and so is the per-session word-list output.

One parameter is added, `min_len`: a kept lemma must have at least that many
characters. Its default, `ORIGINAL_MIN_LEN`, is the original rule
`len(token.base_form) > 2`; None removes the rule.

Everything else is kept as written, including choices we would make
differently, because the port is only useful if it reproduces
`nlp_features.csv`:

* A line counts if "Speaker 0" or "Speaker 1" appears anywhere in it, and its
  text keeps the trailing newline when it is tokenized.
* Jaccard and Cosine compare speaker 0 with speaker 1, not doctor with patient.
* Mean_words_per_turn divides kept lemmas by every patient line, including
  lines that kept none.
* MTLD is lexicalrichness's default, on the kept lemmas joined by spaces, and a
  failure is NaN.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from lexicalrichness import LexicalRichness

#: The original rule, `len(token.base_form) > 2`, as a minimum length.
ORIGINAL_MIN_LEN: Final = 3

#: Feature names as `nlp_features.csv` writes them, in its column order.
FEATURES: Final = (
    "Jaccard",
    "Cosine",
    "Word_count",
    "Word_count_doc",
    "Word_ratio",
    "MTLD_patient",
    "MTLD_doctor",
    "Mean_words_per_turn",
)

STOPWORDS: Final = frozenset(
    {
        "こと", "?", "もの", "ところ", "あっ", "なっ", "さ", "れ", "よう", "1", "ため",
        "の", "ん", "し", "それ", "これ", "あれ", "日", "今日", "中", "方", "時", "何",
        "いる", "れる", "られる", "しれる", "お話", "いただく", "できる", "お願い",
        "お話し", "いただける", "とかす", "いかが", "かなう",
    }
)  # fmt: skip

NORMALIZATION: Final = {"分かる": "わかる"}

#: Sessions where speaker 1, not speaker 0, is the doctor.
OPPOSITE_FILES: Final = frozenset(
    {
        6, 8, 10, 12, 18, 25, 30, 31, 32, 42, 44, 46, 50, 52, 57, 62,
        107, 113, 140, 144, 158, 194, 210, 212, 228, 245, 261,
    }
)  # fmt: skip

CONTENT_POS: Final = frozenset({"名詞", "動詞", "形容詞"})
EXCLUDED_POS2: Final = frozenset({"数"})


def doctor_speaker(session_id: int) -> int:
    """Which diarized speaker is the doctor in this session."""
    return 1 if session_id in OPPOSITE_FILES else 0


def kept_lemma(token: Any, min_len: int | None) -> str | None:
    """The normalised lemma if the token passes every filter, else None."""
    pos1, pos2 = token.part_of_speech.split(",")[:2]
    if pos1 not in CONTENT_POS or pos2 in EXCLUDED_POS2:
        return None
    base: str = token.base_form
    if base in STOPWORDS or token.surface in STOPWORDS:
        return None
    if min_len is not None and len(base) < min_len:
        return None
    return NORMALIZATION.get(base, base)


@dataclass(frozen=True, slots=True)
class SessionTokens:
    """One session's eight features, and its kept lemmas for aggregate counts.

    The lemmas stay in memory: nothing about one session's words is written.
    """

    session_id: int
    features: dict[str, float]
    lemmas: Counter[str] = field(default_factory=Counter)

    @property
    def n_lemmas(self) -> int:
        """Kept lemmas, both speakers."""
        return sum(self.lemmas.values())


def _mtld(words: list[str]) -> float:
    """MTLD from lexicalrichness as analysis.py calls it, NaN on failure."""
    try:
        return float(LexicalRichness(" ".join(words)).mtld())
    except Exception:  # analysis.py catches everything here
        return float("nan")


def session_tokens(
    lines: Iterable[str], session_id: int, tokenizer: Any, min_len: int | None
) -> SessionTokens:
    """The eight features for one transcript, as analysis.py computes them.

    Args:
        lines: The transcript's lines, newlines included, as a text file
            iterates them.
        session_id: The file's stem.
        tokenizer: A `janome.tokenizer.Tokenizer`.
        min_len: Fewest characters a kept lemma may have; None for no rule.
    """
    doctor_id = doctor_speaker(session_id)
    by_speaker: dict[int, list[str]] = {0: [], 1: []}
    patient: list[str] = []
    doctor: list[str] = []
    patient_turns = 0

    for line in lines:
        if "Speaker 0" not in line and "Speaker 1" not in line:
            continue
        speaker_id = int(line.split(":")[0].replace("Speaker ", ""))
        text = line.split(":", 1)[1].replace(" ", "").replace("　", "")
        if speaker_id != doctor_id:
            patient_turns += 1
        for token in tokenizer.tokenize(text):
            lemma = kept_lemma(token, min_len)
            if lemma is None:
                continue
            by_speaker[0 if speaker_id == 0 else 1].append(lemma)
            (patient if speaker_id != doctor_id else doctor).append(lemma)

    counter0, counter1 = Counter(by_speaker[0]), Counter(by_speaker[1])
    set0, set1 = set(counter0), set(counter1)
    jaccard = len(set1 & set0) / len(set1 | set0)
    vocab = sorted(set0 | set1)
    vec0 = np.array([counter0[w] for w in vocab])
    vec1 = np.array([counter1[w] for w in vocab])
    cosine = float(np.dot(vec0, vec1) / (np.linalg.norm(vec0) * np.linalg.norm(vec1)))

    word_count, word_count_doc = len(patient), len(doctor)
    features = {
        "Jaccard": jaccard,
        "Cosine": cosine,
        "Word_count": float(word_count),
        "Word_count_doc": float(word_count_doc),
        "Word_ratio": word_count / (word_count + word_count_doc),
        "MTLD_patient": _mtld(patient),
        "MTLD_doctor": _mtld(doctor),
        "Mean_words_per_turn": (word_count / patient_turns if patient_turns > 0 else float("nan")),
    }
    return SessionTokens(session_id, features, counter0 + counter1)


def transcript_tokens(path: Path, tokenizer: Any, min_len: int | None) -> SessionTokens:
    """Read one transcript as analysis.py does (UTF-8 with BOM) and compute it."""
    with path.open("r", encoding="utf-8-sig") as handle:
        return session_tokens(handle, int(path.stem), tokenizer, min_len)
