"""Recompute the eight lemma-based text features, with and without the length rule.

Run where the transcripts are, by the person allowed to read them. It reads the
`.txt` transcripts `analysis.py` read and the existing `nlp_features.csv`, and:

1. Computes variant (a), the original rule, and checks it reproduces the eight
   columns of `nlp_features.csv`, row for row. If it does not, it stops: nothing
   is written and variant (b) is not computed.
2. Computes variant (b), with no length rule.
3. Writes four tables under the output root, each with a `session_id` taken
   from the transcript file names: the eight features under each variant, and
   the full 20 under each, the twelve features that do not use lemmas copied
   unchanged from `nlp_features.csv`. Rows are matched to `nlp_features.csv` by
   position, which the check in step 1 has just verified.

What it prints is aggregate only: counts, vocabulary sizes and the most
frequent recovered lemmas across all sessions. Nothing about one session's
words is written or printed. No label is read.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from vc_multimodal.exploratory.tokenization.token_features import (
    FEATURES,
    ORIGINAL_MIN_LEN,
    SessionTokens,
    transcript_tokens,
)
from vc_multimodal.io_utils import write_csv
from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

STAGE: Final = "tokenization"

#: `nlp_features.csv`'s columns, in the order analysis.py writes them.
NLP_COLUMNS: Final = (
    "Jaccard",
    "Cosine",
    "Bert",
    "Bert_turn",
    "Bert_turn_max",
    "Bert_turn_min",
    "Bert_turn_std",
    "Word_count",
    "Word_count_doc",
    "Word_ratio",
    "MTLD_patient",
    "MTLD_doctor",
    "Mean_words_per_turn",
    "Agenda_social",
    "Agenda_work",
    "Agenda_future",
    "Agenda_self",
    "Agenda_family",
    "Agenda_anxiety",
    "Agenda_depression",
)

#: Floating-point tolerance for the reproduction check, absolute and relative.
TOLERANCE: Final = 1e-9
TOP_RECOVERED: Final = 30

OUT_DIRNAME: Final = ("exploratory", "tokenization")
TOKENS_ORIGINAL: Final = "token_features_original.csv"
TOKENS_NEW: Final = "token_features_new.csv"
TEXT_ORIGINAL: Final = "text_features_original.csv"
TEXT_NEW: Final = "text_features_new.csv"


class TokenizationError(RuntimeError):
    """Raised when the inputs are not what analysis.py read."""


class ReproductionError(TokenizationError):
    """Raised when the original rule does not reproduce nlp_features.csv."""

    def __init__(self, lines: Sequence[str]) -> None:
        """Keep the per-column summary, which holds no per-session value."""
        self.lines = tuple(lines)
        super().__init__("\n".join(self.lines))


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcript_files(directory: Path) -> list[Path]:
    """The `.txt` transcripts, sorted by `int(stem)` as analysis.py sorts them.

    Raises:
        TokenizationError: if the folder is missing, empty, or holds a file
            whose stem is not a session number (analysis.py would have
            stopped on it).
    """
    if not directory.is_dir():
        msg = f"no transcript folder at {directory}"
        raise TokenizationError(msg)
    files = [path for path in directory.glob("*.txt") if path.is_file()]
    if not files:
        msg = f"no .txt transcripts in {directory}"
        raise TokenizationError(msg)
    odd = [path for path in files if not path.stem.isdigit()]
    if odd:
        msg = f"{len(odd)} .txt file(s) in {directory} are not named by session number"
        raise TokenizationError(msg)
    return sorted(files, key=lambda path: int(path.stem))


def read_nlp_features(path: Path) -> pd.DataFrame:
    """The manuscript's table, refused unless its columns are analysis.py's.

    Raises:
        TokenizationError: if it is missing or its columns differ.
    """
    if not path.is_file():
        msg = f"no nlp_features.csv at {path}"
        raise TokenizationError(msg)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if tuple(frame.columns) != NLP_COLUMNS:
        msg = f"{path.name} does not have analysis.py's 20 columns in its order"
        raise TokenizationError(msg)
    return frame


def compute(files: Sequence[Path], tokenizer: Any, min_len: int | None) -> list[SessionTokens]:
    """One variant over every transcript, in file order."""
    found = []
    for index, path in enumerate(files, start=1):
        found.append(transcript_tokens(path, tokenizer, min_len))
        if index % 10 == 0 or index == len(files):
            logger.info("%s: min_len=%s, %d/%d transcript(s)", STAGE, min_len, index, len(files))
    return found


def feature_frame(sessions: Sequence[SessionTokens]) -> pd.DataFrame:
    """session_id and the eight features, in file order."""
    frame = pd.DataFrame([s.features for s in sessions], columns=list(FEATURES))
    frame.insert(0, "session_id", [s.session_id for s in sessions])
    return frame


def reproduction_problems(original: pd.DataFrame, computed: pd.DataFrame) -> list[str]:
    """Where the port fails to reproduce the original, by column. Empty if it does.

    Counts and the largest difference per column only: no per-session value.
    """
    if len(original) != len(computed):
        return [
            f"row count differs: nlp_features.csv has {len(original)}, "
            f"the transcripts give {len(computed)}"
        ]
    problems: list[str] = []
    for column in FEATURES:
        want = original[column].to_numpy(dtype=np.float64)
        got = computed[column].to_numpy(dtype=np.float64)
        same = np.isclose(got, want, rtol=TOLERANCE, atol=TOLERANCE, equal_nan=True)
        if not same.all():
            both = np.isfinite(want) & np.isfinite(got)
            largest = float(np.max(np.abs(got - want)[both])) if both.any() else float("nan")
            problems.append(
                f"{column}: {int((~same).sum())} of {len(same)} row(s) differ, largest "
                f"difference {largest:.3g}; NaN in the original {int(np.isnan(want).sum())}, "
                f"in the port {int(np.isnan(got).sum())}"
            )
    return problems


def text_table(nlp: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    """The 20 features with session_id: eight from `tokens`, twelve from `nlp`."""
    table = nlp.copy()
    for column in FEATURES:
        table[column] = tokens[column].to_numpy()
    table.insert(0, "session_id", tokens["session_id"].to_numpy())
    return table


@dataclass(frozen=True, slots=True)
class Result:
    """What one run wrote, and the aggregate counts it reports."""

    out_dir: Path
    written: tuple[Path, ...]
    n_sessions: int
    original: tuple[SessionTokens, ...]
    new: tuple[SessionTokens, ...]

    def report_lines(self, digest: Callable[[Path], str]) -> list[str]:
        """Aggregate counts only: no per-session word is printed."""
        lemmas_a: Counter[str] = Counter()
        lemmas_b: Counter[str] = Counter()
        for session in self.original:
            lemmas_a.update(session.lemmas)
        for session in self.new:
            lemmas_b.update(session.lemmas)
        recovered = lemmas_b - lemmas_a
        per_session = np.array(
            [b.n_lemmas - a.n_lemmas for a, b in zip(self.original, self.new, strict=True)]
        )
        lines = [
            f"reproduction check passed: variant (a) matches nlp_features.csv in all "
            f"{len(FEATURES)} columns for {self.n_sessions} session(s), tolerance {TOLERANCE:g}",
            "",
            f"{'':28}{'(a) original':>14}{'(b) no length rule':>20}",
            f"{'kept lemmas, all sessions':28}{sum(lemmas_a.values()):>14,}"
            f"{sum(lemmas_b.values()):>20,}",
            f"{'vocabulary (distinct)':28}{len(lemmas_a):>14,}{len(lemmas_b):>20,}",
            "",
            f"recovered by (b): {sum(recovered.values()):,} lemma token(s), "
            f"{len(recovered):,} distinct lemma(s)",
            f"recovered per session: min {per_session.min()}, "
            f"median {float(np.median(per_session)):.0f}, max {per_session.max()}",
            "",
            f"the {TOP_RECOVERED} most frequent recovered lemmas, across all sessions:",
        ]
        lines.extend(
            f"  {rank:>2}. {lemma}  {count:,}"
            for rank, (lemma, count) in enumerate(recovered.most_common(TOP_RECOVERED), start=1)
        )
        lines.append("")
        lines.extend(f"wrote {path}  SHA-256 {digest(path)}" for path in self.written)
        return lines


def run(transcripts: Path, nlp_features: Path, out_dir: Path, tokenizer: Any) -> Result:
    """Check the port against the original, then compute and write both variants.

    Raises:
        TokenizationError: if the inputs are not what analysis.py read.
        ReproductionError: if variant (a) does not reproduce nlp_features.csv.
            Nothing is written.
    """
    nlp = read_nlp_features(nlp_features)
    files = transcript_files(transcripts)
    if len(files) != len(nlp):
        msg = (
            f"{len(files)} transcript(s) but {len(nlp)} row(s) in {nlp_features.name}: "
            f"these are not the files analysis.py read"
        )
        raise TokenizationError(msg)

    original = compute(files, tokenizer, ORIGINAL_MIN_LEN)
    tokens_a = feature_frame(original)
    problems = reproduction_problems(nlp, tokens_a)
    if problems:
        raise ReproductionError(
            [
                "REPRODUCTION FAILED: variant (a) does not match nlp_features.csv. "
                "Nothing was written and variant (b) was not computed.",
                *problems,
            ]
        )
    logger.info("%s: variant (a) reproduces nlp_features.csv", STAGE)

    new = compute(files, tokenizer, None)
    tokens_b = feature_frame(new)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        TOKENS_ORIGINAL: tokens_a,
        TOKENS_NEW: tokens_b,
        TEXT_ORIGINAL: text_table(nlp, tokens_a),
        TEXT_NEW: text_table(nlp, tokens_b),
    }
    written = tuple(write_csv(out_dir / name, frame) for name, frame in tables.items())
    return Result(
        out_dir=out_dir,
        written=written,
        n_sessions=len(files),
        original=tuple(original),
        new=tuple(new),
    )
