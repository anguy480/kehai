"""Recompute the eight lemma-based text features under the pre-registered variants.

Run where the transcripts are, by the person allowed to read them. It reads the
`.txt` transcripts `analysis.py` read and the existing `nlp_features.csv`, and:

1. Computes variant (a), the original rule, and checks it reproduces the eight
   columns of `nlp_features.csv`, row for row. If it does not, it stops: nothing
   is written and variant (b) is not computed.
2. Computes variant (b), with no length rule, and variant (c), with no length
   rule and 非自立 tokens excluded (Amendment 1 of the pre-registration).
3. Writes six tables under the output root, each with a `session_id` taken
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from vc_multimodal.exploratory.tokenization.token_features import (
    EXCLUDED_POS2,
    FEATURES,
    NON_INDEPENDENT,
    ORIGINAL_MIN_LEN,
    VARIANT_C_EXCLUDED_POS2,
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
TOKENS_NEW_C: Final = "token_features_new_c.csv"
TEXT_NEW_C: Final = "text_features_new_c.csv"


@dataclass(frozen=True, slots=True)
class Variant:
    """One pre-registered filter, and the two files it writes."""

    key: str
    label: str
    min_len: int | None
    excluded_pos2: frozenset[str]
    tokens_file: str
    text_file: str


ORIGINAL: Final = Variant(
    "a", "(a) original", ORIGINAL_MIN_LEN, EXCLUDED_POS2, TOKENS_ORIGINAL, TEXT_ORIGINAL
)
NEW: Final = Variant("b", "(b) no length rule", None, EXCLUDED_POS2, TOKENS_NEW, TEXT_NEW)
NEW_C: Final = Variant(
    "c", f"(c) (b), no {NON_INDEPENDENT}", None, VARIANT_C_EXCLUDED_POS2, TOKENS_NEW_C, TEXT_NEW_C
)
#: In the order they are computed; (a) first, because it is the gate.
VARIANTS: Final = (ORIGINAL, NEW, NEW_C)


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


def compute(files: Sequence[Path], tokenizer: Any, variant: Variant) -> list[SessionTokens]:
    """One variant over every transcript, in file order."""
    found = []
    for index, path in enumerate(files, start=1):
        found.append(transcript_tokens(path, tokenizer, variant.min_len, variant.excluded_pos2))
        if index % 10 == 0 or index == len(files):
            logger.info(
                "%s: variant (%s), %d/%d transcript(s)", STAGE, variant.key, index, len(files)
            )
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


def _pooled(sessions: Sequence[SessionTokens]) -> Counter[str]:
    """Kept lemmas across all sessions."""
    pooled: Counter[str] = Counter()
    for session in sessions:
        pooled.update(session.lemmas)
    return pooled


def _ranked(counts: Counter[str]) -> list[str]:
    """The most frequent lemmas, one per line, ranked."""
    return [
        f"  {rank:>2}. {lemma}  {count:,}"
        for rank, (lemma, count) in enumerate(counts.most_common(TOP_RECOVERED), start=1)
    ]


def _spread(values: Sequence[int]) -> str:
    """Minimum, median and maximum of per-session counts."""
    array = np.array(values)
    return f"min {array.min()}, median {float(np.median(array)):.0f}, max {array.max()}"


@dataclass(frozen=True, slots=True)
class Result:
    """What one run wrote, and the aggregate counts it reports."""

    out_dir: Path
    written: tuple[Path, ...]
    n_sessions: int
    sessions: Mapping[str, tuple[SessionTokens, ...]]

    def report_lines(self, digest: Callable[[Path], str]) -> list[str]:
        """Aggregate counts only: no per-session word is printed."""
        pooled = {key: _pooled(found) for key, found in self.sessions.items()}
        a, b, c = (self.sessions[v.key] for v in VARIANTS)
        lines = [
            f"reproduction check passed: variant (a) matches nlp_features.csv in all "
            f"{len(FEATURES)} columns for {self.n_sessions} session(s), tolerance {TOLERANCE:g}",
            "",
            f"{'':28}" + "".join(f"{v.label:>22}" for v in VARIANTS),
            f"{'kept lemmas, all sessions':28}"
            + "".join(f"{sum(pooled[v.key].values()):>22,}" for v in VARIANTS),
            f"{'vocabulary (distinct)':28}"
            + "".join(f"{len(pooled[v.key]):>22,}" for v in VARIANTS),
        ]

        recovered = pooled[NEW.key] - pooled[ORIGINAL.key]
        lines += [
            "",
            f"(b) against (a): {sum(recovered.values()):,} lemma token(s) recovered, "
            f"{len(recovered):,} distinct lemma(s)",
            "recovered per session: "
            + _spread([y.n_lemmas - x.n_lemmas for x, y in zip(a, b, strict=True)]),
            f"the {TOP_RECOVERED} most frequent lemmas (b) recovers, across all sessions:",
            *_ranked(recovered),
        ]

        dropped = pooled[NEW.key] - pooled[NEW_C.key]
        gained = pooled[NEW_C.key] - pooled[ORIGINAL.key]
        lost = pooled[ORIGINAL.key] - pooled[NEW_C.key]
        lines += [
            "",
            f"(c) against (b): {sum(dropped.values()):,} lemma token(s) dropped as "
            f"{NON_INDEPENDENT}, {len(dropped):,} distinct lemma(s)",
            "dropped per session: "
            + _spread([y.n_lemmas - z.n_lemmas for y, z in zip(b, c, strict=True)]),
            f"(c) against (a): {sum(gained.values()):,} lemma token(s) gained, "
            f"{sum(lost.values()):,} lost",
            "net change per session: "
            + _spread([z.n_lemmas - x.n_lemmas for x, z in zip(a, c, strict=True)]),
            f"the {TOP_RECOVERED} most frequent lemmas (c) drops from (b), across all sessions:",
            *_ranked(dropped),
            "",
        ]
        lines.extend(f"wrote {path}  SHA-256 {digest(path)}" for path in self.written)
        return lines


def run(transcripts: Path, nlp_features: Path, out_dir: Path, tokenizer: Any) -> Result:
    """Check the port against the original, then compute and write every variant.

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

    sessions = {ORIGINAL.key: compute(files, tokenizer, ORIGINAL)}
    tokens = {ORIGINAL.key: feature_frame(sessions[ORIGINAL.key])}
    problems = reproduction_problems(nlp, tokens[ORIGINAL.key])
    if problems:
        raise ReproductionError(
            [
                "REPRODUCTION FAILED: variant (a) does not match nlp_features.csv. "
                "Nothing was written and no other variant was computed.",
                *problems,
            ]
        )
    logger.info("%s: variant (a) reproduces nlp_features.csv", STAGE)

    for variant in VARIANTS[1:]:
        sessions[variant.key] = compute(files, tokenizer, variant)
        tokens[variant.key] = feature_frame(sessions[variant.key])
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {variant.tokens_file: tokens[variant.key] for variant in VARIANTS} | {
        variant.text_file: text_table(nlp, tokens[variant.key]) for variant in VARIANTS
    }
    written = tuple(write_csv(out_dir / name, frame) for name, frame in tables.items())
    return Result(
        out_dir=out_dir,
        written=written,
        n_sessions=len(files),
        sessions={key: tuple(found) for key, found in sessions.items()},
    )
