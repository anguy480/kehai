"""Joining the manuscript's text features to this project's feature table.

The whole point of the confirmatory comparisons is new modalities *against the
text baseline*, so the join has to be right. It is also the one join in this
project that cannot be checked by inspection, because the two tables come from
different pipelines: an error would attach each participant's text features to
someone else, and every downstream number would still look plausible.

So the rule here is that the join is by an explicit identifier or it does not
happen. Row order is not an identifier. A table of 62 rows beside a cohort of
62 sessions is suggestive and nothing more: if the orders differ, the features
are silently transposed between participants and no test, plot or metric would
reveal it.

Also enforced: the text table must carry no outcome. A column derived from K6
or SRS-2 would leak the label into the predictor set, and the comparison would
be against a baseline that already knows the answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

#: Identifier columns accepted for the join, in order of preference.
ID_COLUMNS: Final = ("session_id", "session", "id", "recording_id", "file_id")

#: Column names suggesting an outcome rather than a predictor. Matched against
#: the tokenised name (see `_tokenise`), so `total_score` and `PHQ9` are caught
#: as readily as `total score` and `phq 9`. Deliberately broad: a false positive
#: costs a question, a false negative costs the study.
LABEL_PATTERNS: Final = (
    r"\bk\s?6\b",
    r"\bsrs\b",
    r"\bscore\b",
    r"\btotal\b",
    r"label",
    r"target",
    r"outcome",
    r"^y\b",
    r"diagnos",
    r"severity",
    r"\bphq\b",
    r"\bgad\b",
    r"subscale",
)

#: Names matching a label pattern that are known predictors in this project's
#: source material, and so are allowed. The manuscript's LLM-rated agenda
#: scores are topic ratings of what was discussed, not questionnaire values.
LABEL_ALLOWLIST: Final = ("agenda_anxiety", "agenda_depression")

TEXT_FAMILY: Final = "text"


class TextFeatureError(ValueError):
    """Raised when the text feature table cannot be used safely."""


@dataclass(frozen=True, slots=True)
class TextFeatures:
    """A validated text feature table, ready to join."""

    frame: pd.DataFrame
    id_column: str
    feature_columns: tuple[str, ...]

    @property
    def n_sessions(self) -> int:
        """How many sessions the table covers."""
        return len(self.frame)


def find_id_column(columns: Sequence[str]) -> str | None:
    """The identifier column, if the table has one."""
    lowered = {str(name).strip().lower(): str(name) for name in columns}
    for candidate in ID_COLUMNS:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _tokenise(name: str) -> str:
    r"""A name reduced to space-separated tokens, for pattern matching.

    Word boundaries in a raw column name are unreliable: `_` is a word
    character, so `\btotal\b` does not match `total_score`, and there is no
    boundary between a letter and a digit, so `\bphq\b` does not match `PHQ9`.
    Splitting on punctuation and at letter/digit joins makes both match.
    """
    text = re.sub(r"[^0-9a-z]+", " ", str(name).strip().lower())
    text = re.sub(r"(?<=[a-z])(?=[0-9])", " ", text)
    text = re.sub(r"(?<=[0-9])(?=[a-z])", " ", text)
    return text.strip()


def label_like_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Columns whose names suggest an outcome rather than a predictor."""
    allowed = {_tokenise(name) for name in LABEL_ALLOWLIST}
    found: list[str] = []
    for name in columns:
        token = _tokenise(name)
        if token in allowed:
            continue
        if any(re.search(pattern, token) for pattern in LABEL_PATTERNS):
            found.append(str(name))
    return tuple(found)


def normalise_name(name: str) -> str:
    """Turn a manuscript column name into this project's convention.

    `Bert_turn_max` becomes `text__bert_turn_max`, so the text features select
    by family exactly as every other family does.
    """
    cleaned = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")
    return f"{TEXT_FAMILY}__{cleaned}"


def load(path: Path) -> TextFeatures:
    """Read and validate the manuscript's text features.

    Raises:
        TextFeatureError: if the file is unreadable, carries no identifier, or
            contains a column that looks like an outcome.
    """
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        msg = f"could not read {path.name}: {exc}"
        raise TextFeatureError(msg) from exc

    frame.columns = [str(name).strip() for name in frame.columns]

    suspicious = label_like_columns(list(frame.columns))
    if suspicious:
        msg = (
            f"{path.name} contains column(s) that look like questionnaire outcomes "
            f"rather than text features: {list(suspicious)}. Using them would leak "
            f"the label into the predictor set, so the file is refused. Remove those "
            f"columns, or if they are genuinely predictors, add them to "
            f"LABEL_ALLOWLIST in this module with the reason."
        )
        raise TextFeatureError(msg)

    id_column = find_id_column(list(frame.columns))
    if id_column is None:
        msg = (
            f"{path.name} has no identifier column, so it cannot be joined. Tried "
            f"{list(ID_COLUMNS)}, and found columns: {list(frame.columns)}.\n"
            f"Row order is not an identifier. The table has {len(frame)} row(s) and "
            f"the cohort has 62 sessions, which is suggestive and nothing more: if "
            f"the orders differ, every participant's text features are attached to "
            f"someone else, and no metric would reveal it. Ask for the table with a "
            f"session_id column, or for the ordering in writing."
        )
        raise TextFeatureError(msg)

    features = tuple(name for name in frame.columns if name != id_column)
    if not features:
        msg = f"{path.name} has an identifier but no feature columns"
        raise TextFeatureError(msg)

    renamed = frame.rename(columns={name: normalise_name(name) for name in features})
    renamed = renamed.rename(columns={id_column: "session_id"})
    renamed["session_id"] = pd.to_numeric(renamed["session_id"], errors="coerce").astype("Int64")
    if renamed["session_id"].isna().any():
        msg = f"{path.name} has non-numeric values in its {id_column!r} column"
        raise TextFeatureError(msg)
    renamed["session_id"] = renamed["session_id"].astype("int64")

    duplicated = sorted({int(v) for v in renamed["session_id"][renamed["session_id"].duplicated()]})
    if duplicated:
        msg = f"{path.name} has more than one row for session(s) {duplicated}"
        raise TextFeatureError(msg)

    return TextFeatures(
        frame=renamed,
        id_column=id_column,
        feature_columns=tuple(normalise_name(name) for name in features),
    )


@dataclass(frozen=True, slots=True)
class JoinReport:
    """What the join to the feature table covered.

    Attributes:
        matched: Sessions present in both tables.
        features_only: Sessions in this project's table with no text features.
        text_only: Sessions in the text table that this project does not have.
    """

    matched: tuple[int, ...]
    features_only: tuple[int, ...]
    text_only: tuple[int, ...]

    @property
    def is_complete(self) -> bool:
        """Whether every session on both sides matched."""
        return not self.features_only and not self.text_only

    def report_lines(self) -> list[str]:
        """Human-readable coverage, session IDs only."""
        lines = [f"text features joined for {len(self.matched)} session(s)"]
        if self.features_only:
            lines.append(
                f"  no text features for {len(self.features_only)} session(s): "
                f"{list(self.features_only)}"
            )
        if self.text_only:
            lines.append(
                f"  text features for {len(self.text_only)} session(s) absent from the "
                f"feature table: {list(self.text_only)}"
            )
        if self.is_complete:
            lines.append("  every session matched on both sides")
        return lines


def join(features: pd.DataFrame, text: TextFeatures) -> tuple[pd.DataFrame, JoinReport]:
    """Add the text features to the feature table, by session.

    A left join on this project's table: every session keeps its row, and a
    session with no text features has them absent rather than being dropped.
    Which sessions those are is reported, because a comparison against the text
    baseline computed over a different subset is not the comparison it claims.
    """
    ours = {int(v) for v in features["session_id"]}
    theirs = {int(v) for v in text.frame["session_id"]}

    merged = features.merge(text.frame, on="session_id", how="left", validate="one_to_one")
    report = JoinReport(
        matched=tuple(sorted(ours & theirs)),
        features_only=tuple(sorted(ours - theirs)),
        text_only=tuple(sorted(theirs - ours)),
    )
    for line in report.report_lines():
        logger.info("%s", line)
    return merged, report
