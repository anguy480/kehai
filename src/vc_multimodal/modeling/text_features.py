"""Joining the manuscript's text features to this project's feature table.

The confirmatory comparisons are new modalities *against the text baseline*, so
this join has to be right. It is also the one join in the project that cannot
be checked by inspection, because the two tables come from different pipelines
and different code. An error here attaches each participant's text features to
someone else, every downstream number still looks plausible, and nothing in the
results reveals it.

There are exactly two ways a row becomes a session:

* **By identifier.** The table carries a session ID column. This rests on
  nothing but the file itself and is always preferred.
* **By position, under a stated rule.** Allowed only with an ordering rule
  quoted from the code that wrote the file, reproduced from the same source
  that code iterated, and guarded by an exact row count. This is the mode the
  manuscript's `nlp_features.csv` needs, because it was written without an
  identifier column.

A table with no identifier and no ordering rule is refused. Row order on its
own is not an identifier, however suggestive the row count.

Also enforced: the text table must carry no outcome. A column derived from K6
or SRS-2 would leak the label into the predictor set, and the baseline would
already know the answer.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from vc_multimodal.config import TextFeaturesConfig
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


# ---------------------------------------------------------------------------
# Matching rows to sessions by position
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PositionalPlan:
    """A resolved ordering rule: which session each row belongs to.

    Built from the files the lab iterated, not from our own inventory, so that
    a divergence between the two shows up as a count mismatch rather than being
    quietly absorbed.

    Attributes:
        rule: The named ordering rule.
        provenance: Where the rule came from, in words. Copied to the manifest.
        source_glob: The files the order was reproduced from.
        expected_rows: The row count the rule was confirmed against.
        session_ids: The session for each row, in row order.
    """

    rule: str
    provenance: str
    source_glob: str
    expected_rows: int
    session_ids: tuple[int, ...]

    def describe(self) -> str:
        """One line for a log or a summary."""
        ids = self.session_ids
        span = f"{ids[0]} ... {ids[-1]}" if len(ids) > 1 else str(list(ids))
        return (
            f"rows matched to sessions by position, rule {self.rule!r}, "
            f"{len(ids)} session(s) from {self.source_glob} ({span})"
        )


def resolve_positional(work_root: Path, config: TextFeaturesConfig) -> PositionalPlan:
    """Reproduce the lab's row ordering from the files they iterated.

    The rule `numeric_ascending_session_id` is the lab's
    `sorted(dir_path.glob(extension), key=lambda p: int(p.stem))`: numeric
    ascending by session ID across the whole set, so 62 precedes 102 where a
    lexicographic sort would not.

    Raises:
        TextFeatureError: if the ordering cannot be reproduced exactly - no
            files, a filename that is not a session ID, a repeated ID, or a
            file count other than the one the rule was confirmed against.
    """
    if config.positional is None:  # pragma: no cover - guarded by config validation
        msg = "resolve_positional needs a configured positional ordering rule"
        raise TextFeatureError(msg)
    rule = config.positional

    paths = sorted(work_root.glob(rule.source_glob))
    if not paths:
        msg = (
            f"the row ordering cannot be reproduced: no files matched "
            f"{rule.source_glob!r} under the work root. The ordering rule is defined "
            f"by those files, so without them a positional join is a guess."
        )
        raise TextFeatureError(msg)

    not_numeric = sorted(p.name for p in paths if not p.stem.isdigit())
    if not_numeric:
        msg = (
            f"the row ordering cannot be reproduced: {len(not_numeric)} file(s) matched "
            f"{rule.source_glob!r} whose name is not a session ID: {not_numeric[:5]}. "
            f"The lab's ordering used int(path.stem) over exactly these files, so a "
            f"file we can see and they could not means the two sets differ."
        )
        raise TextFeatureError(msg)

    ids = [int(p.stem) for p in paths]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        msg = (
            f"the row ordering cannot be reproduced: session(s) {repeated} appear more "
            f"than once under {rule.source_glob!r}"
        )
        raise TextFeatureError(msg)

    if len(ids) != rule.expected_rows:
        msg = (
            f"the row ordering was confirmed against {rule.expected_rows} file(s), but "
            f"{rule.source_glob!r} now matches {len(ids)}. Positional matching is only "
            f"as good as that count, so the join stops here. If the change is "
            f"intended, confirm the ordering against the new set and update "
            f"model.text_features.positional.expected_rows."
        )
        raise TextFeatureError(msg)

    # Numeric ascending, which is the rule; `sorted` on paths is lexicographic.
    ordered = tuple(sorted(ids))
    return PositionalPlan(
        rule=rule.rule,
        provenance=" ".join(rule.provenance.split()),
        source_glob=rule.source_glob,
        expected_rows=rule.expected_rows,
        session_ids=ordered,
    )


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TextFeatures:
    """A validated text feature table, ready to join.

    Attributes:
        frame: The table, with a `session_id` column and `text__*` features.
        id_column: The identifier column as it was named in the file, or None
            if rows were matched by position.
        feature_columns: The renamed feature columns.
        plan: The ordering rule used, if rows were matched by position.
        source: The file this came from.
        sha256: Its digest, so the manifest records which file was used.
    """

    frame: pd.DataFrame
    id_column: str | None
    feature_columns: tuple[str, ...]
    plan: PositionalPlan | None
    source: str
    sha256: str

    @property
    def n_sessions(self) -> int:
        """How many sessions the table covers."""
        return len(self.frame)

    @property
    def identification(self) -> str:
        """How rows were matched to sessions."""
        return "positional" if self.plan is not None else "identifier"

    def manifest_record(self) -> dict[str, Any]:
        """What the run manifest records about this join.

        The provenance travels with the results. A reader who wants to know how
        two tables written by different code were aligned finds the rule, its
        source, and the resolved order here rather than having to ask.
        """
        record: dict[str, Any] = {
            "source": self.source,
            "sha256": self.sha256,
            "n_rows": int(self.n_sessions),
            "n_features": len(self.feature_columns),
            "features": list(self.feature_columns),
            "identification": self.identification,
        }
        if self.id_column is not None:
            record["id_column"] = self.id_column
        if self.plan is not None:
            record["ordering"] = {
                "rule": self.plan.rule,
                "provenance": self.plan.provenance,
                "source_glob": self.plan.source_glob,
                "expected_rows": self.plan.expected_rows,
                "session_ids": [int(i) for i in self.plan.session_ids],
            }
        return record


def _digest(path: Path) -> str:
    """The file's SHA-256, so the manifest names the exact file used."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_without_identifier(path: Path, columns: Sequence[str], n_rows: int) -> None:
    """The refusal for a table with neither an identifier nor an ordering rule."""
    msg = (
        f"{path.name} has no identifier column, so it cannot be joined. Tried "
        f"{list(ID_COLUMNS)}, and found columns: {list(columns)}.\n"
        f"Row order is not an identifier. The table has {n_rows} row(s), which is "
        f"suggestive and nothing more: if the order differs from the one assumed, "
        f"every participant's text features are attached to someone else, and no "
        f"metric would reveal it. Either supply the table with a session_id column, "
        f"or state the ordering rule in model.text_features.positional with the "
        f"provenance for it."
    )
    raise TextFeatureError(msg)


def load(path: Path, *, positional: PositionalPlan | None = None) -> TextFeatures:
    """Read and validate the manuscript's text features.

    Args:
        path: The CSV.
        positional: An ordering rule, used only if the table has no identifier
            column. An identifier always wins: it needs no external promise.

    Raises:
        TextFeatureError: if the file is unreadable, contains a column that
            looks like an outcome, or cannot be matched to sessions safely.
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
    if id_column is None and positional is None:
        _refuse_without_identifier(path, list(frame.columns), len(frame))

    if id_column is not None:
        if positional is not None:
            logger.info(
                "%s carries an identifier column %r, so it is used in preference to the "
                "configured positional ordering rule",
                path.name,
                id_column,
            )
        return _from_identifier(path, frame, id_column)

    assert positional is not None
    return _from_position(path, frame, positional)


def _features_of(frame: pd.DataFrame, path: Path, id_column: str | None) -> tuple[str, ...]:
    features = tuple(name for name in frame.columns if name != id_column)
    if not features:
        msg = f"{path.name} has no feature columns"
        raise TextFeatureError(msg)
    return features


def _renamed(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    return frame.rename(columns={name: normalise_name(name) for name in features})


def _from_identifier(path: Path, frame: pd.DataFrame, id_column: str) -> TextFeatures:
    """Match rows to sessions by the table's own identifier."""
    features = _features_of(frame, path, id_column)
    renamed = _renamed(frame, features).rename(columns={id_column: "session_id"})

    numeric = pd.to_numeric(renamed["session_id"], errors="coerce")
    if numeric.isna().any():
        msg = f"{path.name} has non-numeric values in its {id_column!r} column"
        raise TextFeatureError(msg)
    renamed["session_id"] = numeric.astype("int64")

    duplicated = sorted({int(v) for v in renamed["session_id"][renamed["session_id"].duplicated()]})
    if duplicated:
        msg = f"{path.name} has more than one row for session(s) {duplicated}"
        raise TextFeatureError(msg)

    logger.info("%s: %d row(s) joined by %r", path.name, len(renamed), id_column)
    return TextFeatures(
        frame=renamed,
        id_column=id_column,
        feature_columns=tuple(normalise_name(name) for name in features),
        plan=None,
        source=path.name,
        sha256=_digest(path),
    )


def _from_position(path: Path, frame: pd.DataFrame, plan: PositionalPlan) -> TextFeatures:
    """Match rows to sessions by position, under a stated ordering rule.

    Both counts are checked against the rule rather than against each other, so
    the error says which one moved.
    """
    if len(frame) != plan.expected_rows:
        msg = (
            f"{path.name} has {len(frame)} row(s), but the ordering rule "
            f"{plan.rule!r} was confirmed against {plan.expected_rows}. Matching rows "
            f"to sessions by position is only as good as that count, so the join stops "
            f"here rather than aligning a file the rule was not confirmed for. If the "
            f"table has genuinely changed, re-confirm the ordering with whoever wrote "
            f"it and update model.text_features.positional."
        )
        raise TextFeatureError(msg)

    if len(frame) != len(plan.session_ids):
        msg = (
            f"{path.name} has {len(frame)} row(s) and {plan.source_glob!r} names "
            f"{len(plan.session_ids)} session(s). Positional matching requires exactly "
            f"one row per session, in the order those sessions appear."
        )
        raise TextFeatureError(msg)

    features = _features_of(frame, path, None)
    renamed = _renamed(frame, features).reset_index(drop=True)
    renamed.insert(0, "session_id", pd.Series(plan.session_ids, dtype="int64"))

    logger.info("%s: %s", path.name, plan.describe())
    logger.info("ordering provenance: %s", plan.provenance)
    return TextFeatures(
        frame=renamed,
        id_column=None,
        feature_columns=tuple(normalise_name(name) for name in features),
        plan=plan,
        source=path.name,
        sha256=_digest(path),
    )


def load_configured(work_root: Path, config: TextFeaturesConfig) -> TextFeatures:
    """Load the text features described by the configuration."""
    path = work_root / config.path
    if not path.exists():
        msg = f"the text feature table is not at {config.path} under the work root"
        raise TextFeatureError(msg)
    plan = resolve_positional(work_root, config) if config.identification == "positional" else None
    return load(path, positional=plan)


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------
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

    if text.plan is not None and report.features_only:
        logger.warning(
            "%d session(s) in the feature table are absent from the text run that the "
            "row ordering came from (%s). The alignment of the rows that did match is "
            "unaffected, but the two runs saw different data, which is worth checking "
            "before the text baseline is relied on: %s",
            len(report.features_only),
            text.plan.source_glob,
            list(report.features_only),
        )
    return merged, report
