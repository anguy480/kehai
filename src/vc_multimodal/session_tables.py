"""Writing a per-session table without destroying the sessions you did not run.

Every stage here writes one row per session, and every stage can be run on a
subset with `--sessions`. Those two facts together are a trap: building the
table from the current run alone and writing it out silently deletes the rows
for every session outside the subset. Nothing fails, the stage reports success,
and the loss only surfaces later as a table with fewer rows than the cohort.

The same applies to a session the runner skips as already done. Its output is on
disk, so it contributes no fresh row; if the stage then leaves it out, or writes
a stub with only an identifier in it, the session either vanishes or - worse -
appears as a row whose measurements are empty. An empty row is indistinguishable
from a failed measurement, so it corrupts every rate computed over the table:
"corrected in 2 of 3 sessions" when the truth was 2 of 2 measured.

So the rule for a per-session table is: **the rows you computed, plus the rows
already on disk for sessions you did not compute.** A skipped session is simply
one you did not compute, which means it needs no special case at all - its
previous row is carried forward like any other.

That leaves one gap, which stages close with `has_row`: a session whose artifact
exists but whose row does not, usually because an earlier partial run deleted
it. Such a session is not done, whatever is on disk, and re-running it is the
only way to get a complete row.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from vc_multimodal.io_utils import read_csv
from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

ID_COLUMN: Final = "session_id"


class SessionTableError(RuntimeError):
    """Raised when an existing per-session table cannot be merged with."""


@dataclass(frozen=True, slots=True)
class CarriedRows:
    """Rows kept from an existing table for sessions this run did not compute.

    Attributes:
        rows: The carried rows, ready to be combined with fresh ones.
        session_ids: Which sessions they describe.
        absent_columns: Columns the fresh rows have that the existing file did
            not. Carried rows are empty in these, which is reported rather than
            left for a reader to discover.
    """

    rows: tuple[Mapping[str, object], ...]
    session_ids: tuple[int, ...]
    absent_columns: tuple[str, ...]

    def __bool__(self) -> bool:
        """Whether anything was carried forward."""
        return bool(self.rows)

    def notes(self, stage: str) -> list[str]:
        """What a reader of the stage report needs to know."""
        if not self.rows:
            return []
        lines = [f"kept {len(self.rows)} existing row(s) for session(s) not in this run"]
        if self.absent_columns:
            lines.append(
                f"{stage}: the existing table predates column(s) "
                f"{list(self.absent_columns)}, so those are empty for the "
                f"{len(self.rows)} carried session(s). Rerun them to fill the column(s) "
                f"in; rates reported over this table count only sessions with a value."
            )
        return lines


def carry_forward(
    target: Path,
    *,
    computed: Collection[int],
    columns: Sequence[str] = (),
    stage: str = "",
    force: bool = False,
) -> CarriedRows:
    """Rows from the table at `target` for sessions absent from `computed`.

    Args:
        target: Where the table is written.
        computed: Sessions this run produced a fresh row for. Their old rows are
            dropped in favour of the new ones.
        columns: Columns the fresh rows carry, so a shape change is reported.
        stage: Stage name, for messages.
        force: Ignore an unreadable or foreign file instead of refusing. The
            file is left alone; only this run's rows are written.

    Returns:
        The rows to keep, which is empty when there is no existing table.

    Raises:
        SessionTableError: if a file exists at `target` that is not a table this
            pipeline wrote. Merging with it could produce nonsense, and
            overwriting it without being asked could destroy someone's data.
    """
    if not target.exists():
        return CarriedRows(rows=(), session_ids=(), absent_columns=())

    try:
        existing = read_csv(target)
    except (OSError, ValueError) as exc:
        if force:
            logger.warning("%s: could not read %s (%s); starting fresh", stage, target.name, exc)
            return CarriedRows(rows=(), session_ids=(), absent_columns=())
        msg = (
            f"{target.name} exists but could not be read ({exc}), so the rows for "
            f"sessions outside this run cannot be kept. Move the file aside, or rerun "
            f"with --force to overwrite it with this run's rows only."
        )
        raise SessionTableError(msg) from exc

    if ID_COLUMN not in existing.columns:
        if force:
            logger.warning("%s: %s has no %s column; starting fresh", stage, target.name, ID_COLUMN)
            return CarriedRows(rows=(), session_ids=(), absent_columns=())
        msg = (
            f"{target.name} has no {ID_COLUMN} column, so it is not a table this "
            f"pipeline wrote and its rows cannot be matched to sessions. It has "
            f"{list(existing.columns)}. Move it aside, or rerun with --force to "
            f"replace it with this run's rows only."
        )
        raise SessionTableError(msg)

    numeric = pd.to_numeric(existing[ID_COLUMN], errors="coerce")
    usable = existing.loc[numeric.notna()].copy()
    usable[ID_COLUMN] = numeric[numeric.notna()].astype("int64")
    if len(usable) < len(existing):
        logger.warning(
            "%s: %d row(s) in %s have no usable %s and are dropped",
            stage,
            len(existing) - len(usable),
            target.name,
            ID_COLUMN,
        )

    kept = usable[~usable[ID_COLUMN].isin(list(computed))]
    if kept.empty:
        return CarriedRows(rows=(), session_ids=(), absent_columns=())

    absent = tuple(name for name in columns if name not in kept.columns)
    # pandas types records as dict[Hashable, Any]; the keys are column names.
    rows = cast("list[Mapping[str, object]]", kept.to_dict(orient="records"))
    carried = CarriedRows(
        rows=tuple(rows),
        session_ids=tuple(int(v) for v in kept[ID_COLUMN]),
        absent_columns=absent,
    )
    logger.info("%s: keeping %d existing row(s) for session(s) not in this run", stage, len(rows))
    if absent:
        logger.warning("%s: %s", stage, carried.notes(stage)[-1])
    return carried


def combine(
    computed: Mapping[int, Mapping[str, object]],
    carried: CarriedRows,
) -> list[Mapping[str, object]]:
    """Fresh rows and carried rows together, ordered by session.

    Ordering here rather than in each stage's frame builder means a merged table
    reads the same as one written in a single pass.
    """
    merged: dict[int, Mapping[str, object]] = {}
    for row in carried.rows:
        merged[int(cast("Any", row[ID_COLUMN]))] = row
    merged.update(computed)
    return [merged[key] for key in sorted(merged)]


def has_row(target: Path, session_id: int) -> bool:
    """Whether the table at `target` already has a row for this session.

    Stages fold this into their idempotency check. A session whose artifact
    exists but whose row does not is *not* done: skipping it would leave the
    table permanently short of a row that nothing else will ever write.
    """
    if not target.exists():
        return False
    try:
        existing = read_csv(target)
    except (OSError, ValueError):
        return False
    if ID_COLUMN not in existing.columns:
        return False
    numeric = pd.to_numeric(existing[ID_COLUMN], errors="coerce")
    return bool((numeric == session_id).any())


def rows_with_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """The values of `column` that are actually present.

    The honest denominator for any rate reported over a per-session table. A
    carried row from an older table, or a session whose measurement failed,
    holds no value for every column it never measured, and counting those as
    zeroes states something false about the cohort.
    """
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return frame[column].dropna()
