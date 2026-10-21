"""One feature at a time: Spearman correlation with an outcome, and BH FDR.

The models answer whether a set of features predicts an outcome out of sample.
This answers a narrower question that a reader also asks: which single
features move with the outcome at all. It is descriptive and in-sample, so it
is corrected for the number of features looked at, by Benjamini-Hochberg within
each outcome.

Two cases are kept apart from an ordinary result rather than folded into one:

* **Untestable.** A feature that is constant over the sessions that have it,
  or that has fewer than three sessions, has no rank correlation. It gets no
  p-value and is left out of the correction family, and the reason is
  recorded. Counting it as p = 1 would inflate the family and say something
  false about it.
* **Missing values.** Each correlation uses the sessions that have both the
  feature and the outcome. The n is reported per row, so a feature resting on
  fewer sessions is visible.

No label value, rank or residual is returned: only rho, p, q and counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

#: Fewest pairs a rank correlation is computed on.
MIN_PAIRS: Final = 3

#: Why a feature has no correlation.
UNTESTABLE_CONSTANT: Final = "constant"
UNTESTABLE_TOO_FEW: Final = "too_few_sessions"

#: Column order of the per-feature table.
COLUMNS: Final = ("feature", "target", "n", "rho", "p", "q", "untestable")


@dataclass(frozen=True, slots=True)
class Correlation:
    """One feature against one outcome.

    Attributes:
        feature: Column name.
        target: Outcome name.
        n: Sessions with both values.
        rho: Spearman rho, or None when untestable.
        p: Two-sided p-value, or None when untestable.
        untestable: Why there is no correlation, empty when there is one.
    """

    feature: str
    target: str
    n: int
    rho: float | None
    p: float | None
    untestable: str = ""

    @property
    def is_testable(self) -> bool:
        """Whether this row belongs in a correction family."""
        return self.p is not None


def spearman(feature: str, values: np.ndarray, target: str, outcome: np.ndarray) -> Correlation:
    """Spearman rho of one feature with one outcome, over complete pairs."""
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    usable = np.isfinite(x) & np.isfinite(y)
    n = int(usable.sum())
    if n < MIN_PAIRS:
        return Correlation(feature, target, n, None, None, UNTESTABLE_TOO_FEW)
    x, y = x[usable], y[usable]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return Correlation(feature, target, n, None, None, UNTESTABLE_CONSTANT)
    result = stats.spearmanr(x, y)
    return Correlation(feature, target, n, float(result.statistic), float(result.pvalue))


def benjamini_hochberg(p_values: Sequence[float]) -> tuple[float, ...]:
    """Benjamini-Hochberg adjusted p-values (q), in the input order.

    The step-up procedure: sort ascending, scale the i-th smallest by m / i,
    then take the running minimum from the largest down so the result is
    monotone in the original ranking, and cap at 1.

    Raises:
        ValueError: if any p-value is outside [0, 1].
    """
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        msg = f"p-values must lie in [0, 1], got {list(p_values)}"
        raise ValueError(msg)
    count = len(p_values)
    if count == 0:
        return ()
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 1.0
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = running
    return tuple(adjusted)


def correlate(
    frame: pd.DataFrame, columns: Sequence[str], outcomes: pd.DataFrame, targets: Sequence[str]
) -> list[Correlation]:
    """Correlate every column with every target.

    Args:
        frame: Features, one row per session, aligned row for row with
            `outcomes`.
        columns: Feature columns to correlate.
        outcomes: The outcomes, aligned with `frame`.
        targets: Outcome columns.

    Raises:
        ValueError: if the two tables are not the same length.
    """
    if len(frame) != len(outcomes):
        msg = "features and outcomes must be aligned row for row"
        raise ValueError(msg)
    found: list[Correlation] = []
    for target in targets:
        outcome = pd.to_numeric(outcomes[target], errors="coerce").to_numpy(dtype=np.float64)
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            found.append(spearman(column, values, target, outcome))
    return found


def table(correlations: Sequence[Correlation], *, adjust: bool = True) -> pd.DataFrame:
    """The per-feature table, with BH q computed within each target.

    Args:
        correlations: Every correlation in the family, for every target.
        adjust: Compute q. False leaves it empty, for a run that holds only part
            of a family: a q over a partial family is a different number from
            the one that was planned.
    """
    frame = pd.DataFrame(
        [
            {
                "feature": c.feature,
                "target": c.target,
                "n": c.n,
                "rho": c.rho,
                "p": c.p,
                "q": np.nan,
                "untestable": c.untestable,
            }
            for c in correlations
        ],
        columns=list(COLUMNS),
    )
    for column in ("rho", "p", "q"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return add_q(frame) if adjust else frame


def add_q(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill `q` by BH within each target, over the testable rows only."""
    updated = frame.copy()
    updated["q"] = np.nan
    for _, rows in updated.groupby("target", sort=False):
        testable = rows.loc[rows["p"].notna()]
        if testable.empty:
            continue
        q = benjamini_hochberg([float(p) for p in testable["p"]])
        updated.loc[testable.index, "q"] = q
    return updated
