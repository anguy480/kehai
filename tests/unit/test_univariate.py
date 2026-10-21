"""Per-feature Spearman correlation and Benjamini-Hochberg. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vc_multimodal.modeling import univariate


def test_bh_matches_a_hand_computed_example() -> None:
    # Sorted: 0.005, 0.01, 0.03, 0.04 with m = 4 scale to 0.02, 0.02, 0.04, 0.04.
    q = univariate.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert q == pytest.approx((0.02, 0.04, 0.04, 0.02))


def test_bh_is_monotone_capped_and_never_below_p() -> None:
    rng = np.random.default_rng(0)
    p = list(rng.uniform(size=40))
    q = univariate.benjamini_hochberg(p)
    assert all(qi >= pi for pi, qi in zip(p, q, strict=True))
    assert max(q) <= 1.0
    order = np.argsort(p)
    assert np.all(np.diff(np.asarray(q)[order]) >= -1e-12)


def test_bh_rejects_an_impossible_p_value() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        univariate.benjamini_hochberg([0.2, 1.5])


def test_spearman_uses_complete_pairs_and_matches_scipy() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = np.array([2.0, 1.0, 3.0, np.nan, 6.0, 5.0])
    found = univariate.spearman("f", x, "K6", y)
    expected = stats.spearmanr([1.0, 2.0, 5.0, 6.0], [2.0, 1.0, 6.0, 5.0])
    assert found.n == 4
    assert found.rho == pytest.approx(expected.statistic)
    assert found.p == pytest.approx(expected.pvalue)


def test_a_constant_feature_is_untestable_rather_than_p_one() -> None:
    found = univariate.spearman("f", np.zeros(10), "K6", np.arange(10.0))
    assert found.p is None
    assert found.untestable == univariate.UNTESTABLE_CONSTANT


def test_too_few_pairs_is_untestable() -> None:
    found = univariate.spearman("f", np.array([1.0, 2.0]), "K6", np.array([1.0, 2.0]))
    assert found.untestable == univariate.UNTESTABLE_TOO_FEW


def test_q_is_computed_within_each_target_over_testable_rows_only() -> None:
    rng = np.random.default_rng(1)
    features = pd.DataFrame(
        {"a": rng.normal(size=30), "b": rng.normal(size=30), "flat": np.zeros(30)}
    )
    outcomes = pd.DataFrame({"K6": rng.normal(size=30), "SRS2": rng.normal(size=30)})
    frame = univariate.table(
        univariate.correlate(features, ["a", "b", "flat"], outcomes, ["K6", "SRS2"])
    )
    assert list(frame.columns) == list(univariate.COLUMNS)
    assert frame.loc[frame["feature"] == "flat", "q"].isna().all()
    for target in ("K6", "SRS2"):
        rows = frame.loc[(frame["target"] == target) & frame["p"].notna()]
        assert list(rows["q"]) == pytest.approx(univariate.benjamini_hochberg(list(rows["p"])))


def test_a_partial_family_can_be_left_unadjusted() -> None:
    rng = np.random.default_rng(2)
    features = pd.DataFrame({"a": rng.normal(size=20)})
    outcomes = pd.DataFrame({"K6": rng.normal(size=20)})
    frame = univariate.table(univariate.correlate(features, ["a"], outcomes, ["K6"]), adjust=False)
    assert frame["q"].isna().all()
    assert frame["p"].notna().all()
