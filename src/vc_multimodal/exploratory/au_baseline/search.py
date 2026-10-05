"""Greedy forward selection of AU features, for use inside an outer CV fold.

Called once per outer training fold, on that fold's rows only, so the held-out
sessions never influence which features are chosen. The inner score is the
out-of-fold R² of ordinary least squares with an intercept over an inner
k-fold split of the training rows, with median imputation fitted on each inner
training part. Scaling is omitted because it does not change least squares
predictions.

Least squares rather than the outer model because this runs inside every outer
fold of every permutation: written directly in numpy it costs microseconds per
fit, where an elastic net with its own inner cross-validation would cost tens
of milliseconds and put a third level of resampling inside the null.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vc_multimodal.modeling.evaluate import group_folds


@dataclass(frozen=True, slots=True)
class SearchSpec:
    """How the forward selection runs.

    Attributes:
        max_features: Most columns it may select.
        inner_folds: Folds of the inner split that scores each step.
        seed: Seed of the inner split.
    """

    max_features: int
    inner_folds: int
    seed: int


def _impute(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill gaps with training medians, dropping columns with no training value."""
    finite = ~np.isnan(train).all(axis=0)
    train, test = train[:, finite], test[:, finite]
    medians = np.nanmedian(train, axis=0)
    return np.where(np.isnan(train), medians, train), np.where(np.isnan(test), medians, test)


def ols_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """Least squares with an intercept, median imputation fitted on `train_x`."""
    if train_x.shape[1] == 0:
        return np.full(test_x.shape[0], float(np.mean(train_y)))
    filled_train, filled_test = _impute(train_x, test_x)
    design = np.column_stack([np.ones(len(filled_train)), filled_train])
    coef, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    return np.asarray(np.column_stack([np.ones(len(filled_test)), filled_test]) @ coef)


def oof_r2(features: np.ndarray, target: np.ndarray, masks: Sequence[np.ndarray]) -> float:
    """Pooled out-of-fold R² of least squares over `masks`."""
    predicted = np.empty_like(target)
    for mask in masks:
        predicted[mask] = ols_predict(features[~mask], target[~mask], features[mask])
    total = float(np.sum((target - target.mean()) ** 2))
    if total == 0:
        return float("-inf")
    return 1.0 - float(np.sum((target - predicted) ** 2)) / total


def forward_select(
    features: np.ndarray, target: np.ndarray, groups: Sequence[object], spec: SearchSpec
) -> tuple[int, ...]:
    """Column indices chosen greedily, in the order they were added.

    Starts from no features (predicting the mean) and adds whichever column most
    raises the inner R², stopping when none raises it or `max_features` is
    reached. Ties go to the earlier column, so the result is deterministic.
    """
    masks = group_folds(groups, spec.inner_folds, spec.seed)
    selected: list[int] = []
    best = oof_r2(features[:, selected], target, masks)
    while len(selected) < spec.max_features:
        remaining = [j for j in range(features.shape[1]) if j not in selected]
        if not remaining:
            break
        scores = [oof_r2(features[:, [*selected, j]], target, masks) for j in remaining]
        top = int(np.argmax(scores))
        if scores[top] <= best:
            break
        selected.append(remaining[top])
        best = scores[top]
    return tuple(selected)
