"""Cross-validation, shared permutation nulls and the max statistic.

The fitting is the main run's: `build_pipeline` supplies the same elastic net
and random forest, with median imputation and scaling fitted inside every fold,
and the folds come from the same `leave_one_group_out` and `group_folds`. What
this module adds:

* **Spearman rho under k-fold as well as leave-one-out**, averaged over repeats.
* **The nested search**, whose forward selection runs on each outer training
  fold before the outer model is fitted on the columns it chose.
* **One set of permutations for every set and model.** Each permutation of the
  labels is scored by every set under both models on the same 5-fold split, so
  it gives a null R² per set (the per-set null) and, taking the best of them,
  one draw of the max-statistic null. The split is the first k-fold repeat's,
  which is also the split `permutation_baseline` uses in the main run.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from vc_multimodal.exploratory.au_baseline.search import SearchSpec, forward_select
from vc_multimodal.modeling.evaluate import (
    build_pipeline,
    group_folds,
    leave_one_group_out,
    score_predictions,
)


@dataclass(frozen=True, slots=True)
class Task:
    """One feature set, ready to fit.

    Attributes:
        name: The set's name.
        matrix: Sessions x columns, NaN where a feature is missing.
        search: Set for the nested search, whose matrix holds the candidates.
    """

    name: str
    matrix: np.ndarray
    search: SearchSpec | None = None


Selections = list[tuple[int, ...]] | None


def selections_for(
    task: Task, target: np.ndarray, groups: Sequence[object], masks: Sequence[np.ndarray]
) -> Selections:
    """The search's chosen columns for each outer fold, or None for a fixed set."""
    if task.search is None:
        return None
    as_array = np.asarray(groups, dtype=object)
    return [
        forward_select(task.matrix[~mask], target[~mask], list(as_array[~mask]), task.search)
        for mask in masks
    ]


def predict_oof(
    matrix: np.ndarray,
    target: np.ndarray,
    masks: Sequence[np.ndarray],
    *,
    model: str,
    seed: int,
    selections: Selections = None,
) -> tuple[np.ndarray, int]:
    """Out-of-fold predictions, and how many fits hit the iteration cap.

    With `selections`, fold i is fitted on the columns selections[i] names; a
    fold whose selection is empty predicts its training mean.
    """
    predicted = np.full(target.shape, np.nan, dtype=np.float64)
    not_converged = 0
    for index, mask in enumerate(masks):
        train = ~mask
        columns = matrix if selections is None else matrix[:, list(selections[index])]
        if columns.shape[1] == 0:
            predicted[mask] = float(np.mean(target[train]))
            continue
        pipeline = build_pipeline(model, seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(columns[train], target[train])
            not_converged += sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
        predicted[mask] = pipeline.predict(columns[mask])
    return predicted, not_converged


def r2_score(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Pooled out-of-fold R², as `score_predictions` computes it."""
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total == 0:
        return float("nan")
    return 1.0 - float(np.sum((truth - predicted) ** 2)) / total


@dataclass(frozen=True, slots=True)
class SetScores:
    """One set, one model, one target.

    Attributes:
        n: Sessions scored.
        loo_r2: Leave-one-out R².
        loo_spearman: Spearman rho between leave-one-out predictions and outcome.
        loo_spearman_p: Its uncorrected p-value.
        kfold_r2_mean: Mean R² over the k-fold repeats.
        kfold_r2_sd: Its spread.
        kfold_spearman_mean: Mean rho over the repeats where it is defined.
        split_r2: R² on the first repeat's split, the one the null uses.
        n_not_converged: Leave-one-out fits that hit the iteration cap.
        loo_selections: For the search, the columns chosen in each LOO fold.
    """

    n: int
    loo_r2: float
    loo_spearman: float
    loo_spearman_p: float
    kfold_r2_mean: float
    kfold_r2_sd: float
    kfold_spearman_mean: float
    split_r2: float
    n_not_converged: int
    loo_selections: tuple[tuple[int, ...], ...] = ()


def evaluate_set(
    task: Task,
    target: np.ndarray,
    groups: Sequence[object],
    *,
    model: str,
    seed: int,
    folds: int,
    repeats: int,
) -> SetScores:
    """Leave-one-out and repeated k-fold for one set, model and target."""
    loo_masks = leave_one_group_out(groups)
    loo_selected = selections_for(task, target, groups, loo_masks)
    predicted, not_converged = predict_oof(
        task.matrix, target, loo_masks, model=model, seed=seed, selections=loo_selected
    )
    loo = score_predictions(target, predicted)

    r2s: list[float] = []
    rhos: list[float] = []
    for repeat in range(repeats):
        masks = group_folds(groups, folds, seed + repeat)
        selected = selections_for(task, target, groups, masks)
        fold_predicted, _ = predict_oof(
            task.matrix, target, masks, model=model, seed=seed, selections=selected
        )
        scores = score_predictions(target, fold_predicted)
        r2s.append(scores.r2)
        rhos.append(scores.spearman)

    defined = [rho for rho in rhos if np.isfinite(rho)]
    return SetScores(
        n=loo.n,
        loo_r2=loo.r2,
        loo_spearman=loo.spearman,
        loo_spearman_p=loo.spearman_p,
        kfold_r2_mean=float(np.mean(r2s)),
        kfold_r2_sd=float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
        kfold_spearman_mean=float(np.mean(defined)) if defined else float("nan"),
        split_r2=r2s[0],
        n_not_converged=not_converged,
        loo_selections=tuple(loo_selected) if loo_selected is not None else (),
    )


def null_chunk(
    tasks: Sequence[Task],
    target: np.ndarray,
    groups: Sequence[object],
    orders: np.ndarray,
    *,
    models: Sequence[str],
    seed: int,
    folds: int,
) -> np.ndarray:
    """Null R² for every permutation in `orders`, set and model.

    Each row of `orders` permutes the target. The search's selections depend on
    the labels but not on the outer model, so they are made once per
    permutation and shared by both models.

    Returns:
        Array of shape (permutations, sets, models).
    """
    masks = group_folds(groups, folds, seed)
    out = np.full((len(orders), len(tasks), len(models)), np.nan)
    for p, order in enumerate(orders):
        shuffled = target[order]
        for t, task in enumerate(tasks):
            selected = selections_for(task, shuffled, groups, masks)
            for m, model in enumerate(models):
                predicted, _ = predict_oof(
                    task.matrix, shuffled, masks, model=model, seed=seed, selections=selected
                )
                out[p, t, m] = r2_score(shuffled, predicted)
    return out


@dataclass(frozen=True, slots=True)
class SetNull:
    """Where one set's observed R² falls in its own null."""

    observed: float
    null_mean: float
    null_p95: float
    p_value: float


def set_null(observed: float, null: np.ndarray) -> SetNull:
    """Per-set permutation p, counting the observed value so it is never zero."""
    usable = null[np.isfinite(null)]
    return SetNull(
        observed=observed,
        null_mean=float(np.mean(usable)),
        null_p95=float(np.percentile(usable, 95)),
        p_value=(int(np.sum(usable >= observed)) + 1) / (len(usable) + 1),
    )


@dataclass(frozen=True, slots=True)
class MaxStatistic:
    """Where the best observed set falls among the best of each permutation.

    Attributes:
        best_set: The family member with the highest observed split R².
        best_model: Its model.
        observed: That R².
        null_max: The best null R² in the family, one per permutation.
        p_value: Share of permutations whose best reached `observed`, with the
            observed value counted.
        percentile: Share of permutation maxima strictly below `observed`, x100.
    """

    best_set: str
    best_model: str
    observed: float
    null_max: np.ndarray
    p_value: float
    percentile: float


def max_statistic(
    observed: np.ndarray,
    null: np.ndarray,
    set_names: Sequence[str],
    model_names: Sequence[str],
) -> MaxStatistic:
    """The max-statistic test over every set and model passed.

    Args:
        observed: Split R², shape (sets, models).
        null: Null R², shape (permutations, sets, models).
        set_names: Names along the sets axis.
        model_names: Names along the models axis.
    """
    flat = np.where(np.isfinite(observed), observed, -np.inf)
    t, m = np.unravel_index(int(np.argmax(flat)), flat.shape)
    best = float(observed[t, m])
    null_max = np.nanmax(null.reshape(null.shape[0], -1), axis=1)
    return MaxStatistic(
        best_set=set_names[int(t)],
        best_model=model_names[int(m)],
        observed=best,
        null_max=null_max,
        p_value=(int(np.sum(null_max >= best)) + 1) / (len(null_max) + 1),
        percentile=100.0 * float(np.mean(null_max < best)),
    )
