"""Cross-validated evaluation, and the choices behind it.

The manuscript this project extends used leave-one-participant-out with elastic
net, so that is what is reported first, for comparability. Leave-one-out is
also a high-variance way to *compare* models, so a repeated k-fold estimate is
reported beside it; where the two disagree, that disagreement is the result
(docs/decisions/0012).

Three properties are enforced here rather than left to the caller:

* **Every transform is fitted inside the fold.** Imputation and scaling are
  steps of a pipeline, so the held-out session contributes nothing to the
  median or the standard deviation used to transform it. Fitting a scaler on
  the whole table before cross-validating is the most common way a small-sample
  result becomes optimistic, and it leaves no trace in the output.
* **Hyperparameters are chosen inside the fold too.** The elastic net's alpha
  and l1 ratio come from an inner cross-validation on the training part only.
* **Models are compared on the same folds, session by session.** The
  confirmatory tests are comparisons, so they use paired per-session errors
  rather than two independently computed summary numbers.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_ELASTIC_NET: Final = "elastic_net"
MODEL_RANDOM_FOREST: Final = "random_forest"

#: l1 ratios searched by the inner cross-validation, spanning ridge-like to
#: lasso so the search is honest about both ends.
_L1_RATIOS: Final = (0.1, 0.5, 0.9, 1.0)
_INNER_FOLDS: Final = 5
_MAX_ITER: Final = 200_000
_N_TREES: Final = 500

#: How many alphas to try, and how far below the strongest penalty to go.
#:
#: The iteration cap is deliberately generous. Bounding the path removed most
#: non-convergence but not all of it on the real feature matrix, whose columns
#: are correlated by construction - a mean and a standard deviation of the same
#: action unit, a mean and a median of the same latency - in a way the synthetic
#: matrices the bound was tested on were not. Raising the cap from 20k to 200k
#: converges everywhere measured at no cost in time, because the fits that
#: needed it were a small minority.
#:
#: `_ALPHA_EPS` is the ratio of the weakest penalty tried to the strongest.
#: sklearn's default of 1e-3 explores penalties a thousand times weaker than
#: the one that zeroes every coefficient - effectively unpenalised least
#: squares. With 54 features on 62 sessions that end of the path is both
#: indefensible and pathological: coordinate descent stops converging, and
#: measured here it produced 124 non-converged fits per outer fold, whose
#: coefficients are whatever the iteration limit happened to leave behind.
#: Bounding the path at 1e-2 still spans two orders of magnitude, converges
#: everywhere, and is 16 times faster in that regime.
_ALPHA_EPS: Final = 1e-2
_N_ALPHAS: Final = 50

#: Fewest rows worth cross-validating at all.
MIN_ROWS: Final = 10

#: A fit needs two rows; a paired test needs two pairs; a split needs two folds.
_MIN_PAIR: Final = 2


class EvaluationError(ValueError):
    """Raised when a model cannot be evaluated on the data supplied."""


def build_pipeline(model: str, seed: int) -> Pipeline:
    """A pipeline whose every step is fitted on the training fold only.

    Median imputation, because features are missing by cause - a stage that
    could not run for a session - rather than at random, and a median is the
    least assuming filler. Standardisation, because the elastic net's penalty
    is scale-dependent; it is harmless for the forest and keeps the two
    comparable.

    Raises:
        EvaluationError: if `model` is not one we have.
    """
    if model == MODEL_ELASTIC_NET:
        estimator = ElasticNetCV(
            l1_ratio=list(_L1_RATIOS),
            alphas=_N_ALPHAS,
            eps=_ALPHA_EPS,
            cv=_INNER_FOLDS,
            max_iter=_MAX_ITER,
            random_state=seed,
            n_jobs=1,
        )
    elif model == MODEL_RANDOM_FOREST:
        estimator = RandomForestRegressor(
            n_estimators=_N_TREES,
            random_state=seed,
            n_jobs=1,
        )
    else:
        msg = f"unknown model {model!r}; expected {MODEL_ELASTIC_NET} or {MODEL_RANDOM_FOREST}"
        raise EvaluationError(msg)

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


# ---------------------------------------------------------------------------
# Fold schemes
# ---------------------------------------------------------------------------
def group_folds(groups: Sequence[object], n_splits: int, seed: int) -> list[np.ndarray]:
    """Split *groups*, not rows, into `n_splits` held-out sets.

    Splitting groups is what makes the estimate about unseen participants. With
    one session per participant this is an ordinary k-fold; with a participant
    map it is the only correct thing to do, and getting it wrong would put two
    sessions of one person on both sides of a fold.
    """
    unique = list(dict.fromkeys(groups))
    order = np.random.default_rng(seed).permutation(len(unique))
    shuffled = [unique[index] for index in order]
    buckets: list[list[object]] = [[] for _ in range(min(n_splits, len(shuffled)))]
    for index, group in enumerate(shuffled):
        buckets[index % len(buckets)].append(group)

    as_array = np.asarray(groups, dtype=object)
    return [np.isin(as_array, np.asarray(bucket, dtype=object)) for bucket in buckets if bucket]


def leave_one_group_out(groups: Sequence[object]) -> list[np.ndarray]:
    """One held-out set per group: the manuscript's scheme."""
    as_array = np.asarray(groups, dtype=object)
    return [as_array == group for group in dict.fromkeys(groups)]


def _predict_out_of_fold(
    features: np.ndarray,
    target: np.ndarray,
    masks: Sequence[np.ndarray],
    *,
    model: str,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Predict every row from a model that never saw it.

    Rows in no fold keep NaN, which is visible rather than silently scored.

    Returns:
        The predictions, and how many fits failed to converge. A fit that hits
        its iteration cap returns whatever coefficients the optimiser happened
        to be holding, so an estimate built from one is partly arbitrary. The
        count is carried rather than warned about once, because a wall of
        repeated sklearn text is easy to scroll past and a number per estimate
        is not.
    """
    predictions = np.full(target.shape, np.nan, dtype=np.float64)
    not_converged = 0
    for mask in masks:
        train = ~mask
        if train.sum() < _MIN_PAIR or mask.sum() == 0:
            continue
        pipeline = build_pipeline(model, seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(features[train], target[train])
            not_converged += sum(
                1 for entry in caught if issubclass(entry.category, ConvergenceWarning)
            )
        predictions[mask] = pipeline.predict(features[mask])
    return predictions, not_converged


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Scores:
    """How well out-of-fold predictions did.

    Attributes:
        n: Rows scored.
        r2: Out-of-fold R², which is negative when the model does worse than
            predicting the mean. That is a real outcome, not an error: the
            manuscript reported one for SRS-2.
        mae: Mean absolute error, in the target's own units.
        spearman: Rank correlation between prediction and truth.
        spearman_p: Its uncorrected p-value.
    """

    n: int
    r2: float
    mae: float
    spearman: float
    spearman_p: float


def score_predictions(truth: np.ndarray, predicted: np.ndarray) -> Scores:
    """Score out-of-fold predictions, ignoring rows that were never predicted.

    Raises:
        EvaluationError: if nothing was predicted.
    """
    usable = ~(np.isnan(truth) | np.isnan(predicted))
    if not usable.any():
        msg = "no out-of-fold predictions were produced"
        raise EvaluationError(msg)

    y = truth[usable]
    p = predicted[usable]
    residual = float(np.sum((y - p) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float("nan")

    if np.std(p) == 0 or np.std(y) == 0:
        # A model that predicts one value has no ranking to correlate.
        rho, p_value = float("nan"), float("nan")
    else:
        result = stats.spearmanr(y, p)
        rho, p_value = float(result.statistic), float(result.pvalue)

    return Scores(
        n=int(usable.sum()),
        r2=r2,
        mae=float(np.mean(np.abs(y - p))),
        spearman=rho,
        spearman_p=p_value,
    )


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One feature set, one target, one model family.

    Attributes:
        loo: Leave-one-participant-out scores, the primary estimate.
        stability: Mean repeated k-fold R², or None when not requested.
        stability_sd: Its spread across repeats.
        errors: Per-session absolute error under leave-one-out, for the paired
            comparisons. Aligned with the rows given.
        n_features: How many features went in.
        n_not_converged: Leave-one-out fits that hit the iteration cap. Any
            number above zero means part of this estimate rests on coefficients
            the optimiser had not finished computing.
    """

    loo: Scores
    stability: float | None
    stability_sd: float | None
    errors: np.ndarray
    n_features: int
    n_not_converged: int = 0

    @property
    def disagrees_with_stability(self) -> bool:
        """Whether the two schemes disagree about the sign of the result.

        Worth reporting when true: leave-one-out is high-variance for model
        comparison, so a positive R² under one scheme and a negative one under
        the other is a finding about the estimate, not about the features.
        """
        if self.stability is None:
            return False
        return (self.loo.r2 > 0) != (self.stability > 0)


def evaluate(
    features: np.ndarray,
    target: np.ndarray,
    groups: Sequence[object],
    *,
    model: str,
    seed: int,
    stability_folds: int = 0,
    stability_repeats: int = 0,
) -> Evaluation:
    """Cross-validate one feature set against one target.

    Raises:
        EvaluationError: if there is too little data to cross-validate.
    """
    if features.shape[0] < MIN_ROWS:
        msg = (
            f"{features.shape[0]} row(s) is too few to cross-validate; at least "
            f"{MIN_ROWS} are needed for the estimate to mean anything"
        )
        raise EvaluationError(msg)

    loo_predictions, not_converged = _predict_out_of_fold(
        features, target, leave_one_group_out(groups), model=model, seed=seed
    )
    loo = score_predictions(target, loo_predictions)

    stability: float | None = None
    stability_sd: float | None = None
    if stability_folds >= _MIN_PAIR and stability_repeats >= 1:
        per_repeat: list[float] = []
        for repeat in range(stability_repeats):
            masks = group_folds(groups, stability_folds, seed + repeat)
            predicted, _ = _predict_out_of_fold(features, target, masks, model=model, seed=seed)
            try:
                per_repeat.append(score_predictions(target, predicted).r2)
            except EvaluationError:  # pragma: no cover - defensive
                continue
        if per_repeat:
            stability = float(np.mean(per_repeat))
            stability_sd = float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else 0.0

    return Evaluation(
        loo=loo,
        stability=stability,
        stability_sd=stability_sd,
        errors=np.abs(target - loo_predictions),
        n_features=int(features.shape[1]),
        n_not_converged=not_converged,
    )


# ---------------------------------------------------------------------------
# Comparing two feature sets
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Comparison:
    """A paired comparison of two feature sets on the same sessions.

    Attributes:
        n: Sessions both sides predicted.
        median_difference: Median of (second error - first error). Positive
            means the first feature set was closer.
        p_value: Uncorrected two-sided Wilcoxon signed-rank p-value.
        p_adjusted: Filled in by the caller after multiplicity correction.
    """

    n: int
    median_difference: float
    p_value: float
    p_adjusted: float | None = None


def compare_errors(first: np.ndarray, second: np.ndarray) -> Comparison:
    """Compare two models by their per-session errors.

    Paired and non-parametric. Paired because both models predicted the same
    sessions from the same folds, so the pairing removes the between-session
    variance that dominates a difference of two R² values at this sample size.
    Non-parametric because absolute errors are bounded below and skewed, and
    62 of them do not make a t-test's assumptions true.

    Raises:
        EvaluationError: if the two sets of errors are not comparable.
    """
    if first.shape != second.shape:
        msg = "the two error vectors describe different numbers of sessions"
        raise EvaluationError(msg)
    usable = ~(np.isnan(first) | np.isnan(second))
    if usable.sum() < _MIN_PAIR:
        msg = "fewer than two sessions were predicted by both models"
        raise EvaluationError(msg)

    difference = second[usable] - first[usable]
    if np.allclose(difference, 0.0):
        # Identical predictions: there is nothing to test, and Wilcoxon raises.
        return Comparison(n=int(usable.sum()), median_difference=0.0, p_value=1.0)

    result = stats.wilcoxon(difference, alternative="two-sided", zero_method="zsplit")
    return Comparison(
        n=int(usable.sum()),
        median_difference=float(np.median(difference)),
        p_value=float(result.pvalue),
    )


# ---------------------------------------------------------------------------
# Permutation baseline
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Permutation:
    """What the same pipeline achieves on shuffled labels.

    Attributes:
        observed: R² on the real labels, under the permutation scheme.
        null_mean: Mean R² across permutations.
        null_p95: 95th percentile of the null.
        p_value: Fraction of permutations reaching the observed value, with the
            observed value counted, so it is never zero.
        n_permutations: How many were run.
        scheme: The fold scheme the null was computed under.
    """

    observed: float
    null_mean: float
    null_p95: float
    p_value: float
    n_permutations: int
    scheme: str


def permutation_baseline(
    features: np.ndarray,
    target: np.ndarray,
    groups: Sequence[object],
    *,
    model: str,
    seed: int,
    n_permutations: int,
    folds: int,
) -> Permutation | None:
    """What this pipeline scores when the labels mean nothing.

    Computed under k-fold rather than leave-one-out, and the observed value it
    is compared against is computed the same way. A thousand leave-one-out
    nulls would be 62,000 model fits per test; the null is a property of the
    scheme, so the cheaper scheme answers the same question as long as both
    sides of the comparison use it. That equivalence is the reason this is
    sound, and it is why `scheme` is reported rather than assumed.

    Returns None when `n_permutations` is 0.
    """
    if n_permutations <= 0:
        return None

    masks = group_folds(groups, folds, seed)
    observed = score_predictions(
        target, _predict_out_of_fold(features, target, masks, model=model, seed=seed)[0]
    ).r2

    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(target)
        predicted, _ = _predict_out_of_fold(features, shuffled, masks, model=model, seed=seed)
        try:
            null.append(score_predictions(shuffled, predicted).r2)
        except EvaluationError:  # pragma: no cover - defensive
            continue
    if not null:  # pragma: no cover - defensive
        return None

    as_array = np.asarray(null, dtype=np.float64)
    at_least = int(np.sum(as_array >= observed))
    return Permutation(
        observed=observed,
        null_mean=float(np.nanmean(as_array)),
        null_p95=float(np.nanpercentile(as_array, 95)),
        # The observed value is counted in both numerator and denominator, so
        # the smallest reportable p-value is 1/(n+1) rather than zero.
        p_value=(at_least + 1) / (len(as_array) + 1),
        n_permutations=len(as_array),
        scheme=f"{folds}-fold",
    )


def fold_iterator(masks: Sequence[np.ndarray]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Train and test index arrays for each mask, for tests and diagnostics."""
    for mask in masks:
        yield np.flatnonzero(~mask), np.flatnonzero(mask)
