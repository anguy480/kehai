"""Tests for the cross-validated evaluation.

All data is synthetic and generated here. The most important test in this file
is the leakage one: it watches what each transform is fitted on and asserts the
held-out row was never among it. Leakage leaves no trace in the output, so it
has to be caught by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from vc_multimodal.modeling.evaluate import (
    _MAX_ITER,
    MODEL_ELASTIC_NET,
    MODEL_RANDOM_FOREST,
    EvaluationError,
    build_pipeline,
    compare_errors,
    evaluate,
    group_folds,
    leave_one_group_out,
    permutation_baseline,
    score_predictions,
)

SEED = 20260926


def linear_data(n: int = 40, noise: float = 0.2, seed: int = SEED):
    """A target that really is a linear function of two of five features."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 5))
    target = 3.0 * features[:, 0] - 2.0 * features[:, 1] + rng.normal(0, noise, n)
    return features, target, [f"s{i}" for i in range(n)]


class TestFolds:
    def test_leave_one_group_out_holds_out_each_group_once(self) -> None:
        masks = leave_one_group_out(["a", "b", "c"])
        assert len(masks) == 3
        assert [int(mask.sum()) for mask in masks] == [1, 1, 1]

    def test_repeated_sessions_of_one_participant_are_held_out_together(self) -> None:
        # The property that makes the estimate about unseen participants.
        masks = leave_one_group_out(["p1", "p1", "p2"])
        assert len(masks) == 2
        assert list(masks[0]) == [True, True, False]

    def test_group_folds_split_groups_not_rows(self) -> None:
        groups = ["p1", "p1", "p2", "p2", "p3", "p3"]
        masks = group_folds(groups, 3, SEED)
        for mask in masks:
            held = {group for group, flag in zip(groups, mask, strict=True) if flag}
            kept = {group for group, flag in zip(groups, mask, strict=True) if not flag}
            assert not (held & kept)

    def test_every_row_is_held_out_exactly_once(self) -> None:
        masks = group_folds([f"s{i}" for i in range(10)], 5, SEED)
        counts = np.sum(masks, axis=0)
        assert list(counts) == [1] * 10

    def test_fewer_groups_than_folds_is_not_a_crash(self) -> None:
        masks = group_folds(["a", "b"], 5, SEED)
        assert len(masks) == 2


class RecordingTransformer(TransformerMixin, BaseEstimator):
    """Passes data through, recording every row it was fitted on."""

    def __init__(self, seen: list[np.ndarray] | None = None) -> None:
        self.seen = seen if seen is not None else []

    def fit(self, X, y=None):  # noqa: N803
        self.seen.append(np.array(X, copy=True))
        return self

    def transform(self, X):  # noqa: N803
        return X


class TestNoLeakage:
    def test_transforms_never_see_the_held_out_row(self) -> None:
        # Marked rows: row i has the value i in its first column, so anything
        # a transform was fitted on can be identified exactly.
        n = 12
        features = np.zeros((n, 2))
        features[:, 0] = np.arange(n)
        target = np.arange(n, dtype=float)
        seen: list[np.ndarray] = []

        for mask in leave_one_group_out([f"s{i}" for i in range(n)]):
            pipeline = Pipeline([("watch", RecordingTransformer(seen)), ("model", _Mean())])
            pipeline.fit(features[~mask], target[~mask])
            held_out_value = float(features[mask][0, 0])
            fitted_on = seen[-1][:, 0]
            assert held_out_value not in set(fitted_on.tolist())

    def test_the_real_pipeline_fits_its_scaler_per_fold(self) -> None:
        # The scaler's mean must be the training fold's mean, not the table's.
        features, target, groups = linear_data(n=20)
        masks = leave_one_group_out(groups)
        pipeline = build_pipeline(MODEL_ELASTIC_NET, SEED)
        pipeline.fit(features[~masks[0]], target[~masks[0]])
        fold_mean = pipeline.named_steps["scale"].mean_[0]
        assert fold_mean == pytest.approx(features[~masks[0]][:, 0].mean())
        assert fold_mean != pytest.approx(features[:, 0].mean())

    def test_imputation_is_a_pipeline_step_not_a_preprocessing_step(self) -> None:
        # Imputing before cross-validating would let the held-out row's value
        # into the median used to fill the training rows.
        pipeline = build_pipeline(MODEL_ELASTIC_NET, SEED)
        assert list(pipeline.named_steps) == ["impute", "scale", "model"]

    def test_missing_values_are_handled_without_dropping_rows(self) -> None:
        features, target, groups = linear_data(n=30)
        features[3, 2] = np.nan
        features[7, 4] = np.nan
        result = evaluate(features, target, groups, model=MODEL_ELASTIC_NET, seed=SEED)
        assert result.loo.n == 30


class _Mean(BaseEstimator):
    """Predicts the training mean. Enough to exercise a pipeline."""

    def fit(self, X, y):  # noqa: N803
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):  # noqa: N803
        return np.full(len(X), self.value_)


class TestScores:
    def test_a_perfect_prediction_scores_one(self) -> None:
        truth = np.arange(10, dtype=float)
        assert score_predictions(truth, truth.copy()).r2 == pytest.approx(1.0)

    def test_predicting_the_mean_scores_about_zero(self) -> None:
        truth = np.arange(10, dtype=float)
        predicted = np.full(10, truth.mean())
        assert score_predictions(truth, predicted).r2 == pytest.approx(0.0)

    def test_a_worse_than_mean_prediction_scores_negative(self) -> None:
        # A real outcome, not an error: the manuscript reported one for SRS-2.
        truth = np.arange(10, dtype=float)
        predicted = truth[::-1].copy()
        assert score_predictions(truth, predicted).r2 < 0

    def test_rows_never_predicted_are_excluded_rather_than_scored(self) -> None:
        truth = np.arange(10, dtype=float)
        predicted = truth.copy()
        predicted[0] = np.nan
        assert score_predictions(truth, predicted).n == 9

    def test_nothing_predicted_is_an_error_not_a_zero(self) -> None:
        with pytest.raises(EvaluationError, match="no out-of-fold predictions"):
            score_predictions(np.arange(3, dtype=float), np.full(3, np.nan))

    def test_a_constant_prediction_has_no_rank_correlation(self) -> None:
        truth = np.arange(10, dtype=float)
        scores = score_predictions(truth, np.full(10, 4.0))
        assert np.isnan(scores.spearman)

    def test_mae_is_in_the_targets_units(self) -> None:
        truth = np.zeros(4)
        scores = score_predictions(truth, np.array([1.0, -1.0, 2.0, -2.0]))
        assert scores.mae == pytest.approx(1.5)


class TestEvaluate:
    def test_a_learnable_signal_is_learned(self) -> None:
        features, target, groups = linear_data(n=40, noise=0.2)
        result = evaluate(features, target, groups, model=MODEL_ELASTIC_NET, seed=SEED)
        assert result.loo.r2 > 0.8

    def test_pure_noise_does_not_score_well(self) -> None:
        rng = np.random.default_rng(SEED)
        features = rng.normal(size=(40, 5))
        target = rng.normal(size=40)
        result = evaluate(
            features,
            target,
            groups=[f"s{i}" for i in range(40)],
            model=MODEL_ELASTIC_NET,
            seed=SEED,
        )
        assert result.loo.r2 < 0.2

    @pytest.mark.slow
    def test_both_model_families_run(self) -> None:
        features, target, groups = linear_data(n=30)
        for model in (MODEL_ELASTIC_NET, MODEL_RANDOM_FOREST):
            assert evaluate(features, target, groups, model=model, seed=SEED).loo.n == 30

    def test_an_unknown_model_is_refused(self) -> None:
        with pytest.raises(EvaluationError, match="unknown model"):
            build_pipeline("deep_learning", SEED)

    def test_too_few_rows_is_refused_with_the_reason(self) -> None:
        features, target, groups = linear_data(n=6)
        with pytest.raises(EvaluationError, match="too few to cross-validate"):
            evaluate(features, target, groups, model=MODEL_ELASTIC_NET, seed=SEED)

    def test_the_stability_estimate_is_reported_when_asked(self) -> None:
        features, target, groups = linear_data(n=40)
        result = evaluate(
            features,
            target,
            groups,
            model=MODEL_ELASTIC_NET,
            seed=SEED,
            stability_folds=5,
            stability_repeats=3,
        )
        assert result.stability is not None
        assert result.stability_sd is not None

    def test_no_stability_estimate_when_not_asked(self) -> None:
        features, target, groups = linear_data(n=20)
        result = evaluate(features, target, groups, model=MODEL_ELASTIC_NET, seed=SEED)
        assert result.stability is None
        assert not result.disagrees_with_stability

    def test_a_sign_disagreement_between_schemes_is_reported(self) -> None:
        features, target, groups = linear_data(n=20)
        result = evaluate(features, target, groups, model=MODEL_ELASTIC_NET, seed=SEED)
        flipped = type(result)(
            loo=result.loo,
            stability=-abs(result.loo.r2) - 0.1,
            stability_sd=0.05,
            errors=result.errors,
            n_features=result.n_features,
        )
        assert flipped.disagrees_with_stability is (result.loo.r2 > 0)

    def test_per_session_errors_are_returned_for_the_comparisons(self) -> None:
        features, target, groups = linear_data(n=25)
        result = evaluate(features, target, groups, model=MODEL_ELASTIC_NET, seed=SEED)
        assert result.errors.shape == target.shape

    @pytest.mark.slow
    def test_the_same_seed_gives_the_same_answer(self) -> None:
        features, target, groups = linear_data(n=30)
        first = evaluate(features, target, groups, model=MODEL_RANDOM_FOREST, seed=7)
        second = evaluate(features, target, groups, model=MODEL_RANDOM_FOREST, seed=7)
        assert first.loo.r2 == pytest.approx(second.loo.r2)


@pytest.mark.slow
class TestComparison:
    def test_the_better_model_has_a_positive_median_difference(self) -> None:
        good = np.full(20, 1.0)
        bad = np.full(20, 3.0)
        result = compare_errors(good, bad)
        assert result.median_difference == pytest.approx(2.0)
        assert result.p_value < 0.01

    def test_identical_predictions_are_not_a_significant_difference(self) -> None:
        errors = np.abs(np.random.default_rng(SEED).normal(size=20))
        result = compare_errors(errors, errors.copy())
        assert result.median_difference == 0.0
        assert result.p_value == 1.0

    def test_only_sessions_predicted_by_both_are_compared(self) -> None:
        first = np.array([1.0, 2.0, np.nan, 4.0])
        second = np.array([2.0, 3.0, 1.0, 5.0])
        assert compare_errors(first, second).n == 3

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(EvaluationError, match="different numbers of sessions"):
            compare_errors(np.zeros(3), np.zeros(4))

    def test_too_few_shared_sessions_is_refused(self) -> None:
        first = np.array([1.0, np.nan, np.nan])
        second = np.array([1.0, 2.0, 3.0])
        with pytest.raises(EvaluationError, match="fewer than two sessions"):
            compare_errors(first, second)


@pytest.mark.slow
class TestPermutationBaseline:
    def test_a_real_signal_beats_its_own_null(self) -> None:
        features, target, groups = linear_data(n=30, noise=0.2)
        result = permutation_baseline(
            features,
            target,
            groups,
            model=MODEL_ELASTIC_NET,
            seed=SEED,
            n_permutations=30,
            folds=5,
        )
        assert result is not None
        assert result.observed > result.null_p95
        assert result.p_value < 0.05

    def test_noise_does_not_beat_its_null(self) -> None:
        rng = np.random.default_rng(SEED)
        features = rng.normal(size=(30, 5))
        target = rng.normal(size=30)
        result = permutation_baseline(
            features,
            target,
            [f"s{i}" for i in range(30)],
            model=MODEL_ELASTIC_NET,
            seed=SEED,
            n_permutations=30,
            folds=5,
        )
        assert result is not None
        assert result.p_value > 0.05

    def test_the_p_value_can_never_be_zero(self) -> None:
        # The observed value is counted on both sides, so the floor is 1/(n+1).
        features, target, groups = linear_data(n=30, noise=0.01)
        result = permutation_baseline(
            features,
            target,
            groups,
            model=MODEL_ELASTIC_NET,
            seed=SEED,
            n_permutations=10,
            folds=5,
        )
        assert result is not None
        assert result.p_value >= 1 / 11

    def test_the_scheme_is_reported_rather_than_assumed(self) -> None:
        features, target, groups = linear_data(n=30)
        result = permutation_baseline(
            features,
            target,
            groups,
            model=MODEL_ELASTIC_NET,
            seed=SEED,
            n_permutations=5,
            folds=5,
        )
        assert result is not None
        assert result.scheme == "5-fold"

    def test_zero_permutations_means_no_baseline(self) -> None:
        features, target, groups = linear_data(n=30)
        assert (
            permutation_baseline(
                features,
                target,
                groups,
                model=MODEL_ELASTIC_NET,
                seed=SEED,
                n_permutations=0,
                folds=5,
            )
            is None
        )


@pytest.mark.slow
class TestConvergenceIsCountedNotIgnored:
    """A fit that hits its iteration cap returns whatever it was holding.

    Bounding the alpha path removed most non-convergence but not all of it on
    the real feature matrix, whose columns are correlated by construction. The
    count is what makes any remaining case visible per estimate rather than a
    wall of sklearn text to scroll past.
    """

    def correlated_data(self, n: int = 40, p: int = 36):
        """Features correlated the way the real ones are.

        A mean and a standard deviation of the same measure, a mean and a
        median of the same latency: near-duplicate columns, which is what makes
        coordinate descent struggle.
        """
        rng = np.random.default_rng(SEED)
        base = rng.normal(size=(n, p // 3))
        features = np.hstack([base, base + rng.normal(0, 0.01, base.shape), base * 1.001])
        return features, rng.normal(size=n), [f"s{i}" for i in range(n)]

    def test_a_converged_run_reports_zero(self) -> None:
        features, target, groups = linear_data(n=30)
        result = evaluate(features, target, groups, model=MODEL_ELASTIC_NET, seed=SEED)
        assert result.n_not_converged == 0

    def test_the_count_is_carried_on_the_evaluation(self) -> None:
        features, target, groups = self.correlated_data()
        result = evaluate(features, target, groups, model=MODEL_ELASTIC_NET, seed=SEED)
        # Whatever the number, it must be a number rather than a surprise.
        assert isinstance(result.n_not_converged, int)
        assert result.n_not_converged >= 0

    def test_the_iteration_cap_is_generous_enough_to_matter(self) -> None:
        # The cap was raised from 20k to 200k because bounding the path alone
        # left non-converged fits on the real matrix.
        assert _MAX_ITER >= 200_000

    def test_a_forest_never_reports_non_convergence(self) -> None:
        features, target, groups = linear_data(n=30)
        result = evaluate(features, target, groups, model=MODEL_RANDOM_FOREST, seed=SEED)
        assert result.n_not_converged == 0
