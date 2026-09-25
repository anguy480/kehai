# 15. How the models are cross-validated, and why the penalty path is bounded

Date: 2026-09-26

## Status

Accepted.

## Context

The manuscript this project extends used leave-one-participant-out CV with
elastic net, and reported a moderate result for K6 and a negative R² for SRS-2.
Comparability with that is worth a lot, so it is the primary estimate here too.
But leave-one-out has two known problems that matter at N=62, and a third
turned up while implementing it.

## Decision

### Leave-one-out is primary; repeated k-fold is reported beside it

Leave-one-out is nearly unbiased for the *level* of prediction error and
notoriously high-variance for *comparing* models: each of the 62 training sets
differs from the others by one observation, so the fits are highly correlated
and the variance of the difference between two models is poorly estimated.

Both are therefore reported: leave-one-out for comparability, and repeated
5-fold over 20 repeats beside it. **Where the two disagree about the sign, the
disagreement is the finding** and the summary says so. A result that is
positive under one scheme and negative under the other is not a result about
the features.

### Comparisons are paired, per session, and non-parametric

The confirmatory tests are comparisons between feature sets, not tests of a
single model. Comparing two summary R² values throws away the pairing: both
models predicted the same sessions from the same folds, and between-session
variance dominates at this sample size.

So each confirmatory test is a **Wilcoxon signed-rank test on per-session
absolute errors**, paired across the same folds. Non-parametric because
absolute errors are bounded below and skewed, and 62 of them do not make a
t-test's assumptions true. The four tests are corrected together by Holm.

### The permutation null uses k-fold, and says so

A permutation null under leave-one-out would be 1000 x 62 model fits per test.
The null is a property of the *scheme*, so it is computed under 5-fold, and the
observed value it is compared against is computed the same way. Both numbers
appear in the results table with the scheme named, so nobody compares a
leave-one-out R² against a k-fold null by accident.

### The elastic net's penalty path is bounded at eps = 1e-2

This one was found by measurement, not by reasoning, and it is the reason this
ADR exists.

`ElasticNetCV`'s `eps` is the ratio of the weakest penalty tried to the
strongest - the strongest being the one that zeroes every coefficient.
sklearn's default is 1e-3, so the path explores penalties a thousand times
weaker than that: effectively unpenalised least squares.

With 54 exploratory features on 62 sessions, that end of the path is
indefensible statistically *and* pathological computationally. Coordinate
descent stops converging there: measured on this shape, the default settings
produced **124 non-converged fits per outer fold**, each returning whatever
coefficients the iteration limit happened to leave behind. The fits did not
fail, no error was raised, and the resulting R² looked like any other number.

Bounding the path at `eps = 1e-2` still spans two orders of magnitude of
regularisation, converges everywhere tested, and is 16 times faster in the
p ≈ n regime. The l1 ratio grid is four values rather than six, and the path
has 50 alphas rather than 100, both of which are ample over two decades.

## Consequences

* The reported estimates come from converged fits. This was not true before the
  bound, and nothing in the output would have revealed it.
* The analysis is confined to genuinely regularised models. At 54 features on
  62 sessions that is the only defensible region anyway, so the bound removes
  something we would not have wanted to use.
* `vc model` takes tens of minutes on the full cohort, dominated by the
  permutation nulls, and logs each estimate as it lands. Silence for that long
  from a command someone else runs blind is indistinguishable from a hang.
* Imputation and scaling are pipeline steps, so they are fitted on the training
  fold only. A test asserts this directly by recording every row each transform
  was fitted on: leakage leaves no trace in the output, so it has to be caught
  by construction.
* Nothing here rescues a small sample. The point of the tiering, the pairing and
  the correction is to make a modest test honest, not to make it significant.
