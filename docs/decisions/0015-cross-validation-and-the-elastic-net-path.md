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

## Amendment, 2026-09-29: the bound was not sufficient on the real matrix

This ADR claimed the bounded path "converges everywhere tested". That claim was
too general, and a run on the real feature table showed it: convergence warnings
reappeared, 13 fits in the 54-feature exploratory set and 1 in the 74-feature
one.

The claim was tested on **synthetic** matrices of independent normal columns.
The real feature matrix is correlated by construction - a mean and a standard
deviation of the same action unit, a mean and a median of the same latency, two
facial windows of the same session - and median imputation makes some columns
nearer still. Coordinate descent on near-duplicate columns needs more iterations
than on independent ones, whatever the penalty range.

### What changed

**The iteration cap is raised from 20,000 to 200,000.** Measured on the real
matrix this converges everywhere, and it costs nothing: 52.7s against 51.9s for
the same 62 leave-one-out fits, because the fits that needed the headroom were a
small minority.

Raising the cap is deliberately preferred over restricting the path further.
Tightening `eps` from 1e-2 to 5e-2 also achieves convergence and is six times
faster, but it is a *second* change to which models the search may consider,
and the speed is not needed. Raising the cap changes nothing about the analysis;
it only lets the optimiser finish the work it was already asked to do. If the
cost ever becomes a problem, `eps` is where to look, and it should be a recorded
decision rather than a performance tweak.

### The durable part

A cap is a guess about a future table, so non-convergence is now **counted and
reported** rather than warned about once. Each estimate carries
`n_fits_not_converged` into `model_results.csv`, so a reader can see whether a
number rests on fits the optimiser had not finished. The count exists because
the previous behaviour - a wall of repeated sklearn text during a run that takes
tens of minutes - is trivially scrolled past, and because any cap chosen today
may be too small for a table with more features or more collinearity.

A non-zero count is not automatically fatal. It says that part of that estimate
rests on coefficients the optimiser was still moving, which is a reason to
discount that estimate specifically, not the run.
