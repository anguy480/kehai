# 12. Reduce the analysis surface by tiering, not by capping features

- Status: accepted
- Date: 2026-09-23
- Supersedes the enforcement mechanism of
  [ADR 6](0006-feature-budget-for-n62.md); the naming convention and family
  structure there still stand.

## Context

With turns (12) and prosody (12) built and face still to come, the feature
table will hold roughly 48 columns for 62 sessions. ADR 6 answered this with a
hard ceiling of 60 features, enforced as an error. That is a blunt instrument,
and it addresses the wrong quantity.

Counting columns is not the risk. The risk is the size of the *analysis
surface*. As configured, `vc model` compares 8 feature sets against 2 targets
with 2 model families: **32 cross-validated estimates**. Each is a
leave-one-out R² on 62 observations, which is nearly unbiased but has high
variance, so the best of 32 will look good whether or not anything is there.
The manuscript being compared against already reported a negative
cross-validated R² for SRS-2 from text features, which makes a positive result
from some corner of 32 attempts both tempting and easy to obtain.

The two remedies under consideration solve different problems, and conflating
them would leave the main one untreated.

**Nested feature selection inside the CV folds** addresses *leakage* and the
within-model p-versus-n problem. It is necessary and is already the design
(ADR 6, with a leakage test). But it does nothing about multiplicity: selecting
features inside folds still leaves 32 reported comparisons. It also has costs
that are easy to overlook at this sample size. Selection inside 62 folds
returns a different subset per fold, so "which features matter" has no stable
answer to report, and the inner loop is fitting on 61 observations, where
selection is close to noise.

**Pre-registering a small primary set** addresses multiplicity directly, which
is the dominant threat. Its usual weakness is that it depends on a promise:
readers must take on trust that the primary set was chosen before the outcomes
were seen.

That weakness does not apply here. Under [ADR 1](0001-handoff-split-no-labels-on-student-machine.md)
the extraction half runs on a machine that has never held the questionnaire
scores, and the handoff manifest records the commit that produced the features.
A confirmatory feature set is therefore credible *by construction* rather than
by declaration: tuning features against the outcome is not something that was
avoided, it is something that could not have happened. That makes
pre-registration unusually strong in this project, and it is the argument to
put in front of a reviewer.

## Decision

A two-tier analysis, with nested selection retained for what it is actually
good for.

### Confirmatory tier

* **A primary feature set of three features per family**, named in
  `model.tiers.primary_features`, chosen on prior literature rather than on
  this data. Twelve features for the combined set, so roughly five observations
  per feature.
* **Four primary tests**: for each target, the combined new-modality set
  against the text baseline. Not 32.
* Each primary test reports R², MAE and Spearman with a permutation null, and
  **Holm correction across the four**, with the family of tests stated.
* The face families' three features each are now fixed, in
  [ADR 13](0013-face-features-follow-the-lab-precedent.md), from the AUs the
  lab's own published study found predictive of social performance. Still
  before any label was seen; the manifest records the commit, so the ordering
  is verifiable rather than asserted.

The initial primary features, and why each:

| Family | Feature | Rationale |
| --- | --- | --- |
| turns | `turns__latency_median` | Response timing is the most theory-central turn-taking measure for social communication. |
| turns | `turns__participant_speaking_ratio` | Conversational reciprocity and balance. |
| turns | `turns__overlap_ratio` | Simultaneous speech indexes turn-taking coordination. |
| prosody | `prosody__f0_semitone_sd` | Reduced pitch variability is the most replicated acoustic correlate of autistic-type speech, so it is the natural primary for SRS-2. |
| prosody | `prosody__speech_rate_proxy` | Slowed speech is a long-standing marker of psychological distress, so it is the natural primary for K6. |
| prosody | `prosody__intensity_sd_db` | Reduced vocal dynamism, the amplitude counterpart of the above. |

### Exploratory tier

* Everything else: the full feature table, the remaining feature sets, and the
  random forest.
* Reported **with the number of comparisons stated**, as exploratory, with no
  confirmatory language and no inference from the best of them.
* This is where an unexpected result is allowed to be interesting without being
  a finding.

### Nested selection, as a sensitivity analysis

* Kept, inside the folds, with the leakage test extended to cover selection and
  not only scaling.
* Reported as a robustness check on the confirmatory conclusion — does it
  survive letting the data choose? — rather than as the primary mechanism.
* Reported with **selection frequency per feature across folds**, which is the
  honest way to present an unstable selector: a feature retained in 60 of 62
  folds means something, one retained in 31 does not.

### What replaces the cap

`aggregate.max_features` stays, raised to a sanity ceiling rather than a
scientific argument. Its job is now to catch a bug that generates hundreds of
columns, not to make the analysis defensible. The tiering does that.

## Consequences

* The headline result is four tests with a stated correction, which is what a
  reviewer can evaluate. The other 28 estimates remain available and are
  labelled for what they are.
* A negative confirmatory result stays interpretable. If the twelve primary
  features do not beat text on SRS-2, that is a clean finding given the prior
  negative result, rather than a failure to search hard enough.
* Choosing the primary features is now a decision with a date and a commit
  behind it. Changing one later is possible but visible, and a change made
  after any label has been seen moves the analysis into the exploratory tier.
* Interpretability is retained. A per-family principal component would reduce
  dimension further with less selection risk, but "the first component of the
  facial features" is not something a clinical reader can act on, and the
  feature dictionary would lose its meaning.
* Leave-one-out remains the primary estimator, for comparability with the
  manuscript. Because it is high-variance for *comparing* models, repeated
  5-fold is reported alongside it as a stability check. Where the two disagree,
  the disagreement is the result worth reporting.
* None of this rescues a small sample. Sixty-two participants supports a
  modest, well-specified test and no more; effect sizes are reported with
  intervals, not as a significance verdict.

## Amendment, 2026-09-24: one confirmatory turn feature replaced

This ADR says above that changing a primary feature later must be visible and
dated, and that a change made after any label has been seen moves the analysis
into the exploratory tier. This is that record, and no questionnaire score has
been seen by anyone running this pipeline.

`turns__overlap_ratio` was a confirmatory feature. When the lab's original
whisper-diarization output arrived it turned out that the diarizer *partitions*
time: every moment is assigned to exactly one speaker, so two speakers never
overlap by construction. Measured across all 62 sessions, pairwise speaker
overlap is exactly 0.0000 seconds. The feature is therefore structurally
constant at zero, along with `turns__interruption_rate`, which is defined from
the same quantity.

A constant feature in a pre-registered slot is worse than a weak one: it cannot
support or refute anything, and it silently spends one of the four corrected
tests. The slot goes to `turns__n_per_minute` (turn-taking rate), which is also
theory-central for social communication - conversational pace and reciprocity
are part of what SRS-2 asks about - and which does vary across the cohort, 33
to 149 diarized segments per session.

Both overlap features stay in the feature table at zero rather than being
dropped. Deleting them would hide the limitation; a column of zeros with a
feature-dictionary entry saying why records it. If a future run uses a diarizer
that permits overlap (pyannote does), the features become live without a schema
change, and a note in the handoff README says so.

The general lesson is automated rather than remembered: `vc aggregate` now
detects any feature with no variance across sessions and escalates in its
summary when that feature holds a confirmatory slot. This class of problem is
now caught by the pipeline instead of noticed by a reader.
