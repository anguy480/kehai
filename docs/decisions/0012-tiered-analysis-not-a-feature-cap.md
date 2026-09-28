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

## Amendment, 2026-09-29: the confirmatory tier was unreachable

This ADR describes a confirmatory tier of twelve pre-registered features and
four corrected tests. The implementation did not produce it, and a smoke run on
the real feature table with randomly generated labels is what revealed it.

A feature set was treated as confirmatory only if **all** of its columns were
among the pre-registered twelve. No set can satisfy that: `all` carries 54
columns, `audio` 24, and the text baseline's columns are not pre-registered
features at all. So the condition never held, and three things followed.

* Every one of the 32 estimates was tiered exploratory.
* The permutation null is gated on the confirmatory tier, so `n_permutations:
  1000` was silently ignored for the whole run. No null was ever computed.
* The four reported "confirmatory tests" compared the **full** feature sets -
  54 features against 20 - rather than the pre-registered set. The headline
  tests were the exploratory analysis wearing the confirmatory label.

Nothing was lost: the run used random labels, and no questionnaire score has
been seen by anyone on the extraction side. That is also what makes fixing it
legitimate now rather than post hoc, and it is precisely the window this ADR
exists to protect. It closes the first time the analysis is run against real
outcomes.

### Who chose this, and what is still open

**This choice was made by the assistant implementing the fix, not by the
project's owner.** It is recorded that way because this ADR requires a change to
the confirmatory tier to be visible and attributed, and an unattributed change is
exactly what it is written to prevent. Three options were drafted and one was
taken; the record should not imply that anyone else weighed them.

The reasoning for the option taken:

* The bug had to be fixed either way. Leaving the tier unreachable meant the
  headline tests were the exploratory analysis under a confirmatory label, which
  is worse than any of the alternatives.
* Restricting our own families to their pre-registered features is what this ADR
  already says the confirmatory tier is, so that part restores the stated design
  rather than choosing a new one.
* The text baseline entering whole is the genuinely new judgement. The
  alternative - pre-registering a subset of the manuscript's text features -
  would mean selecting the baseline we are measured against, on no prior basis,
  which is a larger liberty than the asymmetry it would remove.
* A third option, accepting full feature sets as confirmatory and amending this
  ADR to match, was rejected because 54 features on 62 sessions is the regime
  this ADR exists to avoid.

**This remains open for the project owner to overrule.** No questionnaire score
has been seen by anyone on the extraction side, so the choice can still be
changed without becoming post hoc - and that is the only reason it was safe to
implement before being reviewed. The window closes the first time the analysis
runs against real outcomes, and if the decision is to be revisited it should be
revisited before then.

### What a confirmatory comparison uses

A set named in a pre-registered comparison is now evaluated **twice**:
confirmatory on its pre-registered columns, and exploratory on all of them. Both
are wanted, and computing them separately is what stops one becoming the other.

Per family, the confirmatory variant keeps the pre-registered features where
that family has them, and the whole family where it does not:

| set | confirmatory | exploratory |
| --- | --- | --- |
| `all` | 12 | 54 |
| `audio` | 6 | 24 |
| `text` | 20 | (same, so evaluated once) |

**The text baseline enters whole, and the asymmetry is deliberate.** We
pre-registered a small set of features from our own families on prior
literature. We never pre-registered a subset of the manuscript's text features,
and choosing one now would mean selecting the baseline we are measured against,
on no prior basis - a far worse liberty than the asymmetry.

### Consequences

* 40 estimates rather than 32, three of them confirmatory variants, and six
  permutation nulls that now actually run.
* The comparisons are keyed by tier as well as by feature set, so a
  confirmatory test cannot pick up the exploratory variant of the same set.
* A run in which no set reaches the confirmatory tier now says so in its notes
  rather than reporting confirmatory tests built from exploratory estimates.
