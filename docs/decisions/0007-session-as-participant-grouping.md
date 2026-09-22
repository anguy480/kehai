# 7. Treat each session as one participant, overridable by a map

- Status: accepted
- Date: 2026-09-23

## Context

The analysis uses leave-one-participant-out cross-validation. That equals
leave-one-session-out only if no participant appears in more than one session.
If any participant recurs, ordinary session-wise cross-validation leaks: the same
person's data would appear in both the training and test folds, and the reported
metrics would be optimistic.

The lab's grant materials describe the dataset as N=62 participants, and there
are 62 sessions with unique IDs, so one session per participant is the
documented structure. It has not yet been confirmed with the professor, and the
two recruitment waves make an unnoticed overlap conceivable.

## Decision

Default to `model.grouping: session`, i.e. one participant per session. Support
`model.grouping: participant_map` with a `participant_map.csv`
(`session_id,participant_id`) that overrides the default when supplied. The
grouping actually used is recorded in the manifest and in the results summary.

## Consequences

- The default matches the documented dataset and needs no extra file.
- If the assumption turns out to be wrong, the fix is a CSV rather than a code
  change, and previously reported results can be identified by the grouping
  recorded in their manifest.
- Cross-validation code always groups by participant ID, even when that ID is
  derived from the session ID, so there is one code path rather than two.
