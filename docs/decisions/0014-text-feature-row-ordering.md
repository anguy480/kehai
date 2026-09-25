# 14. Aligning the manuscript's text features, which carry no identifier

Date: 2026-09-25

## Status

Accepted.

## Context

The confirmatory comparisons in this project are *new modalities against the
text baseline* (`new_modalities_vs_text`, `audio_vs_text`, see
docs/decisions/0012). The baseline is the feature table from the lab's
manuscript, `nlp_features.csv`: 62 rows, 20 text features - lexical overlap,
Sentence-BERT similarity, MTLD, turn statistics, and six LLM-rated agenda
scores.

It has no identifier column. No `session_id`, no filename, nothing. Sixty-two
rows beside a cohort of 62 sessions.

This is the most dangerous shape a data file can have in this project, and it
is worth being explicit about why. Every other error we might make announces
itself: a bad path raises, a schema violation is caught by a contract, a broken
feature is constant and now flagged by `vc aggregate`. A wrong row order does
none of that. It produces a complete table of the right shape with every value
in range, attached to the wrong people. Elastic net fits it, leave-one-out
scores it, the permutation baseline reports against it, and every number is
plausible. The most likely symptom is a null result - the same null result the
manuscript already found for SRS-2 - which we would have no way to distinguish
from a real one.

There is no statistical defence against this. It cannot be detected from the
data, because a permuted assignment of 62 rows is exactly as well-formed as the
correct one.

## Decision

**Rows are matched to sessions by an explicit identifier, or by a named
ordering rule whose provenance is recorded. Never by assumption.**

Identifier mode is preferred and always wins where a column exists, because it
rests on nothing but the file itself.

For this file, positional mode applies, under these conditions - all of them,
every run:

1. **The rule is named, not inferred.** `numeric_ascending_session_id`, a
   `Literal` in the config, so a second rule cannot appear by accident.

2. **The provenance is quoted, not paraphrased.** The lab supplied the code that
   wrote the file:

   > Rows were written by iterating transcript files with
   > `sorted(dir_path.glob(extension), key=lambda p: int(p.stem))`, so the order
   > is numeric ascending by session ID across the whole set
   > (1, 3, 6 ... 62, 102 ... 261), not lexicographic.

   That wording is in `config/default.yaml` and is copied verbatim into the run
   manifest. The config refuses a provenance string too short to say where a
   rule came from: an assertion is not evidence.

3. **The order is reproduced from the source they iterated**, not from our own
   inventory: `sorted(int(path.stem))` over
   `diarization/diarizations_original/*.srt`. Deriving it from our inventory
   would hide precisely the failure worth catching, because our inventory would
   agree with itself whether or not it agreed with theirs.

4. **The counts must match exactly.** The `.srt` files must number 62, the
   confirmed count, and the table must have 62 rows. Either one moving stops
   the join with a message naming which number changed. A file we can see whose
   stem is not a session ID also stops it, because the lab's `int(path.stem)`
   would have raised on that file, so its presence means the two sets differ.

5. **The absence of an identifier remains a refusal by default.** A future table
   with no identifier and no configured rule is refused with a message
   explaining that row order is not an identifier.

The distinction in (2) is not pedantry. Lexicographic and numeric ordering
disagree on this cohort, because the IDs span 1-62 and 102-261: lexicographic
puts 102 before 62. Getting that wrong would misalign 26 of the 62
participants while leaving the rest correct - a partial corruption, which is
harder to notice than a total one.

## Consequences

* The alignment is auditable after the fact. The manifest carries the rule, the
  quoted provenance, the source it was reproduced from, the file's SHA-256, and
  the full resolved session order. A reviewer asking "how did you align two
  tables when one had no IDs?" gets an answer with a date and a source, not a
  reassurance.
* The guards are load-bearing rather than decorative: adding or removing one
  transcript file, or receiving a re-exported table with a different row count,
  stops the analysis instead of silently shifting every subsequent row.
* We still depend on a human statement. Reproducing the lab's sort makes that
  statement checkable, and the count guards make a divergence loud, but if the
  quoted description of their code were wrong, we would inherit the error. The
  only complete fix is an identifier column, which has been requested; this
  design is what makes the interim defensible rather than what makes it safe.
* A session in our feature table that the text run did not cover is reported as
  a warning rather than an error. It does not affect the alignment of the rows
  that did match - alignment is against the transcript set, not against our
  table - but it is evidence the two runs saw different data and is worth
  seeing.
* If the lab later supplies the table with `session_id`, nothing needs
  rewriting: `identification: "identifier"` uses it and the ordering rule stops
  being consulted. That path is already tested.
