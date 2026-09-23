# 9. Verify the psychiatrist's side by label OCR, with the assumption as fallback

- Status: accepted
- Date: 2026-09-23

## Context

Every session checked by hand showed the psychiatrist in the LEFT tile. That is
useful, but it is a sample, not a guarantee: 62 sessions were recorded in two
waves months apart, and a single swapped session would silently invert the
speaking/listening split for facial features and send participant-only prosody
through the wrong speaker.

Zoom draws a name label in each tile, so the layout can be checked rather than
assumed. Doing so raises a privacy problem: those labels are people's names,
including participants'.

## Decision

`vc verify-layout` reads the name label in each tile and decides per session.

- **OCR is primary.** Where it reaches a conclusion, that conclusion is what
  gets recorded.
- **`speakers.assumed_psychiatrist_side` (default `left`) is a fallback only**,
  used where OCR is unavailable or inconclusive.
- **A disagreement is a QC flag, never a silent override.** `layout.csv` records
  the side OCR found, the assumed side, whether they matched, and the flags. A
  mismatching session keeps OCR's answer and is flagged
  `layout_side_mismatch`.

**The psychiatrist is identified without ever being named.** They appear in
every session; each participant appears in one. So the label that *recurs*
across sessions is the psychiatrist's. This is the primary rule, and it means no
name is needed anywhere. Explicit label fragments may be supplied through the
environment variable named by `speakers.label_ocr.psychiatrist_label_env`, read
from the gitignored `.env`. Only the variable's *name* appears in committed
config: a real person's name must never enter this repository.

**Recognised text never leaves the process.** It is normalised and compared in
memory. `layout.csv` has no text column, log records carry counts only, and the
summary reports counts and session IDs. A test plants distinctive label text and
asserts it appears in neither the table, the logs, nor the stage messages.

Backends are pluggable: `apple_vision` (default; on-device macOS Vision via the
optional `ocr` extra, supports `ja-JP`), `tesseract` (external binary if
installed), and `none`. An unavailable backend is a reported state, not an
error, so the fallback path is ordinary rather than exceptional.

## Consequences

- The assumption gets tested across all 62 sessions instead of trusted. If every
  session comes back left, it is confirmed and later stages can rely on it.
- Running on a subset cannot settle anything, because recurrence needs a cohort.
  The stage says so explicitly rather than reporting a bare "inconclusive".
- Two psychiatrists across the two waves would produce two recurring labels,
  which the rule handles without change.
- OCR depends on `speakers.label_ocr.label_region` matching where Zoom actually
  draws the label; that is a configured guess to be checked against the preview
  sheets, and a wrong region shows up as every session being inconclusive rather
  than as a wrong answer.
- Frames are decoded to read labels. They are extracted, read and deleted one at
  a time, and the Vision backend converts them in memory with no file and no
  image encoding, so the "never save a frame" rule still holds.
- This check is independent of, and complementary to, the mouth-movement
  cross-check in ADR 8: labels identify tiles by name, mouth movement ties a
  diarized voice to a tile.
