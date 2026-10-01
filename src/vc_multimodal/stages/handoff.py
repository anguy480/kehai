"""Build the bundle the label holder receives.

This stage is the seam between the two halves of the project (ADR 0001). One
half extracts features from video on a machine that never holds a
questionnaire score; the other fits models on a machine that holds the scores
and never the video. This is what crosses between them.

What that means for this code:

* **Nothing sensitive crosses.** The bundle carries session-level numbers, a
  dictionary explaining them, quality columns, and a manifest. No audio, no
  frames, no transcripts, no free text from a recording. The checks for that
  are in `_refuse_unsafe_columns` and are deliberately paranoid, because a
  bundle is a file that gets emailed.
* **No label crosses in either direction.** A column that looks like a
  questionnaire outcome stops the build. The extraction half should never have
  one, so if it does, something is wrong upstream and shipping it would make it
  permanent.
* **It has to explain itself.** The person who runs the analysis did not write
  this code. Every column has a description, every limitation that would change
  how a result is read is in the README, and the manifest says which commit,
  which tools and which model files produced the numbers.

The bundle is written to a temporary directory and renamed into place, so an
interrupted build leaves no half-finished bundle for someone to send.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from vc_multimodal import feature_dictionary, handoff_text
from vc_multimodal.config import AppConfig
from vc_multimodal.contracts import ContractError
from vc_multimodal.feature_dictionary import DictionaryError
from vc_multimodal.io_utils import read_csv, write_csv, write_json, write_text
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.modeling.text_features import TextFeatureError, TextFeatures, load_configured
from vc_multimodal.modeling.text_features import label_like_columns as label_like
from vc_multimodal.modeling.tiers import describe_plan, resolve_tiers
from vc_multimodal.paths import DataRoots
from vc_multimodal.provenance import GitState, environment_record, git_state
from vc_multimodal.qc_notes import QcNotes
from vc_multimodal.qc_notes import load as load_qc_notes
from vc_multimodal.stages import aggregate as aggregate_stage
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import model as model_stage

logger = get_logger(__name__)

STAGE: Final = "handoff"
BUNDLE_DIRNAME: Final = "handoff"

FEATURES_FILE: Final = "features.csv"
DICTIONARY_FILE: Final = "feature_dictionary.csv"
QC_FILE: Final = "qc_report.csv"
TEXT_FILE: Final = "text_features.csv"
MANIFEST_FILE: Final = "manifest.json"
README_FILE: Final = "README.md"
CONFIG_FILE: Final = "config.snapshot.yaml"
QC_NOTES_FILE: Final = "qc_notes.csv"

#: Column-name fragments that must never appear in a bundle. Anything holding
#: what someone said, rather than a measurement of how they said it.
FORBIDDEN_FRAGMENTS: Final = (
    "transcript",
    "utterance",
    "text_content",
    "words_spoken",
    "speech_text",
    "caption",
    "srt",
    "name",
    "label_text",
    "ocr",
)


class HandoffError(RuntimeError):
    """Raised when a bundle cannot be built, or would not be safe to send."""


@dataclass(frozen=True, slots=True)
class HandoffResult:
    """What the stage produced."""

    path: Path
    n_sessions: int
    n_features: int
    files: tuple[str, ...]
    git: GitState | None
    text: TextFeatures | None
    tier_lines: tuple[str, ...]
    notes: tuple[str, ...]


def bundle_root(roots: DataRoots) -> Path:
    """Where bundles are collected."""
    return roots.out_path(BUNDLE_DIRNAME)


def bundle_name(git: GitState | None, *, now: datetime | None = None) -> str:
    """The bundle's directory name: the date it was built and the commit."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d")
    return f"{stamp}_{git.short if git is not None else 'nogit'}"


def _refuse_unsafe_columns(columns: Sequence[str]) -> None:
    """Stop the build if any column should not leave this machine.

    Two separate refusals, because they are two separate mistakes: a label in
    the extraction half means the halves have been mixed, and a text column
    means something derived from a transcript has leaked into the table.
    """
    outcomes = label_like(list(columns))
    if outcomes:
        msg = (
            f"the feature table contains column(s) that look like questionnaire "
            f"outcomes: {list(outcomes)}. The extraction half of this project never "
            f"holds a label (docs/decisions/0001), so this is a sign the two halves "
            f"have been mixed. Nothing is written."
        )
        raise HandoffError(msg)

    textual = sorted(
        name
        for name in columns
        if any(fragment in str(name).lower() for fragment in FORBIDDEN_FRAGMENTS)
    )
    if textual:
        msg = (
            f"the feature table contains column(s) that may carry what someone said "
            f"or who they are: {textual}. Transcripts and names never leave the work "
            f"root. Nothing is written."
        )
        raise HandoffError(msg)


def _load_features(roots: DataRoots) -> pd.DataFrame:
    """Read the feature table `vc aggregate` wrote."""
    path = aggregate_stage.features_path(roots)
    if not path.exists():
        msg = f"no feature table at {path.name}; run `vc aggregate` first"
        raise HandoffError(msg)
    try:
        frame = read_csv(path)
    except (OSError, ValueError) as exc:
        msg = f"could not read the feature table: {exc}"
        raise HandoffError(msg) from exc
    if frame.empty:
        msg = "the feature table has no rows; there is nothing to hand off"
        raise HandoffError(msg)
    return frame


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Separate the features from the quality columns.

    They go in different files because they answer different questions, and
    because a QC column silently modelled as a predictor would be a real bug:
    `qc__speaking_seconds` correlates with session length, not with anything
    about the participant.
    """
    identifiers = [name for name in ("session_id", "wave") if name in frame.columns]
    qc_columns = [name for name in frame.columns if str(name).startswith("qc__")]
    feature_columns = tuple(
        name
        for name in frame.columns
        if name not in identifiers and name not in qc_columns and "__" in str(name)
    )
    features = frame[[*identifiers, *feature_columns]].copy()
    qc = frame[[*identifiers, *qc_columns]].copy()
    return features, qc, feature_columns


def _text_features(config: AppConfig, roots: DataRoots) -> tuple[TextFeatures | None, list[str]]:
    """Load the manuscript's text features, if they are configured and present."""
    if config.model.text_features is None:
        return None, ["no text feature table is configured, so the bundle has no text baseline"]
    try:
        loaded = load_configured(roots.work, config.model.text_features)
    except TextFeatureError as exc:
        return None, [f"text features not included: {exc}"]
    return loaded, []


def _manifest(
    config: AppConfig,
    *,
    git: GitState | None,
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    text: TextFeatures | None,
    backends: Sequence[str],
    confirmed: QcNotes,
    now: datetime,
) -> dict[str, Any]:
    """Everything needed to know how these numbers were produced."""
    record: dict[str, Any] = {
        "built_at": now.isoformat(),
        "bundle_contract": 1,
        "git": git.record() if git is not None else None,
        "environment": environment_record(),
        "sessions": {
            "n": len(features),
            "session_ids": [int(v) for v in features["session_id"]],
        },
        "features": {
            "n": len(feature_columns),
            "names": list(feature_columns),
            "families": sorted({feature_dictionary.family_of(str(c)) for c in feature_columns}),
        },
        "face_backends": sorted(set(backends)),
        "seeds": {"seed": config.runtime.seed},
        "config": config.snapshot(),
    }
    record["text_features"] = text.manifest_record() if text is not None else None
    # The notes in full, so a bundle is a snapshot of what was annotated when.
    record["qc_notes"] = [note.row() for note in confirmed.notes]
    return record


#: Where a session's tracking quality is unusual enough to name individually.
_DROPPED_NOTABLE: Final = 0.05


def _dropped_frame_section(qc: pd.DataFrame) -> str:
    """Where each session sits on facial tracking quality.

    A distribution rather than a pass/fail line, because there is no principled
    cutoff: the cohort runs from a fraction of a percent to almost everything,
    and where to draw a line is the analyst's decision. What they need is to see
    the shape and be able to look any session up.
    """
    column = "qc__face_dropped_fraction"
    if column not in qc.columns:
        return ""
    values = pd.to_numeric(qc[column], errors="coerce").dropna()
    if values.empty:
        return (
            "### Face tracking quality\n\n"
            "No session in this bundle has a facial dropped-frame fraction "
            "recorded, so nothing here describes tracking quality.\n"
        )

    measured = len(values)
    unmeasured = len(qc) - measured
    lines = [
        "### Face tracking quality, session by session",
        "",
        f"`qc__face_dropped_fraction` is the share of sampled frames in which no "
        f"usable face was found. Measured for {measured} session(s)"
        + (f"; {unmeasured} session(s) have no facial measurements at all." if unmeasured else "."),
        "",
        "| | dropped frames |",
        "| --- | --- |",
        f"| best | {values.min():.1%} |",
        f"| 25th percentile | {values.quantile(0.25):.1%} |",
        f"| median | {values.median():.1%} |",
        f"| 75th percentile | {values.quantile(0.75):.1%} |",
        f"| worst | {values.max():.1%} |",
    ]

    notable = qc.loc[
        pd.to_numeric(qc[column], errors="coerce") > _DROPPED_NOTABLE,
        ["session_id", column],
    ].sort_values(column, ascending=False)
    if not notable.empty:
        lines.extend(
            [
                "",
                f"Sessions above {_DROPPED_NOTABLE:.0%}, which is well clear of the "
                f"rest of the cohort:",
                "",
                "| session | dropped frames | confirmed cause |",
                "| --- | --- | --- |",
            ]
        )
        reasons = (
            qc.set_index("session_id")["qc__annotation_reason"]
            if "qc__annotation_reason" in qc.columns
            else pd.Series(dtype="object")
        )
        for session_id, value in notable.itertuples(index=False):
            # An absent reason arrives as NaN from a CSV round-trip, and
            # str(nan) is "nan", which would print as this session's confirmed
            # cause. Check for absence before converting.
            raw = reasons.get(int(session_id))
            reason = "" if raw is None or pd.isna(raw) else str(raw).strip()
            lines.append(f"| {int(session_id)} | {float(value):.1%} | {reason or 'not checked'} |")
    else:
        lines.extend(["", f"No session is above {_DROPPED_NOTABLE:.0%}."])

    lines.extend(
        [
            "",
            f"Every session's value is in `{QC_FILE}`, so any of them can be looked "
            f"up rather than inferred from this summary.",
            "",
        ]
    )
    return "\n".join(lines)


#: Feature whose constancy has a specific explanation worth attaching to it.
_OVERLAP_FEATURE: Final = "turns__overlap_ratio"
_INTERRUPTION_FEATURE: Final = "turns__interruption_rate"


def _setup_section(git: GitState | None) -> str:
    """How to get the tool, for someone who has only ever received a bundle."""
    if git is not None and git.remote:
        name = git.remote.rstrip("/").rsplit("/", 1)[-1]
        clone = f"git clone {git.remote}.git\ncd {name}"
        commit = f"git checkout {git.short}   # the commit this bundle was built from"
    else:
        name = "repository"
        clone = "# obtain the pipeline repository from whoever sent this bundle"
        commit = ""

    return f"""## Before you run this

The analysis is one command from this project's own repository. Nothing in this
bundle runs on its own.

```
# 1. Install uv, which manages the Python version and the dependencies
#    (https://docs.astral.sh/uv/getting-started/installation/)

# 2. Get the code
{clone}
{commit}

# 3. Install everything. This needs Python 3.11, which uv fetches itself;
#    no system Python is used and no extras are needed for the analysis.
uv sync
```

Then run the analysis from the directory holding this bundle. That directory is
not inside the repository, so first tell uv where the repository is:

```
export UV_PROJECT=/path/to/{name}   # the repository from step 2
uv run vc model --features features.csv --labels <your labels file> --out results
```
"""


def _labels_section(config: AppConfig) -> str:
    """Exactly what the labels file has to look like."""
    targets = list(config.model.targets)
    header = ",".join(["session_id", *targets])
    example_values = ",".join(["1", *(str(6 + index * 30) for index in range(len(targets)))])
    accepted = ", ".join(f"`{name}`" for name in model_stage.LABEL_ID_COLUMNS)

    return f"""## The labels file

A CSV with one row per session: a session identifier, and one column per
outcome. The first two lines should look like this:

```
{header}
{example_values}
```

* The identifier column may be named any of {accepted}, and holds the session
  numbers used throughout this bundle.
* The outcome columns must be named exactly {", ".join(f"`{name}`" for name in targets)}.
  A configured outcome that is absent is reported and skipped; if none of them is
  present the command stops rather than guessing which column is which.
* Order does not matter. **Rows are matched by session identifier, never by
  position.**
* Extra columns are ignored, so a working spreadsheet can be used as it is.

**A session missing from the labels file is excluded from the analysis and named
in the output**, both in the printed summary and in the results. The reverse is
also reported: a session with a label but no features. Every figure in the
results is computed over the sessions present in both, and the count is stated,
so a smaller sample than expected is visible rather than silent.

**No label is ever written into any output of that command** - not the values,
not a per-session prediction, not a residual - and nothing needs to be sent
back.
"""


def _independence_section(text: TextFeatures) -> str:
    """Why the text baseline does not already contain the outcome.

    The confirmatory tests compare new modalities against this baseline, so a
    baseline built partly from the questionnaires would make those comparisons
    meaningless - and the two columns in question are named after the very
    constructs the outcomes measure. Recorded the same way as the ordering rule:
    quoted, attributed and dated, rather than asserted.
    """
    lines = [
        "### Why the text baseline is independent of the outcomes",
        "",
        "Every confirmatory test in this analysis compares the new modalities "
        "against the text features, so a text feature derived from a questionnaire "
        "would make those comparisons meaningless - the baseline would already know "
        "the answer.",
        "",
        "Two of the text columns are named after the constructs the outcomes "
        "measure, which is exactly what a leaked subscale would look like. The "
        "pipeline flags such names by default and refuses to use them; they are "
        "used here only because their provenance was confirmed by someone who knows "
        "how they were produced:",
        "",
    ]
    for entry in text.confirmations:
        columns = ", ".join(f"`{name}`" for name in entry.columns)
        lines.extend(
            [
                f"On {columns}:",
                "",
                f"> {entry.statement}",
                "",
                f"— {entry.confirmed_by}, {entry.confirmed_on}",
                "",
            ]
        )
    lines.extend(
        [
            f"The same statements are in `{MANIFEST_FILE}`. No other column in the "
            f"text table is exempt from that check, and nothing is exempt without a "
            f"statement like the above recorded against it.",
            "",
        ]
    )
    return "\n".join(lines)


def _confirmatory_plan_section(
    config: AppConfig, feature_columns: Sequence[str], text: TextFeatures | None
) -> str:
    """What each side of each confirmatory test actually is.

    Derived through the same functions the model stage uses, so the document and
    the analysis cannot describe different tests. Naming the comparisons alone -
    "all vs text" - invites the reading that all 54 features are on one side,
    which is not what a confirmatory comparison uses.
    """
    columns = [*feature_columns, *(text.feature_columns if text is not None else ())]
    sets = model_stage.resolve_feature_sets(pd.DataFrame(columns=columns), config)

    rows: list[str] = []
    for comparison in config.model.tiers.primary_comparisons:
        sizes = []
        for name in comparison.against:
            feature_set = sets.get(name)
            if feature_set is None or not feature_set.is_usable:
                sizes.append(f"`{name}` (not in this bundle)")
                continue
            restricted = model_stage.confirmatory_columns(feature_set, config)
            whole = len(feature_set.columns)
            sizes.append(
                f"`{name}`: {len(restricted)} of its {whole} feature(s)"
                if len(restricted) != whole
                else f"`{name}`: all {whole} feature(s)"
            )
        rows.append(f"| `{comparison.name}` | {sizes[0]} | {sizes[1]} |")

    if not rows:
        return ""

    plan = resolve_tiers(columns, config.model)
    n_tests = len(config.model.tiers.primary_comparisons) * len(config.model.targets)
    composition = ", ".join(
        f"{count} {source}" for source, count in plan.counted_by_source().items() if count
    )

    return (
        f"""### What each confirmatory test compares

**{n_tests} confirmatory test(s)**: {len(config.model.tiers.primary_comparisons)} """
        f"""comparison(s) on {len(config.model.targets)} target(s), corrected together
by {config.model.tiers.multiplicity_correction}. Everything else is exploratory:
{len(plan.exploratory)} of the {plan.n_features} feature(s) described here
({composition}), reported without confirmatory claims.

A confirmatory comparison does not use every feature of the sets it names. Our
own families are cut to the features pre-registered for them; the text baseline
enters whole.

| test | one side | against |
| --- | --- | --- |
"""
        + "\n".join(rows)
        + """

**The asymmetry is deliberate and it runs against us.** We pre-registered a small
set of features from our own families on prior literature. We never
pre-registered a subset of the manuscript's text features, and choosing one now
would mean deciding how strong the baseline we are measured against gets to be.
A baseline should be the strongest version of what it stands for, so it enters
whole, and our side of every confirmatory comparison is the smaller one.

The same feature sets are also evaluated unrestricted, and those estimates are
reported in the exploratory tier. See `docs/decisions/0012`.
"""
    )


def _zero_variance_section(features: pd.DataFrame, feature_columns: Sequence[str]) -> str:
    """Which features carry no information in *this* bundle.

    Derived rather than written in advance, and derived through the same
    function `vc aggregate` reports from, so the README and the stage cannot
    disagree. The previous version of this section was prose asserting that two
    turn features were constant; that was true of a three-session pilot and
    false of the full cohort, where one of the two varies.
    """
    constant = aggregate_stage.constant_features(features, feature_columns)
    lines: list[str] = []

    if constant:
        lines.extend(
            [
                f"### {len(constant)} feature(s) carry no information in this bundle",
                "",
                "These take the same value in every session, so no model can learn "
                "anything from them. They are listed in the feature dictionary like any "
                "other column and are kept in the table deliberately - a column of one "
                "value records a limitation, where dropping it would hide one.",
                "",
            ]
        )
        for name in constant:
            values = pd.to_numeric(features[name], errors="coerce").dropna()
            shown = f"{values.iloc[0]:g}" if not values.empty else "absent"
            lines.append(f"* `{name}` - every session is {shown}")
        lines.append("")
    else:
        lines.extend(
            [
                "### Every feature varies across this bundle",
                "",
                "No feature takes the same value in every session, so none of them is "
                "carrying zero information by construction.",
                "",
            ]
        )

    if _OVERLAP_FEATURE in constant:
        lines.append(handoff_text.OVERLAP_CONSTANT_NOTE)

    interruption = _interruption_lines(features, constant)
    if interruption:
        lines.extend(interruption)

    return "\n".join(lines)


def _interruption_lines(features: pd.DataFrame, constant: Sequence[str]) -> list[str]:
    """What to say about the interruption rate, given what it actually did.

    Three cases, and the difference between them matters to a reader: constant
    at zero, varying but mostly zero for a reason, or genuinely varying.
    """
    if _INTERRUPTION_FEATURE not in features.columns:
        return []
    values = pd.to_numeric(features[_INTERRUPTION_FEATURE], errors="coerce").dropna()
    if values.empty or _INTERRUPTION_FEATURE in constant:
        # Constant: already listed above, and the floor note explains why.
        return [handoff_text.INTERRUPTION_FLOOR_NOTE] if not values.empty else []

    non_zero = int((values > 0).sum())
    return [
        f"`{_INTERRUPTION_FEATURE}` is not constant, but it is zero in "
        f"{len(values) - non_zero} of {len(values)} session(s), and its largest value "
        f"is {values.max():.3f} events per minute - roughly one detected event in a "
        f"ten-minute conversation.",
        "",
        handoff_text.INTERRUPTION_FLOOR_NOTE,
    ]


def _readme(
    config: AppConfig,
    *,
    result_files: Sequence[str],
    features: pd.DataFrame,
    qc: pd.DataFrame,
    feature_columns: Sequence[str],
    text: TextFeatures | None,
    git: GitState | None,
    confirmed: QcNotes,
    now: datetime,
) -> str:
    """The document the analyst actually reads."""
    commit = git.short if git is not None else "unknown (not a git checkout)"
    dirty = " (built from a MODIFIED working tree)" if git is not None and git.is_dirty else ""
    n_confirmatory = len(config.model.tiers.primary_columns)

    rows = [
        (
            FEATURES_FILE,
            f"One row per session, {len(feature_columns)} feature columns. The predictors.",
        ),
        (
            DICTIONARY_FILE,
            "What every column means, its unit, and whether it is confirmatory or "
            "exploratory. Read this before modelling.",
        ),
        (
            QC_FILE,
            "Quality columns per session: how much speech and face each row rests on, "
            "and any flags. **Not predictors.**",
        ),
        (
            TEXT_FILE,
            "The manuscript's text features, with session IDs attached. The baseline "
            "for the comparisons.",
        ),
        (
            MANIFEST_FILE,
            "Commit, tool versions, model files, seeds, and the full resolved configuration.",
        ),
        (CONFIG_FILE, "The configuration as it ran, in readable form."),
        (
            QC_NOTES_FILE,
            "Findings a person confirmed by watching a recording, with who "
            "confirmed each and when. A session marked unavailable has those "
            "features withheld from `features.csv` on purpose.",
        ),
    ]
    setup_section = _setup_section(git)
    labels_section = _labels_section(config)
    files_table = "\n".join(
        ["| File | What it is |", "| --- | --- |"]
        + [f"| `{name}` | {what} |" for name, what in rows if name in result_files]
    )

    parts: list[str] = [
        f"""# Feature bundle: multimodal features for the ICU conversation study

Built {now.strftime("%Y-%m-%d")} from commit `{commit}`{dirty}.

This bundle holds **{len(feature_columns)} session-level features for
{len(features)} sessions**, extracted from the recordings. It holds **no
questionnaire scores**, by design: the extraction half of this project never
had access to them (see `docs/decisions/0001` in the repository).

{setup_section}
{labels_section}

## Files

{files_table}

## Read this before interpreting anything

### The confirmatory features are the result; the rest are exploratory

{n_confirmatory} features across four families are pre-registered as
confirmatory, chosen from prior literature before any label was seen. They are
marked in `{DICTIONARY_FILE}`. The other features are exploratory: report them
as such, with the number of comparisons stated.
""",
    ]

    parts.append(_dropped_frame_section(qc))

    parts.append(
        """### Quality columns are not predictors

`qc__*` columns describe how much evidence each row rests on.
`qc__speaking_seconds` mostly measures how long the session was, and
`qc__face_measured_speaking` mostly measures whether the camera was pointed at
the participant. Modelling them would produce a real-looking result about
recording conditions. They are in a separate file for that reason.

A row with entries in `qc__stages_missing` has features absent by cause, not by
chance. Which sessions to exclude is your decision; make it once, from the QC
table, and state it.
"""
    )

    parts.append(_zero_variance_section(features, feature_columns))

    if confirmed:
        parts.append(
            "### Sessions a person checked and marked unusable\n\n"
            "Each of these was watched, and the finding recorded before any "
            "questionnaire score was seen by anyone on the extraction side.\n\n"
            "**A modality marked `unavailable` has its features blank in "
            f"`{FEATURES_FILE}` on purpose.** They were measured, judged unusable and "
            "withheld; they are not missing through a bug, and they should not be "
            "imputed from the reason given here. Every other modality for these "
            "sessions is unaffected and should be used normally.\n\n"
            "| session | modality | status | reason | confirmed by | date |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            + "\n".join(confirmed.markdown_rows())
            + f"\n\nThe same rows are in `{QC_NOTES_FILE}` and in "
            f"`{MANIFEST_FILE}`, and `qc__annotations` in `{QC_FILE}` carries them "
            "per session."
        )

    parts.extend(handoff_text.notes())

    if text is not None and text.plan is not None:
        parts.append(
            f"""### How the text features were aligned

`{TEXT_FILE}` began life with no identifier column at all: one row per session
in the manuscript's own order, and nothing naming which session. Rows were
matched under an ordering rule supplied by the lab, quoted in full in
`{MANIFEST_FILE}`:

> {text.plan.provenance}

The order was reproduced by sorting the transcript files numerically by session
ID, and the build refuses to proceed unless the file count and the row count
both equal {text.plan.expected_rows}. **The session IDs are now written into
`{TEXT_FILE}` explicitly**, so this ambiguity does not recur downstream. See
`docs/decisions/0014` for why this was handled so carefully.
"""
        )

    if text is not None and text.confirmations:
        parts.append(_independence_section(text))

    if text is None:
        parts.append(
            """### The text baseline is not in this bundle

The manuscript's text features could not be included, so the comparisons
against text cannot be run from this bundle alone. The reason is in the build
log and in the notes below.
"""
        )

    parts.append(
        """### What this bundle cannot tell you

* **Sixty-two participants.** This supports a modest, well-specified test and
  no more. Effect sizes with intervals, not significance verdicts.
* **One session per participant.** There is no test-retest information, so
  nothing here separates a stable trait from how someone was on the day.
* **Session length varies.** Features are rates and ratios rather than totals
  for that reason, but a 4-minute session is still thinner evidence than a
  16-minute one. `qc__speaking_seconds` is how you see that.
* **These are Zoom recordings.** Gain, framing and network conditions vary
  between sessions and are not controlled. Absolute intensity in particular is
  not comparable across sessions.
"""
    )

    parts.append(_confirmatory_plan_section(config, feature_columns, text))

    parts.append(
        f"""## Provenance

Everything needed to reproduce these numbers is in `{MANIFEST_FILE}`: the
commit, the Python and package versions, the facial and prosodic backends with
their versions, the pinned model file and its digest, the random seed, and the
full configuration. If a number here needs to be defended, that file is where
the answer is.

Files in this bundle: {", ".join(f"`{name}`" for name in result_files)}.
"""
    )
    return "\n".join(part.strip() + "\n" for part in parts)


def _check_repository(repo: Path, *, allow_dirty: bool) -> tuple[GitState | None, list[str]]:
    """Establish what commit the bundle can honestly claim.

    A dirty tree is refused rather than noted, because the commit hash is the
    bundle's main provenance and a hash that does not describe the code that
    ran is worse than no hash at all. `--allow-dirty` exists for a pilot, and
    the bundle says so when it is used.
    """
    git = git_state(repo)
    notes: list[str] = []
    if git is None:
        notes.append("not a git checkout, so the bundle cannot name the commit that produced it")
        logger.warning("%s: %s", STAGE, notes[-1])
        return git, notes
    if git.is_dirty and not allow_dirty:
        msg = (
            f"the working tree has {len(git.dirty_paths)} uncommitted change(s), so the "
            f"commit recorded in the bundle would not describe the code that built it: "
            f"{list(git.dirty_paths[:10])}. Commit them, or pass --allow-dirty to build "
            f"anyway and have the bundle say so."
        )
        raise HandoffError(msg)
    if git.is_dirty:
        notes.append(
            f"built from a modified working tree ({len(git.dirty_paths)} uncommitted "
            f"change(s)); the manifest and README both record this"
        )
        logger.warning("%s: %s", STAGE, notes[-1])
    return git, notes


def _prepare_target(
    roots: DataRoots, git: GitState | None, *, moment: datetime, force: bool
) -> tuple[Path, Path]:
    """The bundle's final path, and the staging directory it is built in."""
    target = bundle_root(roots) / bundle_name(git, now=moment)
    if target.exists() and not force:
        msg = (
            f"a bundle for today and this commit already exists at {target.name}. Two "
            f"bundles with the same name are two things that cannot be told apart "
            f"later. Pass --force to replace it, or commit your changes so the name "
            f"differs."
        )
        raise HandoffError(msg)
    if target.exists():
        logger.warning("%s: replacing the existing bundle at %s", STAGE, target)
    staging = target.with_name(f"{target.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return target, staging


def _write_tables(
    staging: Path,
    *,
    config: AppConfig,
    features: pd.DataFrame,
    qc: pd.DataFrame,
    dictionary: pd.DataFrame,
    text: TextFeatures | None,
    confirmed: QcNotes,
) -> list[str]:
    """Write every table the bundle carries, returning what was written."""
    written = [FEATURES_FILE, QC_FILE, DICTIONARY_FILE]
    write_csv(staging / FEATURES_FILE, features)
    write_csv(staging / QC_FILE, qc)
    write_csv(staging / DICTIONARY_FILE, dictionary)
    if text is not None:
        write_csv(staging / TEXT_FILE, text.frame)
        written.append(TEXT_FILE)
    write_text(staging / CONFIG_FILE, config.to_yaml())
    written.append(CONFIG_FILE)
    if confirmed:
        # The notes themselves, not only the compact QC columns: a reason is a
        # sentence, and whoever reads a blanked modality needs it whole.
        write_csv(staging / QC_NOTES_FILE, pd.DataFrame([note.row() for note in confirmed.notes]))
        written.append(QC_NOTES_FILE)
    return written


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    repo: Path | None = None,
    allow_dirty: bool = False,
    force: bool = False,
    now: datetime | None = None,
) -> HandoffResult:
    """Assemble the handoff bundle.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        repo: The repository to record, defaulting to the current directory.
        allow_dirty: Build even though the working tree has uncommitted
            changes. The bundle then records that it was built from a modified
            tree, which is weaker provenance than a commit.
        force: Replace an existing bundle for today's date and this commit.
        now: The build time, for deterministic tests.

    Returns:
        Where the bundle was written and what it holds.

    Raises:
        HandoffError: if the bundle cannot be built, or would not be safe to
            send.
        ContractError: if the feature table breaks its schema.
    """
    moment = now or datetime.now(UTC)
    git, notes = _check_repository(repo or Path.cwd(), allow_dirty=allow_dirty)

    frame = _load_features(roots)
    _refuse_unsafe_columns(list(frame.columns))
    features, qc, feature_columns = _split(frame)
    if not feature_columns:
        msg = "the feature table has no feature columns; there is nothing to hand off"
        raise HandoffError(msg)

    try:
        dictionary = feature_dictionary.build(config, [str(c) for c in frame.columns])
    except DictionaryError as exc:
        raise HandoffError(str(exc)) from exc

    text, text_notes = _text_features(config, roots)
    confirmed = load_qc_notes(roots.work / config.qc.notes_path)
    if confirmed:
        for line in confirmed.report_lines():
            logger.info("%s: %s", STAGE, line)
    notes.extend(text_notes)
    for note in text_notes:
        logger.warning("%s: %s", STAGE, note)

    backend_records = face_stage.stored_backends(roots)
    backends = [str(record["backend"]) for record in backend_records.values()]

    plan = resolve_tiers([str(name) for name in frame.columns], config.model)
    tier_lines = tuple(describe_plan(plan, config.model))

    target, staging = _prepare_target(roots, git, moment=moment, force=force)

    written: list[str] = []
    try:
        written = _write_tables(
            staging,
            config=config,
            features=features,
            qc=qc,
            dictionary=dictionary,
            text=text,
            confirmed=confirmed,
        )

        manifest = _manifest(
            config,
            git=git,
            features=features,
            feature_columns=feature_columns,
            text=text,
            backends=backends,
            confirmed=confirmed,
            now=moment,
        )
        manifest["files"] = [*written, MANIFEST_FILE, README_FILE]
        manifest["notes"] = list(notes)
        write_json(staging / MANIFEST_FILE, manifest)
        written.append(MANIFEST_FILE)

        write_text(
            staging / README_FILE,
            _readme(
                config,
                result_files=manifest["files"],
                features=features,
                qc=qc,
                feature_columns=feature_columns,
                text=text,
                git=git,
                confirmed=confirmed,
                now=moment,
            ),
        )
        written.append(README_FILE)
    except (OSError, ValueError, ContractError):
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    logger.info("%s: wrote %d file(s) to %s", STAGE, len(written), target)

    return HandoffResult(
        path=target,
        n_sessions=len(features),
        n_features=len(feature_columns),
        files=tuple(manifest["files"]),
        git=git,
        text=text,
        tier_lines=tier_lines,
        notes=tuple(notes),
    )


def summarise(result: HandoffResult) -> list[str]:
    """What the CLI prints. Counts and names only, never a value."""
    lines = [
        f"bundle: {result.path}",
        f"sessions: {result.n_sessions}   features: {result.n_features}",
        f"files: {', '.join(result.files)}",
    ]
    if result.git is not None:
        state = "modified tree" if result.git.is_dirty else "clean tree"
        lines.append(f"commit: {result.git.short} on {result.git.branch} ({state})")
    if result.text is not None:
        how = (
            f"positional, rule {result.text.plan.rule!r}"
            if result.text.plan is not None
            else f"by {result.text.id_column!r}"
        )
        lines.append(
            f"text baseline: {result.text.n_sessions} session(s), "
            f"{len(result.text.feature_columns)} feature(s), matched {how}"
        )
    else:
        lines.append("text baseline: ABSENT - the comparisons against text cannot be run")
    for note in result.notes:
        lines.append(f"NOTE: {note}")
    return lines
