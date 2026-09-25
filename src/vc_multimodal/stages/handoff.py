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
from vc_multimodal.stages import aggregate as aggregate_stage
from vc_multimodal.stages import face as face_stage

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
    return record


def _readme(
    config: AppConfig,
    *,
    result_files: Sequence[str],
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    text: TextFeatures | None,
    git: GitState | None,
    tier_lines: Sequence[str],
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
    ]
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

## What to do with it

The analysis runs on the machine that holds the labels:

```
vc model --features features.csv --labels <your labels file> --out <directory>
```

The labels file needs a `session_id` column and one column per outcome. **No
label is ever written into any output of that command**, and no label needs to
be sent back.

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

    parts.append(
        """### Quality columns are not predictors

`qc__*` columns describe how much evidence each row rests on. `
qc__speaking_seconds` mostly measures how long the session was;
`qc__face_measured_speaking` mostly measures whether the camera was pointed at
the participant. Modelling them would produce a real-looking result about
recording conditions. They are in a separate file for that reason.

A row with entries in `qc__stages_missing` has features absent by cause, not by
chance. Which sessions to exclude is your decision; make it once, from the QC
table, and state it.
"""
    )

    parts.append(
        """### Two turn features are constant at zero

`turns__overlap_ratio` and `turns__interruption_rate` are zero for every
session, and that is a property of the diarizer rather than of the
conversations. The diarization assigns every moment to exactly one speaker, so
simultaneous speech cannot be represented in its output; measured across all
sessions, speaker overlap is exactly 0.0000 seconds.

They are kept in the table rather than dropped, so that this limitation is
visible in the bundle instead of being invisible in its absence. **Do not read
them as evidence that these participants never interrupted or overlapped.**
Both become informative unchanged if the recordings are re-diarized with a
tool that permits overlapping speech.
"""
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
    elif text is None:
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

    parts.append("## Tier plan as it ran\n\n```\n" + "\n".join(tier_lines) + "\n```\n")

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
        write_csv(staging / FEATURES_FILE, features)
        written.append(FEATURES_FILE)
        write_csv(staging / QC_FILE, qc)
        written.append(QC_FILE)
        write_csv(staging / DICTIONARY_FILE, dictionary)
        written.append(DICTIONARY_FILE)
        if text is not None:
            write_csv(staging / TEXT_FILE, text.frame)
            written.append(TEXT_FILE)
        write_text(staging / CONFIG_FILE, config.to_yaml())
        written.append(CONFIG_FILE)

        manifest = _manifest(
            config,
            git=git,
            features=features,
            feature_columns=feature_columns,
            text=text,
            backends=backends,
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
                feature_columns=feature_columns,
                text=text,
                git=git,
                tier_lines=tier_lines,
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
