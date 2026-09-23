"""Stage 1: inventory every raw recording from its container metadata.

This is the stage that answers the project's open questions about the recordings
themselves: how long they are, whether the frame rate is constant, and above all
whether each file carries one mixed audio stream or one per speaker. It reads
metadata only, never pixels or samples.

Nothing here is destructive and nothing is inferred silently: unreadable or
unusual files are recorded with flags so they can be looked at, rather than
dropped or quietly corrected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig, DurationChecks
from vc_multimodal.contracts import INVENTORY_SCHEMA, ContractError, validate
from vc_multimodal.ffmpeg import FfmpegTools, MediaInfo, parse_media_info
from vc_multimodal.io_utils import read_csv, write_csv
from vc_multimodal.logging_setup import get_logger, run_stamp
from vc_multimodal.paths import DataRoots, Discovery, RawSession, discover_sessions, select_sessions
from vc_multimodal.runner import StageReport, run_sessions

logger = get_logger(__name__)

STAGE: Final = "inventory"
INVENTORY_FILENAME: Final = "inventory.csv"

FLAG_UNREADABLE: Final = "unreadable"
FLAG_NO_VIDEO: Final = "no_video_stream"
FLAG_NO_AUDIO: Final = "no_audio_stream"
FLAG_MULTI_AUDIO: Final = "multiple_audio_streams"
FLAG_VFR: Final = "variable_frame_rate"
FLAG_SHORT: Final = "duration_below_window"
FLAG_LONG: Final = "duration_above_window"
FLAG_OUTLIER: Final = "duration_outlier"
FLAG_KNOWN_SHORT: Final = "known_short_session"

# Columns whose dtype must be set explicitly. Pandas would infer `object` for a
# column that is entirely missing, which happens when every file is unreadable,
# and the contract would then reject an otherwise valid table.
_INT_COLUMNS: Final = (
    "size_bytes",
    "width",
    "height",
    "n_audio_streams",
    "audio_channels",
    "audio_sample_rate",
)
_FLOAT_COLUMNS: Final = ("duration_s", "fps")
_STR_COLUMNS: Final = ("video_codec", "audio_codec")

COLUMN_ORDER: Final = (
    "session_id",
    "wave",
    "date_folder",
    "relpath",
    "readable",
    "duration_s",
    "size_bytes",
    "video_codec",
    "width",
    "height",
    "fps",
    "fps_variable",
    "n_audio_streams",
    "audio_codec",
    "audio_channels",
    "audio_sample_rate",
    "flags",
)

# Scale factors making a robust deviation comparable to a standard deviation
# for normally distributed data (Iglewicz & Hoaglin's modified z-score).
_MAD_TO_SIGMA: Final = 1.4826
_MEAN_AD_TO_SIGMA: Final = 1.2533

# Fewer values than this leave no meaningful cohort to compare against.
_MIN_FOR_MAD: Final = 3

# How many missing column names to name when rejecting an existing file.
_MAX_LISTED_COLUMNS: Final = 6


class ExistingInventoryError(RuntimeError):
    """Raised when the file at the inventory path is not one we can merge with.

    The output path is inside a directory the user also works in by hand, so a
    file being there does not mean this pipeline wrote it. Merging into an
    unrecognised file would either crash or, worse, silently produce a mixed
    table, and overwriting it would destroy someone's work.
    """


def inventory_path(roots: DataRoots) -> Path:
    """Where the inventory table is written."""
    return roots.out_path(INVENTORY_FILENAME)


def backup_path(target: Path, *, stamp: str | None = None) -> Path:
    """Where an unrecognised file at the inventory path is moved aside to."""
    return target.with_name(f"{target.name}.bak-{stamp or run_stamp()}")


# Failures that mean "this file is not an inventory table" rather than
# "something is wrong with this program".
_UNREADABLE_TABLE_ERRORS: Final = (
    ContractError,
    ValueError,  # covers pandas ParserError and EmptyDataError
    KeyError,
    OSError,
    UnicodeDecodeError,
)


def _existing_error_message(target: Path, reason: str) -> str:
    """Explain that the file at the inventory path cannot be merged with."""
    return (
        f"{target} exists but is not an inventory table written by this "
        f"pipeline, so it cannot be merged with.\n"
        f"  reason: {reason}\n"
        f"Move or delete that file, or rerun with --force, which moves it aside "
        f"to {backup_path(target).name} and starts a fresh table."
    )


def read_existing(target: Path) -> pd.DataFrame:
    """Read and validate an existing inventory table.

    Args:
        target: Path to the existing table.

    Returns:
        The validated table.

    Raises:
        ExistingInventoryError: if the file cannot be read as an inventory
            table, with instructions for resolving it.
    """
    try:
        frame = read_csv(target)
    except _UNREADABLE_TABLE_ERRORS as exc:
        raise ExistingInventoryError(
            _existing_error_message(target, f"{type(exc).__name__}: {exc}")
        ) from exc

    # Checked before validating so the common case - a file with entirely
    # different columns, such as the output of a hand-written ffprobe loop -
    # gets a plain-language reason instead of a schema dump. Only our own
    # column names are named: the other file's headers may be data values.
    missing = [name for name in INVENTORY_SCHEMA.columns if name not in frame.columns]
    if missing:
        shown = ", ".join(missing[:_MAX_LISTED_COLUMNS])
        if len(missing) > _MAX_LISTED_COLUMNS:
            shown += f", and {len(missing) - _MAX_LISTED_COLUMNS} more"
        reason = (
            f"it has {len(frame.columns)} column(s) and is missing "
            f"{len(missing)} required one(s): {shown}"
        )
        raise ExistingInventoryError(_existing_error_message(target, reason))

    try:
        typed = coerce_dtypes(frame[list(COLUMN_ORDER)])
        return validate(typed, INVENTORY_SCHEMA, context=f"existing table {target.name}")
    except (*_UNREADABLE_TABLE_ERRORS, ContractError) as exc:
        raise ExistingInventoryError(_existing_error_message(target, str(exc))) from exc


def is_existing_inventory(target: Path) -> bool:
    """Whether `target` holds a table this pipeline can recognise."""
    try:
        read_existing(target)
    except ExistingInventoryError:
        return False
    return True


def mad_outliers(values: Mapping[int, float], k: float) -> tuple[int, ...]:
    """Identify robust outliers by median absolute deviation.

    Chosen over a mean/SD rule because a single very short recording would
    inflate the SD and hide itself. When the median absolute deviation is zero,
    which happens when most recordings share a duration, the scale falls back to
    the mean absolute deviation; only a cohort of entirely identical durations
    yields no outliers.

    Args:
        values: Session ID to value.
        k: How many scaled MADs from the median counts as an outlier.

    Returns:
        Outlying session IDs, ascending.
    """
    if len(values) < _MIN_FOR_MAD:
        return ()
    ids = sorted(values)
    series = np.array([values[i] for i in ids], dtype=float)
    median = float(np.median(series))
    deviations = np.abs(series - median)

    scale = float(np.median(deviations)) * _MAD_TO_SIGMA
    if scale == 0.0:
        # A zero MAD means more than half the recordings share a duration, which
        # is exactly the situation this check exists for: one odd file among many
        # near-identical ones. Falling back to the mean absolute deviation keeps
        # the test working there instead of silently disabling it.
        scale = float(np.mean(deviations)) * _MEAN_AD_TO_SIGMA
    if scale == 0.0:
        # Every recording has an identical duration: there is nothing to flag.
        return ()

    scores = deviations / scale
    return tuple(
        int(session_id) for session_id, score in zip(ids, scores, strict=True) if score > k
    )


def duration_flags(
    durations: Mapping[int, float | None],
    checks: DurationChecks,
    known_short: Sequence[int] = (),
) -> dict[int, list[str]]:
    """Flag recordings of unexpected length.

    Two independent tests, either of which can flag: an absolute window, and a
    robust outlier test against the cohort. A session listed in `known_short` is
    additionally marked as already known, so it reads as expected rather than as
    a surprise.

    Args:
        durations: Session ID to duration in seconds; None for unreadable files.
        checks: Absolute window and MAD multiplier.
        known_short: Sessions already known to be unusually short.

    Returns:
        Session ID to flags, for flagged sessions only.
    """
    flags: dict[int, list[str]] = {}
    usable = {sid: value for sid, value in durations.items() if value is not None}

    for session_id, duration in usable.items():
        if duration < checks.min_seconds:
            flags.setdefault(session_id, []).append(FLAG_SHORT)
        elif duration > checks.max_seconds:
            flags.setdefault(session_id, []).append(FLAG_LONG)

    for session_id in mad_outliers(usable, checks.mad_k):
        flags.setdefault(session_id, []).append(FLAG_OUTLIER)

    for session_id in known_short:
        if session_id in flags:
            flags[session_id].append(FLAG_KNOWN_SHORT)

    return flags


def stream_flags(info: MediaInfo) -> list[str]:
    """Flags derived from one file's stream layout."""
    flags: list[str] = []
    if info.width is None or info.height is None:
        flags.append(FLAG_NO_VIDEO)
    if info.n_audio_streams == 0:
        flags.append(FLAG_NO_AUDIO)
    elif info.n_audio_streams > 1:
        flags.append(FLAG_MULTI_AUDIO)
    if info.fps_variable:
        flags.append(FLAG_VFR)
    return flags


def _row(session: RawSession, info: MediaInfo | None, error: str | None) -> dict[str, object]:
    """Build one inventory row, with nulls where metadata is unavailable."""
    audio = info.primary_audio if info else None
    return {
        "session_id": session.session_id,
        "wave": session.wave,
        "date_folder": session.date_folder,
        "relpath": session.relpath,
        "readable": info is not None,
        "duration_s": info.duration_s if info else None,
        "size_bytes": info.size_bytes if info else None,
        "video_codec": info.video_codec if info else None,
        "width": info.width if info else None,
        "height": info.height if info else None,
        "fps": info.fps if info else None,
        "fps_variable": bool(info.fps_variable) if info else False,
        "n_audio_streams": info.n_audio_streams if info else None,
        "audio_codec": audio.codec if audio else None,
        "audio_channels": audio.channels if audio else None,
        "audio_sample_rate": audio.sample_rate if audio else None,
        "flags": ";".join(stream_flags(info)) if info else f"{FLAG_UNREADABLE}: {error or ''}",
    }


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a boolean column that may have round-tripped through text.

    `astype(bool)` is wrong for strings: it maps the string "False" to True.
    """
    if series.dtype == bool:
        return series
    return (
        series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        .fillna(False)
        .astype(bool)
    )


def coerce_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Give an inventory table its canonical dtypes.

    Used both when building a table and when reading one back, because a CSV
    round-trip loses them: a nullable `Int64` column with no missing values
    returns as `int64`, and a column that is entirely missing returns as
    `object`. Without this, a table this pipeline wrote would fail its own
    contract on the next run.
    """
    typed = frame.copy()
    for column in _INT_COLUMNS:
        typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("Int64")
    for column in _FLOAT_COLUMNS:
        typed[column] = pd.to_numeric(typed[column], errors="coerce").astype("float64")
    for column in _STR_COLUMNS:
        typed[column] = typed[column].astype("object")
    typed["readable"] = _as_bool(typed["readable"])
    typed["fps_variable"] = _as_bool(typed["fps_variable"])
    typed["flags"] = typed["flags"].fillna("").astype(str)
    return typed


def build_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Assemble inventory rows into a correctly typed table."""
    frame = pd.DataFrame(list(rows), columns=list(COLUMN_ORDER))
    return coerce_dtypes(frame).sort_values("session_id", ignore_index=True)


def apply_duration_flags(frame: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Append duration flags to an inventory table's `flags` column."""
    durations: dict[int, float | None] = {
        int(session_id): (None if pd.isna(duration) else float(duration))
        for session_id, duration in zip(frame["session_id"], frame["duration_s"], strict=True)
    }
    extra = duration_flags(durations, config.dataset.duration, config.dataset.known_short_sessions)
    if not extra:
        return frame

    updated = frame.copy()
    for position, session_id in enumerate(updated["session_id"]):
        added = extra.get(int(session_id))
        if not added:
            continue
        existing = str(updated.at[position, "flags"] or "")
        parts = [part for part in [existing, *added] if part]
        updated.at[position, "flags"] = ";".join(parts)
    return updated


@dataclass(frozen=True, slots=True)
class InventoryResult:
    """What the inventory stage produced."""

    report: StageReport
    frame: pd.DataFrame
    path: Path
    discovery: Discovery

    @property
    def ok(self) -> bool:
        """Whether every requested session was probed and nothing was flagged."""
        return self.report.ok and not self.discovery.problems


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    session_ids: Sequence[int] | None = None,
    workers: int | None = None,
    force: bool = False,
    tools: FfmpegTools | None = None,
) -> InventoryResult:
    """Probe every requested recording and write `inventory.csv`.

    Running on a subset updates only those rows and keeps any existing rows for
    other sessions, so piloting on three sessions does not discard the rest of
    the table. `force` discards the existing table instead of merging.

    Any file already at the output path is validated against the inventory
    schema before it is merged with, because that directory is one the user also
    works in by hand. An unrecognised file stops the run with instructions;
    under `force` it is moved aside rather than overwritten.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to probe, or None for every discovered session.
        workers: Parallel workers, or None to choose automatically.
        force: Ignore any existing inventory rows.
        tools: Located binaries; discovered if omitted.

    Returns:
        The stage report, the written table and the discovery result.

    Raises:
        ExistingInventoryError: if a file at the output path is not an inventory
            table and `force` was not passed.
    """
    binaries = tools or FfmpegTools.discover()
    discovery = discover_sessions(roots.data, config.dataset)
    for problem in discovery.problems:
        logger.warning("raw data: %s", problem)

    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []

    # Resolve what to do with any existing file BEFORE probing, so an
    # unusable one fails in a second rather than after 62 ffprobe calls.
    target = inventory_path(roots)
    previous: pd.DataFrame | None = None
    if target.exists():
        if force:
            if not is_existing_inventory(target):
                moved = backup_path(target)
                target.replace(moved)
                logger.warning(
                    "%s was not an inventory table; moved aside to %s", target, moved.name
                )
                notes.append(f"moved an unrecognised {target.name} aside to {moved.name}")
        else:
            previous = read_existing(target)

    probed: dict[int, dict[str, object]] = {}

    def probe_one(session: RawSession) -> str:
        info = parse_media_info(binaries.probe(session.path))
        probed[session.session_id] = _row(session, info, None)
        return f"{info.n_audio_streams} audio stream(s)"

    report = run_sessions(
        STAGE, selected, probe_one, workers=workers, backend="threads", notes=notes
    )

    # Failed sessions still get a row, so the table shows every file.
    by_id = {session.session_id: session for session in selected}
    for outcome in report.failed:
        session = by_id[outcome.session_id]
        probed[outcome.session_id] = _row(session, None, outcome.message)

    rows: list[Mapping[str, object]] = list(probed.values())
    if previous is not None:
        kept = previous[~previous["session_id"].isin(list(probed))]
        # pandas types records as dict[Hashable, Any]; the keys are column names.
        kept_rows = cast("list[Mapping[str, object]]", kept.to_dict(orient="records"))
        rows = [*kept_rows, *rows]
        logger.info("merged %d existing inventory row(s)", len(kept))

    frame = apply_duration_flags(build_frame(rows), config)
    validate(frame, INVENTORY_SCHEMA, context=STAGE)
    write_csv(target, frame)
    logger.info("wrote %s with %d row(s)", target, len(frame))

    return InventoryResult(report=report, frame=frame, path=target, discovery=discovery)


def summarise(frame: pd.DataFrame, config: AppConfig) -> list[str]:
    """Build a metadata-only summary of an inventory table.

    Deliberately aggregate: counts, durations and codec values, never content.
    The audio-stream count is called out because it decides how `vc
    extract-audio` and every later audio stage are configured.
    """
    if frame.empty:
        return ["inventory is empty"]

    lines = [f"sessions: {len(frame)} (expected {config.dataset.expected_sessions})"]

    by_wave = frame.groupby("wave", dropna=False)["session_id"].count().sort_index()
    lines.append("by wave: " + ", ".join(f"{wave}={count}" for wave, count in by_wave.items()))

    unreadable = frame.loc[~frame["readable"], "session_id"].tolist()
    if unreadable:
        lines.append(f"UNREADABLE: {sorted(unreadable)}")

    durations = frame["duration_s"].dropna()
    if not durations.empty:
        lines.append(
            f"duration: min {durations.min() / 60:.1f} min, "
            f"median {durations.median() / 60:.1f} min, "
            f"max {durations.max() / 60:.1f} min, "
            f"total {durations.sum() / 3600:.1f} h"
        )

    def counts(column: str, label: str) -> str:
        values = frame[column].dropna()
        if values.empty:
            return f"{label}: unknown"
        tally = values.value_counts().sort_index()
        return f"{label}: " + ", ".join(f"{value} ({count})" for value, count in tally.items())

    resolution = frame.dropna(subset=["width", "height"])
    if not resolution.empty:
        sizes = (
            resolution["width"].astype("Int64").astype(str)
            + "x"
            + resolution["height"].astype("Int64").astype(str)
        )
        tally = sizes.value_counts()
        lines.append(
            "resolution: " + ", ".join(f"{value} ({count})" for value, count in tally.items())
        )

    lines.append(counts("video_codec", "video codec"))
    lines.append(counts("fps", "frame rate"))
    lines.append(counts("n_audio_streams", "audio streams per file"))
    lines.append(counts("audio_codec", "audio codec"))
    lines.append(counts("audio_channels", "audio channels"))
    lines.append(counts("audio_sample_rate", "audio sample rate"))

    flagged = frame.loc[frame["flags"].astype(str) != ""]
    if flagged.empty:
        lines.append("flags: none")
    else:
        by_flag: dict[str, list[int]] = {}
        for session_id, raw_flags in zip(flagged["session_id"], flagged["flags"], strict=True):
            for flag in str(raw_flags).split(";"):
                name = flag.split(":", 1)[0].strip()
                if name:
                    by_flag.setdefault(name, []).append(int(session_id))
        lines.append("flags:")
        lines.extend(
            f"  {name}: {len(ids)} session(s) {sorted(ids)}"
            for name, ids in sorted(by_flag.items())
        )

    return lines
