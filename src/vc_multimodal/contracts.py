"""Data contracts validated at stage boundaries.

Each stage declares the shape of what it writes and what it expects to read, so
a malformed intermediate is caught where it is produced rather than three stages
later as a confusing statistic.

Validation failures are reported *without* failure-case values by default.
pandera normally quotes the offending rows, and offending rows here can contain
transcript text or values derived from clinical recordings. `validate()`
therefore summarises by column and check name only; pass `redact=False` when
working with synthetic test data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

# Feature families, in report order. `text` exists only when the manuscript's
# text-feature CSV is joined in by `vc model`.
FEATURE_FAMILIES: Final[tuple[str, ...]] = (
    "turns",
    "prosody",
    "face_speaking",
    "face_listening",
    "text",
)

FEATURE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:" + "|".join(FEATURE_FAMILIES) + r")__[a-z0-9]+(?:_[a-z0-9]+)*$"
)
QC_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^qc__[a-z0-9]+(?:_[a-z0-9]+)*$")

ROLES: Final[tuple[str, ...]] = ("psychiatrist", "participant", "unknown")
SIDES: Final[tuple[str, ...]] = ("left", "right")


class ContractError(ValueError):
    """Raised when a table does not satisfy its schema."""


def _positive_id() -> pa.Column:
    return pa.Column(int, pa.Check.gt(0), nullable=False)


INVENTORY_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    name="inventory",
    strict=True,
    unique=["session_id"],
    columns={
        "session_id": _positive_id(),
        "wave": pa.Column(str, nullable=False),
        "date_folder": pa.Column(str, nullable=False),
        "relpath": pa.Column(str, nullable=False),
        "readable": pa.Column(bool, nullable=False),
        "duration_s": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        "size_bytes": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "video_codec": pa.Column(str, nullable=True),
        "width": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "height": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "fps": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        "fps_variable": pa.Column(bool, nullable=False),
        "n_audio_streams": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "audio_codec": pa.Column(str, nullable=True),
        "audio_channels": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "audio_sample_rate": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        # Semicolon-separated QC flags; empty string when clean.
        "flags": pa.Column(str, nullable=False),
    },
)

SEGMENT_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    name="segments",
    strict=False,  # an optional `text` column may be present in work outputs
    columns={
        "session_id": _positive_id(),
        "speaker": pa.Column(str, nullable=False),
        "start_s": pa.Column(float, pa.Check.ge(0.0), nullable=False),
        "end_s": pa.Column(float, pa.Check.ge(0.0), nullable=False),
    },
    checks=[
        pa.Check(
            lambda df: df["end_s"] > df["start_s"],
            name="end_after_start",
            error="every segment must end after it starts",
        )
    ],
)

SPEECH_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    name="speech",
    strict=False,
    columns={
        "session_id": _positive_id(),
        "speaker": pa.Column(str, nullable=False),
        "role": pa.Column(str, pa.Check.isin(ROLES), nullable=False),
        "start_s": pa.Column(float, pa.Check.ge(0.0), nullable=False),
        "end_s": pa.Column(float, pa.Check.ge(0.0), nullable=False),
    },
    checks=[
        pa.Check(
            lambda df: df["end_s"] > df["start_s"],
            name="end_after_start",
            error="every speech span must end after it starts",
        )
    ],
)

TURN_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    name="turns",
    strict=False,
    columns={
        "session_id": _positive_id(),
        "turn_index": pa.Column(int, pa.Check.ge(0), nullable=False),
        "role": pa.Column(str, pa.Check.isin(ROLES), nullable=False),
        "start_s": pa.Column(float, pa.Check.ge(0.0), nullable=False),
        "end_s": pa.Column(float, pa.Check.ge(0.0), nullable=False),
        # Null for the first turn and wherever no preceding offset exists.
        "latency_s": pa.Column(float, nullable=True),
    },
)


AUDIO_QC_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    name="audio_qc",
    strict=True,
    unique=["session_id"],
    columns={
        "session_id": _positive_id(),
        "wave": pa.Column(str, nullable=False),
        "sample_rate": pa.Column("Int64", pa.Check.gt(0), nullable=False),
        "source_channels": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "duration_s": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        # The container's stated duration, and how far the decoded audio fell
        # short of it. A truncated file keeps honest-looking metadata, so the
        # two are recorded separately rather than assumed equal.
        "expected_duration_s": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        "duration_shortfall_s": pa.Column(float, nullable=True),
        "n_samples": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "active_fraction": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True),
        # Null when the recording was mono, or had nothing active to measure.
        "lr_correlation": pa.Column(float, pa.Check.in_range(-1.0, 1.0), nullable=True),
        "ild_db": pa.Column(float, nullable=True),
        "rms_left": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        "rms_right": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        "peak_left": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        "peak_right": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        "bit_identical": pa.Column("boolean", nullable=True),
        "flags": pa.Column(str, nullable=False),
    },
)


LAYOUT_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    name="layout",
    strict=True,
    unique=["session_id"],
    columns={
        "session_id": _positive_id(),
        "wave": pa.Column(str, nullable=False),
        "decided_side": pa.Column(str, pa.Check.isin(SIDES), nullable=False),
        "method": pa.Column(str, pa.Check.isin(("ocr", "assumed")), nullable=False),
        "ocr_side": pa.Column(str, pa.Check.isin((*SIDES, "inconclusive")), nullable=False),
        "assumed_side": pa.Column(str, pa.Check.isin(SIDES), nullable=False),
        # Null where OCR reached no conclusion, so there was nothing to compare.
        "matches_assumed": pa.Column("boolean", nullable=True),
        "n_labels_left": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "n_labels_right": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "best_confidence": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True),
        "flags": pa.Column(str, nullable=False),
    },
)
# Deliberately no text column: recognised labels are people's names.


def feature_schema(
    feature_columns: Iterable[str],
    *,
    require_wave: bool = True,
) -> pa.DataFrameSchema:
    """Build the schema for a feature table with the given feature columns.

    Feature names must follow `family__name`, with the family drawn from
    `FEATURE_FAMILIES`, so that `vc model` can select feature sets by family
    without a hand-maintained list. QC columns use the `qc__` prefix and may be
    of any type.

    Args:
        feature_columns: Every non-identifier column in the table.
        require_wave: Whether a `wave` column must be present.

    Returns:
        A strict schema for the table.

    Raises:
        ContractError: if a column name breaks the naming convention.
    """
    columns: dict[str, pa.Column] = {"session_id": _positive_id()}
    if require_wave:
        columns["wave"] = pa.Column(str, nullable=False)

    bad: list[str] = []
    for name in feature_columns:
        if QC_NAME_PATTERN.match(name):
            columns[name] = pa.Column(nullable=True)
        elif FEATURE_NAME_PATTERN.match(name):
            columns[name] = pa.Column(float, nullable=True)
        else:
            bad.append(name)

    if bad:
        msg = (
            f"feature columns break the naming convention: {sorted(bad)}. "
            f"Expected '<family>__<name>' with family in {list(FEATURE_FAMILIES)}, "
            f"or 'qc__<name>'."
        )
        raise ContractError(msg)

    return pa.DataFrameSchema(name="features", strict=True, unique=["session_id"], columns=columns)


def family_of(column: str) -> str | None:
    """Return the feature family of `column`, or None if it is not a feature."""
    if not FEATURE_NAME_PATTERN.match(column):
        return None
    return column.split("__", 1)[0]


def columns_in_families(columns: Sequence[str], families: Iterable[str]) -> tuple[str, ...]:
    """Select columns belonging to any of `families`, preserving input order."""
    wanted = set(families)
    return tuple(c for c in columns if family_of(c) in wanted)


def _summarise(error: SchemaErrors | SchemaError, *, redact: bool) -> str:
    """Render a pandera failure, optionally without quoting offending data."""
    if not redact:
        return str(error)
    if isinstance(error, SchemaError):
        return f"check {error.check} failed on column {error.schema.name!r}"
    failures = error.failure_cases
    if "column" not in failures or "check" not in failures:  # pragma: no cover - defensive
        return "schema validation failed"
    grouped = (
        failures.groupby(["column", "check"], dropna=False).size().sort_values(ascending=False)
    )
    lines = [
        f"  column {column!r}: check {check} failed on {count} row(s)"
        for (column, check), count in grouped.items()
    ]
    return "\n".join(lines)


def validate(
    frame: pd.DataFrame,
    schema: pa.DataFrameSchema,
    *,
    context: str = "",
    redact: bool = True,
) -> pd.DataFrame:
    """Validate `frame` against `schema` and return it.

    Args:
        frame: Table to check.
        schema: Contract to check it against.
        context: Added to the error message, e.g. a stage or session.
        redact: Summarise failures by column and check instead of quoting the
            offending values. Keep this on for anything derived from real data.

    Returns:
        The validated frame.

    Raises:
        ContractError: if validation fails.
    """
    try:
        return schema.validate(frame, lazy=True)
    except (SchemaErrors, SchemaError) as exc:
        where = f" for {context}" if context else ""
        msg = f"{schema.name} table{where} failed validation:\n{_summarise(exc, redact=redact)}"
        raise ContractError(msg) from exc
