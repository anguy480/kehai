"""Stage: determine which side of the frame the psychiatrist is on.

Sessions checked by hand all showed the psychiatrist in the LEFT tile, and
`speakers.assumed_psychiatrist_side` records that. This stage exists so the
assumption is checked rather than trusted: it reads the Zoom name label in each
tile and decides per session, falling back to the assumed side only where OCR
reaches no conclusion. A disagreement is recorded as a QC flag and never
silently overrides what OCR found.

Identifying the psychiatrist without naming them
------------------------------------------------
The psychiatrist appears in every session; each participant appears in one. So
the label that RECURS across sessions is the psychiatrist's, and no name is
needed anywhere. That is the primary rule. Explicit label fragments can be
supplied through an environment variable for the cases it cannot settle, but a
real person's name never enters this repository.

What leaves this stage
----------------------
Sides, counts, confidences and flags. Recognised text is compared in memory and
is never printed, logged or written to any file. `layout.csv` has no text
column, and the summary reports session IDs only.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import cv2
import pandas as pd

from vc_multimodal.config import AppConfig, CropBox
from vc_multimodal.contracts import LAYOUT_SCHEMA, validate
from vc_multimodal.ffmpeg import FfmpegTools, parse_media_info
from vc_multimodal.io_utils import write_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.ocr import OcrBackend, OcrError, get_backend
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.runner import StageReport, run_sessions
from vc_multimodal.stages.preview import sample_times

logger = get_logger(__name__)

STAGE: Final = "verify-layout"
LAYOUT_FILENAME: Final = "layout.csv"

SIDE_LEFT: Final = "left"
SIDE_RIGHT: Final = "right"
SIDE_INCONCLUSIVE: Final = "inconclusive"

METHOD_OCR: Final = "ocr"
METHOD_ASSUMED: Final = "assumed"

FLAG_MISMATCH: Final = "layout_side_mismatch"
FLAG_INCONCLUSIVE: Final = "layout_inconclusive"
FLAG_OCR_UNAVAILABLE: Final = "layout_ocr_unavailable"
FLAG_NOT_TWO_TILE: Final = "layout_not_two_tile"
FLAG_BOTH_SIDES: Final = "layout_label_on_both_sides"
FLAG_OCR_ERROR: Final = "layout_ocr_error"

# A label cannot be shown to recur without at least two sessions to compare.
_MIN_SESSIONS_FOR_RECURRENCE: Final = 2

COLUMN_ORDER: Final = (
    "session_id",
    "wave",
    "decided_side",
    "method",
    "ocr_side",
    "assumed_side",
    "matches_assumed",
    "n_labels_left",
    "n_labels_right",
    "best_confidence",
    "flags",
)

# Punctuation and spacing that Zoom labels carry, and that OCR adds or drops,
# removed before two labels are compared. Written as escapes rather than
# literals so the fullwidth forms are unambiguous to a reader: ideographic
# space, middle dots, fullwidth comma and stops, brackets, colon, semicolon.
_STRIP_PATTERN: Final = re.compile(
    "[\\s"
    "\u3000"  # ideographic space
    "\u00b7\u30fb"  # middle dot, katakana middle dot
    "\uff0c\u3002\uff0e"  # fullwidth comma, ideographic full stop, fullwidth stop
    "\uff08\uff09"  # fullwidth parentheses
    "\u3010\u3011"  # lenticular brackets
    "\uff1a\uff1b"  # fullwidth colon and semicolon
    ",.()\\[\\]:;\\-_/\\\\|*"  # the ASCII equivalents
    "]+"
)


def layout_path(roots: DataRoots) -> Path:
    """Where the layout table is written."""
    return roots.out_path(LAYOUT_FILENAME)


# ---------------------------------------------------------------------------
# Pure label handling. No I/O, no OCR, fully unit tested.
# ---------------------------------------------------------------------------
def normalise_label(text: str) -> str:
    """Reduce a recognised label to a comparable key.

    OCR of the same name across sessions differs in case, spacing, width of
    Japanese characters and stray punctuation. Normalising makes the recurrence
    rule work without needing the name itself.

    Returns an empty string for text with nothing comparable left in it.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _STRIP_PATTERN.sub("", folded)


def find_recurring_labels(
    per_session: Mapping[int, Sequence[str]],
    *,
    min_recurrence: float,
) -> frozenset[str]:
    """Find the labels that recur across sessions.

    A label appearing in at least `min_recurrence` of the sessions belongs to
    someone present in most of them, i.e. the psychiatrist. Each participant's
    label appears in one session and falls far below the threshold.

    Args:
        per_session: Session ID to the normalised labels seen anywhere in it.
        min_recurrence: Required fraction of sessions, in (0, 1].

    Returns:
        The recurring labels, empty if none reaches the threshold.
    """
    if not per_session:
        return frozenset()

    counts: Counter[str] = Counter()
    for labels in per_session.values():
        counts.update(set(labels))

    threshold = max(2, round(min_recurrence * len(per_session)))
    return frozenset(label for label, count in counts.items() if count >= threshold)


def matches_any_pattern(label: str, patterns: Sequence[str]) -> bool:
    """Whether a normalised label contains any of the normalised `patterns`."""
    return any(pattern and pattern in label for pattern in patterns)


@dataclass(frozen=True, slots=True)
class SideDecision:
    """Which side OCR points at, and why."""

    side: str
    flags: tuple[str, ...] = ()


def decide_side(
    left_labels: Sequence[str],
    right_labels: Sequence[str],
    *,
    recurring: frozenset[str] = frozenset(),
    patterns: Sequence[str] = (),
) -> SideDecision:
    """Decide which side holds the psychiatrist from one session's labels.

    Explicit patterns win where supplied, since they express direct knowledge.
    Otherwise the recurring label decides. A label matching on both sides, or on
    neither, is inconclusive: the caller then falls back to the assumed side
    and flags it, rather than guessing.

    Args:
        left_labels: Normalised labels read in the left tile.
        right_labels: Normalised labels read in the right tile.
        recurring: Labels found to recur across the cohort.
        patterns: Normalised explicit label fragments, if configured.

    Returns:
        The side, or `inconclusive`, with any flags raised.
    """
    if patterns:
        left_hit = any(matches_any_pattern(label, patterns) for label in left_labels)
        right_hit = any(matches_any_pattern(label, patterns) for label in right_labels)
    else:
        left_hit = any(label in recurring for label in left_labels)
        right_hit = any(label in recurring for label in right_labels)

    if left_hit and not right_hit:
        return SideDecision(SIDE_LEFT)
    if right_hit and not left_hit:
        return SideDecision(SIDE_RIGHT)
    if left_hit and right_hit:
        return SideDecision(SIDE_INCONCLUSIVE, (FLAG_INCONCLUSIVE, FLAG_BOTH_SIDES))
    return SideDecision(SIDE_INCONCLUSIVE, (FLAG_INCONCLUSIVE,))


def resolve(decision: SideDecision, assumed_side: str) -> tuple[str, str, bool | None, list[str]]:
    """Combine an OCR decision with the assumed side.

    OCR wins whenever it reached a conclusion. The assumed side is used only as
    a fallback, and a disagreement is flagged rather than applied.

    Returns:
        The decided side, the method used, whether OCR matched the assumption
        (None when OCR was inconclusive), and the flags to record.
    """
    flags = list(decision.flags)
    if decision.side == SIDE_INCONCLUSIVE:
        return assumed_side, METHOD_ASSUMED, None, flags

    matches = decision.side == assumed_side
    if not matches:
        flags.append(FLAG_MISMATCH)
    return decision.side, METHOD_OCR, matches, flags


# ---------------------------------------------------------------------------
# Reading labels from frames
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TileLabels:
    """What OCR found in one tile of one session."""

    labels: tuple[str, ...] = ()
    best_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class SessionLabels:
    """What OCR found in one session, per side."""

    session_id: int
    by_side: Mapping[str, TileLabels] = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    @property
    def all_labels(self) -> tuple[str, ...]:
        """Every normalised label seen in the session, either side."""
        return tuple(label for tile in self.by_side.values() for label in tile.labels)

    @property
    def best_confidence(self) -> float:
        """Highest confidence seen anywhere in the session."""
        return max((tile.best_confidence for tile in self.by_side.values()), default=0.0)


def crop_label_region(tile: cv2.typing.MatLike, region: CropBox | None) -> cv2.typing.MatLike:
    """Crop the label area out of a tile image, or return the whole tile."""
    if region is None:
        return tile
    height, width = tile.shape[:2]
    left, top, box_w, box_h = region.to_pixels(width, height)
    return tile[top : top + box_h, left : left + box_w]


def read_session_labels(
    session: RawSession,
    *,
    config: AppConfig,
    backend: OcrBackend,
    tools: FfmpegTools,
    scratch: Path,
) -> SessionLabels:
    """OCR the name label in each tile of one session.

    Frames are extracted, read and deleted one at a time, so no decoded frame of
    a real recording is left on disk.
    """
    ocr_config = config.speakers.label_ocr
    video = config.video

    if not video.is_two_tile:
        return SessionLabels(session.session_id, {}, (FLAG_NOT_TWO_TILE, FLAG_INCONCLUSIVE))

    duration = parse_media_info(tools.probe(session.path)).duration_s
    times = sample_times(ocr_config.sample_times_seconds, duration)

    found: dict[str, list[str]] = {SIDE_LEFT: [], SIDE_RIGHT: []}
    confidence: dict[str, float] = {SIDE_LEFT: 0.0, SIDE_RIGHT: 0.0}
    errors = 0

    scratch.mkdir(parents=True, exist_ok=True)
    for index, timestamp in enumerate(times):
        frame_file = scratch / f"{index}.png"
        try:
            tools.extract_frame(session.path, timestamp, frame_file)
            image = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
        finally:
            frame_file.unlink(missing_ok=True)
        if image is None:
            continue

        height, width = image.shape[:2]
        for side in (SIDE_LEFT, SIDE_RIGHT):
            tile_name = video.tile_on_side(side)
            if tile_name is None:  # pragma: no cover - guarded by is_two_tile
                continue
            box = video.tiles[tile_name]
            left, top, box_w, box_h = box.to_pixels(width, height)
            tile = image[top : top + box_h, left : left + box_w]
            patch = crop_label_region(tile, ocr_config.label_region)

            try:
                lines = backend.read(patch, languages=ocr_config.languages)
            except OcrError:
                # One unreadable patch must not lose the whole session. The
                # error type is counted; the exception text is not logged,
                # because a backend may quote what it was reading.
                errors += 1
                continue

            for line in lines:
                if line.confidence < ocr_config.min_confidence:
                    continue
                key = normalise_label(line.text)
                if not key:
                    continue
                found[side].append(key)
                confidence[side] = max(confidence[side], line.confidence)

    read_anything = any(found[side] for side in (SIDE_LEFT, SIDE_RIGHT))
    flags = () if read_anything or not errors else (FLAG_OCR_ERROR, FLAG_INCONCLUSIVE)
    if errors:
        logger.warning("session %s: %d OCR read(s) failed", session.session_id, errors)

    return SessionLabels(
        session_id=session.session_id,
        by_side={
            side: TileLabels(tuple(dict.fromkeys(found[side])), confidence[side])
            for side in (SIDE_LEFT, SIDE_RIGHT)
        },
        flags=flags,
    )


def explicit_patterns(config: AppConfig) -> tuple[str, ...]:
    """Normalised explicit label fragments from the configured environment var.

    Read from the environment, never from a config file: the value is a real
    person's name.
    """
    raw = os.environ.get(config.speakers.label_ocr.psychiatrist_label_env, "")
    return tuple(key for key in (normalise_label(part) for part in raw.split(",")) if key)


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LayoutResult:
    """What the verify-layout stage produced."""

    report: StageReport
    frame: pd.DataFrame
    path: Path

    @property
    def sides(self) -> Mapping[int, str]:
        """Session ID to the side the psychiatrist was decided to be on."""
        return dict(zip(self.frame["session_id"], self.frame["decided_side"], strict=True))


def build_rows(
    labels: Mapping[int, SessionLabels],
    waves: Mapping[int, str],
    *,
    recurring: frozenset[str] = frozenset(),
    patterns: Sequence[str] = (),
    assumed_side: str,
) -> list[Mapping[str, object]]:
    """Turn per-session labels into layout rows.

    Pure: the whole decision - which side OCR points at, whether that agrees
    with the assumption, and which flags follow - is decided here from data
    alone, with no OCR, filesystem or configuration access.

    Args:
        labels: Session ID to the labels read for it.
        waves: Session ID to recruitment wave.
        recurring: Labels found to recur across the cohort.
        patterns: Normalised explicit label fragments, if configured.
        assumed_side: The fallback side from configuration.

    Returns:
        One row per session, ascending by session ID.
    """
    rows: list[Mapping[str, object]] = []
    for session_id in sorted(labels):
        result = labels[session_id]
        left = result.by_side.get(SIDE_LEFT, TileLabels())
        right = result.by_side.get(SIDE_RIGHT, TileLabels())

        if result.flags:
            decision = SideDecision(SIDE_INCONCLUSIVE, result.flags)
        else:
            decision = decide_side(
                left.labels, right.labels, recurring=recurring, patterns=patterns
            )

        decided, method, matches, flags = resolve(decision, assumed_side)
        rows.append(
            {
                "session_id": session_id,
                "wave": waves.get(session_id, "unknown"),
                "decided_side": decided,
                "method": method,
                "ocr_side": decision.side,
                "assumed_side": assumed_side,
                "matches_assumed": matches,
                "n_labels_left": len(left.labels),
                "n_labels_right": len(right.labels),
                "best_confidence": round(result.best_confidence, 3),
                "flags": ";".join(dict.fromkeys(flags)),
            }
        )
    return rows


def build_frame(
    rows: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Assemble layout rows into a correctly typed table."""
    frame = pd.DataFrame(list(rows), columns=list(COLUMN_ORDER))
    for column in ("n_labels_left", "n_labels_right"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["best_confidence"] = pd.to_numeric(frame["best_confidence"], errors="coerce").astype(
        "float64"
    )
    frame["matches_assumed"] = frame["matches_assumed"].astype("boolean")
    frame["flags"] = frame["flags"].fillna("").astype(str)
    return frame.sort_values("session_id", ignore_index=True)


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    session_ids: Sequence[int] | None = None,
    workers: int | None = None,
    backend: OcrBackend | None = None,
    tools: FfmpegTools | None = None,
) -> LayoutResult:
    """Read every session's name labels and decide which side is the psychiatrist.

    Runs in two passes. The first reads labels per session, in parallel, with
    per-session failures isolated. The second is a pure cohort decision: the
    recurring label identifies the psychiatrist, so no name is required.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to check, or None for all.
        workers: Parallel workers, or None to choose automatically.
        backend: OCR backend; built from config if omitted.
        tools: Located binaries; discovered if omitted.

    Returns:
        The stage report and the written layout table.
    """
    binaries = tools or FfmpegTools.discover()
    ocr_config = config.speakers.label_ocr
    engine = backend or get_backend(ocr_config.backend if ocr_config.enabled else "none")

    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []

    usable = engine.available()
    if not usable:
        reason = getattr(engine, "unavailable_reason", lambda: "OCR is unavailable")()
        logger.warning("%s: %s; falling back to the assumed side", STAGE, reason)
        notes.append(f"OCR unavailable ({reason}); every session falls back to the assumption")
    else:
        logger.info("%s: OCR backend %s (%s)", STAGE, engine.name, engine.version())

    # ---- pass 1: read labels ------------------------------------------
    labels: dict[int, SessionLabels] = {}

    def read_one(session: RawSession) -> str:
        if not usable:
            labels[session.session_id] = SessionLabels(
                session.session_id, {}, (FLAG_OCR_UNAVAILABLE, FLAG_INCONCLUSIVE)
            )
            return "OCR unavailable"
        result = read_session_labels(
            session,
            config=config,
            backend=engine,
            tools=binaries,
            scratch=roots.work_path("tmp", "verify_layout", str(session.session_id)),
        )
        labels[session.session_id] = result
        # Counts only. The labels themselves are never logged.
        return (
            f"{len(result.by_side[SIDE_LEFT].labels)} left / "
            f"{len(result.by_side[SIDE_RIGHT].labels)} right label(s)"
            if result.by_side
            else "no tiles to read"
        )

    report = run_sessions(
        STAGE, selected, read_one, workers=workers, backend="threads", notes=notes
    )

    # ---- pass 2: cohort decision --------------------------------------
    patterns = explicit_patterns(config)
    if patterns:
        logger.info(
            "%s: using %d explicit label pattern(s) from the environment", STAGE, len(patterns)
        )
    recurring = find_recurring_labels(
        {sid: result.all_labels for sid, result in labels.items()},
        min_recurrence=ocr_config.min_recurrence,
    )
    extra_notes: list[str] = []
    if not patterns:
        logger.info("%s: %d recurring label(s) identified across the cohort", STAGE, len(recurring))
        if usable and len(labels) < _MIN_SESSIONS_FOR_RECURRENCE:
            # The psychiatrist is identified by their label recurring, which
            # needs a cohort. Say so rather than reporting a bare
            # "inconclusive" that looks like an OCR failure.
            extra_notes.append(
                f"only {len(labels)} session(s) were read, so no label can be shown to "
                f"recur; run without --sessions, or set "
                f"{ocr_config.psychiatrist_label_env} to identify the psychiatrist directly"
            )
        elif usable and not recurring:
            extra_notes.append(
                "no label recurred across the sessions read, so the psychiatrist could not "
                "be identified; check speakers.label_ocr.label_region against the previews"
            )

    rows = build_rows(
        labels,
        {session.session_id: session.wave for session in selected},
        recurring=recurring,
        patterns=patterns,
        assumed_side=config.speakers.assumed_psychiatrist_side,
    )

    if extra_notes:
        for note in extra_notes:
            logger.warning("%s: %s", STAGE, note)
        report = StageReport(
            stage=report.stage,
            outcomes=report.outcomes,
            seconds=report.seconds,
            notes=(*report.notes, *extra_notes),
        )

    frame = build_frame(rows)
    validate(frame, LAYOUT_SCHEMA, context=STAGE)
    target = layout_path(roots)
    write_csv(target, frame)
    logger.info("wrote %s with %d row(s)", target, len(frame))

    return LayoutResult(report=report, frame=frame, path=target)


def summarise(frame: pd.DataFrame, config: AppConfig) -> list[str]:
    """Summarise the layout check: counts and session IDs only, never text."""
    if frame.empty:
        return ["no sessions were checked"]

    assumed = config.speakers.assumed_psychiatrist_side
    lines = [
        f"checked {len(frame)} session(s); assumed psychiatrist side: {assumed}",
        "",
        "psychiatrist side, as found by label OCR:",
    ]

    ocr_side = frame["ocr_side"]
    for side in (SIDE_LEFT, SIDE_RIGHT, SIDE_INCONCLUSIVE):
        ids = sorted(int(i) for i in frame.loc[ocr_side == side, "session_id"])
        label = f"  {side:13s} {len(ids):3d}"
        lines.append(label if not ids else f"{label}  {ids}")

    conclusive = frame.loc[ocr_side != SIDE_INCONCLUSIVE]
    mismatched = sorted(
        int(i) for i in conclusive.loc[~conclusive["matches_assumed"].fillna(False), "session_id"]
    )
    total = len(frame)
    fallbacks = total - len(conclusive)

    lines.append("")
    if mismatched:
        lines.append(f"DISAGREES with the assumed side: {len(mismatched)} session(s) {mismatched}")
        lines.append("  These are flagged, not overridden: OCR's finding is what was recorded.")
    elif conclusive.empty:
        lines.append("OCR reached no conclusion for any session, so the assumption is untested.")
    else:
        lines.append(f"every session OCR could settle agrees with the assumption ({assumed}).")

    if fallbacks == 0 and not mismatched:
        lines.append(
            f"All {total} session(s) confirmed: the psychiatrist is on the {assumed}. "
            f"Later stages can rely on it."
        )
    elif fallbacks == 0:
        lines.append(
            f"All {total} session(s) were settled by OCR, but {len(mismatched)} disagree "
            f"with speakers.assumed_psychiatrist_side. Resolve those before relying on it."
        )
    else:
        lines.append(
            f"{len(conclusive)} of {total} session(s) were settled by OCR; "
            f"{fallbacks} fell back to the assumption and are flagged."
        )

    return lines
