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
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

import cv2
import pandas as pd

from vc_multimodal.config import AppConfig, CropBox
from vc_multimodal.contracts import LAYOUT_SCHEMA, validate
from vc_multimodal.features.geometry import (
    FULL_FRAME,
    ContentBox,
    RegionGeometry,
    detect_content_box,
    format_box,
    resolve_regions,
)
from vc_multimodal.ffmpeg import FfmpegTools, parse_media_info
from vc_multimodal.io_utils import write_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.ocr import OcrBackend, OcrError, get_backend
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.runner import StageReport, run_sessions
from vc_multimodal.session_tables import carry_forward, combine
from vc_multimodal.stages.preview import sample_times

logger = get_logger(__name__)

STAGE: Final = "verify-layout"
LAYOUT_FILENAME: Final = "layout.csv"
LAYOUT_DEBUG_FILENAME: Final = "layout_debug.csv"

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

#: What OCR itself concluded, as opposed to `decided_side`, which falls back to
#: the configured assumption. Named so that readers of this table cannot drift
#: from the writer.
OCR_SIDE_COLUMN: Final = "ocr_side"

COLUMN_ORDER: Final = (
    "session_id",
    "wave",
    "decided_side",
    "method",
    OCR_SIDE_COLUMN,
    "assumed_side",
    "matches_assumed",
    "recurring_label_key",
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


@dataclass(frozen=True, slots=True)
class RecurringLabel:
    """A label that recurs across sessions, identified without being named.

    The key is an ordinal assigned within this run - `PSY_A`, `PSY_B` - ordered
    by how many sessions the label appears in. It is not derived from the text
    in any way, not even by hashing, so it can be printed and written freely
    while still letting sessions be grouped by which recurring speaker they
    matched.
    """

    key: str
    n_sessions: int


def recurring_counts(per_session: Mapping[int, Sequence[str]]) -> Counter[str]:
    """How many sessions each label appears in, counting a label once each."""
    counts: Counter[str] = Counter()
    for labels in per_session.values():
        counts.update(set(labels))
    return counts


def recurrence_threshold(n_sessions: int, min_recurrence: float) -> int:
    """How many sessions a label must appear in to count as recurring."""
    return max(_MIN_SESSIONS_FOR_RECURRENCE, round(min_recurrence * n_sessions))


def assign_recurring_keys(
    per_session: Mapping[int, Sequence[str]],
    *,
    min_recurrence: float,
) -> dict[str, RecurringLabel]:
    """Give each recurring label an opaque key, most widespread first.

    Ordered by session count so the keys are stable for a given cohort, with
    the label itself as a tie-break so a rerun produces the same assignment.

    Args:
        per_session: Session ID to the normalised labels seen in it.
        min_recurrence: Required fraction of sessions, in (0, 1].

    Returns:
        Normalised label to its key and session count. Empty when no label
        reaches the threshold.
    """
    if not per_session:
        return {}

    counts = recurring_counts(per_session)
    threshold = recurrence_threshold(len(per_session), min_recurrence)
    qualifying = sorted(
        ((label, count) for label, count in counts.items() if count >= threshold),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        label: RecurringLabel(key=f"PSY_{chr(ord('A') + index)}", n_sessions=count)
        for index, (label, count) in enumerate(qualifying)
    }


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
    return frozenset(assign_recurring_keys(per_session, min_recurrence=min_recurrence))


def matches_any_pattern(label: str, patterns: Sequence[str]) -> bool:
    """Whether a normalised label contains any of the normalised `patterns`."""
    return any(pattern and pattern in label for pattern in patterns)


@dataclass(frozen=True, slots=True)
class SideDecision:
    """Which side OCR points at, why, and which recurring label settled it."""

    side: str
    flags: tuple[str, ...] = ()
    #: Opaque key of the recurring label that identified the psychiatrist, or
    #: None when nothing matched or explicit patterns were used instead.
    matched_key: str | None = None


def decide_side(
    left_labels: Sequence[str],
    right_labels: Sequence[str],
    *,
    recurring: frozenset[str] = frozenset(),
    patterns: Sequence[str] = (),
    keys: Mapping[str, RecurringLabel] | None = None,
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
        keys: Opaque key per recurring label, so the decision can record which
            recurring speaker settled it without naming them.

    Returns:
        The side, or `inconclusive`, with any flags raised and the key of the
        recurring label that settled it.
    """
    if patterns:
        left_hit = any(matches_any_pattern(label, patterns) for label in left_labels)
        right_hit = any(matches_any_pattern(label, patterns) for label in right_labels)
        matched_key = None
    else:
        left_match = next((label for label in left_labels if label in recurring), None)
        right_match = next((label for label in right_labels if label in recurring), None)
        left_hit, right_hit = left_match is not None, right_match is not None
        matched = left_match or right_match
        entry = keys.get(matched) if keys is not None and matched is not None else None
        matched_key = entry.key if entry is not None else None

    if left_hit and not right_hit:
        return SideDecision(SIDE_LEFT, matched_key=matched_key)
    if right_hit and not left_hit:
        return SideDecision(SIDE_RIGHT, matched_key=matched_key)
    if left_hit and right_hit:
        return SideDecision(
            SIDE_INCONCLUSIVE, (FLAG_INCONCLUSIVE, FLAG_BOTH_SIDES), matched_key=matched_key
        )
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
class RegionObservations:
    """What OCR saw in one region of one session, and where that region was.

    Deliberately carries counts and coordinates only. Recognised text never
    enters this record, because the whole point of the diagnostic is to be
    printable.
    """

    session_id: int
    frame_width: int
    frame_height: int
    content_detected: bool
    content_box: CropBox
    content_bars: tuple[int, int, int, int]
    geometry: RegionGeometry
    upscale: float
    n_frames_read: int = 0
    n_observations: int = 0
    n_above_confidence: int = 0
    n_usable_labels: int = 0
    max_confidence: float = 0.0
    n_ocr_errors: int = 0

    def report_lines(self) -> list[str]:
        """Human-readable diagnostic for this region."""
        geometry = self.geometry
        return [
            f"  tile {geometry.tile!r} ({geometry.role})",
            f"    tile  {format_box(geometry.tile_box, geometry.tile_pixels)}",
            f"    label {format_box(geometry.label_box, geometry.label_pixels)}"
            + ("  [whole tile]" if geometry.label_is_whole_tile else ""),
            f"    read {self.n_frames_read} frame(s) at {self.upscale:g}x: "
            f"{self.n_observations} observation(s), "
            f"{self.n_above_confidence} above confidence "
            f"{'' if self.n_ocr_errors == 0 else f'({self.n_ocr_errors} OCR error(s)) '}"
            f"-> {self.n_usable_labels} usable label(s), "
            f"max confidence {self.max_confidence:.2f}",
        ]


DEBUG_COLUMN_ORDER: Final = (
    "session_id",
    "frame_width",
    "frame_height",
    "content_detected",
    "content_x",
    "content_y",
    "content_width",
    "content_height",
    "bar_left",
    "bar_top",
    "bar_right",
    "bar_bottom",
    "tile",
    "role",
    "tile_x",
    "tile_y",
    "tile_width",
    "tile_height",
    "tile_px_left",
    "tile_px_top",
    "tile_px_width",
    "tile_px_height",
    "label_x",
    "label_y",
    "label_width",
    "label_height",
    "label_px_left",
    "label_px_top",
    "label_px_width",
    "label_px_height",
    "upscale",
    "n_frames_read",
    "n_observations",
    "n_above_confidence",
    "n_usable_labels",
    "max_confidence",
    "n_ocr_errors",
)


def debug_frame(observations: Sequence[RegionObservations]) -> pd.DataFrame:
    """Assemble the per-region diagnostic table.

    One row per region per session, with every coordinate in both fractional
    and pixel form. No text column exists on this table by construction.
    """
    rows: list[dict[str, object]] = []
    for item in observations:
        geometry = item.geometry
        tile_px = geometry.tile_pixels
        label_px = geometry.label_pixels
        rows.append(
            {
                "session_id": item.session_id,
                "frame_width": item.frame_width,
                "frame_height": item.frame_height,
                "content_detected": item.content_detected,
                "content_x": round(item.content_box.x, 6),
                "content_y": round(item.content_box.y, 6),
                "content_width": round(item.content_box.width, 6),
                "content_height": round(item.content_box.height, 6),
                "bar_left": item.content_bars[0],
                "bar_top": item.content_bars[1],
                "bar_right": item.content_bars[2],
                "bar_bottom": item.content_bars[3],
                "tile": geometry.tile,
                "role": geometry.role,
                "tile_x": round(geometry.tile_box.x, 6),
                "tile_y": round(geometry.tile_box.y, 6),
                "tile_width": round(geometry.tile_box.width, 6),
                "tile_height": round(geometry.tile_box.height, 6),
                "tile_px_left": tile_px[0],
                "tile_px_top": tile_px[1],
                "tile_px_width": tile_px[2],
                "tile_px_height": tile_px[3],
                "label_x": round(geometry.label_box.x, 6),
                "label_y": round(geometry.label_box.y, 6),
                "label_width": round(geometry.label_box.width, 6),
                "label_height": round(geometry.label_box.height, 6),
                "label_px_left": label_px[0],
                "label_px_top": label_px[1],
                "label_px_width": label_px[2],
                "label_px_height": label_px[3],
                "upscale": item.upscale,
                "n_frames_read": item.n_frames_read,
                "n_observations": item.n_observations,
                "n_above_confidence": item.n_above_confidence,
                "n_usable_labels": item.n_usable_labels,
                "max_confidence": round(item.max_confidence, 4),
                "n_ocr_errors": item.n_ocr_errors,
            }
        )
    frame = pd.DataFrame(rows, columns=list(DEBUG_COLUMN_ORDER))
    return frame.sort_values(["session_id", "tile"], ignore_index=True)


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


def upscaled(patch: cv2.typing.MatLike, factor: float) -> cv2.typing.MatLike:
    """Enlarge a label patch before recognition.

    Name labels are small enough to sit near the limit of what on-device OCR
    reads reliably, and enlarging costs almost nothing on a patch this size.
    """
    if factor <= 1.0 or patch.size == 0:
        return patch
    height, width = patch.shape[:2]
    return cv2.resize(
        patch,
        (max(1, round(width * factor)), max(1, round(height * factor))),
        interpolation=cv2.INTER_CUBIC,
    )


def _read_frame_regions(
    image: cv2.typing.MatLike,
    geometry: Sequence[RegionGeometry],
    *,
    config: AppConfig,
    backend: OcrBackend,
    found: MutableMapping[str, list[str]],
    confidence: MutableMapping[str, float],
    counts: Mapping[str, MutableMapping[str, float]],
) -> int:
    """Read every region of one frame, accumulating labels and counts.

    Returns:
        How many reads failed, so the caller can flag a session whose OCR did
        not work at all.
    """
    ocr_config = config.speakers.label_ocr
    errors = 0

    for region in geometry:
        side = config.video.side_of_tile(region.tile)
        if side is None:  # pragma: no cover - guarded by is_two_tile
            continue
        left, top, box_w, box_h = region.label_pixels
        patch = upscaled(image[top : top + box_h, left : left + box_w], ocr_config.upscale)
        counts[side]["frames"] += 1

        try:
            lines = backend.read(patch, languages=ocr_config.languages)
        except OcrError:
            # One unreadable patch must not lose the whole session. The error
            # type is counted; the exception text is not logged, because a
            # backend may quote what it was reading.
            errors += 1
            counts[side]["errors"] += 1
            continue

        counts[side]["observations"] += len(lines)
        for line in lines:
            if line.confidence < ocr_config.min_confidence:
                continue
            counts[side]["above"] += 1
            confidence[side] = max(confidence[side], line.confidence)
            key = normalise_label(line.text)
            if not key:
                continue
            counts[side]["usable"] += 1
            found[side].append(key)

    return errors


def _build_observations(
    session_id: int,
    *,
    config: AppConfig,
    geometry: Sequence[RegionGeometry],
    content: ContentBox,
    frame_size: tuple[int, int],
    counts: Mapping[str, Mapping[str, float]],
    confidence: Mapping[str, float],
) -> tuple[RegionObservations, ...]:
    """Assemble the per-region diagnostic records for one session."""
    frame_width, frame_height = frame_size
    records: list[RegionObservations] = []
    for region in geometry:
        side = config.video.side_of_tile(region.tile)
        if side is None:  # pragma: no cover - guarded by is_two_tile
            continue
        tally = counts[side]
        records.append(
            RegionObservations(
                session_id=session_id,
                frame_width=frame_width,
                frame_height=frame_height,
                content_detected=content.detected,
                content_box=content.box,
                content_bars=content.bars,
                geometry=region,
                upscale=config.speakers.label_ocr.upscale,
                n_frames_read=int(tally["frames"]),
                n_observations=int(tally["observations"]),
                n_above_confidence=int(tally["above"]),
                n_usable_labels=int(tally["usable"]),
                max_confidence=confidence[side],
                n_ocr_errors=int(tally["errors"]),
            )
        )
    return tuple(records)


def read_session_labels(
    session: RawSession,
    *,
    config: AppConfig,
    backend: OcrBackend,
    tools: FfmpegTools,
    scratch: Path,
) -> tuple[SessionLabels, tuple[RegionObservations, ...]]:
    """OCR the name label in each tile of one session.

    Frames are extracted, read and deleted one at a time, so no decoded frame
    of a real recording is left on disk.

    Returns:
        The labels found per side, and a diagnostic record per region saying
        exactly where it looked and how much it saw there.
    """
    ocr_config = config.speakers.label_ocr
    video = config.video

    if not video.is_two_tile:
        return (
            SessionLabels(session.session_id, {}, (FLAG_NOT_TWO_TILE, FLAG_INCONCLUSIVE)),
            (),
        )

    duration = parse_media_info(tools.probe(session.path)).duration_s
    times = sample_times(ocr_config.sample_times_seconds, duration)

    found: dict[str, list[str]] = {SIDE_LEFT: [], SIDE_RIGHT: []}
    confidence: dict[str, float] = {SIDE_LEFT: 0.0, SIDE_RIGHT: 0.0}
    counts: dict[str, dict[str, float]] = {
        side: {"frames": 0, "observations": 0, "above": 0, "usable": 0, "errors": 0}
        for side in (SIDE_LEFT, SIDE_RIGHT)
    }
    geometry: tuple[RegionGeometry, ...] = ()
    content = ContentBox(box=FULL_FRAME, detected=False)
    frame_width = frame_height = 0
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

        frame_height, frame_width = image.shape[:2]
        content = (
            detect_content_box(image)
            if video.letterbox_detection == "auto"
            else ContentBox(box=FULL_FRAME, detected=False)
        )
        geometry = resolve_regions(config, frame_width, frame_height, content=content)

        errors += _read_frame_regions(
            image,
            geometry,
            config=config,
            backend=backend,
            found=found,
            confidence=confidence,
            counts=counts,
        )

    read_anything = any(found[side] for side in (SIDE_LEFT, SIDE_RIGHT))
    flags = () if read_anything or not errors else (FLAG_OCR_ERROR, FLAG_INCONCLUSIVE)
    if errors:
        logger.warning("session %s: %d OCR read(s) failed", session.session_id, errors)

    observations = _build_observations(
        session.session_id,
        config=config,
        geometry=geometry,
        content=content,
        frame_size=(frame_width, frame_height),
        counts=counts,
        confidence=confidence,
    )

    labels = SessionLabels(
        session_id=session.session_id,
        by_side={
            side: TileLabels(tuple(dict.fromkeys(found[side])), confidence[side])
            for side in (SIDE_LEFT, SIDE_RIGHT)
        },
        flags=flags,
    )
    return labels, observations


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
    #: Per-region diagnostics: where each region was and how much was seen
    #: there. Empty when OCR did not run.
    debug: pd.DataFrame = field(default_factory=pd.DataFrame)
    debug_path: Path | None = None
    observations: tuple[RegionObservations, ...] = ()

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
    keys: Mapping[str, RecurringLabel] | None = None,
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
        keys: Opaque key per recurring label, recorded per session.
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
                left.labels,
                right.labels,
                recurring=recurring,
                patterns=patterns,
                keys=keys,
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
                "recurring_label_key": decision.matched_key,
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
    frame["recurring_label_key"] = frame["recurring_label_key"].astype("object")
    frame["flags"] = frame["flags"].fillna("").astype(str)
    return frame.sort_values("session_id", ignore_index=True)


def _cohort_notes(
    label_keys: Mapping[str, RecurringLabel],
    *,
    n_sessions: int,
    usable: bool,
    patterns: Sequence[str],
    token_env: str,
) -> list[str]:
    """Explain a cohort that could not identify a recurring speaker.

    A bare "inconclusive" reads like an OCR failure, when the cause is often
    that there was no cohort to compare against.
    """
    if patterns:
        logger.info("%s: using %d explicit label pattern(s)", STAGE, len(patterns))
        return []

    logger.info(
        "%s: %d recurring label(s) identified across the cohort: %s",
        STAGE,
        len(label_keys),
        ", ".join(f"{item.key} in {item.n_sessions} session(s)" for item in label_keys.values())
        or "none",
    )
    if not usable:
        return []
    if n_sessions < _MIN_SESSIONS_FOR_RECURRENCE:
        return [
            f"only {n_sessions} session(s) were read, so no label can be shown to "
            f"recur; run without --sessions, or set {token_env} to identify the "
            f"psychiatrist directly"
        ]
    if not label_keys:
        return [
            "no label recurred across the sessions read, so the psychiatrist could not "
            "be identified; check speakers.label_ocr.label_region against the previews, "
            "and `vc verify-layout --debug-region` for where it looked"
        ]
    return []


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
    diagnostics: list[RegionObservations] = []

    def read_one(session: RawSession) -> str:
        if not usable:
            labels[session.session_id] = SessionLabels(
                session.session_id, {}, (FLAG_OCR_UNAVAILABLE, FLAG_INCONCLUSIVE)
            )
            return "OCR unavailable"
        result, observations = read_session_labels(
            session,
            config=config,
            backend=engine,
            tools=binaries,
            scratch=roots.work_path("tmp", "verify_layout", str(session.session_id)),
        )
        labels[session.session_id] = result
        diagnostics.extend(observations)
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
    label_keys = assign_recurring_keys(
        {sid: result.all_labels for sid, result in labels.items()},
        min_recurrence=ocr_config.min_recurrence,
    )
    recurring = frozenset(label_keys)
    extra_notes = _cohort_notes(
        label_keys,
        n_sessions=len(labels),
        usable=usable,
        patterns=patterns,
        token_env=ocr_config.psychiatrist_label_env,
    )

    rows = build_rows(
        labels,
        {session.session_id: session.wave for session in selected},
        recurring=recurring,
        patterns=patterns,
        keys=label_keys,
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

    # Rows for sessions outside this run are kept rather than deleted. One
    # caveat specific to this stage: the psychiatrist's label is identified by
    # recurrence *across the cohort in the run*, so a carried row was decided
    # against a different set of sessions than a fresh one. The `method` column
    # records how each row was decided, and a subset run cannot establish
    # recurrence at all, which is why a note says so.
    target = layout_path(roots)
    computed = {int(cast("int", row["session_id"])) for row in rows}
    carried = carry_forward(target, computed=computed, columns=list(COLUMN_ORDER), stage=STAGE)
    if carried:
        caveat = (
            f"{len(carried.rows)} kept row(s) were decided against the cohort of an "
            f"earlier run; the recurring label is identified across whichever sessions "
            f"are in a run, so only a full run settles it for the whole cohort"
        )
        logger.warning("%s: %s", STAGE, caveat)
        report = report.with_notes([*carried.notes(STAGE), caveat])
    frame = build_frame(
        combine({int(cast("int", row["session_id"])): row for row in rows}, carried)
    )
    validate(frame, LAYOUT_SCHEMA, context=STAGE)
    write_csv(target, frame)
    logger.info(
        "wrote %s with %d row(s) (%d from this run, %d kept)",
        target,
        len(frame),
        len(rows),
        len(carried.rows),
    )

    debug_path: Path | None = None
    ordered_diagnostics = sorted(
        diagnostics, key=lambda item: (item.session_id, item.geometry.tile)
    )
    debug_table = debug_frame(ordered_diagnostics)
    if not debug_table.empty:
        debug_path = roots.out_path(LAYOUT_DEBUG_FILENAME)
        write_csv(debug_path, debug_table)
        logger.info("wrote %s with %d row(s)", debug_path, len(debug_table))

    return LayoutResult(
        report=report,
        frame=frame,
        path=target,
        debug=debug_table,
        debug_path=debug_path,
        observations=tuple(ordered_diagnostics),
    )


def debug_report(observations: Sequence[RegionObservations], config: AppConfig) -> list[str]:
    """Render the per-region diagnostic, grouped by session.

    Reports where every region was, in fractional and pixel coordinates, and
    how much OCR saw there. Never the text itself.
    """
    if not observations:
        return [
            "no regions were examined: OCR did not run, so there is nothing to "
            "diagnose. `vc doctor` reports whether the backend is usable."
        ]

    by_session: dict[int, list[RegionObservations]] = {}
    for item in observations:
        by_session.setdefault(item.session_id, []).append(item)

    lines = [
        "region diagnostic (coordinates only; recognised text is never shown)",
        f"letterbox detection: {config.video.letterbox_detection}"
        f" | label upscale: {config.speakers.label_ocr.upscale:g}x"
        f" | min confidence: {config.speakers.label_ocr.min_confidence:g}",
    ]
    for session_id in sorted(by_session):
        regions = by_session[session_id]
        first = regions[0]
        lines.append("")
        lines.append(f"session {session_id}: frame {first.frame_width}x{first.frame_height}")
        if first.content_detected:
            left, top, right, bottom = first.content_bars
            content_pixels = first.content_box.to_pixels(first.frame_width, first.frame_height)
            lines.append(
                f"  letterbox: bars l={left} t={top} r={right} b={bottom}; "
                f"content {format_box(first.content_box, content_pixels)}"
            )
            lines.append(
                "    tile fractions are interpreted within this content area, not the whole frame"
            )
        else:
            lines.append("  letterbox: none detected; tiles are fractions of the whole frame")
        for region in sorted(regions, key=lambda item: item.geometry.tile_box.x):
            lines.extend(region.report_lines())

    totals = {
        "observations": sum(item.n_observations for item in observations),
        "above confidence": sum(item.n_above_confidence for item in observations),
        "usable labels": sum(item.n_usable_labels for item in observations),
        "OCR errors": sum(item.n_ocr_errors for item in observations),
    }
    lines.append("")
    lines.append("totals across every region: " + ", ".join(f"{k} {v}" for k, v in totals.items()))
    if totals["observations"] == 0:
        lines.append(
            "  Nothing at all was recognised. That points at the region rather than "
            "the recogniser: check the label box against an annotated preview "
            "(`vc preview --label-regions --force`)."
        )
    elif totals["usable labels"] == 0:
        lines.append(
            "  Text was found but none of it survived normalisation or the "
            "confidence threshold. Try lowering speakers.label_ocr.min_confidence "
            "or raising upscale."
        )
    return lines


def _recurring_speaker_lines(frame: pd.DataFrame) -> list[str]:
    """Cross-tabulate which recurring speaker settled each session, by wave.

    Whether one person ran every session, or one per recruitment wave, decides
    how many psychiatrist reference clips are needed and which sessions each
    one covers. The keys are opaque ordinals, so this says nothing about who
    anyone is.
    """
    if "recurring_label_key" not in frame:
        return []
    settled = frame.loc[frame["recurring_label_key"].notna()]
    if settled.empty:
        return []

    lines = ["", "recurring speaker per session, by wave (opaque keys, not names):"]
    for key in sorted(set(settled["recurring_label_key"])):
        rows = settled.loc[settled["recurring_label_key"] == key]
        by_wave = rows.groupby("wave", dropna=False)["session_id"].count().sort_index()
        spread = ", ".join(f"{wave}={count}" for wave, count in by_wave.items())
        lines.append(f"  {key}: {len(rows)} session(s)  [{spread}]")
        ids = sorted(int(i) for i in rows["session_id"])
        lines.append(f"    {ids}")

    keys_per_wave = settled.groupby("wave")["recurring_label_key"].nunique()
    mixed = sorted(str(wave) for wave, n in keys_per_wave.items() if n > 1)
    if len(set(settled["recurring_label_key"])) == 1:
        lines.append("  one recurring speaker across every settled session.")
    elif mixed:
        lines.append(
            f"  more than one recurring speaker appears within wave(s) {mixed}, so the "
            f"split is NOT clean by wave: a session-to-psychiatrist map is needed, not "
            f"just one clip per wave."
        )
    else:
        lines.append(
            "  each wave has exactly one recurring speaker, so the split is clean by "
            "wave: one reference clip per wave covers every settled session."
        )
    return lines


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

    lines.extend(_recurring_speaker_lines(frame))

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
