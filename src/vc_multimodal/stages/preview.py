"""Stage: write one contact sheet per session so the crop can be checked by eye.

This is the only stage that writes an image, and it exists because the video
layout is an open question: Zoom two-person gallery view is expected, with the
psychiatrist on the left, but it could be active-speaker view, in which a single
tile switches between people and per-tile cropping would be invalid.

Each sheet shows, for several timestamps, the whole frame with the configured
crop boxes drawn on it, followed by each tile as it will actually be cropped.
Several timestamps are used so a layout that switches mid-session is obvious.

Previews contain participants' faces. They are written under `$VC_OUT_ROOT`,
are never committed, and are never part of a handoff bundle.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig, CropBox
from vc_multimodal.ffmpeg import FfmpegTools, parse_media_info
from vc_multimodal.io_utils import atomic_path, read_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots, RawSession, discover_sessions, select_sessions
from vc_multimodal.runner import StageReport, run_sessions
from vc_multimodal.stages.inventory import inventory_path

logger = get_logger(__name__)

STAGE: Final = "preview"
PREVIEWS_DIRNAME: Final = "previews"

_LABEL_HEIGHT: Final = 18
_LABEL_FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE: Final = 0.4
# Keep the last frame of a short recording in reach: never seek past this
# fraction of the duration.
_MAX_SEEK_FRACTION: Final = 0.95


def previews_dir(roots: DataRoots) -> Path:
    """Directory holding preview sheets."""
    return roots.out_path(PREVIEWS_DIRNAME, create_parent=True)


def preview_path(roots: DataRoots, session_id: int, image_format: str) -> Path:
    """Where one session's preview sheet is written."""
    return roots.out_path(PREVIEWS_DIRNAME, f"{session_id}.{image_format}")


def _fit_width(image: np.ndarray, width: int) -> np.ndarray:
    """Scale `image` to exactly `width`, preserving aspect ratio."""
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    interpolation = cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _with_label(image: np.ndarray, text: str) -> np.ndarray:
    """Return `image` with a caption strip added above it."""
    strip = np.full((_LABEL_HEIGHT, image.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(strip, text, (4, _LABEL_HEIGHT - 5), _LABEL_FONT, _LABEL_SCALE, (235, 235, 235), 1)
    return np.vstack([strip, image])


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """Bottom-pad `image` so its height matches `height`."""
    if image.shape[0] >= height:
        return image
    filler = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
    return np.vstack([image, filler])


def ordered_tiles(config: AppConfig) -> tuple[tuple[str, CropBox, str], ...]:
    """Tiles left to right, each with the role assigned to it.

    Ordering by position rather than by dictionary key means the sheet reads the
    same way as the video does.
    """
    roles = {
        config.video.participant_tile: "participant",
        config.video.psychiatrist_tile: "psychiatrist",
    }
    tiles = sorted(config.video.tiles.items(), key=lambda item: (item[1].x, item[1].y))
    return tuple((name, box, roles.get(name, "unassigned")) for name, box in tiles)


def compose_contact_sheet(
    frames: Sequence[tuple[float, np.ndarray]],
    tiles: Sequence[tuple[str, CropBox, str]],
    *,
    max_width: int,
) -> np.ndarray:
    """Build one contact sheet from sampled frames.

    Args:
        frames: `(timestamp, frame)` pairs, in the order they should appear.
        tiles: Tiles to crop, as returned by `ordered_tiles`.
        max_width: Width budget for the finished sheet.

    Returns:
        The composed image.

    Raises:
        ValueError: if no frames were supplied.
    """
    if not frames:
        msg = "cannot compose a preview from zero frames"
        raise ValueError(msg)

    cell_width = max(64, max_width // (len(tiles) + 1))
    rows: list[np.ndarray] = []

    for timestamp, frame in frames:
        height, width = frame.shape[:2]

        annotated = frame.copy()
        for name, box, role in tiles:
            left, top, box_w, box_h = box.to_pixels(width, height)
            cv2.rectangle(annotated, (left, top), (left + box_w, top + box_h), (0, 220, 255), 2)
            cv2.putText(
                annotated,
                f"{name}/{role}",
                (left + 4, top + 16),
                _LABEL_FONT,
                _LABEL_SCALE,
                (0, 220, 255),
                1,
            )

        cells = [_with_label(_fit_width(annotated, cell_width), f"t={timestamp:.0f}s full frame")]
        for name, box, role in tiles:
            left, top, box_w, box_h = box.to_pixels(width, height)
            crop = frame[top : top + box_h, left : left + box_w]
            cells.append(_with_label(_fit_width(crop, cell_width), f"{name} = {role}"))

        tallest = max(cell.shape[0] for cell in cells)
        rows.append(np.hstack([_pad_to_height(cell, tallest) for cell in cells]))

    widest = max(row.shape[1] for row in rows)
    padded = [
        np.hstack([row, np.zeros((row.shape[0], widest - row.shape[1], 3), dtype=np.uint8)])
        if row.shape[1] < widest
        else row
        for row in rows
    ]
    return np.vstack(padded)


def _session_durations(roots: DataRoots) -> Mapping[int, float]:
    """Durations from a previously written inventory, empty if there is none.

    A fast path only: sessions absent here are probed individually.
    """
    path = inventory_path(roots)
    if not path.exists():
        return {}
    frame = read_csv(path)
    return {
        int(session_id): float(duration)
        for session_id, duration in zip(frame["session_id"], frame["duration_s"], strict=True)
        if pd.notna(duration)
    }


def sample_times(requested: Sequence[float], duration: float | None) -> tuple[float, ...]:
    """Choose timestamps to sample, given a recording's actual length.

    Requested timestamps that fit are used as-is. If any falls past the end of
    the recording, the whole schedule is replaced by the same number of evenly
    spaced timestamps within the recording instead. Merely clamping would pile
    every out-of-range request onto the same final frame and, after
    de-duplication, leave a single-frame sheet. That matters for the recording
    already known to be unusually short, where a spread of frames is exactly
    what is needed to judge the layout.

    A recording of unknown duration is left to the requested values.
    """
    wanted = [t for t in requested if t >= 0.0]
    if not wanted:
        return (0.0,)
    if duration is None or duration <= 0:
        return tuple(dict.fromkeys(wanted))

    limit = duration * _MAX_SEEK_FRACTION
    if all(t <= limit for t in wanted):
        return tuple(dict.fromkeys(round(t, 3) for t in wanted))

    count = len(wanted)
    spread = [round(limit * (index + 1) / (count + 1), 3) for index in range(count)]
    return tuple(dict.fromkeys(spread))


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    session_ids: Sequence[int] | None = None,
    workers: int | None = None,
    force: bool = False,
    tools: FfmpegTools | None = None,
) -> StageReport:
    """Write one preview sheet per session.

    Args:
        config: Resolved configuration.
        roots: Data roots.
        session_ids: Sessions to preview, or None for all.
        workers: Parallel workers, or None to choose automatically.
        force: Rewrite sheets that already exist.
        tools: Located binaries; discovered if omitted.

    Returns:
        The stage report.
    """
    binaries = tools or FfmpegTools.discover()
    discovery = discover_sessions(roots.data, config.dataset)
    selected, missing = select_sessions(discovery.sessions, session_ids)
    notes = [f"requested session(s) not found: {sorted(missing)}"] if missing else []

    durations = _session_durations(roots)
    tiles = ordered_tiles(config)
    image_format = config.video.preview_format
    previews_dir(roots)

    def is_done(session: RawSession) -> bool:
        return preview_path(roots, session.session_id, image_format).exists()

    def preview_one(session: RawSession) -> str:
        # Prefer the inventory's duration, but fall back to probing this one
        # file. Without that, `vc preview` would depend on `vc inventory` having
        # run first and would fail every seek on a recording shorter than the
        # first requested timestamp.
        duration = durations.get(session.session_id)
        if duration is None:
            duration = parse_media_info(binaries.probe(session.path)).duration_s
        times = sample_times(config.video.preview_times_seconds, duration)
        scratch = roots.work_path("tmp", STAGE, str(session.session_id))
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            frames: list[tuple[float, np.ndarray]] = []
            for index, timestamp in enumerate(times):
                raw = binaries.extract_frame(session.path, timestamp, scratch / f"{index}.png")
                image = cv2.imread(str(raw), cv2.IMREAD_COLOR)
                if image is None:
                    msg = f"could not read the frame extracted at {timestamp:.1f}s"
                    raise RuntimeError(msg)
                frames.append((timestamp, image))
                # Remove each frame as soon as it is in memory: no decoded
                # frame of a real recording is left on disk.
                raw.unlink(missing_ok=True)

            sheet = compose_contact_sheet(frames, tiles, max_width=config.video.preview_max_width)
            target = preview_path(roots, session.session_id, image_format)
            with atomic_path(target, suffix=f".{image_format}") as tmp:
                if not cv2.imwrite(str(tmp), sheet):
                    msg = f"OpenCV could not write {target.name}"
                    raise RuntimeError(msg)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        return f"{len(times)} timestamp(s)"

    report = run_sessions(
        STAGE,
        selected,
        preview_one,
        workers=workers,
        force=force,
        is_done=is_done,
        backend="threads",
        notes=notes,
    )
    logger.info("previews written to %s", previews_dir(roots))
    return report
