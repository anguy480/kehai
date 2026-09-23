"""Frame geometry: letterboxing, tiles, and where the name labels sit.

Label OCR returning inconclusive for every session points at geometry rather
than at the recogniser: if the region handed to OCR is not where the label is,
nothing will ever be read from it. Two things can put it in the wrong place.

First, letterboxing. Tile boxes are configured as fractions of the frame, but a
Zoom recording can carry black bars, so the two participant tiles occupy the
*content* area rather than the whole 1280x720 frame. Splitting the frame down
the middle then splits the content off-centre.

Second, the label region within a tile is a guess about where Zoom draws the
name, and a guess cannot be checked without seeing where it landed.

Everything here is pure: it takes a frame or its dimensions and returns
coordinates, in both fractional and pixel form, so a diagnostic can report
exactly what was looked at.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from vc_multimodal.config import AppConfig, CropBox

# Luminance at or below which a pixel counts as part of a black bar. Video
# compression does not produce a perfectly flat zero, so this is not 0.
DEFAULT_DARKNESS: Final = 16

# A detected content area smaller than this fraction of the frame is treated as
# a detection failure rather than believed: something other than letterboxing
# is going on, and cropping to it would discard most of the picture.
MIN_CONTENT_FRACTION: Final = 0.25

FULL_FRAME: Final = CropBox(x=0.0, y=0.0, width=1.0, height=1.0)

# A grayscale image has two dimensions; a colour one has three.
_GRAYSCALE_DIMS: Final = 2


@dataclass(frozen=True, slots=True)
class ContentBox:
    """The part of a frame that is not letterbox bar.

    Attributes:
        box: The content area as fractions of the whole frame.
        detected: Whether bars were actually found. False means the content
            fills the frame, or detection was not believable, and `box` is the
            full frame.
        bars: Bar thickness in pixels as `(left, top, right, bottom)`.
    """

    box: CropBox
    detected: bool
    bars: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def is_full_frame(self) -> bool:
        """Whether the content area is the entire frame."""
        return not self.detected


def detect_content_box(
    frame: np.ndarray,
    *,
    darkness: int = DEFAULT_DARKNESS,
    min_content_fraction: float = MIN_CONTENT_FRACTION,
) -> ContentBox:
    """Find the non-letterboxed area of a frame.

    A row or column is a bar when every pixel in it is at or below `darkness`.
    Only bars touching the frame edge are removed, so a dark row in the middle
    of the picture is left alone.

    Args:
        frame: A BGR or grayscale image.
        darkness: Luminance at or below which a pixel counts as black.
        min_content_fraction: Refuse a detection smaller than this fraction of
            the frame's area.

    Returns:
        The content area, and whether bars were found.
    """
    if frame.size == 0:
        return ContentBox(box=FULL_FRAME, detected=False)

    grey = frame if frame.ndim == _GRAYSCALE_DIMS else frame.max(axis=2)
    height, width = grey.shape[:2]

    row_bright = grey.max(axis=1) > darkness
    column_bright = grey.max(axis=0) > darkness
    if not row_bright.any() or not column_bright.any():
        # An entirely dark frame: nothing to centre on.
        return ContentBox(box=FULL_FRAME, detected=False)

    top = int(np.argmax(row_bright))
    bottom = height - int(np.argmax(row_bright[::-1]))
    left = int(np.argmax(column_bright))
    right = width - int(np.argmax(column_bright[::-1]))

    bars = (left, top, width - right, height - bottom)
    if not any(bars):
        return ContentBox(box=FULL_FRAME, detected=False)

    area_fraction = ((right - left) * (bottom - top)) / (width * height)
    if area_fraction < min_content_fraction:
        return ContentBox(box=FULL_FRAME, detected=False)

    return ContentBox(
        box=CropBox(
            x=left / width,
            y=top / height,
            width=(right - left) / width,
            height=(bottom - top) / height,
        ),
        detected=True,
        bars=bars,
    )


def place_within(inner: CropBox, outer: CropBox) -> CropBox:
    """Interpret `inner`'s fractions as being relative to `outer`.

    This is what makes a configured 50/50 tile split mean "half of the picture"
    rather than "half of the frame including its black bars".
    """
    return CropBox(
        x=outer.x + inner.x * outer.width,
        y=outer.y + inner.y * outer.height,
        width=inner.width * outer.width,
        height=inner.height * outer.height,
    )


@dataclass(frozen=True, slots=True)
class RegionGeometry:
    """Where one tile, and its name label, were looked for.

    All fractions are relative to the whole frame, so they can be compared
    directly against what is drawn on a preview sheet.
    """

    tile: str
    role: str
    tile_box: CropBox
    tile_pixels: tuple[int, int, int, int]
    label_box: CropBox
    label_pixels: tuple[int, int, int, int]

    @property
    def label_is_whole_tile(self) -> bool:
        """Whether no label sub-region was configured."""
        return self.label_box == self.tile_box


def resolve_regions(
    config: AppConfig,
    frame_width: int,
    frame_height: int,
    *,
    content: ContentBox | None = None,
) -> tuple[RegionGeometry, ...]:
    """Work out every tile and label region for a frame of this size.

    Args:
        config: Resolved configuration, for the tiles, roles and label region.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        content: Detected content area. None or undetected means the tiles are
            fractions of the whole frame.

    Returns:
        One entry per tile, ordered left to right.
    """
    outer = content.box if content is not None and content.detected else FULL_FRAME
    label_region = config.speakers.label_ocr.label_region
    roles: Mapping[str, str] = {
        config.video.participant_tile: "participant",
        config.video.psychiatrist_tile: "psychiatrist",
    }

    regions: list[RegionGeometry] = []
    for name, configured in config.video.tiles_left_to_right():
        tile_box = place_within(configured, outer)
        label_box = place_within(label_region, tile_box) if label_region else tile_box
        regions.append(
            RegionGeometry(
                tile=name,
                role=roles.get(name, "unassigned"),
                tile_box=tile_box,
                tile_pixels=tile_box.to_pixels(frame_width, frame_height),
                label_box=label_box,
                label_pixels=label_box.to_pixels(frame_width, frame_height),
            )
        )
    return tuple(regions)


def format_box(box: CropBox, pixels: Sequence[int]) -> str:
    """Render a box in both fractional and pixel coordinates."""
    left, top, width, height = pixels
    return (
        f"x={box.x:.4f} y={box.y:.4f} w={box.width:.4f} h={box.height:.4f}"
        f"  |  {width}x{height} at ({left},{top})"
    )
