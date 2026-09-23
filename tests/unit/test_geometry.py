"""Frame geometry: letterboxing and region resolution.

Label OCR found nothing in any of the 62 recordings, and the cause was
geometric: the recordings carry 180px bars top and bottom, so the content is
two 16:9 tiles side by side inside a 720p frame. A label region expressed as a
fraction of the whole frame landed in the bottom bar. These tests cover that
case specifically, along with the ones that would break the correction.
"""

from __future__ import annotations

import numpy as np
import pytest

from vc_multimodal.config import AppConfig, CropBox, load_config
from vc_multimodal.features.geometry import (
    FULL_FRAME,
    ContentBox,
    detect_content_box,
    format_box,
    place_within,
    resolve_regions,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"


def frame_with_bars(
    width: int = 1280,
    height: int = 720,
    *,
    left: int = 0,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    brightness: int = 120,
) -> np.ndarray:
    """A frame whose content area is surrounded by black bars."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[top : height - bottom, left : width - right] = brightness
    return image


# ---------------------------------------------------------------------------
# the real case
# ---------------------------------------------------------------------------
def test_the_real_recordings_letterboxing_is_detected():
    """1280x720 with 180px bars: two 16:9 tiles side by side, as found."""
    content = detect_content_box(frame_with_bars(top=180, bottom=180))

    assert content.detected
    assert content.bars == (0, 180, 0, 180)
    assert content.box.y == pytest.approx(0.25)
    assert content.box.height == pytest.approx(0.5)
    assert content.box.width == pytest.approx(1.0)


def test_the_label_region_moves_out_of_the_black_bar(default_config: AppConfig):
    """The bug: y=0.82 of a 720p frame is inside a 180px bottom bar."""
    content = detect_content_box(frame_with_bars(top=180, bottom=180))

    uncorrected = resolve_regions(default_config, 1280, 720, content=None)
    corrected = resolve_regions(default_config, 1280, 720, content=content)

    bar_starts_at = 720 - 180
    # Without the correction the label box sits inside the bottom bar.
    assert uncorrected[0].label_pixels[1] >= bar_starts_at
    # With it, the box is inside the picture.
    top = corrected[0].label_pixels[1]
    assert top + corrected[0].label_pixels[3] <= bar_starts_at


def test_the_corrected_tiles_are_the_two_sixteen_by_nine_halves(
    default_config: AppConfig,
):
    content = detect_content_box(frame_with_bars(top=180, bottom=180))
    regions = resolve_regions(default_config, 1280, 720, content=content)

    assert [region.tile_pixels for region in regions] == [
        (0, 180, 640, 360),
        (640, 180, 640, 360),
    ]


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def test_pillarboxing_is_detected():
    content = detect_content_box(frame_with_bars(left=80, right=80))
    assert content.bars == (80, 0, 80, 0)
    assert content.box.x == pytest.approx(0.0625)


def test_bars_on_all_four_sides():
    content = detect_content_box(frame_with_bars(left=10, top=20, right=30, bottom=40))
    assert content.bars == (10, 20, 30, 40)


def test_a_frame_with_no_bars_reports_the_full_frame():
    content = detect_content_box(frame_with_bars())
    assert not content.detected
    assert content.box == FULL_FRAME
    assert content.bars == (0, 0, 0, 0)


def test_an_entirely_black_frame_is_not_believed():
    """Nothing to centre on, so the full frame is the only safe answer."""
    content = detect_content_box(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert not content.detected
    assert content.box == FULL_FRAME


def test_a_detection_that_would_discard_most_of_the_picture_is_refused():
    """Something other than letterboxing is going on; do not crop to it."""
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[350:370, 630:650] = 200  # a tiny bright patch
    content = detect_content_box(image)
    assert not content.detected


def test_a_dark_row_inside_the_picture_is_not_treated_as_a_bar():
    image = frame_with_bars(top=180, bottom=180)
    image[400:410, :] = 0  # a dark band in the middle of the content
    content = detect_content_box(image)
    assert content.bars == (0, 180, 0, 180)


def test_near_black_compression_noise_still_counts_as_a_bar():
    """Encoders do not produce a perfectly flat zero."""
    image = frame_with_bars(top=180, bottom=180)
    image[:180, :] = 8
    content = detect_content_box(image)
    assert content.detected
    assert content.bars[1] == 180


def test_the_darkness_threshold_is_adjustable():
    image = frame_with_bars(top=180, bottom=180)
    # Bars brighter than the default threshold are not bars by default.
    image[:180, :] = 40
    image[540:, :] = 40
    assert not detect_content_box(image).detected
    assert detect_content_box(image, darkness=50).detected


def test_a_grayscale_frame_is_accepted():
    image = np.zeros((720, 1280), dtype=np.uint8)
    image[180:540, :] = 120
    assert detect_content_box(image).bars == (0, 180, 0, 180)


def test_an_empty_frame_is_handled():
    assert not detect_content_box(np.zeros((0, 0, 3), dtype=np.uint8)).detected


# ---------------------------------------------------------------------------
# placing one box inside another
# ---------------------------------------------------------------------------
def test_placing_a_box_within_the_full_frame_changes_nothing():
    box = CropBox(x=0.5, y=0.82, width=0.3, height=0.18)
    assert place_within(box, FULL_FRAME) == box


def test_placing_a_half_within_a_half():
    outer = CropBox(x=0.0, y=0.25, width=1.0, height=0.5)
    inner = CropBox(x=0.5, y=0.0, width=0.5, height=1.0)
    placed = place_within(inner, outer)
    assert placed.x == pytest.approx(0.5)
    assert placed.y == pytest.approx(0.25)
    assert placed.width == pytest.approx(0.5)
    assert placed.height == pytest.approx(0.5)


def test_placement_stays_inside_the_outer_box():
    outer = CropBox(x=0.1, y=0.1, width=0.8, height=0.8)
    placed = place_within(CropBox(x=0.9, y=0.9, width=0.1, height=0.1), outer)
    assert placed.x + placed.width <= outer.x + outer.width + 1e-9
    assert placed.y + placed.height <= outer.y + outer.height + 1e-9


# ---------------------------------------------------------------------------
# region resolution
# ---------------------------------------------------------------------------
def test_regions_are_ordered_left_to_right_with_roles(default_config: AppConfig):
    regions = resolve_regions(default_config, 1280, 720)
    assert [region.tile for region in regions] == ["left", "right"]
    assert [region.role for region in regions] == ["psychiatrist", "participant"]


def test_regions_report_both_fractional_and_pixel_coordinates(
    default_config: AppConfig,
):
    region = resolve_regions(default_config, 1280, 720)[0]
    assert region.tile_pixels == region.tile_box.to_pixels(1280, 720)
    assert region.label_pixels == region.label_box.to_pixels(1280, 720)


def test_with_no_label_region_the_whole_tile_is_read():
    config = load_config(DEFAULT, overrides={"speakers.label_ocr.label_region": None})
    region = resolve_regions(config, 1280, 720)[0]
    assert region.label_is_whole_tile
    assert region.label_box == region.tile_box


def test_a_configured_label_region_is_not_the_whole_tile(default_config: AppConfig):
    assert not resolve_regions(default_config, 1280, 720)[0].label_is_whole_tile


def test_an_undetected_content_box_is_the_same_as_none(default_config: AppConfig):
    undetected = ContentBox(box=FULL_FRAME, detected=False)
    assert resolve_regions(default_config, 1280, 720, content=undetected) == resolve_regions(
        default_config, 1280, 720, content=None
    )


def test_label_regions_of_the_two_tiles_do_not_overlap(default_config: AppConfig):
    content = detect_content_box(frame_with_bars(top=180, bottom=180))
    left, right = resolve_regions(default_config, 1280, 720, content=content)
    left_end = left.label_pixels[0] + left.label_pixels[2]
    assert left_end <= right.label_pixels[0]


def test_a_three_tile_layout_still_resolves():
    config = load_config(
        DEFAULT,
        overrides={
            "video.tiles": {
                "left": {"x": 0.0, "y": 0.0, "width": 0.34, "height": 1.0},
                "middle": {"x": 0.34, "y": 0.0, "width": 0.32, "height": 1.0},
                "right": {"x": 0.66, "y": 0.0, "width": 0.34, "height": 1.0},
            }
        },
    )
    regions = resolve_regions(config, 1280, 720)
    assert [region.tile for region in regions] == ["left", "middle", "right"]
    assert regions[1].role == "unassigned"


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def test_a_box_is_reported_in_both_coordinate_systems():
    box = CropBox(x=0.5, y=0.25, width=0.5, height=0.5)
    text = format_box(box, box.to_pixels(1280, 720))
    assert "x=0.5000" in text
    assert "640x360 at (640,180)" in text
