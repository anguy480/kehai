"""The preview stage: timestamp choice, sheet composition and cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from tests.conftest import WINTER_FOLDER, place_fake_media
from vc_multimodal.config import AppConfig, CropBox, load_config
from vc_multimodal.paths import DataRoots
from vc_multimodal.stages import inventory as inventory_stage
from vc_multimodal.stages import preview as stage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"


# ---------------------------------------------------------------------------
# timestamp choice
# ---------------------------------------------------------------------------
def test_timestamps_inside_the_recording_are_used_as_requested():
    assert stage.sample_times((60.0, 300.0, 540.0), 660.0) == (60.0, 300.0, 540.0)


def test_a_short_recording_gets_the_same_number_of_spread_timestamps():
    """Clamping would pile all three onto the final frame and de-duplicate to one."""
    times = stage.sample_times((60.0, 300.0, 540.0), 6.0)
    assert len(times) == 3
    assert times == tuple(sorted(times))
    assert all(0.0 < t < 6.0 for t in times)


def test_spread_timestamps_stay_inside_the_recording():
    times = stage.sample_times((60.0, 300.0, 540.0), 10.0)
    assert max(times) <= 10.0 * 0.95


def test_an_unknown_duration_leaves_the_request_alone():
    assert stage.sample_times((60.0, 300.0), None) == (60.0, 300.0)
    assert stage.sample_times((60.0, 300.0), 0.0) == (60.0, 300.0)


def test_duplicate_requests_collapse():
    assert stage.sample_times((60.0, 60.0), 660.0) == (60.0,)


def test_negative_timestamps_are_dropped():
    assert stage.sample_times((-5.0, 60.0), 660.0) == (60.0,)


def test_an_empty_request_falls_back_to_the_first_frame():
    assert stage.sample_times((), 660.0) == (0.0,)
    assert stage.sample_times((-1.0,), 660.0) == (0.0,)


# ---------------------------------------------------------------------------
# tile ordering
# ---------------------------------------------------------------------------
def test_tiles_are_ordered_left_to_right_with_their_roles(default_config: AppConfig):
    tiles = stage.ordered_tiles(default_config)
    assert [name for name, _, _ in tiles] == ["left", "right"]
    assert [role for _, _, role in tiles] == ["psychiatrist", "participant"]


def test_swapping_the_roles_is_reflected_in_the_labels():
    """The layout is unconfirmed, so the labels must follow the config."""
    config = load_config(
        DEFAULT,
        overrides={"video.participant_tile": "left", "video.psychiatrist_tile": "right"},
    )
    assert [role for _, _, role in stage.ordered_tiles(config)] == [
        "participant",
        "psychiatrist",
    ]


def test_a_tile_with_no_role_is_labelled_unassigned():
    config = load_config(
        DEFAULT,
        overrides={
            "video.tiles": {
                "left": {"x": 0.0, "y": 0.0, "width": 0.4, "height": 1.0},
                "middle": {"x": 0.4, "y": 0.0, "width": 0.2, "height": 1.0},
                "right": {"x": 0.6, "y": 0.0, "width": 0.4, "height": 1.0},
            }
        },
    )
    roles = {name: role for name, _, role in stage.ordered_tiles(config)}
    assert roles["middle"] == "unassigned"


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------
def _frame(width: int = 640, height: int = 360) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (60, 60, 60)
    image[:, width // 2 :] = (180, 180, 180)
    return image


def _tiles() -> list[tuple[str, CropBox, str]]:
    return [
        ("left", CropBox(x=0.0, y=0.0, width=0.5, height=1.0), "psychiatrist"),
        ("right", CropBox(x=0.5, y=0.0, width=0.5, height=1.0), "participant"),
    ]


def test_a_sheet_has_one_row_per_timestamp_and_a_column_per_tile_plus_the_frame():
    frames = [(60.0, _frame()), (300.0, _frame())]
    sheet = stage.compose_contact_sheet(frames, _tiles(), max_width=1200)

    # Three columns: full frame, then each tile.
    assert sheet.shape[1] == pytest.approx(1200, abs=4)
    single = stage.compose_contact_sheet(frames[:1], _tiles(), max_width=1200)
    assert sheet.shape[0] == pytest.approx(single.shape[0] * 2, abs=2)


def test_a_sheet_is_a_three_channel_image():
    sheet = stage.compose_contact_sheet([(60.0, _frame())], _tiles(), max_width=800)
    assert sheet.ndim == 3
    assert sheet.shape[2] == 3
    assert sheet.dtype == np.uint8


def test_composition_does_not_modify_the_source_frame():
    """Boxes are drawn on a copy; the crops must show the unannotated frame."""
    frame = _frame()
    original = frame.copy()
    stage.compose_contact_sheet([(60.0, frame)], _tiles(), max_width=800)
    assert np.array_equal(frame, original)


def test_the_crops_come_from_the_right_halves_of_the_frame():
    """A left/right mix-up here would silently mislabel every later feature."""
    frame = _frame()
    sheet = stage.compose_contact_sheet(
        [(0.0, frame)],
        _tiles(),
        max_width=900,
    )
    third = sheet.shape[1] // 3
    label_rows = slice(30, sheet.shape[0] - 5)
    left_cell = sheet[label_rows, third : 2 * third].mean()
    right_cell = sheet[label_rows, 2 * third :].mean()
    # The synthetic frame is dark on the left, light on the right.
    assert left_cell < right_cell


def test_composing_zero_frames_is_an_error():
    with pytest.raises(ValueError, match="zero frames"):
        stage.compose_contact_sheet([], _tiles(), max_width=800)


def test_a_narrow_width_budget_still_produces_a_usable_sheet():
    sheet = stage.compose_contact_sheet([(60.0, _frame())], _tiles(), max_width=10)
    assert sheet.shape[0] > 0
    assert sheet.shape[1] > 0


# ---------------------------------------------------------------------------
# running the stage
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_run_writes_one_sheet_per_session(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)
    make_real_media(3, folder="January 17 2026")

    report = stage.run(default_config, roots, workers=1)

    assert report.ok
    written = sorted(p.name for p in stage.previews_dir(roots).iterdir())
    assert written == ["28.jpg", "3.jpg"]


@pytest.mark.slow
def test_a_written_sheet_is_a_readable_image(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)
    stage.run(default_config, roots, workers=1)
    image = cv2.imread(str(stage.preview_path(roots, 28, "jpg")))
    assert image is not None
    assert image.shape[0] > 0


@pytest.mark.slow
def test_no_extracted_frame_is_left_on_disk(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    """Decoded frames of real recordings must not accumulate in the work tree."""
    make_real_media(28)
    stage.run(default_config, roots, workers=1)
    leftovers = [p for p in roots.work.rglob("*") if p.is_file() and p.suffix in {".png", ".jpg"}]
    assert leftovers == []


@pytest.mark.slow
def test_existing_sheets_are_skipped_unless_forced(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)
    assert len(stage.run(default_config, roots, workers=1).succeeded) == 1

    second = stage.run(default_config, roots, workers=1)
    assert len(second.skipped) == 1
    assert not second.succeeded

    third = stage.run(default_config, roots, workers=1, force=True)
    assert len(third.succeeded) == 1


@pytest.mark.slow
def test_a_short_recording_is_previewed_without_an_inventory(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    """The stage must not depend on `vc inventory` having run first."""
    make_real_media(210, folder="July 4 2026", duration=6.0)
    report = stage.run(default_config, roots, workers=1)
    assert report.ok
    assert "3 timestamp(s)" in report.succeeded[0].message


@pytest.mark.slow
def test_the_inventory_is_used_to_avoid_seeking_past_the_end(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    """Without the inventory's durations, a 6 s recording would fail every seek."""
    make_real_media(210, folder="July 4 2026", duration=6.0)
    inventory_stage.run(default_config, roots, workers=1)

    report = stage.run(default_config, roots, workers=1)
    assert report.ok
    assert "3 timestamp(s)" in report.succeeded[0].message


@pytest.mark.slow
def test_a_session_that_cannot_be_decoded_is_reported_and_the_run_continues(
    roots: DataRoots, default_config: AppConfig, make_real_media: Any
):
    make_real_media(28)
    place_fake_media(roots.data, WINTER_FOLDER, [29])

    report = stage.run(default_config, roots, workers=1)

    assert [o.session_id for o in report.failed] == [29]
    assert [o.session_id for o in report.succeeded] == [28]
    assert stage.preview_path(roots, 28, "jpg").exists()
    assert not stage.preview_path(roots, 29, "jpg").exists()
