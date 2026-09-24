"""Configuration loading, merging, override and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vc_multimodal.config import (
    AppConfig,
    ConfigError,
    CropBox,
    DatasetConfig,
    deep_merge,
    load_config,
    read_yaml,
    set_by_dotted_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"
PILOT = REPO_ROOT / "config" / "pilot.yaml"


def test_shipped_default_config_is_valid(default_config: AppConfig):
    assert default_config.dataset.expected_sessions == 62
    assert len(default_config.dataset.folders) == 5
    assert set(default_config.dataset.waves) == {"winter", "summer"}


def test_shipped_default_matches_the_documented_raw_layout(default_config: AppConfig):
    dataset = default_config.dataset
    assert dataset.wave_of_folder("December 21 2025") == "winter"
    assert dataset.wave_of_folder("August 1 2026") == "summer"
    assert dataset.wave_of_id(28) == "winter"
    assert dataset.wave_of_id(210) == "summer"
    assert dataset.wave_of_id(80) is None  # the gap between the two ID ranges
    assert 210 in dataset.known_short_sessions


def test_pilot_overlay_wins_over_default():
    config = load_config(DEFAULT, overlays=[PILOT])
    assert config.runtime.workers == 2
    assert config.runtime.log_level == "DEBUG"
    # Untouched sections still come from the default.
    assert config.dataset.expected_sessions == 62


def test_cli_override_beats_every_file():
    config = load_config(DEFAULT, overlays=[PILOT], overrides={"runtime.workers": 7})
    assert config.runtime.workers == 7


def test_override_can_reach_a_nested_section():
    config = load_config(DEFAULT, overrides={"face.sample_fps": 5.0})
    assert config.face.sample_fps == 5.0


def test_config_is_frozen(default_config: AppConfig):
    with pytest.raises(Exception, match=r"frozen|immutable"):
        default_config.runtime.workers = 4  # type: ignore[misc]


def test_snapshot_round_trips_through_yaml(default_config: AppConfig):
    restored = AppConfig.model_validate(yaml.safe_load(default_config.to_yaml()))
    assert restored == default_config


def test_unknown_key_is_an_error(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("runtime:\n  wrkers: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"wrkers|extra"):
        load_config(DEFAULT, overlays=[bad])


def test_missing_file_names_the_path(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_is_reported(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("runtime: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        read_yaml(bad)


def test_non_mapping_config_is_rejected(tmp_path: Path):
    bad = tmp_path / "list.yaml"
    bad.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        read_yaml(bad)


def test_empty_config_file_is_an_empty_mapping(tmp_path: Path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert read_yaml(empty) == {}


# ---------------------------------------------------------------------------
# merge and override mechanics
# ---------------------------------------------------------------------------
def test_deep_merge_merges_nested_mappings():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    assert deep_merge(base, {"a": {"y": 20}}) == {"a": {"x": 1, "y": 20}, "b": 3}


def test_deep_merge_replaces_lists_wholesale():
    """An overlay listing pilot sessions means those, not those appended."""
    base = {"runtime": {"pilot_sessions": [1, 2, 3]}}
    merged = deep_merge(base, {"runtime": {"pilot_sessions": [9]}})
    assert merged["runtime"]["pilot_sessions"] == [9]


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


def test_dotted_override_creates_missing_sections():
    assert set_by_dotted_path({}, "a.b.c", 1) == {"a": {"b": {"c": 1}}}


def test_dotted_override_through_a_scalar_is_an_error():
    with pytest.raises(ConfigError, match="not a section"):
        set_by_dotted_path({"a": 1}, "a.b", 2)


def test_empty_dotted_path_is_an_error():
    with pytest.raises(ConfigError, match="empty config override"):
        set_by_dotted_path({}, "", 1)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_crop_box_rejects_a_box_leaving_the_frame():
    with pytest.raises(ValueError, match="right edge"):
        CropBox(x=0.6, y=0.0, width=0.5, height=1.0)
    with pytest.raises(ValueError, match="bottom edge"):
        CropBox(x=0.0, y=0.6, width=1.0, height=0.5)


def test_crop_box_rejects_zero_size():
    with pytest.raises(ValueError, match="greater than 0"):
        CropBox(x=0.0, y=0.0, width=0.0, height=1.0)


def test_adjacent_tiles_tile_exactly_with_no_seam_or_overlap(default_config: AppConfig):
    """Rounding edges rather than origin+size keeps halves exactly adjacent."""
    width, height = 1919, 1081  # deliberately odd, to catch rounding drift
    left = default_config.video.tiles["left"].to_pixels(width, height)
    right = default_config.video.tiles["right"].to_pixels(width, height)
    assert left[0] + left[2] == right[0]
    assert left[2] + right[2] == width
    assert left[3] == right[3] == height


def test_crop_box_always_yields_at_least_one_pixel():
    tiny = CropBox(x=0.999, y=0.999, width=0.001, height=0.001)
    _, _, w, h = tiny.to_pixels(10, 10)
    assert w >= 1
    assert h >= 1


# ---------------------------------------------------------------------------
# dataset cross-checks
# ---------------------------------------------------------------------------
def _dataset(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "expected_sessions": 4,
        "media_glob": "*.mp4",
        "waves": {
            "a": {"folders": ["Folder A"], "id_min": 1, "id_max": 10},
            "b": {"folders": ["Folder B"], "id_min": 11, "id_max": 20},
        },
        "duration": {"min_seconds": 1.0, "max_seconds": 2.0, "mad_k": 3.0},
    }
    base.update(overrides)
    return base


def test_dataset_rejects_a_folder_claimed_by_two_waves():
    spec = _dataset(
        waves={
            "a": {"folders": ["Shared"], "id_min": 1, "id_max": 10},
            "b": {"folders": ["Shared"], "id_min": 11, "id_max": 20},
        }
    )
    with pytest.raises(ValueError, match="claimed by both"):
        DatasetConfig.model_validate(spec)


def test_dataset_rejects_overlapping_id_ranges():
    spec = _dataset(
        waves={
            "a": {"folders": ["A"], "id_min": 1, "id_max": 62},
            "b": {"folders": ["B"], "id_min": 50, "id_max": 261},
        }
    )
    with pytest.raises(ValueError, match=r"ranges of .* overlap"):
        DatasetConfig.model_validate(spec)


def test_dataset_rejects_an_inverted_id_range():
    spec = _dataset(waves={"a": {"folders": ["A"], "id_min": 62, "id_max": 1}})
    with pytest.raises(ValueError, match="exceeds id_max"):
        DatasetConfig.model_validate(spec)


def test_dataset_rejects_an_inverted_duration_window():
    with pytest.raises(ValueError, match="below max_seconds"):
        DatasetConfig.model_validate(
            _dataset(duration={"min_seconds": 900.0, "max_seconds": 480.0, "mad_k": 3.0})
        )


def test_dataset_requires_at_least_one_wave():
    with pytest.raises(ValueError, match="at least one wave"):
        DatasetConfig.model_validate(_dataset(waves={}))


# ---------------------------------------------------------------------------
# stage preconditions that are deliberately not load-time errors
# ---------------------------------------------------------------------------
def test_the_default_import_dir_is_the_supplied_diarization_run(default_config: AppConfig):
    """The professor supplied the original whisper-diarization output."""
    assert default_config.diarization.backend == "import"
    assert default_config.diarization.import_dir == "diarization/diarizations_original"


def test_a_config_loads_even_though_import_dir_is_unset(default_config: AppConfig):
    """`vc inventory` must work before diarization has been configured."""
    config = load_config(DEFAULT, overrides={"diarization.import_dir": None})
    assert config.diarization.import_dir is None


def test_require_import_dir_explains_what_to_configure():
    config = load_config(DEFAULT, overrides={"diarization.import_dir": None})
    with pytest.raises(ConfigError, match="import_dir"):
        config.diarization.require_import_dir()


def test_require_import_dir_returns_the_configured_value():
    config = load_config(DEFAULT, overrides={"diarization.import_dir": "diarization"})
    assert config.diarization.require_import_dir() == "diarization"


def test_participant_map_grouping_requires_a_map():
    with pytest.raises(ConfigError, match="participant_map"):
        load_config(DEFAULT, overrides={"model.grouping": "participant_map"})


def test_participant_map_grouping_accepts_a_map():
    config = load_config(
        DEFAULT,
        overrides={"model.grouping": "participant_map", "model.participant_map": "map.csv"},
    )
    assert config.model.grouping == "participant_map"


def test_video_roles_must_name_real_tiles():
    with pytest.raises(ConfigError, match="not defined in tiles"):
        load_config(DEFAULT, overrides={"video.participant_tile": "middle"})


def test_video_roles_must_differ():
    with pytest.raises(ConfigError, match="must differ"):
        load_config(DEFAULT, overrides={"video.participant_tile": "left"})


def test_f0_range_must_be_ordered():
    with pytest.raises(ConfigError, match="below f0_ceiling"):
        load_config(DEFAULT, overrides={"prosody.f0_floor_hz": 600.0})


def test_workers_must_be_positive_or_null():
    with pytest.raises(ConfigError, match="at least 1"):
        load_config(DEFAULT, overrides={"runtime.workers": 0})
    assert load_config(DEFAULT, overrides={"runtime.workers": None}).runtime.workers is None
