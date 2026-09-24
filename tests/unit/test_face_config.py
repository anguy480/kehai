"""The facial feature configuration, held to the decisions behind it.

The action units are the lab's house set and the confirmatory features are the
three its own published study found predictive (docs/decisions/0013). Those are
commitments, so they are asserted here rather than left to a document that can
drift away from the code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pytest
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from tests.conftest import REAL_FACE_MODEL
from vc_multimodal.config import AppConfig, ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"

# Tanaka et al. 2025 (JMIR Form Res 9:e59261) extracted exactly these with
# OpenFace; Miyamoto et al. 2025 (Acta Psychologica 254:104782) found AU01,
# AU06 and AU12 positively associated with social performance.
HOUSE_UNITS = ("au01", "au02", "au04", "au06", "au12")
PREDICTIVE_UNITS = ("au01", "au06", "au12")


# ---------------------------------------------------------------------------
# the action unit set
# ---------------------------------------------------------------------------
def test_the_configured_units_are_the_labs_house_set(default_config: AppConfig):
    assert default_config.face.unit_keys == HOUSE_UNITS


def test_every_unit_has_a_description_for_the_feature_dictionary(
    default_config: AppConfig,
):
    for unit in default_config.face.action_units:
        assert unit.description
        assert unit.description.islower()


def test_every_unit_maps_to_both_backends(default_config: AppConfig):
    """Values are not comparable across backends, but both must be measurable."""
    for unit in default_config.face.action_units:
        assert unit.blendshapes
        assert unit.openface_column.startswith("AU")
        assert unit.openface_column.endswith("_r")


def test_the_left_right_units_average_two_blendshapes(default_config: AppConfig):
    """A brow or a smile is measured per side and has to be combined."""
    for key in ("au02", "au04", "au06", "au12"):
        unit = default_config.face.unit(key)
        assert unit is not None
        assert len(unit.blendshapes) == 2


def test_the_jaw_and_blink_signals_are_not_action_units(default_config: AppConfig):
    """Jaw opening serves the speaker cross-check; blink is tracking quality."""
    face = default_config.face
    assert face.jaw_blendshape == "jawOpen"
    assert face.jaw_blendshape not in [
        shape for unit in face.action_units for shape in unit.blendshapes
    ]
    assert face.blink_blendshapes


def test_the_required_blendshapes_cover_everything_needed(default_config: AppConfig):
    required = default_config.face.required_blendshapes
    for unit in default_config.face.action_units:
        for shape in unit.blendshapes:
            assert shape in required
    assert default_config.face.jaw_blendshape in required
    assert len(set(required)) == len(required)


def test_a_unit_key_must_suit_a_feature_name():
    with pytest.raises(ConfigError, match="lower-case alphanumeric"):
        load_config(
            DEFAULT,
            overrides={
                "face.action_units": [
                    {
                        "key": "AU_12",
                        "description": "bad key",
                        "blendshapes": ["mouthSmileLeft"],
                        "openface_column": "AU12_r",
                    }
                ]
            },
        )


def test_a_unit_needs_at_least_one_blendshape():
    with pytest.raises(ConfigError, match="at least one blendshape"):
        load_config(
            DEFAULT,
            overrides={
                "face.action_units": [
                    {
                        "key": "au12",
                        "description": "lip corner puller",
                        "blendshapes": [],
                        "openface_column": "AU12_r",
                    }
                ]
            },
        )


def test_duplicate_unit_keys_are_refused():
    unit = {
        "key": "au12",
        "description": "lip corner puller",
        "blendshapes": ["mouthSmileLeft"],
        "openface_column": "AU12_r",
    }
    with pytest.raises(ConfigError, match="duplicate action unit"):
        load_config(DEFAULT, overrides={"face.action_units": [unit, unit]})


# ---------------------------------------------------------------------------
# the confirmatory features follow the precedent
# ---------------------------------------------------------------------------
def test_the_face_primaries_are_the_three_predictive_units(
    default_config: AppConfig,
):
    """Miyamoto et al. 2025: AU01, AU06 and AU12 predicted social performance."""
    tiers = default_config.model.tiers
    for family in ("face_speaking", "face_listening"):
        named = tiers.primary_features[family]
        assert len(named) == 3
        units = {name.split("__")[1].removesuffix("_mean") for name in named}
        assert units == set(PREDICTIVE_UNITS)


def test_the_same_units_are_primary_in_both_windows(default_config: AppConfig):
    """So the speaking/listening contrast is not confounded with the measure."""
    tiers = default_config.model.tiers
    speaking = {n.removeprefix("face_speaking__") for n in tiers.primary_features["face_speaking"]}
    listening = {
        n.removeprefix("face_listening__") for n in tiers.primary_features["face_listening"]
    }
    assert speaking == listening


def test_every_primary_face_feature_is_measurable_from_a_configured_unit(
    default_config: AppConfig,
):
    """A pre-registered feature the extractor cannot produce is not a plan."""
    configured = set(default_config.face.unit_keys)
    tiers = default_config.model.tiers
    for family in ("face_speaking", "face_listening"):
        for name in tiers.primary_features[family]:
            unit = name.split("__")[1].removesuffix("_mean")
            assert unit in configured, name


def test_the_exploratory_units_are_extracted_but_not_promoted(
    default_config: AppConfig,
):
    """AU02 has a published autism association but was not among the three."""
    primary_units = {
        name.split("__")[1].removesuffix("_mean")
        for family in ("face_speaking", "face_listening")
        for name in default_config.model.tiers.primary_features[family]
    }
    assert "au02" in default_config.face.unit_keys
    assert "au02" not in primary_units
    assert "au04" not in primary_units


def test_no_family_is_awaiting_its_primaries_any_more(default_config: AppConfig):
    assert default_config.model.tiers.families_awaiting_primaries == ()


def test_there_are_twelve_confirmatory_features(default_config: AppConfig):
    """Three per family across four families, for 62 observations."""
    assert len(default_config.model.tiers.primary_columns) == 12


# ---------------------------------------------------------------------------
# head pose is not gaze
# ---------------------------------------------------------------------------
def test_head_pose_is_extracted(default_config: AppConfig):
    assert default_config.face.mediapipe.head_pose


def test_nothing_in_the_configuration_claims_to_measure_gaze(
    default_config: AppConfig,
):
    """The precedent's gaze findings rest on Tobii; these are Zoom recordings."""
    units = default_config.face.unit_keys
    assert not any("gaze" in key or "look" in key for key in units)
    primaries = default_config.model.tiers.primary_columns
    assert not any("gaze" in name for name in primaries)


def test_the_model_asset_is_pinned_by_hash(default_config: AppConfig):
    """So a silently updated download cannot change what the features mean."""
    digest = default_config.face.mediapipe.model_sha256
    assert digest is not None
    assert len(digest) == 64
    assert digest == digest.lower()


# ---------------------------------------------------------------------------
# the real model emits what the configuration asks for
# ---------------------------------------------------------------------------
def _landmarker_model() -> Path:
    if REAL_FACE_MODEL is None:
        pytest.skip("the face landmarker model has not been downloaded")
    return REAL_FACE_MODEL


@pytest.mark.slow
def test_the_model_file_matches_its_pinned_hash(default_config: AppConfig):
    model = _landmarker_model()
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    assert digest == default_config.face.mediapipe.model_sha256


@pytest.mark.slow
def test_the_model_emits_every_blendshape_the_configuration_needs(
    default_config: AppConfig,
):
    """AU06 would be unmeasurable if cheekSquint were absent, and AU06 is one
    of the three the precedent found predictive."""
    model = _landmarker_model()
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    image = np.full((480, 480, 3), 200, np.uint8)
    cv2.ellipse(image, (240, 250), (120, 160), 0, 0, 360, (215, 185, 165), -1)
    cv2.circle(image, (200, 210), 14, (40, 40, 40), -1)
    cv2.circle(image, (280, 210), 14, (40, 40, 40), -1)
    cv2.ellipse(image, (240, 330), (45, 20), 0, 0, 180, (90, 50, 50), -1)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )
        )

    assert result.face_blendshapes, "the landmarker found no face to measure"
    emitted = {category.category_name for category in result.face_blendshapes[0]}
    missing = [shape for shape in default_config.face.required_blendshapes if shape not in emitted]
    assert missing == []
    # Head pose comes back as a 4x4 transformation matrix.
    assert np.asarray(result.facial_transformation_matrixes[0]).shape == (4, 4)


@pytest.mark.slow
def test_the_installed_mediapipe_is_below_the_broken_major_version():
    """1.0.1 aborts the process on macOS arm64; see docs/decisions/0003."""
    major = int(mp.__version__.split(".")[0])
    assert major < 1, f"mediapipe {mp.__version__} is pinned out for a reason"
