"""The confirmatory/exploratory split and the multiplicity correction.

This is the part of the analysis a reviewer questions first, so the arithmetic
behind it is tested directly rather than left to the write-up. See
docs/decisions/0012.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vc_multimodal.config import AppConfig, ConfigError, load_config
from vc_multimodal.features.prosody_math import FEATURE_NAMES as PROSODY_FEATURES
from vc_multimodal.features.turn_math import FEATURE_NAMES as TURN_FEATURES
from vc_multimodal.modeling.tiers import (
    describe_plan,
    holm_adjust,
    primary_feature_report,
    resolve_tiers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"

BUILT_COLUMNS = ("session_id", "wave", *TURN_FEATURES, *PROSODY_FEATURES, "qc__flags")


# ---------------------------------------------------------------------------
# what the shipped configuration commits to
# ---------------------------------------------------------------------------
def test_the_primary_features_exist_in_the_stages_that_produce_them(
    default_config: AppConfig,
):
    """A pre-registered feature that does not exist is not pre-registration."""
    produced = {*TURN_FEATURES, *PROSODY_FEATURES}
    for family in ("turns", "prosody"):
        for feature in default_config.model.tiers.primary_features[family]:
            assert feature in produced, feature


def test_the_confirmatory_set_is_small_relative_to_the_sample(
    default_config: AppConfig,
):
    """62 observations and twelve primary features: about five per feature."""
    assert len(default_config.model.tiers.primary_columns) == 12


def test_the_confirmatory_tests_are_far_fewer_than_the_estimates(
    default_config: AppConfig,
):
    plan = resolve_tiers(BUILT_COLUMNS, default_config.model)
    assert plan.n_primary_tests == 4
    assert plan.n_estimates == 32
    assert plan.n_exploratory_estimates == 28


def test_every_family_now_has_its_primaries_fixed(default_config: AppConfig):
    """The face families were filled in from the lab's own published findings
    (docs/decisions/0013), still before any label was seen."""
    tiers = default_config.model.tiers
    assert tiers.families_awaiting_primaries == ()
    assert set(tiers.primary_features) == {
        "turns",
        "prosody",
        "face_speaking",
        "face_listening",
    }
    assert all(len(features) == 3 for features in tiers.primary_features.values())


# ---------------------------------------------------------------------------
# resolving a plan against a real table
# ---------------------------------------------------------------------------
def test_the_plan_splits_the_columns(default_config: AppConfig):
    plan = resolve_tiers(BUILT_COLUMNS, default_config.model)

    assert set(plan.primary) <= set(BUILT_COLUMNS)
    assert not set(plan.primary) & set(plan.exploratory)
    # Identifier and QC columns belong to neither tier.
    assert "session_id" not in plan.primary + plan.exploratory
    assert "qc__flags" not in plan.primary + plan.exploratory


def test_every_feature_lands_in_exactly_one_tier(default_config: AppConfig):
    plan = resolve_tiers(BUILT_COLUMNS, default_config.model)
    features = {*TURN_FEATURES, *PROSODY_FEATURES}
    assert set(plan.primary) | set(plan.exploratory) == features


def test_the_face_primaries_show_as_missing_until_the_stage_produces_them(
    default_config: AppConfig,
):
    """`vc face` is not built yet, so the columns it will write do not exist.

    They are reported as missing rather than quietly ignored, which is the
    same mechanism that would catch a pre-registered feature disappearing.
    """
    plan = resolve_tiers(BUILT_COLUMNS, default_config.model)
    assert all(name.startswith("face_") for name in plan.missing)
    assert len(plan.missing) == 6
    assert not plan.is_complete


def test_a_missing_primary_feature_is_reported_not_dropped(default_config: AppConfig):
    """A pre-registered feature that vanished is a change to the analysis."""
    without = tuple(c for c in BUILT_COLUMNS if c != "turns__latency_median")

    plan = resolve_tiers(without, default_config.model)

    assert "turns__latency_median" in plan.missing
    assert not plan.is_complete


def test_a_complete_plan_says_so():
    config = load_config(
        DEFAULT,
        overrides={
            "model.tiers.primary_features": {
                "turns": ["turns__latency_median"],
                "prosody": ["prosody__f0_semitone_sd"],
            }
        },
    )
    plan = resolve_tiers(BUILT_COLUMNS, config.model)
    assert plan.is_complete
    assert plan.missing == ()
    assert plan.awaiting == ()


def test_an_empty_table_leaves_every_primary_missing(default_config: AppConfig):
    plan = resolve_tiers(["session_id", "wave"], default_config.model)
    assert len(plan.missing) == len(default_config.model.tiers.primary_columns)
    assert plan.primary == ()


def test_the_report_states_both_counts(default_config: AppConfig):
    plan = resolve_tiers(BUILT_COLUMNS, default_config.model)
    text = "\n".join(describe_plan(plan, default_config.model))

    assert "4 test(s)" in text
    assert "28 further estimate(s)" in text
    assert "holm correction" in text
    assert "without confirmatory claims" in text


def test_the_report_names_the_confirmatory_comparisons(default_config: AppConfig):
    plan = resolve_tiers(BUILT_COLUMNS, default_config.model)
    text = "\n".join(describe_plan(plan, default_config.model))
    assert "new_modalities_vs_text: all vs text" in text


def test_the_report_flags_a_family_whose_primaries_are_unfixed():
    """The shipped config has none, so the mechanism is exercised directly."""
    config = load_config(
        DEFAULT,
        overrides={
            "model.tiers.primary_features": {
                "turns": ["turns__latency_median"],
                "face_speaking": [],
            }
        },
    )
    plan = resolve_tiers(BUILT_COLUMNS, config.model)
    text = "\n".join(describe_plan(plan, config.model))
    assert "NOT YET FIXED" in text
    assert "face_speaking" in text


def test_the_shipped_config_has_nothing_awaiting(default_config: AppConfig):
    plan = resolve_tiers(BUILT_COLUMNS, default_config.model)
    assert plan.awaiting == ()
    assert "NOT YET FIXED" not in "\n".join(describe_plan(plan, default_config.model))


def test_the_report_flags_a_missing_primary_feature(default_config: AppConfig):
    without = tuple(c for c in BUILT_COLUMNS if c != "prosody__f0_semitone_sd")
    plan = resolve_tiers(without, default_config.model)
    text = "\n".join(describe_plan(plan, default_config.model))
    assert "MISSING" in text
    assert "not a detail" in text


def test_the_per_family_report_lists_what_is_available(default_config: AppConfig):
    report = primary_feature_report(BUILT_COLUMNS, default_config.model)
    assert len(report["turns"]) == 3
    assert report["face_speaking"] == ()


# ---------------------------------------------------------------------------
# configuration validation
# ---------------------------------------------------------------------------
def test_a_primary_feature_must_match_the_family_it_is_listed_under():
    with pytest.raises(ConfigError, match="is not named"):
        load_config(
            DEFAULT,
            overrides={"model.tiers.primary_features": {"turns": ["prosody__f0_semitone_sd"]}},
        )


def test_a_comparison_must_name_configured_feature_sets():
    with pytest.raises(ConfigError, match=re.escape("not defined in model.feature_sets")):
        load_config(
            DEFAULT,
            overrides={
                "model.tiers.primary_comparisons": [
                    {"name": "bad", "against": ["all", "nonexistent"]}
                ]
            },
        )


def test_a_comparison_cannot_compare_a_set_with_itself():
    with pytest.raises(ConfigError, match="with itself"):
        load_config(
            DEFAULT,
            overrides={
                "model.tiers.primary_comparisons": [{"name": "silly", "against": ["all", "all"]}]
            },
        )


def test_the_stability_check_needs_at_least_two_folds():
    with pytest.raises(ConfigError, match="greater than or equal to 2"):
        load_config(DEFAULT, overrides={"model.tiers.stability_folds": 1})


# ---------------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------------
def test_holm_is_more_powerful_than_bonferroni_at_the_same_error_rate():
    raw = [0.004, 0.02, 0.03, 0.4]
    holm = holm_adjust(raw)
    bonferroni = [min(1.0, p * len(raw)) for p in raw]
    assert holm[0] == pytest.approx(bonferroni[0])
    assert all(h <= b + 1e-12 for h, b in zip(holm, bonferroni, strict=True))
    assert holm[3] < bonferroni[3]


def test_the_smallest_p_value_is_multiplied_by_the_number_of_tests():
    assert holm_adjust([0.01, 0.5, 0.6, 0.7])[0] == pytest.approx(0.04)


def test_adjusted_values_are_monotone_in_the_original_ranking():
    raw = [0.001, 0.049, 0.05, 0.9]
    adjusted = holm_adjust(raw)
    ordered = sorted(range(len(raw)), key=lambda i: raw[i])
    values = [adjusted[i] for i in ordered]
    assert values == sorted(values)


def test_the_step_down_stops_at_the_first_failure():
    """Holm's defining behaviour: later p-values inherit the running maximum."""
    adjusted = holm_adjust([0.02, 0.03])
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[1] == pytest.approx(0.04)


def test_adjustment_is_capped_at_one():
    assert all(value <= 1.0 for value in holm_adjust([0.5, 0.6, 0.7, 0.8]))


def test_the_order_of_the_input_is_preserved():
    adjusted = holm_adjust([0.4, 0.004])
    assert adjusted[1] < adjusted[0]


def test_a_single_test_is_unchanged():
    assert holm_adjust([0.03]) == (pytest.approx(0.03),)


def test_no_tests_adjust_to_nothing():
    assert holm_adjust([]) == ()


@pytest.mark.parametrize("bad", [[-0.1], [1.1], [0.5, 2.0]])
def test_a_p_value_outside_the_unit_interval_is_refused(bad: list[float]):
    with pytest.raises(ValueError, match=re.escape("must lie in [0, 1]")):
        holm_adjust(bad)


def test_identical_p_values_adjust_identically():
    adjusted = holm_adjust([0.02, 0.02, 0.02])
    assert adjusted[0] == adjusted[1] == adjusted[2]


def test_four_confirmatory_tests_is_the_correction_family(default_config: AppConfig):
    """The family is the confirmatory tier, not all 32 estimates."""
    plan = resolve_tiers(BUILT_COLUMNS, default_config.model)
    adjusted = holm_adjust([0.01] * plan.n_primary_tests)
    assert len(adjusted) == 4
    assert adjusted[0] == pytest.approx(0.04)
