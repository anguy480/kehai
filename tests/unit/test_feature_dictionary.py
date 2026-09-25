"""Tests for the feature dictionary.

The load-bearing property: a column that ships must have a description. These
tests hold that line for the real configured feature set, so adding a feature
without documenting it fails here rather than reaching the analyst.
"""

from __future__ import annotations

import pytest

from vc_multimodal.config import AppConfig
from vc_multimodal.feature_dictionary import (
    DictionaryError,
    build,
    descriptions,
    entries,
    family_of,
)
from vc_multimodal.features.prosody_math import FEATURE_NAMES as PROSODY_NAMES
from vc_multimodal.features.turn_math import FEATURE_NAMES as TURN_NAMES
from vc_multimodal.stages.aggregate import QC_COLUMNS, all_feature_names


def every_column(config: AppConfig) -> list[str]:
    """The full table as `vc aggregate` builds it."""
    upstream = [*TURN_NAMES, *PROSODY_NAMES]
    return [
        "session_id",
        "wave",
        *all_feature_names(config, upstream=upstream),
        *QC_COLUMNS,
    ]


class TestEveryShippedColumnIsDescribed:
    def test_the_real_feature_set_is_fully_described(self, default_config: AppConfig) -> None:
        frame = build(default_config, every_column(default_config))
        assert len(frame) == len(every_column(default_config))
        assert not frame["description"].str.strip().eq("").any()

    def test_every_turn_feature_is_described(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        assert [name for name in TURN_NAMES if name not in known] == []

    def test_every_prosody_feature_is_described(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        assert [name for name in PROSODY_NAMES if name not in known] == []

    def test_every_qc_column_is_described(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        assert [name for name in QC_COLUMNS if name not in known] == []

    def test_an_undescribed_column_stops_the_dictionary(self, default_config: AppConfig) -> None:
        with pytest.raises(DictionaryError, match="no description"):
            build(default_config, ["session_id", "turns__something_new"])

    def test_the_error_names_the_columns_and_where_to_fix_them(
        self, default_config: AppConfig
    ) -> None:
        with pytest.raises(DictionaryError) as excinfo:
            build(default_config, ["prosody__made_up"])
        message = str(excinfo.value)
        assert "prosody__made_up" in message
        assert "feature_dictionary.py" in message

    def test_text_features_are_described_without_being_enumerated(
        self, default_config: AppConfig
    ) -> None:
        # They come from the manuscript, so we describe their provenance rather
        # than redefining measures we did not compute.
        frame = build(default_config, ["session_id", "text__bert_turn_max"])
        row = frame.set_index("name").loc["text__bert_turn_max"]
        assert "manuscript" in str(row["description"]).lower()
        assert row["family"] == "text"


class TestTiers:
    def test_confirmatory_features_are_marked(self, default_config: AppConfig) -> None:
        frame = build(default_config, every_column(default_config))
        marked = set(frame.loc[frame["tier"] == "confirmatory", "name"])
        assert marked == set(default_config.model.tiers.primary_columns)

    def test_there_are_twelve_confirmatory_features(self, default_config: AppConfig) -> None:
        frame = build(default_config, every_column(default_config))
        assert int((frame["tier"] == "confirmatory").sum()) == 12

    def test_qc_columns_are_not_a_tier(self, default_config: AppConfig) -> None:
        frame = build(default_config, every_column(default_config))
        qc = frame[frame["family"] == "qc"]
        assert set(qc["tier"]) == {"not a feature"}

    def test_identifiers_are_not_a_tier(self, default_config: AppConfig) -> None:
        frame = build(default_config, ["session_id", "wave"])
        assert set(frame["tier"]) == {"not a feature"}


class TestDescriptionsStateTheAwkwardParts:
    def test_the_constant_turn_features_say_so(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        for name in ("turns__overlap_ratio", "turns__interruption_rate"):
            assert "CONSTANT AT ZERO" in known[name][1]

    def test_the_two_pause_rates_state_their_different_denominators(
        self, default_config: AppConfig
    ) -> None:
        known = descriptions(default_config)
        assert "per minute of recording" in known["turns__n_per_minute"][1].lower()
        assert "participant's speech" in known["turns__pause_within_rate"][1].lower()

    def test_absolute_intensity_is_marked_as_not_comparable(
        self, default_config: AppConfig
    ) -> None:
        known = descriptions(default_config)
        assert "NOT COMPARABLE" in known["prosody__intensity_mean_db"][1]

    def test_the_speech_rate_is_marked_a_proxy(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        assert "PROXY" in known["prosody__speech_rate_proxy"][1]

    def test_head_pose_is_not_offered_as_gaze(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        pose = [name for name in known if "head_" in name]
        assert pose
        for name in pose:
            assert "NOT gaze" in known[name][1]

    def test_the_face_backend_column_warns_against_mixing(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        assert "NOT comparable" in known["qc__face_backend"][1]

    def test_peak_statistics_explain_why_they_are_not_means(
        self, default_config: AppConfig
    ) -> None:
        known = descriptions(default_config)
        peaks = [name for name in known if name.endswith("_p90")]
        assert peaks
        for name in peaks:
            assert "mostly zero" in known[name][1]


class TestFaceColumnsFollowTheGeneratedNames:
    def test_both_windows_are_described(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        speaking = [n for n in known if n.startswith("face_speaking__")]
        listening = [n for n in known if n.startswith("face_listening__")]
        assert len(speaking) == len(listening)
        assert speaking

    def test_the_window_is_stated_in_words(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        name = next(n for n in known if n.startswith("face_speaking__au12"))
        assert "while the participant was speaking" in known[name][1]
        other = name.replace("face_speaking", "face_listening")
        assert "psychiatrist was speaking" in known[other][1]

    def test_the_action_unit_description_comes_from_config(self, default_config: AppConfig) -> None:
        known = descriptions(default_config)
        configured = default_config.face.unit("au12")
        assert configured is not None
        name = next(n for n in known if n.startswith("face_speaking__au12"))
        assert configured.description in known[name][1]

    def test_a_reconfigured_unit_set_changes_the_dictionary(
        self, default_config: AppConfig
    ) -> None:
        described = descriptions(default_config)
        assert any("au01" in name for name in described)
        assert all("au99" not in name for name in described)


class TestFamilies:
    @pytest.mark.parametrize(
        ("name", "family"),
        [
            ("turns__latency_median", "turns"),
            ("prosody__f0_semitone_sd", "prosody"),
            ("face_speaking__au12_mean", "face_speaking"),
            ("face_listening__au06_p90", "face_listening"),
            ("text__jaccard", "text"),
            ("qc__flags", "qc"),
            ("session_id", "identifier"),
        ],
    )
    def test_the_family_is_the_name_prefix(self, name: str, family: str) -> None:
        assert family_of(name) == family

    def test_entries_keep_table_order(self, default_config: AppConfig) -> None:
        columns = ["session_id", "prosody__hnr_db", "turns__latency_mean", "qc__flags"]
        assert [entry.name for entry in entries(default_config, columns)] == columns
