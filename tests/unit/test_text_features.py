"""Tests for joining the manuscript's text features.

All data here is synthetic. The most important test is the one asserting that
a table without an identifier is refused: that refusal is the only thing
standing between us and a comparison where each participant's text features
belong to someone else.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vc_multimodal.modeling import text_features as tf


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def manuscript_columns() -> dict[str, list[float]]:
    """The manuscript's column names, with synthetic values."""
    names = [
        "Jaccard",
        "Cosine",
        "Bert",
        "Bert_turn_max",
        "MTLD_patient",
        "Mean_words_per_turn",
        "Agenda_social",
        "Agenda_anxiety",
        "Agenda_depression",
    ]
    return {name: [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] for i, name in enumerate(names)}


class TestIdentifierIsRequired:
    def test_a_table_with_no_identifier_is_refused(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "nlp.csv", pd.DataFrame(manuscript_columns()))
        with pytest.raises(tf.TextFeatureError, match="no identifier column"):
            tf.load(path)

    def test_the_refusal_says_row_order_is_not_an_identifier(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "nlp.csv", pd.DataFrame(manuscript_columns()))
        with pytest.raises(tf.TextFeatureError) as excinfo:
            tf.load(path)
        assert "Row order is not an identifier" in str(excinfo.value)

    def test_the_refusal_lists_the_columns_it_did_find(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "nlp.csv", pd.DataFrame(manuscript_columns()))
        with pytest.raises(tf.TextFeatureError) as excinfo:
            tf.load(path)
        assert "Jaccard" in str(excinfo.value)

    @pytest.mark.parametrize("name", ["session_id", "session", "id", "recording_id", "file_id"])
    def test_any_accepted_identifier_name_works(self, tmp_path: Path, name: str) -> None:
        frame = pd.DataFrame({name: [1, 2, 3], **manuscript_columns()})
        features = tf.load(write_csv(tmp_path / "nlp.csv", frame))
        assert features.id_column == name
        assert list(features.frame["session_id"]) == [1, 2, 3]

    def test_the_identifier_is_found_whatever_its_case(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"Session_ID": [1, 2, 3], **manuscript_columns()})
        features = tf.load(write_csv(tmp_path / "nlp.csv", frame))
        assert features.id_column == "Session_ID"

    def test_a_non_numeric_identifier_is_refused(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"session_id": ["s1", "s2", "s3"], **manuscript_columns()})
        with pytest.raises(tf.TextFeatureError, match="non-numeric"):
            tf.load(write_csv(tmp_path / "nlp.csv", frame))

    def test_a_repeated_session_is_refused(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"session_id": [1, 2, 2], **manuscript_columns()})
        with pytest.raises(tf.TextFeatureError, match="more than one row"):
            tf.load(write_csv(tmp_path / "nlp.csv", frame))

    def test_an_identifier_with_no_features_is_refused(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"session_id": [1, 2, 3]})
        with pytest.raises(tf.TextFeatureError, match="no feature columns"):
            tf.load(write_csv(tmp_path / "nlp.csv", frame))


class TestOutcomeColumnsAreRefused:
    @pytest.mark.parametrize(
        "name",
        [
            "K6",
            "k6_total",
            "SRS",
            "SRS_2",
            "srs2_total",
            "total_score",
            "y",
            "label",
            "target",
            "outcome",
            "diagnosis",
            "severity",
            "PHQ9",
            "GAD7",
            "subscale_a",
        ],
    )
    def test_a_label_like_column_stops_the_load(self, tmp_path: Path, name: str) -> None:
        frame = pd.DataFrame(
            {"session_id": [1, 2, 3], name: [1.0, 2.0, 3.0], **manuscript_columns()}
        )
        with pytest.raises(tf.TextFeatureError, match="questionnaire outcomes"):
            tf.load(write_csv(tmp_path / "nlp.csv", frame))

    def test_the_refusal_explains_the_leak(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"session_id": [1, 2, 3], "K6": [1.0, 2.0, 3.0]})
        with pytest.raises(tf.TextFeatureError) as excinfo:
            tf.load(write_csv(tmp_path / "nlp.csv", frame))
        assert "leak the label" in str(excinfo.value)

    def test_the_agenda_ratings_are_not_mistaken_for_outcomes(self) -> None:
        # Agenda_anxiety and Agenda_depression are LLM ratings of what was
        # discussed, not questionnaire values, so they must pass.
        assert tf.label_like_columns(["Agenda_anxiety", "Agenda_depression"]) == ()

    def test_the_manuscript_columns_all_pass(self) -> None:
        assert tf.label_like_columns(list(manuscript_columns())) == ()


class TestNames:
    def test_names_are_prefixed_by_family(self) -> None:
        assert tf.normalise_name("Bert_turn_max") == "text__bert_turn_max"

    def test_punctuation_becomes_underscores(self) -> None:
        assert tf.normalise_name("MTLD (patient)") == "text__mtld_patient"

    def test_the_family_prefix_lets_text_be_selected_like_any_other(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"session_id": [1, 2, 3], **manuscript_columns()})
        features = tf.load(write_csv(tmp_path / "nlp.csv", frame))
        assert all(name.startswith("text__") for name in features.feature_columns)


class TestJoin:
    def ours(self, session_ids: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {"session_id": session_ids, "turns__latency_median": [1.0] * len(session_ids)}
        )

    def theirs(self, tmp_path: Path, session_ids: list[int]) -> tf.TextFeatures:
        frame = pd.DataFrame({"session_id": session_ids, "Jaccard": [0.5] * len(session_ids)})
        return tf.load(write_csv(tmp_path / "nlp.csv", frame))

    def test_a_complete_join_reports_as_complete(self, tmp_path: Path) -> None:
        merged, report = tf.join(self.ours([1, 2, 3]), self.theirs(tmp_path, [1, 2, 3]))
        assert report.is_complete
        assert report.matched == (1, 2, 3)
        assert len(merged) == 3

    def test_the_join_is_by_identifier_not_row_order(self, tmp_path: Path) -> None:
        ours = pd.DataFrame({"session_id": [1, 2, 3], "turns__latency_median": [10.0, 20.0, 30.0]})
        # Same sessions, reverse order, distinguishable values.
        frame = pd.DataFrame({"session_id": [3, 2, 1], "Jaccard": [0.3, 0.2, 0.1]})
        text = tf.load(write_csv(tmp_path / "nlp.csv", frame))
        merged, _ = tf.join(ours, text)
        by_session = merged.set_index("session_id")["text__jaccard"]
        assert by_session.loc[1] == pytest.approx(0.1)
        assert by_session.loc[3] == pytest.approx(0.3)

    def test_a_session_with_no_text_features_keeps_its_row(self, tmp_path: Path) -> None:
        merged, report = tf.join(self.ours([1, 2, 3]), self.theirs(tmp_path, [1, 2]))
        assert report.features_only == (3,)
        assert len(merged) == 3
        assert pd.isna(merged.set_index("session_id").loc[3, "text__jaccard"])

    def test_an_unmatched_text_row_is_reported_and_dropped(self, tmp_path: Path) -> None:
        merged, report = tf.join(self.ours([1, 2]), self.theirs(tmp_path, [1, 2, 9]))
        assert report.text_only == (9,)
        assert len(merged) == 2

    def test_both_sides_can_be_incomplete_at_once(self, tmp_path: Path) -> None:
        _, report = tf.join(self.ours([1, 2]), self.theirs(tmp_path, [2, 3]))
        assert report.matched == (2,)
        assert report.features_only == (1,)
        assert report.text_only == (3,)
        assert not report.is_complete

    def test_the_report_names_sessions_and_nothing_else(self, tmp_path: Path) -> None:
        _, report = tf.join(self.ours([1, 2]), self.theirs(tmp_path, [2, 3]))
        text = "\n".join(report.report_lines())
        assert "[1]" in text
        assert "[3]" in text
        assert "0.5" not in text
