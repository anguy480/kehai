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
from pydantic import ValidationError

from vc_multimodal.config import TextFeaturesConfig, load_config
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


# ---------------------------------------------------------------------------
# Matching rows to sessions by position, under a stated ordering rule
# ---------------------------------------------------------------------------
LAB_PROVENANCE = (
    "Rows were written by iterating transcript files with "
    "sorted(dir_path.glob(extension), key=lambda p: int(p.stem)), so the order is "
    "numeric ascending by session ID. Confirmed by the lab on 2026-09-24."
)


def positional_config(
    *,
    expected_rows: int = 3,
    source_glob: str = "diarization/*.srt",
    provenance: str = LAB_PROVENANCE,
    path: str = "nlp.csv",
) -> TextFeaturesConfig:
    return TextFeaturesConfig.model_validate(
        {
            "path": path,
            "identification": "positional",
            "positional": {
                "rule": "numeric_ascending_session_id",
                "provenance": provenance,
                "source_glob": source_glob,
                "expected_rows": expected_rows,
            },
        }
    )


def make_transcripts(work: Path, session_ids: list[int], suffix: str = ".srt") -> None:
    """Stand-ins for the files the lab iterated. Content is irrelevant here."""
    directory = work / "diarization"
    directory.mkdir(parents=True, exist_ok=True)
    for session_id in session_ids:
        (directory / f"{session_id}{suffix}").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n")


class TestResolvingTheOrderingRule:
    def test_the_order_is_numeric_ascending_not_lexicographic(self, tmp_path: Path) -> None:
        # The distinction that matters for this cohort: 62 precedes 102, and a
        # lexicographic sort would put 102 first.
        make_transcripts(tmp_path, [62, 102, 7])
        plan = tf.resolve_positional(tmp_path, positional_config())
        assert plan.session_ids == (7, 62, 102)

    def test_the_provenance_is_carried_on_the_plan(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        plan = tf.resolve_positional(tmp_path, positional_config())
        assert "int(p.stem)" in plan.provenance
        assert "2026-09-24" in plan.provenance

    def test_a_row_count_change_stops_the_join(self, tmp_path: Path) -> None:
        # The guard the lab's confirmation rests on: this is no longer the set
        # the ordering was confirmed against.
        make_transcripts(tmp_path, [1, 2, 3, 4])
        with pytest.raises(tf.TextFeatureError, match="confirmed against 3 file"):
            tf.resolve_positional(tmp_path, positional_config(expected_rows=3))

    def test_the_count_error_names_both_numbers(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [1, 2])
        with pytest.raises(tf.TextFeatureError) as excinfo:
            tf.resolve_positional(tmp_path, positional_config(expected_rows=3))
        message = str(excinfo.value)
        assert "confirmed against 3" in message
        assert "matches 2" in message

    def test_no_matching_files_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "diarization").mkdir()
        with pytest.raises(tf.TextFeatureError, match="no files matched"):
            tf.resolve_positional(tmp_path, positional_config())

    def test_a_filename_that_is_not_a_session_id_is_refused(self, tmp_path: Path) -> None:
        # The lab's int(path.stem) would have raised on this file, so its
        # presence means we are looking at a different set than they were.
        make_transcripts(tmp_path, [1, 2])
        (tmp_path / "diarization" / "notes.srt").write_text("x")
        with pytest.raises(tf.TextFeatureError, match="not a session ID"):
            tf.resolve_positional(tmp_path, positional_config())

    def test_only_the_configured_extension_is_counted(self, tmp_path: Path) -> None:
        # The directory also holds .txt files, which carry no timestamps.
        make_transcripts(tmp_path, [1, 2, 3])
        make_transcripts(tmp_path, [1, 2, 3], suffix=".txt")
        plan = tf.resolve_positional(tmp_path, positional_config())
        assert plan.session_ids == (1, 2, 3)


class TestLoadingByPosition:
    def table(self, path: Path, n_rows: int) -> Path:
        frame = pd.DataFrame(
            {
                "Jaccard": [0.1 * (i + 1) for i in range(n_rows)],
                "Bert": [0.5] * n_rows,
            }
        )
        return write_csv(path, frame)

    def test_rows_are_attached_in_the_rule_order(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [62, 102, 7])
        path = self.table(tmp_path / "nlp.csv", 3)
        loaded = tf.load(path, positional=tf.resolve_positional(tmp_path, positional_config()))
        assert list(loaded.frame["session_id"]) == [7, 62, 102]
        # First row belongs to the lowest session ID, not the first file listed.
        assert loaded.frame.iloc[0]["text__jaccard"] == pytest.approx(0.1)
        assert loaded.frame.loc[loaded.frame.session_id == 102, "text__jaccard"].iloc[
            0
        ] == pytest.approx(0.3)

    def test_the_identification_mode_is_recorded(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        loaded = tf.load(
            self.table(tmp_path / "nlp.csv", 3),
            positional=tf.resolve_positional(tmp_path, positional_config()),
        )
        assert loaded.identification == "positional"
        assert loaded.id_column is None

    def test_a_table_with_the_wrong_row_count_is_refused(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        plan = tf.resolve_positional(tmp_path, positional_config())
        path = self.table(tmp_path / "nlp.csv", 4)
        with pytest.raises(tf.TextFeatureError, match="has 4 row"):
            tf.load(path, positional=plan)

    def test_the_row_count_error_names_the_confirmed_count(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        plan = tf.resolve_positional(tmp_path, positional_config())
        with pytest.raises(tf.TextFeatureError) as excinfo:
            tf.load(self.table(tmp_path / "nlp.csv", 2), positional=plan)
        assert "confirmed against 3" in str(excinfo.value)

    def test_an_identifier_wins_over_the_ordering_rule(self, tmp_path: Path) -> None:
        # An identifier needs no external promise, so it is always preferred.
        make_transcripts(tmp_path, [1, 2, 3])
        frame = pd.DataFrame({"session_id": [102, 7, 62], "Jaccard": [0.1, 0.2, 0.3]})
        loaded = tf.load(
            write_csv(tmp_path / "nlp.csv", frame),
            positional=tf.resolve_positional(tmp_path, positional_config()),
        )
        assert loaded.identification == "identifier"
        assert loaded.plan is None
        by_session = loaded.frame.set_index("session_id")["text__jaccard"]
        assert by_session.loc[102] == pytest.approx(0.1)

    def test_an_outcome_column_is_still_refused_in_positional_mode(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        frame = pd.DataFrame({"Jaccard": [0.1, 0.2, 0.3], "K6_total": [1, 2, 3]})
        with pytest.raises(tf.TextFeatureError, match="questionnaire outcomes"):
            tf.load(
                write_csv(tmp_path / "nlp.csv", frame),
                positional=tf.resolve_positional(tmp_path, positional_config()),
            )

    def test_the_refusal_path_survives_for_a_table_with_no_rule(self, tmp_path: Path) -> None:
        with pytest.raises(tf.TextFeatureError, match="no identifier column"):
            tf.load(self.table(tmp_path / "nlp.csv", 3))

    def test_load_configured_resolves_the_rule_itself(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [62, 102, 7])
        self.table(tmp_path / "nlp.csv", 3)
        loaded = tf.load_configured(tmp_path, positional_config())
        assert list(loaded.frame["session_id"]) == [7, 62, 102]

    def test_load_configured_says_so_when_the_table_is_missing(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        with pytest.raises(tf.TextFeatureError, match=r"not at nlp\.csv"):
            tf.load_configured(tmp_path, positional_config())


class TestManifestRecord:
    def test_the_ordering_rule_and_its_provenance_reach_the_manifest(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [62, 102, 7])
        write_csv(tmp_path / "nlp.csv", pd.DataFrame({"Jaccard": [0.1, 0.2, 0.3]}))
        record = tf.load_configured(tmp_path, positional_config()).manifest_record()
        assert record["identification"] == "positional"
        assert record["ordering"]["rule"] == "numeric_ascending_session_id"
        assert "int(p.stem)" in record["ordering"]["provenance"]
        assert record["ordering"]["session_ids"] == [7, 62, 102]
        assert record["ordering"]["expected_rows"] == 3

    def test_the_record_names_the_exact_file_used(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        write_csv(tmp_path / "nlp.csv", pd.DataFrame({"Jaccard": [0.1, 0.2, 0.3]}))
        record = tf.load_configured(tmp_path, positional_config()).manifest_record()
        assert record["source"] == "nlp.csv"
        assert len(record["sha256"]) == 64

    def test_the_digest_changes_when_the_file_does(self, tmp_path: Path) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        write_csv(tmp_path / "nlp.csv", pd.DataFrame({"Jaccard": [0.1, 0.2, 0.3]}))
        first = tf.load_configured(tmp_path, positional_config()).sha256
        write_csv(tmp_path / "nlp.csv", pd.DataFrame({"Jaccard": [0.9, 0.2, 0.3]}))
        assert tf.load_configured(tmp_path, positional_config()).sha256 != first

    def test_an_identifier_join_records_the_column_and_no_ordering(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"session_id": [1, 2], "Jaccard": [0.1, 0.2]})
        record = tf.load(write_csv(tmp_path / "nlp.csv", frame)).manifest_record()
        assert record["identification"] == "identifier"
        assert record["id_column"] == "session_id"
        assert "ordering" not in record

    def test_the_record_names_features_and_never_values(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"session_id": [1, 2], "Jaccard": [0.123456, 0.2]})
        record = tf.load(write_csv(tmp_path / "nlp.csv", frame)).manifest_record()
        assert record["features"] == ["text__jaccard"]
        assert "0.123456" not in str(record)


class TestProvenanceIsRequired:
    def test_the_positional_block_must_be_stated_either_way(self) -> None:
        # Every setting in this project is explicit, so an omitted block is a
        # missing field rather than a silent default.
        with pytest.raises(ValidationError, match="positional"):
            TextFeaturesConfig.model_validate({"path": "nlp.csv", "identification": "positional"})

    def test_positional_mode_with_an_empty_block_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="requires a"):
            TextFeaturesConfig.model_validate(
                {"path": "nlp.csv", "identification": "positional", "positional": None}
            )

    def test_identifier_mode_needs_no_ordering_rule(self) -> None:
        config = TextFeaturesConfig.model_validate(
            {"path": "nlp.csv", "identification": "identifier", "positional": None}
        )
        assert config.positional is None

    def test_a_bare_assertion_is_not_provenance(self) -> None:
        with pytest.raises(ValidationError, match="where the ordering rule came from"):
            positional_config(provenance="sorted by id")

    def test_the_default_config_carries_the_labs_wording(self, tmp_path: Path) -> None:
        config = load_config(Path("config/default.yaml"))
        text = config.model.text_features
        assert text is not None
        assert text.identification == "positional"
        assert text.positional is not None
        assert text.positional.expected_rows == 62
        assert "int(p.stem)" in text.positional.provenance
        assert "not lexicographic" in text.positional.provenance


class TestPositionalJoinWarnsOnDivergentCohorts:
    def test_a_session_we_have_and_they_did_not_is_reported(
        self, tmp_path: Path, package_logs: pytest.LogCaptureFixture
    ) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        write_csv(tmp_path / "nlp.csv", pd.DataFrame({"Jaccard": [0.1, 0.2, 0.3]}))
        text = tf.load_configured(tmp_path, positional_config())
        ours = pd.DataFrame({"session_id": [1, 2, 3, 9], "turns__latency_median": [1.0] * 4})
        _, report = tf.join(ours, text)
        assert report.features_only == (9,)
        assert "saw different data" in package_logs.text

    def test_matching_cohorts_produce_no_warning(
        self, tmp_path: Path, package_logs: pytest.LogCaptureFixture
    ) -> None:
        make_transcripts(tmp_path, [1, 2, 3])
        write_csv(tmp_path / "nlp.csv", pd.DataFrame({"Jaccard": [0.1, 0.2, 0.3]}))
        text = tf.load_configured(tmp_path, positional_config())
        ours = pd.DataFrame({"session_id": [1, 2, 3], "turns__latency_median": [1.0] * 3})
        tf.join(ours, text)
        assert "saw different data" not in package_logs.text
