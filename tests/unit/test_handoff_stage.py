"""Tests for the handoff bundle.

This is the stage that sends data to another person, so most of these tests are
about what must *not* happen: no label, nothing derived from a transcript, no
bundle whose commit does not describe the code that built it, and no
half-written bundle left where someone might send it.

Every feature table here is synthetic.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from vc_multimodal import qc_notes
from vc_multimodal.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from vc_multimodal.paths import DataRoots
from vc_multimodal.qc_notes import QcNoteError
from vc_multimodal.stages import aggregate as aggregate_stage
from vc_multimodal.stages import handoff as handoff_stage
from vc_multimodal.stages.handoff import HandoffError

MOMENT = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clean git checkout standing in for the project repository."""
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Test")
    (path / "code.py").write_text("x = 1\n")
    git(path, "add", "code.py")
    git(path, "commit", "-q", "-m", "first")
    return path


def feature_table(
    default_config: AppConfig, session_ids: Sequence[int] = (1, 2, 3)
) -> pd.DataFrame:
    """A small table with the real column names."""
    n = len(session_ids)
    columns: dict[str, object] = {
        "session_id": list(session_ids),
        "wave": ["winter"] * n,
        "turns__latency_median": [1.0 + i for i in range(n)],
        "turns__participant_speaking_ratio": [0.4] * n,
        "turns__overlap_ratio": [0.0] * n,
        "prosody__f0_semitone_sd": [2.5 + i for i in range(n)],
        "face_speaking__au12_mean": [0.1] * n,
    }
    # Which QC columns are text comes from the stage, not a copy of the list:
    # an earlier version hardcoded it here and silently gave new text columns a
    # float value, which made a test about README prose fail for no visible
    # reason.
    for name in aggregate_stage.QC_COLUMNS:
        if name in aggregate_stage._STRING_QC:
            columns[name] = ["mediapipe" if name == "qc__face_backend" else ""] * n
        else:
            columns[name] = [10.0] * n
    return pd.DataFrame(columns)


def write_features(roots: DataRoots, frame: pd.DataFrame) -> Path:
    path = aggregate_stage.features_path(roots)
    frame.to_csv(path, index=False)
    return path


def build(
    default_config: AppConfig, roots: DataRoots, repo: Path, **kwargs: object
) -> handoff_stage.HandoffResult:
    return handoff_stage.run(
        default_config,
        roots,
        repo=repo,
        now=MOMENT,
        **kwargs,  # type: ignore[arg-type]
    )


class TestTheBundleIsComplete:
    def test_every_expected_file_is_written(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo)
        written = {path.name for path in result.path.iterdir()}
        assert handoff_stage.FEATURES_FILE in written
        assert handoff_stage.QC_FILE in written
        assert handoff_stage.DICTIONARY_FILE in written
        assert handoff_stage.MANIFEST_FILE in written
        assert handoff_stage.README_FILE in written
        assert handoff_stage.CONFIG_FILE in written

    def test_the_directory_is_named_for_the_date_and_commit(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo)
        assert result.path.name.startswith("20260925_")
        assert result.git is not None
        assert result.path.name.endswith(result.git.short)

    def test_the_counts_are_reported(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config, [1, 2, 3, 4]))
        result = build(default_config, roots, repo)
        assert result.n_sessions == 4
        assert result.n_features == 5

    def test_no_partial_directory_is_left_behind(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        build(default_config, roots, repo)
        assert list(handoff_stage.bundle_root(roots).glob("*.partial")) == []

    def test_a_failed_build_leaves_nothing_to_send(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        # An undescribed column fails after the target name is known but before
        # anything is written: there must be no bundle, partial or otherwise.
        frame = feature_table(default_config)
        frame["turns__undocumented"] = 1.0
        write_features(roots, frame)
        with pytest.raises(HandoffError):
            build(default_config, roots, repo)
        assert not handoff_stage.bundle_root(roots).exists() or (
            list(handoff_stage.bundle_root(roots).iterdir()) == []
        )


class TestSeparationOfFeaturesAndQuality:
    def test_qc_columns_are_not_in_the_feature_file(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo)
        features = pd.read_csv(result.path / handoff_stage.FEATURES_FILE)
        assert [c for c in features.columns if c.startswith("qc__")] == []

    def test_the_qc_file_keeps_the_identifier(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo)
        qc = pd.read_csv(result.path / handoff_stage.QC_FILE)
        assert "session_id" in qc.columns
        assert any(c.startswith("qc__") for c in qc.columns)

    def test_both_files_cover_the_same_sessions(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config, [5, 9, 12]))
        result = build(default_config, roots, repo)
        features = pd.read_csv(result.path / handoff_stage.FEATURES_FILE)
        qc = pd.read_csv(result.path / handoff_stage.QC_FILE)
        assert list(features["session_id"]) == list(qc["session_id"]) == [5, 9, 12]


class TestNothingUnsafeLeaves:
    def test_a_label_like_column_stops_the_build(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        frame = feature_table(default_config)
        frame["k6_total"] = [10, 11, 12]
        write_features(roots, frame)
        with pytest.raises(HandoffError, match="questionnaire outcomes"):
            build(default_config, roots, repo)

    def test_the_label_refusal_explains_that_the_halves_are_mixed(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        frame = feature_table(default_config)
        frame["srs2_score"] = [1, 2, 3]
        write_features(roots, frame)
        with pytest.raises(HandoffError) as excinfo:
            build(default_config, roots, repo)
        assert "never holds a label" in str(excinfo.value)

    @pytest.mark.parametrize(
        "column", ["transcript", "utterance_count_text", "speaker_name", "ocr_text", "caption_line"]
    )
    def test_a_column_that_may_carry_speech_or_names_stops_the_build(
        self, default_config: AppConfig, roots: DataRoots, repo: Path, column: str
    ) -> None:
        frame = feature_table(default_config)
        frame[column] = ["something"] * 3
        write_features(roots, frame)
        with pytest.raises(HandoffError, match="what someone said"):
            build(default_config, roots, repo)

    def test_nothing_is_written_when_a_column_is_refused(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        frame = feature_table(default_config)
        frame["k6"] = [1, 2, 3]
        write_features(roots, frame)
        with pytest.raises(HandoffError):
            build(default_config, roots, repo)
        assert not handoff_stage.bundle_root(roots).exists() or (
            list(handoff_stage.bundle_root(roots).glob("2026*")) == []
        )


class TestTheCommitMustDescribeTheCode:
    def test_a_dirty_tree_is_refused(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        (repo / "code.py").write_text("x = 2\n")
        write_features(roots, feature_table(default_config))
        with pytest.raises(HandoffError, match="uncommitted change"):
            build(default_config, roots, repo)

    def test_the_refusal_names_the_changed_paths(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        (repo / "code.py").write_text("x = 2\n")
        write_features(roots, feature_table(default_config))
        with pytest.raises(HandoffError) as excinfo:
            build(default_config, roots, repo)
        assert "code.py" in str(excinfo.value)
        assert "--allow-dirty" in str(excinfo.value)

    def test_allow_dirty_builds_and_records_it(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        (repo / "code.py").write_text("x = 2\n")
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo, allow_dirty=True)
        assert any("modified working tree" in note for note in result.notes)
        manifest = json.loads((result.path / handoff_stage.MANIFEST_FILE).read_text())
        assert manifest["git"]["dirty"] is True
        assert "MODIFIED working tree" in (result.path / handoff_stage.README_FILE).read_text()

    def test_a_non_repository_is_noted_not_refused(
        self, default_config: AppConfig, roots: DataRoots, tmp_path: Path
    ) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        write_features(roots, feature_table(default_config))
        result = handoff_stage.run(default_config, roots, repo=plain, now=MOMENT)
        assert result.git is None
        assert any("not a git checkout" in note for note in result.notes)
        assert result.path.name.endswith("nogit")


class TestRebuilding:
    def test_an_existing_bundle_is_not_silently_replaced(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        build(default_config, roots, repo)
        with pytest.raises(HandoffError, match="already exists"):
            build(default_config, roots, repo)

    def test_force_replaces_it(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        first = build(default_config, roots, repo)
        (first.path / "stray.txt").write_text("left over")
        second = build(default_config, roots, repo, force=True)
        assert second.path == first.path
        assert not (second.path / "stray.txt").exists()


class TestPreconditions:
    def test_a_missing_feature_table_says_which_command_to_run(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        with pytest.raises(HandoffError, match="vc aggregate"):
            build(default_config, roots, repo)

    def test_an_empty_feature_table_is_refused(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config).iloc[0:0])
        with pytest.raises(HandoffError, match="no rows"):
            build(default_config, roots, repo)

    def test_a_table_with_no_features_is_refused(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        frame = pd.DataFrame({"session_id": [1, 2], "wave": ["winter", "winter"]})
        write_features(roots, frame)
        with pytest.raises(HandoffError, match="no feature columns"):
            build(default_config, roots, repo)


class TestTheManifest:
    def manifest(self, result: handoff_stage.HandoffResult) -> dict[str, object]:
        return json.loads((result.path / handoff_stage.MANIFEST_FILE).read_text())

    def test_it_records_the_commit(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo)
        assert result.git is not None
        assert self.manifest(result)["git"]["commit"] == result.git.commit  # type: ignore[index]

    def test_it_records_the_tool_versions(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        environment = self.manifest(build(default_config, roots, repo))["environment"]
        assert "mediapipe" in environment["packages"]  # type: ignore[index]
        assert environment["python"]  # type: ignore[index]

    def test_it_records_the_seed_and_the_whole_config(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        manifest = self.manifest(build(default_config, roots, repo))
        assert manifest["seeds"]["seed"] == default_config.runtime.seed  # type: ignore[index]
        assert "face" in manifest["config"]  # type: ignore[operator]

    def test_it_records_the_sessions_and_features(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config, [7, 8]))
        manifest = self.manifest(build(default_config, roots, repo))
        assert manifest["sessions"]["session_ids"] == [7, 8]  # type: ignore[index]
        assert "turns__latency_median" in manifest["features"]["names"]  # type: ignore[index]

    def test_it_holds_no_feature_values(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        # The manifest is the file most likely to be pasted into an email.
        frame = feature_table(default_config)
        frame["prosody__f0_semitone_sd"] = [123.456789, 2.0, 3.0]
        write_features(roots, frame)
        text = (build(default_config, roots, repo).path / handoff_stage.MANIFEST_FILE).read_text()
        assert "123.456789" not in text


class TestTheReadme:
    def readme(self, result: handoff_stage.HandoffResult) -> str:
        return (result.path / handoff_stage.README_FILE).read_text()

    def test_it_states_the_command_to_run(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        assert "vc model" in self.readme(build(default_config, roots, repo))

    def test_it_says_the_bundle_holds_no_labels(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        assert "questionnaire scores" in self.readme(build(default_config, roots, repo))

    def test_it_explains_the_constant_overlap_features(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        text = self.readme(build(default_config, roots, repo))
        assert "turns__overlap_ratio" in text
        assert "0.0000 seconds" in text
        assert "never interrupted" in text

    def test_it_carries_the_backend_and_gaze_notes(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        text = self.readme(build(default_config, roots, repo))
        assert "re-extracting every session" in text
        assert "no gaze features" in text.lower()

    def test_it_warns_that_qc_columns_are_not_predictors(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        assert "not predictors" in self.readme(build(default_config, roots, repo)).lower()

    def test_it_states_the_sample_size_limits(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        assert "Sixty-two participants" in self.readme(build(default_config, roots, repo))

    def test_it_lists_only_the_files_actually_written(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        # No text feature table here, so the README must not promise one.
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo)
        text = self.readme(result)
        assert handoff_stage.TEXT_FILE not in result.files
        assert f"| `{handoff_stage.TEXT_FILE}` |" not in text


class TestTheTextBaseline:
    def place_text_features(self, roots: DataRoots, session_ids: Sequence[int]) -> None:
        directory = roots.work / "diarization" / "diarizations_original"
        directory.mkdir(parents=True, exist_ok=True)
        for session_id in session_ids:
            (directory / f"{session_id}.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n")
        frame = pd.DataFrame({"Jaccard": [0.1 * (i + 1) for i in range(len(session_ids))]})
        frame.to_csv(roots.work / "nlp_features.csv", index=False)

    def config_for(self, n: int) -> AppConfig:
        """The real config with the confirmed row count moved, for a small table."""
        return load_config(
            DEFAULT_CONFIG_PATH,
            overrides={"model.text_features.positional.expected_rows": n},
        )

    def test_the_text_table_is_shipped_with_explicit_session_ids(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config, [1, 2, 3]))
        self.place_text_features(roots, [1, 2, 3])
        config = self.config_for(3)
        result = handoff_stage.run(config, roots, repo=repo, now=MOMENT)
        shipped = pd.read_csv(result.path / handoff_stage.TEXT_FILE)
        assert list(shipped["session_id"]) == [1, 2, 3]
        assert "text__jaccard" in shipped.columns

    def test_the_readme_quotes_the_ordering_provenance(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config, [1, 2, 3]))
        self.place_text_features(roots, [1, 2, 3])
        config = self.config_for(3)
        result = handoff_stage.run(config, roots, repo=repo, now=MOMENT)
        text = (result.path / handoff_stage.README_FILE).read_text()
        assert "int(p.stem)" in text
        assert "0014" in text

    def test_the_manifest_carries_the_ordering_rule(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config, [1, 2, 3]))
        self.place_text_features(roots, [1, 2, 3])
        config = self.config_for(3)
        result = handoff_stage.run(config, roots, repo=repo, now=MOMENT)
        manifest = json.loads((result.path / handoff_stage.MANIFEST_FILE).read_text())
        ordering = manifest["text_features"]["ordering"]
        assert ordering["rule"] == "numeric_ascending_session_id"
        assert ordering["session_ids"] == [1, 2, 3]

    def test_a_missing_text_table_is_noted_not_fatal(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo)
        assert result.text is None
        assert any("text features not included" in note for note in result.notes)
        text = (result.path / handoff_stage.README_FILE).read_text()
        assert "text baseline is not in this bundle" in text.lower()

    def test_the_summary_says_the_comparisons_cannot_run(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        lines = handoff_stage.summarise(build(default_config, roots, repo))
        assert any("ABSENT" in line for line in lines)


class TestTheSummary:
    def test_it_reports_counts_and_never_a_value(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        frame = feature_table(default_config)
        frame["prosody__f0_semitone_sd"] = [987.654321, 1.0, 2.0]
        write_features(roots, frame)
        lines = handoff_stage.summarise(build(default_config, roots, repo))
        joined = "\n".join(lines)
        assert "sessions: 3" in joined
        assert "987.654321" not in joined

    def test_it_names_the_commit_and_tree_state(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo)
        assert result.git is not None
        joined = "\n".join(handoff_stage.summarise(result))
        assert result.git.short in joined
        assert "clean tree" in joined


# ---------------------------------------------------------------------------
# human-confirmed QC notes reaching the bundle
# ---------------------------------------------------------------------------
BLUR_NOTE = (
    "participant's camera is too out of focus for face tracking; confirmed by "
    "watching the recording"
)


def record_note(roots: DataRoots, session_id: int = 43, modality: str = "face") -> None:
    qc_notes.record(
        roots.work / "qc_notes.csv",
        session_id=session_id,
        modality=modality,
        status="unavailable",
        reason=BLUR_NOTE,
        recorded_by="tester",
        now=MOMENT,
    )


class TestConfirmedNotesTravelWithTheBundle:
    def test_the_notes_file_is_shipped(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        record_note(roots)
        result = build(default_config, roots, repo)
        assert handoff_stage.QC_NOTES_FILE in result.files
        shipped = pd.read_csv(result.path / handoff_stage.QC_NOTES_FILE)
        assert list(shipped["session_id"]) == [43]
        assert BLUR_NOTE in str(shipped.iloc[0]["reason"])

    def test_the_readme_shows_the_session_and_the_reason(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        record_note(roots)
        text = (build(default_config, roots, repo).path / handoff_stage.README_FILE).read_text()
        assert "marked unusable" in text
        assert "| 43 | face | unavailable |" in text
        assert BLUR_NOTE in text

    def test_the_readme_says_the_blanks_are_deliberate(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        # Otherwise the analyst reads the absence as a bug, or imputes it.
        write_features(roots, feature_table(default_config))
        record_note(roots)
        text = (build(default_config, roots, repo).path / handoff_stage.README_FILE).read_text()
        assert "on purpose" in text
        assert "not missing through a bug" in text
        assert "should not be imputed" in text

    def test_the_readme_says_other_modalities_are_still_good(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        record_note(roots)
        text = (build(default_config, roots, repo).path / handoff_stage.README_FILE).read_text()
        assert "unaffected and should be used normally" in text

    def test_the_manifest_records_the_notes_in_full(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        record_note(roots)
        result = build(default_config, roots, repo)
        manifest = json.loads((result.path / handoff_stage.MANIFEST_FILE).read_text())
        assert manifest["qc_notes"][0]["session_id"] == 43
        assert manifest["qc_notes"][0]["recorded_by"] == "tester"

    def test_a_bundle_with_no_notes_says_nothing_about_them(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        result = build(default_config, roots, repo)
        assert handoff_stage.QC_NOTES_FILE not in result.files
        text = (result.path / handoff_stage.README_FILE).read_text()
        assert "marked unusable" not in text

    def test_a_malformed_notes_file_stops_the_bundle(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        # A finding that silently fails to travel is the thing this feature
        # exists to prevent.
        write_features(roots, feature_table(default_config))
        (roots.work / "qc_notes.csv").write_text(
            "session_id,modality,status,reason\n43,eyebrows,unavailable,too blurry to track\n"
        )
        with pytest.raises(QcNoteError, match="modality"):
            build(default_config, roots, repo)


# ---------------------------------------------------------------------------
# face tracking quality in the README
# ---------------------------------------------------------------------------
def features_with_dropped(default_config: AppConfig, dropped: list[float | None]) -> pd.DataFrame:
    """A feature table carrying a dropped-frame fraction per session."""
    frame = feature_table(default_config, list(range(1, len(dropped) + 1)))
    frame["qc__face_dropped_fraction"] = dropped
    return frame


class TestTheTrackingQualitySection:
    def readme(self, default_config: AppConfig, roots: DataRoots, repo: Path) -> str:
        return (build(default_config, roots, repo).path / handoff_stage.README_FILE).read_text()

    def test_the_distribution_is_reported(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, features_with_dropped(default_config, [0.001, 0.002, 0.324]))
        text = self.readme(default_config, roots, repo)
        assert "Face tracking quality, session by session" in text
        assert "median" in text
        assert "| worst | 32.4% |" in text

    def test_an_unusual_session_is_named_with_its_value(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        # So the professor can see where each session sits rather than being
        # given a summary and asked to trust it.
        write_features(roots, features_with_dropped(default_config, [0.001, 0.002, 0.992]))
        text = self.readme(default_config, roots, repo)
        assert "| 3 | 99.2% |" in text

    def test_an_unusual_session_shows_its_confirmed_cause(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        frame = features_with_dropped(default_config, [0.001, 0.002, 0.992])
        frame.loc[frame["session_id"] == 3, "qc__annotation_reason"] = BLUR_NOTE
        write_features(roots, frame)
        text = self.readme(default_config, roots, repo)
        assert BLUR_NOTE in text

    def test_an_unchecked_outlier_says_so_rather_than_looking_explained(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, features_with_dropped(default_config, [0.001, 0.002, 0.992]))
        text = self.readme(default_config, roots, repo)
        assert "not checked" in text

    def test_a_clean_cohort_says_there_are_no_outliers(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, features_with_dropped(default_config, [0.001, 0.002, 0.003]))
        assert "No session is above 5%" in self.readme(default_config, roots, repo)

    def test_sessions_with_no_facial_measurement_are_counted_separately(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        # Absent is not zero dropped, and must not be averaged in as if it were.
        write_features(roots, features_with_dropped(default_config, [0.001, 0.002, None]))
        text = self.readme(default_config, roots, repo)
        assert "Measured for 2 session(s)" in text
        assert "1 session(s) have no facial measurements" in text

    def test_no_values_at_all_is_stated_not_faked(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, features_with_dropped(default_config, [None, None, None]))
        text = self.readme(default_config, roots, repo)
        assert "No session in this bundle has a facial dropped-frame fraction" in text

    def test_the_reader_is_told_where_to_look_up_a_session(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, features_with_dropped(default_config, [0.001, 0.002, 0.324]))
        assert handoff_stage.QC_FILE in self.readme(default_config, roots, repo)


class TestTheRemoteRecordingLimitation:
    def test_the_readme_carries_it(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        text = (build(default_config, roots, repo).path / handoff_stage.README_FILE).read_text()
        assert "varies with the participant's own setup" in text

    def test_it_names_the_lab_study_it_is_not_comparable_with(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        text = (build(default_config, roots, repo).path / handoff_stage.README_FILE).read_text()
        assert "Miyamoto" in text
        assert "controlled lighting" in text

    def test_it_says_the_quality_measure_is_not_behavioural(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        text = (build(default_config, roots, repo).path / handoff_stage.README_FILE).read_text()
        assert "not a behavioural measure" in text

    def test_it_says_a_threshold_must_be_chosen_once(
        self, default_config: AppConfig, roots: DataRoots, repo: Path
    ) -> None:
        write_features(roots, feature_table(default_config))
        text = (build(default_config, roots, repo).path / handoff_stage.README_FILE).read_text()
        assert "decide the threshold once" in text
