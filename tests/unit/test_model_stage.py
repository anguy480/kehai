"""Tests for the model stage.

Synthetic features and synthetic labels. The properties under test are mostly
about the boundary this stage sits on: that labels are matched by session and
never written out, that the confirmatory tier is what the configuration says
and not what the data suggests, and that everything it could not do is named.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vc_multimodal.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from vc_multimodal.paths import DataRoots
from vc_multimodal.stages import model as stage

RNG = np.random.default_rng(20260926)


def feature_table(
    session_ids: list[int], *, families: tuple[str, ...] = ("turns", "prosody")
) -> pd.DataFrame:
    """A feature table with real column names and a learnable signal."""
    frame = pd.DataFrame({"session_id": session_ids, "wave": ["winter"] * len(session_ids)})
    names = {
        "turns": [
            "turns__latency_median",
            "turns__participant_speaking_ratio",
            "turns__n_per_minute",
        ],
        "prosody": [
            "prosody__f0_semitone_sd",
            "prosody__intensity_sd_db",
            "prosody__speech_rate_proxy",
        ],
        "face_speaking": [
            "face_speaking__au12_mean",
            "face_speaking__au06_mean",
            "face_speaking__au01_mean",
        ],
        "face_listening": [
            "face_listening__au12_mean",
            "face_listening__au06_mean",
            "face_listening__au01_mean",
        ],
    }
    for family in families:
        for column in names[family]:
            frame[column] = RNG.normal(size=len(session_ids))
    frame["qc__flags"] = ""
    return frame


def label_table(session_ids: list[int], targets: tuple[str, ...] = ("K6", "SRS2")) -> pd.DataFrame:
    return pd.DataFrame(
        {"session_id": session_ids, **{t: RNG.normal(10, 3, len(session_ids)) for t in targets}}
    )


def write(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def no_text_config() -> AppConfig:
    """The real config with the text baseline switched off.

    Most tests here are not about the text join, and leaving it on would make
    them depend on a file outside the repository.
    """
    return load_config(DEFAULT_CONFIG_PATH, overrides={"model.text_features": None})


class TestLabels:
    def test_the_targets_are_read(self, tmp_path: Path) -> None:
        path = write(tmp_path / "labels.csv", label_table([1, 2, 3]))
        table = stage.load_labels(path, ("K6", "SRS2"))
        assert table.targets == ("K6", "SRS2")
        assert table.n_rows == 3

    def test_a_missing_identifier_is_refused(self, tmp_path: Path) -> None:
        path = write(tmp_path / "labels.csv", pd.DataFrame({"K6": [1, 2]}))
        with pytest.raises(stage.ModelError, match="no session identifier"):
            stage.load_labels(path, ("K6",))

    def test_the_refusal_says_labels_are_never_matched_by_row_order(self, tmp_path: Path) -> None:
        path = write(tmp_path / "labels.csv", pd.DataFrame({"K6": [1, 2]}))
        with pytest.raises(stage.ModelError) as excinfo:
            stage.load_labels(path, ("K6",))
        assert "never by row order" in str(excinfo.value)

    def test_no_configured_target_present_is_refused(self, tmp_path: Path) -> None:
        path = write(tmp_path / "labels.csv", pd.DataFrame({"session_id": [1], "other": [2]}))
        with pytest.raises(stage.ModelError, match="none of the configured targets"):
            stage.load_labels(path, ("K6", "SRS2"))

    def test_one_missing_target_is_a_warning_not_a_refusal(
        self, tmp_path: Path, package_logs: pytest.LogCaptureFixture
    ) -> None:
        path = write(tmp_path / "labels.csv", label_table([1, 2, 3], targets=("K6",)))
        table = stage.load_labels(path, ("K6", "SRS2"))
        assert table.targets == ("K6",)
        assert "SRS2" in package_logs.text

    def test_a_repeated_session_is_refused(self, tmp_path: Path) -> None:
        path = write(tmp_path / "labels.csv", pd.DataFrame({"session_id": [1, 1], "K6": [3, 4]}))
        with pytest.raises(stage.ModelError, match="more than one row"):
            stage.load_labels(path, ("K6",))

    def test_no_label_value_is_logged(
        self, tmp_path: Path, package_logs: pytest.LogCaptureFixture
    ) -> None:
        frame = pd.DataFrame({"session_id": [1, 2], "K6": [17.4321, 3.1415]})
        stage.load_labels(write(tmp_path / "labels.csv", frame), ("K6",))
        assert "17.4321" not in package_logs.text
        assert "3.1415" not in package_logs.text


class TestCohort:
    def test_only_sessions_on_both_sides_are_modelled(self, tmp_path: Path) -> None:
        labels = stage.load_labels(write(tmp_path / "l.csv", label_table([1, 2, 9])), ("K6",))
        cohort = stage.build_cohort(feature_table([1, 2, 3]), labels, {})
        assert cohort.n == 2
        assert cohort.features_only == (3,)
        assert cohort.labels_only == (9,)

    def test_both_sides_are_aligned_by_session(self, tmp_path: Path) -> None:
        features = feature_table([3, 1, 2])
        labels = stage.load_labels(write(tmp_path / "l.csv", label_table([2, 3, 1])), ("K6",))
        cohort = stage.build_cohort(features, labels, {})
        assert list(cohort.features["session_id"]) == [1, 2, 3]
        assert list(cohort.labels["session_id"]) == [1, 2, 3]

    def test_no_overlap_is_refused_with_a_reason(self, tmp_path: Path) -> None:
        labels = stage.load_labels(write(tmp_path / "l.csv", label_table([90, 91])), ("K6",))
        with pytest.raises(stage.ModelError, match="same session numbering"):
            stage.build_cohort(feature_table([1, 2]), labels, {})

    def test_the_report_names_the_dropped_sessions(self, tmp_path: Path) -> None:
        labels = stage.load_labels(write(tmp_path / "l.csv", label_table([1, 9])), ("K6",))
        cohort = stage.build_cohort(feature_table([1, 2]), labels, {})
        text = "\n".join(cohort.report_lines())
        assert "[2]" in text
        assert "[9]" in text

    def test_groups_default_to_the_session(self, tmp_path: Path) -> None:
        labels = stage.load_labels(write(tmp_path / "l.csv", label_table([1, 2])), ("K6",))
        cohort = stage.build_cohort(feature_table([1, 2]), labels, {})
        assert cohort.groups == (1, 2)

    def test_a_participant_map_groups_sessions_together(self, tmp_path: Path) -> None:
        labels = stage.load_labels(write(tmp_path / "l.csv", label_table([1, 2, 3])), ("K6",))
        cohort = stage.build_cohort(feature_table([1, 2, 3]), labels, {1: "p1", 2: "p1", 3: "p2"})
        assert cohort.groups == ("p1", "p1", "p2")


class TestFeatureSets:
    def test_a_set_resolves_to_the_columns_of_its_families(self) -> None:
        config = no_text_config()
        sets = stage.resolve_feature_sets(feature_table([1, 2]), config)
        assert all(c.startswith("turns__") for c in sets["turns"].columns)
        assert sets["audio"].columns == sets["turns"].columns + sets["prosody"].columns

    def test_qc_columns_are_never_features(self) -> None:
        frame = feature_table([1, 2])
        frame["qc__speaking_seconds"] = 10.0
        sets = stage.resolve_feature_sets(frame, no_text_config())
        assert all(not c.startswith("qc__") for s in sets.values() for c in s.columns)

    def test_an_absent_family_is_named_rather_than_ignored(self) -> None:
        sets = stage.resolve_feature_sets(feature_table([1, 2]), no_text_config())
        assert "face_speaking" in sets["face"].absent_families
        assert not sets["face"].is_usable

    def test_a_present_family_is_not_reported_absent(self) -> None:
        sets = stage.resolve_feature_sets(feature_table([1, 2]), no_text_config())
        assert sets["turns"].absent_families == ()


def shrunken_config(**extra: object) -> AppConfig:
    """The real analysis, shrunk so the tests run in seconds.

    What is under test is the shape - tiers, comparisons, correction, outputs -
    not the cost, so the feature sets are cut to the ones the confirmatory
    comparison needs, with one target and a handful of permutations.
    """
    overrides: dict[str, object] = {
        "model.n_permutations": 3,
        "model.tiers.stability_repeats": 1,
        "model.tiers.stability_folds": 3,
        "model.models": ["elastic_net"],
        "model.targets": ["K6"],
        "model.feature_sets": {
            "all": ["turns", "prosody", "face_speaking", "face_listening"],
            "audio": ["turns", "prosody"],
            "text": ["text"],
        },
    }
    overrides.update(extra)
    return load_config(DEFAULT_CONFIG_PATH, overrides=overrides)


def run_inputs(tmp_path: Path, n: int = 16) -> tuple[Path, Path]:
    """A feature table with every confirmatory feature, and a learnable target."""
    sessions = list(range(1, n + 1))
    features = feature_table(
        sessions, families=("turns", "prosody", "face_speaking", "face_listening")
    )
    labels = pd.DataFrame(
        {
            "session_id": sessions,
            "K6": 2.0 * features["turns__latency_median"] + RNG.normal(0, 0.5, n),
        }
    )
    return (
        write(tmp_path / "features.csv", features),
        write(tmp_path / "labels.csv", labels),
    )


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: pytest.TempPathFactory) -> stage.ModelResult:
    """One real run of the stage, shared by every assertion about its output.

    Module-scoped on purpose: the run is the expensive part, and re-running it
    per assertion made this file slower than the rest of the suite combined.
    """
    tmp_path = tmp_path_factory.mktemp("model_run")
    features_path, labels_path = run_inputs(tmp_path)
    roots = DataRoots(data=tmp_path, work=tmp_path, out=tmp_path)
    return stage.run(
        shrunken_config(**{"model.text_features": None}),
        roots,
        features_path=features_path,
        labels_path=labels_path,
        out_dir=tmp_path / "out",
    )


@pytest.mark.slow
class TestTheRun:
    def test_it_writes_results_and_a_summary(self, completed_run: stage.ModelResult) -> None:
        assert completed_run.results_path.exists()
        assert completed_run.comparisons_path.exists()
        assert completed_run.summary_path.exists()
        assert completed_run.estimates

    def test_the_confirmatory_tier_comes_from_the_config_not_the_data(
        self, completed_run: stage.ModelResult
    ) -> None:
        confirmatory = {e.feature_set for e in completed_run.estimates if e.tier == "confirmatory"}
        named = {
            name
            for comparison in shrunken_config().model.tiers.primary_comparisons
            for name in comparison.against
        }
        assert confirmatory <= named

    def test_the_summary_states_what_the_analysis_cannot_say(
        self, completed_run: stage.ModelResult
    ) -> None:
        summary = completed_run.summary_path.read_text()
        assert "Sixty-two participants" in summary
        assert "No label appears in any file" in summary

    def test_both_cv_schemes_are_reported(self, completed_run: stage.ModelResult) -> None:
        results = pd.read_csv(completed_run.results_path)
        assert "loo_r2" in results.columns
        assert "kfold_r2_mean" in results.columns
        assert results["kfold_r2_mean"].notna().any()

    def test_a_learnable_target_is_learned(self, completed_run: stage.ModelResult) -> None:
        # The target is a function of one feature, so at least one estimate
        # should be well clear of zero. Otherwise the plumbing is wrong in a
        # way none of the structural assertions would catch.
        assert max(e.evaluation.loo.r2 for e in completed_run.estimates) > 0.5

    def test_no_label_value_appears_in_any_output(self, tmp_path: Path, roots: DataRoots) -> None:
        sessions = list(range(1, 17))
        features = feature_table(sessions, families=("turns", "prosody"))
        marked = 913.7717  # a value that cannot occur by chance downstream
        labels = pd.DataFrame(
            {"session_id": sessions, "K6": [marked, *list(RNG.normal(10, 3, 15))]}
        )
        result = stage.run(
            shrunken_config(**{"model.text_features": None, "model.n_permutations": 0}),
            roots,
            features_path=write(tmp_path / "f.csv", features),
            labels_path=write(tmp_path / "l.csv", labels),
        )
        for path in (result.results_path, result.comparisons_path, result.summary_path):
            assert "913.77" not in path.read_text()

    def test_missing_text_features_are_reported_not_hidden(
        self, tmp_path: Path, roots: DataRoots
    ) -> None:
        features_path, labels_path = run_inputs(tmp_path)
        # The text baseline left switched on, but pointed at nothing.
        config = shrunken_config(
            **{"model.text_features.path": "absent.csv", "model.n_permutations": 0}
        )
        result = stage.run(config, roots, features_path=features_path, labels_path=labels_path)
        assert any("text features unavailable" in note for note in result.notes)
        assert "text baseline is missing" in result.summary_path.read_text().lower()

    def test_a_missing_feature_table_says_so(self, roots: DataRoots, tmp_path: Path) -> None:
        _, labels_path = run_inputs(tmp_path)
        with pytest.raises(stage.ModelError, match="no feature table"):
            stage.run(
                shrunken_config(**{"model.text_features": None}),
                roots,
                features_path=tmp_path / "absent.csv",
                labels_path=labels_path,
            )
