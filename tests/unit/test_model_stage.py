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
from vc_multimodal.modeling.tiers import EstimateCounts, TierPlan, describe_plan, resolve_tiers
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


# ---------------------------------------------------------------------------
# the confirmatory path, end to end, through the correction
#
# This is the last step of the whole run, and it crashed after all 32 estimates
# had been computed: the correction rebuilt a slotted dataclass from __dict__,
# which does not exist on one. No test reached it, because every earlier fixture
# left the `text` feature set with no columns - so no comparison could be built,
# `collected` stayed empty, and the correction never ran at all.
# ---------------------------------------------------------------------------
def comparable_inputs(tmp_path: Path, n: int = 20) -> tuple[Path, Path]:
    """Features where BOTH sides of every confirmatory comparison exist.

    The text columns are the point: without them the comparison is skipped and
    the correction is never exercised.
    """
    sessions = list(range(1, n + 1))
    features = feature_table(
        sessions, families=("turns", "prosody", "face_speaking", "face_listening")
    )
    for name in ("text__jaccard", "text__cosine", "text__bert", "text__mtld_patient"):
        features[name] = RNG.normal(size=n)
    labels = pd.DataFrame(
        {
            "session_id": sessions,
            # Learnable from the turn features and not from the text ones, so
            # the comparison has something to find.
            "K6": 2.0 * features["turns__latency_median"] + RNG.normal(0, 0.5, n),
        }
    )
    return (
        write(tmp_path / "features.csv", features),
        write(tmp_path / "labels.csv", labels),
    )


def comparison_config(**extra: object) -> AppConfig:
    """The real confirmatory plan, shrunk. Both compared sets have columns."""
    overrides: dict[str, object] = {
        "model.text_features": None,
        "model.n_permutations": 0,
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


@pytest.fixture(scope="module")
def corrected_run(tmp_path_factory: pytest.TempPathFactory) -> stage.ModelResult:
    """One full run that reaches the multiplicity correction."""
    tmp_path = tmp_path_factory.mktemp("model_corrected")
    features_path, labels_path = comparable_inputs(tmp_path)
    roots = DataRoots(data=tmp_path, work=tmp_path, out=tmp_path)
    return stage.run(
        comparison_config(),
        roots,
        features_path=features_path,
        labels_path=labels_path,
        out_dir=tmp_path / "out",
    )


@pytest.mark.slow
class TestTheConfirmatoryPathCompletes:
    def test_the_correction_runs_without_crashing(self, corrected_run: stage.ModelResult) -> None:
        """The regression: this raised AttributeError on a slotted dataclass."""
        assert corrected_run.tests

    def test_both_output_files_are_written(self, corrected_run: stage.ModelResult) -> None:
        assert corrected_run.results_path.exists()
        assert corrected_run.comparisons_path.exists()
        assert not pd.read_csv(corrected_run.results_path).empty
        assert not pd.read_csv(corrected_run.comparisons_path).empty

    def test_every_planned_comparison_was_actually_run(
        self, corrected_run: stage.ModelResult
    ) -> None:
        # Guards the silent skip that hid the crash: a comparison whose feature
        # set has no columns produces no test, and the run still "succeeds".
        planned = comparison_config().model.tiers.primary_comparisons
        assert len(corrected_run.tests) == len(planned)
        assert not any("could not be run" in note for note in corrected_run.notes)

    def test_each_test_carries_an_adjusted_p_value(self, corrected_run: stage.ModelResult) -> None:
        for test in corrected_run.tests:
            assert test.comparison.p_adjusted is not None

    def test_the_adjustment_never_lowers_a_p_value(self, corrected_run: stage.ModelResult) -> None:
        for test in corrected_run.tests:
            assert test.comparison.p_adjusted is not None
            assert test.comparison.p_adjusted >= test.comparison.p_value

    def test_the_raw_and_adjusted_values_both_reach_the_file(
        self, corrected_run: stage.ModelResult
    ) -> None:
        frame = pd.read_csv(corrected_run.comparisons_path)
        assert frame["p_value"].notna().all()
        assert frame["p_holm"].notna().all()

    def test_the_summary_reports_the_corrected_tests(
        self, corrected_run: stage.ModelResult
    ) -> None:
        text = "\n".join(stage.summarise(corrected_run))
        assert "confirmatory tests:" in text
        assert "p(Holm)" in text

    def test_the_markdown_summary_shows_the_corrected_table(
        self, corrected_run: stage.ModelResult
    ) -> None:
        summary = corrected_run.summary_path.read_text()
        assert "p (Holm)" in summary
        for test in corrected_run.tests:
            assert test.name in summary


@pytest.mark.slow
class TestCorrectionCanBeTurnedOff:
    def test_without_correction_the_tests_still_run(self, tmp_path: Path, roots: DataRoots) -> None:
        features_path, labels_path = comparable_inputs(tmp_path)
        result = stage.run(
            comparison_config(**{"model.tiers.multiplicity_correction": "none"}),
            roots,
            features_path=features_path,
            labels_path=labels_path,
        )
        assert result.tests
        for test in result.tests:
            assert test.comparison.p_adjusted is None
        assert pd.read_csv(result.comparisons_path)["p_value"].notna().all()


# ---------------------------------------------------------------------------
# which columns a confirmatory comparison actually uses
#
# The bug this covers: the confirmatory tier was unreachable. A set was called
# confirmatory only if all its columns were among the pre-registered ones, which
# `all` (54 columns) and `text` (no pre-registered subset at all) can never be.
# So every estimate was exploratory, no permutation null ever ran, and the four
# "confirmatory" tests compared full feature sets instead of the 12 features
# ADR 0012 pre-registers.
# ---------------------------------------------------------------------------
def resolved(config: AppConfig, frame: pd.DataFrame, name: str) -> stage.FeatureSet:
    return stage.resolve_feature_sets(frame, config)[name]


#: Real feature names that are NOT pre-registered. Without some of these the
#: fixture's families are exactly their primaries, the confirmatory restriction
#: is a no-op, and the tests cannot tell the two tiers apart - which is the
#: shape the real 54-feature table does not have.
NON_PRIMARY = (
    "turns__latency_sd",
    "turns__pause_within_mean",
    "prosody__hnr_db",
    "prosody__jitter_local",
    "face_speaking__au12_sd",
    "face_speaking__au02_mean",
    "face_listening__au04_mean",
    "face_listening__au06_p90",
)


def full_table(n: int = 20) -> pd.DataFrame:
    """A table shaped like the real one: primaries plus exploratory features."""
    frame = feature_table(
        list(range(1, n + 1)),
        families=("turns", "prosody", "face_speaking", "face_listening"),
    )
    for name in (*NON_PRIMARY, "text__jaccard", "text__cosine", "text__bert"):
        frame[name] = RNG.normal(size=n)
    return frame


class TestConfirmatoryColumns:
    def test_our_families_are_cut_to_their_pre_registered_features(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        frame = full_table()
        feature_set = resolved(config, frame, "all")
        restricted = stage.confirmatory_columns(feature_set, config)
        # Only the pre-registered ones, and every one of them present.
        primary = set(config.model.tiers.primary_columns)
        assert set(restricted) <= primary
        assert set(restricted) == primary & set(feature_set.columns)

    def test_the_restriction_is_a_strict_reduction(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        feature_set = resolved(config, full_table(), "all")
        restricted = stage.confirmatory_columns(feature_set, config)
        assert len(restricted) < len(feature_set.columns)

    def test_the_text_baseline_enters_whole(self) -> None:
        """We never pre-registered a subset of someone else's feature set.

        Choosing one now would mean selecting the baseline we are measured
        against, on no prior basis.
        """
        config = load_config(DEFAULT_CONFIG_PATH)
        feature_set = resolved(config, full_table(), "text")
        assert set(stage.confirmatory_columns(feature_set, config)) == set(feature_set.columns)

    def test_a_partly_pre_registered_set_keeps_only_its_primaries(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        feature_set = resolved(config, full_table(), "audio")
        restricted = stage.confirmatory_columns(feature_set, config)
        assert all(name.startswith(("turns__", "prosody__")) for name in restricted)
        assert set(restricted) <= set(config.model.tiers.primary_columns)


class TestVariants:
    def test_a_named_set_is_evaluated_confirmatory_and_exploratory(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        variants = stage.variants_for(resolved(config, full_table(), "all"), config)
        assert [v.tier for v in variants] == ["confirmatory", "exploratory"]
        assert len(variants[0].columns) < len(variants[1].columns)

    def test_an_unnamed_set_is_exploratory_only(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        variants = stage.variants_for(resolved(config, full_table(), "turns"), config)
        assert [v.tier for v in variants] == ["exploratory"]

    def test_a_set_whose_columns_are_all_pre_registered_is_evaluated_once(self) -> None:
        # The text baseline: restricting it changes nothing, so one estimate
        # serves and is reported as confirmatory.
        config = load_config(DEFAULT_CONFIG_PATH)
        variants = stage.variants_for(resolved(config, full_table(), "text"), config)
        assert [v.tier for v in variants] == ["confirmatory"]

    def test_the_confirmatory_tier_is_reachable_at_all(self) -> None:
        """The regression: it never was.

        Asserted against the shipped configuration, because the bug was that
        the real plan could not produce a single confirmatory estimate.
        """
        config = load_config(DEFAULT_CONFIG_PATH)
        sets = stage.resolve_feature_sets(full_table(), config)
        tiers = [
            variant.tier
            for feature_set in sets.values()
            if feature_set.is_usable
            for variant in stage.variants_for(feature_set, config)
        ]
        assert "confirmatory" in tiers

    def test_every_set_named_in_a_comparison_has_a_confirmatory_variant(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        sets = stage.resolve_feature_sets(full_table(), config)
        named = {
            name
            for comparison in config.model.tiers.primary_comparisons
            for name in comparison.against
        }
        for name in named:
            variants = stage.variants_for(sets[name], config)
            assert any(v.tier == "confirmatory" for v in variants), name


@pytest.fixture(scope="module")
def tiered_run(tmp_path_factory: pytest.TempPathFactory) -> stage.ModelResult:
    """One full run over a realistically shaped table, with both tiers."""
    tmp_path = tmp_path_factory.mktemp("model_tiered")
    frame = full_table()
    labels = pd.DataFrame(
        {
            "session_id": frame["session_id"],
            "K6": 2.0 * frame["turns__latency_median"] + RNG.normal(0, 0.5, len(frame)),
        }
    )
    roots = DataRoots(data=tmp_path, work=tmp_path, out=tmp_path)
    return stage.run(
        comparison_config(**{"model.n_permutations": 3}),
        roots,
        features_path=write(tmp_path / "features.csv", frame),
        labels_path=write(tmp_path / "labels.csv", labels),
        out_dir=tmp_path / "out",
    )


@pytest.mark.slow
class TestTheRunHonoursTheTiers:
    def test_both_tiers_are_represented(self, tiered_run: stage.ModelResult) -> None:
        tiers = {estimate.tier for estimate in tiered_run.estimates}
        assert tiers == {"confirmatory", "exploratory"}

    def test_the_confirmatory_estimates_use_fewer_features(
        self, tiered_run: stage.ModelResult
    ) -> None:
        by_tier: dict[str, set[int]] = {}
        for estimate in tiered_run.estimates:
            if estimate.feature_set == "all":
                by_tier.setdefault(estimate.tier, set()).add(estimate.evaluation.n_features)
        assert max(by_tier["confirmatory"]) < min(by_tier["exploratory"])

    def test_the_permutation_null_runs_on_the_confirmatory_variants(
        self, tiered_run: stage.ModelResult
    ) -> None:
        # It is gated on the confirmatory tier, so an unreachable tier meant
        # n_permutations was silently ignored for the whole run.
        with_null = [e for e in tiered_run.estimates if e.permutation is not None]
        assert with_null
        assert all(e.tier == "confirmatory" for e in with_null)

    def test_the_results_file_carries_the_permutation_columns(
        self, tiered_run: stage.ModelResult
    ) -> None:
        frame = pd.read_csv(tiered_run.results_path)
        assert "permutation_p" in frame.columns
        assert frame["permutation_p"].notna().any()

    def test_the_comparison_uses_the_confirmatory_variant(
        self, tiered_run: stage.ModelResult
    ) -> None:
        """Otherwise the pre-registered test is quietly a different test."""
        confirmatory = {
            (e.feature_set, e.target): e.evaluation.loo.r2
            for e in tiered_run.estimates
            if e.tier == "confirmatory"
        }
        assert tiered_run.tests
        for test in tiered_run.tests:
            assert test.first_r2 == pytest.approx(confirmatory[test.first, test.target])
            assert test.second_r2 == pytest.approx(confirmatory[test.second, test.target])

    def test_no_note_says_the_confirmatory_tier_is_empty(
        self, tiered_run: stage.ModelResult
    ) -> None:
        assert not any("confirmatory tier" in note for note in tiered_run.notes)


# ---------------------------------------------------------------------------
# what the tier message counts
#
# It reported "62 further feature(s)" against a 54-feature table, because the
# plan covers the joined table (54 ours + 20 text) while nothing in the message
# said so. The estimate figure was worse: len(feature_sets) x targets x models
# minus a count of *tests*, which is both stale and a category error.
# ---------------------------------------------------------------------------
class TestTheTierMessageCounts:
    def joined_plan(self) -> tuple[TierPlan, AppConfig]:
        config = load_config(DEFAULT_CONFIG_PATH)
        return resolve_tiers([str(c) for c in full_table().columns], config.model), config

    def test_the_exploratory_count_states_its_denominator(self) -> None:
        plan, config = self.joined_plan()
        text = "\n".join(describe_plan(plan, config.model))
        assert f"of {plan.n_features} feature(s) in the table" in text

    def test_the_text_baseline_is_counted_separately(self) -> None:
        # So a reader comparing against a 54-feature extraction can reconcile.
        plan, config = self.joined_plan()
        text = "\n".join(describe_plan(plan, config.model))
        assert "text baseline" in text
        assert "measured here" in text

    def test_the_sources_add_up_to_the_total(self) -> None:
        plan, _ = self.joined_plan()
        assert sum(plan.counted_by_source().values()) == plan.n_features

    def test_no_estimate_count_is_printed_when_it_is_not_known(self) -> None:
        # A wrong count reads as information; an absent one does not.
        plan, config = self.joined_plan()
        assert not any("estimates:" in line for line in describe_plan(plan, config.model))

    def test_the_supplied_estimate_count_is_printed(self) -> None:
        plan, config = self.joined_plan()
        counts = EstimateCounts(confirmatory=12, exploratory=28)
        text = "\n".join(describe_plan(plan, config.model, counts))
        assert "estimates: 40 (12 confirmatory, 28 exploratory)" in text


@pytest.mark.slow
class TestTheRunReportsItsOwnCounts:
    def test_the_estimate_count_matches_the_estimates_produced(
        self, tiered_run: stage.ModelResult
    ) -> None:
        """The property the old formula could not hold: agreement with reality."""
        confirmatory = sum(1 for e in tiered_run.estimates if e.tier == "confirmatory")
        exploratory = sum(1 for e in tiered_run.estimates if e.tier == "exploratory")
        assert confirmatory + exploratory == len(tiered_run.estimates)
        # And the results file has exactly that many rows.
        assert len(pd.read_csv(tiered_run.results_path)) == len(tiered_run.estimates)

    def test_the_results_file_records_non_convergence(self, tiered_run: stage.ModelResult) -> None:
        frame = pd.read_csv(tiered_run.results_path)
        assert "n_fits_not_converged" in frame.columns
        assert frame["n_fits_not_converged"].notna().all()
