"""The per-modality evaluation: plan, handoffs, corrections. Synthetic data only."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vc_multimodal.config import DEFAULT_CONFIG_PATH, load_config
from vc_multimodal.exploratory.per_modality import analyze
from vc_multimodal.exploratory.per_modality.plan import (
    RUN_FIRST,
    RUN_SECOND,
    Inputs,
    Plan,
    PlanError,
    Validation,
    load_plan,
)
from vc_multimodal.features.aggregate_math import feature_names
from vc_multimodal.features.prosody_math import FEATURE_NAMES as PROSODY_FEATURES
from vc_multimodal.features.turn_math import FEATURE_NAMES as TURN_FEATURES
from vc_multimodal.modeling import evaluate

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG = REPO_ROOT / "docs" / "prereg" / "per-modality.md"
MIYAMOTO_AUS = ("au01", "au06", "au12")

SESSIONS = [*range(1, 19), 43, 52]
CONSTANT_TURNS = ("turns__overlap_ratio", "turns__interruption_rate")


# ---------------------------------------------------------------------------
# The plan says what the pre-registration says
# ---------------------------------------------------------------------------
class TestThePlan:
    def test_it_loads(self) -> None:
        plan = load_plan()
        assert plan.targets == ("K6", "SRS2")
        assert plan.models.primary == "elastic_net"
        assert plan.models.secondary == "random_forest"

    def test_prosody_and_turns_are_the_full_extracted_families(self) -> None:
        plan = load_plan()
        assert plan.modalities["prosody"].features == PROSODY_FEATURES
        assert plan.modalities["turns"].features == TURN_FEATURES

    def test_face_is_au01_06_12_with_mean_sd_p90_in_both_windows(self) -> None:
        plan = load_plan()
        for window in ("face_speaking", "face_listening"):
            assert plan.modalities[window].features == feature_names(
                window, MIYAMOTO_AUS, MIYAMOTO_AUS
            )

    def test_the_counts_are_the_pre_registered_ones(self) -> None:
        plan = load_plan()
        counts = {name: len(m.features) for name, m in plan.modalities.items()}
        assert counts == {"prosody": 12, "turns": 12, "face_speaking": 9, "face_listening": 9}

    def test_every_planned_feature_is_named_in_the_pre_registration(self) -> None:
        text = PREREG.read_text(encoding="utf-8")
        plan = load_plan()
        for name, modality in plan.modalities.items():
            for feature in modality.features:
                generic = (
                    feature.replace(f"{name}__", "w__") if name.startswith("face") else feature
                )
                assert f"`{generic}`" in text, feature

    def test_validation_matches_the_confirmatory_configuration(self) -> None:
        plan = load_plan()
        config = load_config(DEFAULT_CONFIG_PATH)
        assert plan.validation.seed == config.runtime.seed
        assert plan.validation.stability_folds == config.model.tiers.stability_folds
        assert plan.validation.stability_repeats == config.model.tiers.stability_repeats
        assert plan.validation.n_permutations == config.model.n_permutations
        # vc model computes its null on max(2, stability_folds) folds.
        assert plan.validation.permutation_folds == max(2, config.model.tiers.stability_folds)
        assert tuple(plan.targets) == tuple(config.model.targets)

    def test_the_bundle_digest_is_the_one_sent(self) -> None:
        log = (REPO_ROOT / "docs" / "analysis-log.md").read_text(encoding="utf-8")
        assert load_plan().inputs.base_features_sha256 in log

    def test_the_analysis_digest_ignores_the_input_digests(self) -> None:
        plan = load_plan()
        moved = plan.model_copy(
            update={"inputs": Inputs(base_features_sha256="0" * 64, face_features_sha256="1" * 64)}
        )
        assert moved.analysis_digest() == plan.analysis_digest()

    def test_the_analysis_digest_follows_the_analysis(self) -> None:
        plan = load_plan()
        reseeded = plan.model_copy(
            update={"validation": plan.validation.model_copy(update={"seed": 1})}
        )
        assert reseeded.analysis_digest() != plan.analysis_digest()

    def test_a_modality_missing_from_the_runs_is_refused(self, tmp_path: Path) -> None:
        raw = (REPO_ROOT / "config" / "exploratory" / "per_modality.yaml").read_text()
        broken = raw.replace(
            'second: ["face_speaking", "face_listening"]', 'second: ["face_speaking"]'
        )
        path = tmp_path / "plan.yaml"
        path.write_text(broken)
        with pytest.raises(PlanError, match="exactly once"):
            load_plan(path)


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------
def base_table(rng: np.random.Generator) -> pd.DataFrame:
    frame = pd.DataFrame({"session_id": SESSIONS})
    for column in (*PROSODY_FEATURES, *TURN_FEATURES):
        frame[column] = rng.normal(size=len(SESSIONS))
    for column in CONSTANT_TURNS:
        frame[column] = 0.0
    frame.loc[frame["session_id"] == 5, "prosody__hnr_db"] = np.nan
    frame["qc__flags"] = ""
    return frame


def face_table(rng: np.random.Generator, plan: Plan) -> pd.DataFrame:
    frame = pd.DataFrame({"session_id": SESSIONS})
    for window in ("face_speaking", "face_listening"):
        for column in plan.modalities[window].features:
            frame[column] = rng.gamma(2.0, size=len(SESSIONS))
    face_columns = [c for c in frame.columns if c.startswith("face_")]
    frame.loc[frame["session_id"] == 43, face_columns] = np.nan
    frame[analyze.FACE_BACKEND_COLUMN] = "openface"
    return frame


def labels(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": SESSIONS,
            "K6": rng.normal(10.0, 4.0, size=len(SESSIONS)),
            "SRS2": rng.normal(60.0, 10.0, size=len(SESSIONS)),
        }
    )


def write(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def small_plan(base: Path, face: Path | None) -> Plan:
    """The real plan, pointed at synthetic files and shrunk to run in seconds."""
    plan = load_plan()
    return plan.model_copy(
        update={
            "inputs": Inputs(
                base_features_sha256=analyze.file_sha256(base),
                face_features_sha256=None if face is None else analyze.file_sha256(face),
            ),
            "validation": Validation(
                seed=plan.validation.seed,
                stability_folds=3,
                stability_repeats=1,
                permutation_folds=3,
                n_permutations=2,
            ),
        }
    )


@pytest.fixture
def few_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forest's size is not under test; its time is."""
    monkeypatch.setattr(evaluate, "_N_TREES", 5)


def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(7)
    bundle = tmp_path / "bundle"
    write(bundle / analyze.BUNDLE_FEATURES, base_table(rng))
    face = write(tmp_path / "face_openface_features.csv", face_table(rng, load_plan()))
    label_path = write(tmp_path / "private" / "labels.csv", labels(rng))
    return bundle, face, label_path


def both_runs(tmp_path: Path, plan: Plan, bundle: Path, face: Path, label_path: Path) -> None:
    analyze.run(
        plan,
        RUN_FIRST,
        labels_path=label_path,
        out_dir=tmp_path / "first",
        commit="c1",
        bundle=bundle,
    )
    analyze.run(
        plan,
        RUN_SECOND,
        labels_path=label_path,
        out_dir=tmp_path / "second",
        commit="c2",
        face_table=face,
        prior_dir=tmp_path / "first",
    )


# ---------------------------------------------------------------------------
# The two runs
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.usefixtures("few_trees")
class TestTheTwoRuns:
    def test_the_first_run_reports_raw_values_and_no_correction(self, tmp_path: Path) -> None:
        bundle, _, label_path = inputs(tmp_path)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, None)
        result = analyze.run(
            plan,
            RUN_FIRST,
            labels_path=label_path,
            out_dir=tmp_path / "first",
            commit="c1",
            bundle=bundle,
        )
        assert not result.corrected
        univariate = pd.read_csv(tmp_path / "first" / analyze.UNIVARIATE_FILE)
        models = pd.read_csv(tmp_path / "first" / analyze.MODELS_FILE)
        assert univariate["q"].isna().all()
        assert models["p_holm"].isna().all()
        assert models["perm_p"].notna().all()
        assert set(models["modality"]) == {"prosody", "turns"}
        assert len(models) == 2 * 2 * 2  # modalities x targets x models
        assert not (tmp_path / "first" / analyze.FINAL_MODELS_FILE).exists()
        # The two structurally zero turn features are untestable, not p = 1.
        flat = univariate.loc[univariate["feature"].isin(CONSTANT_TURNS)]
        assert (flat["untestable"] == "constant").all()
        assert flat["p"].isna().all()

    def test_the_second_run_corrects_the_complete_families(self, tmp_path: Path) -> None:
        bundle, face, label_path = inputs(tmp_path)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, face)
        both_runs(tmp_path, plan, bundle, face, label_path)

        models = pd.read_csv(tmp_path / "second" / analyze.FINAL_MODELS_FILE)
        primary = models.loc[models["role"] == "primary"]
        assert len(primary) == 8
        assert primary["p_holm"].notna().all()
        assert (primary["p_holm"] >= primary["perm_p"] - 1e-12).all()
        assert models.loc[models["role"] == "secondary", "p_holm"].isna().all()

        univariate = pd.read_csv(tmp_path / "second" / analyze.FINAL_UNIVARIATE_FILE)
        for target in ("K6", "SRS2"):
            rows = univariate.loc[univariate["target"] == target]
            assert len(rows) == 42
            # 42 planned, 2 constant: a family of 40, as pre-registered.
            assert rows["q"].notna().sum() == 40

        record = json.loads((tmp_path / "second" / analyze.RUN_FILE).read_text())
        assert record["prior"]["commit"] == "c1"
        assert record["flagged_sessions_present"] == [52]
        # Session 43's face is withheld, so both face modalities leave it out.
        assert record["excluded_no_features"] == {"face_speaking": [43], "face_listening": [43]}
        assert record["n_sessions_by_modality"]["face_speaking"] == len(SESSIONS) - 1

    def test_no_output_holds_a_label_value(self, tmp_path: Path) -> None:
        bundle, face, label_path = inputs(tmp_path)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, face)
        both_runs(tmp_path, plan, bundle, face, label_path)
        written = "\n".join(
            path.read_text(encoding="utf-8")
            for folder in (tmp_path / "first", tmp_path / "second")
            for path in folder.rglob("*")
            if path.is_file()
        )
        values = pd.read_csv(label_path)[["K6", "SRS2"]].to_numpy().ravel()
        for value in values:
            # Five decimals of a label value would not appear by chance.
            assert re.search(re.escape(f"{value:.5f}"), written) is None, value

    def test_a_first_run_under_another_analysis_is_refused(self, tmp_path: Path) -> None:
        bundle, face, label_path = inputs(tmp_path)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, face)
        analyze.run(
            plan,
            RUN_FIRST,
            labels_path=label_path,
            out_dir=tmp_path / "first",
            commit="c1",
            bundle=bundle,
        )
        changed = plan.model_copy(
            update={"validation": plan.validation.model_copy(update={"seed": 1})}
        )
        with pytest.raises(analyze.AnalyzeError, match="different analysis"):
            analyze.run(
                changed,
                RUN_SECOND,
                labels_path=label_path,
                out_dir=tmp_path / "second",
                commit="c2",
                face_table=face,
                prior_dir=tmp_path / "first",
            )


# ---------------------------------------------------------------------------
# Refusals that need no model fit
# ---------------------------------------------------------------------------
class TestRefusals:
    def test_a_bundle_that_is_not_the_planned_one_is_refused(self, tmp_path: Path) -> None:
        bundle, _, label_path = inputs(tmp_path)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, None)
        frame = pd.read_csv(bundle / analyze.BUNDLE_FEATURES)
        frame.loc[0, "prosody__hnr_db"] = 99.0
        write(bundle / analyze.BUNDLE_FEATURES, frame)
        with pytest.raises(analyze.AnalyzeError, match="not the planned"):
            analyze.run(
                plan,
                RUN_FIRST,
                labels_path=label_path,
                out_dir=tmp_path / "o",
                commit="c",
                bundle=bundle,
            )

    def test_the_face_run_needs_its_digest_fixed_first(self, tmp_path: Path) -> None:
        bundle, face, _ = inputs(tmp_path)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, None)
        with pytest.raises(analyze.AnalyzeError, match="not fixed in the plan"):
            analyze.load_tables(plan, RUN_SECOND, face_table=face)

    def test_a_face_table_still_holding_session_43_is_refused(self, tmp_path: Path) -> None:
        bundle, face, _ = inputs(tmp_path)
        frame = pd.read_csv(face)
        frame.loc[frame["session_id"] == 43, "face_speaking__au12_mean"] = 1.0
        write(face, frame)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, face)
        with pytest.raises(analyze.AnalyzeError, match="QC note was not applied"):
            analyze.load_tables(plan, RUN_SECOND, face_table=face)

    def test_a_face_table_not_from_openface_is_refused(self, tmp_path: Path) -> None:
        bundle, face, _ = inputs(tmp_path)
        frame = pd.read_csv(face)
        frame.loc[0, analyze.FACE_BACKEND_COLUMN] = "mediapipe"
        write(face, frame)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, face)
        with pytest.raises(analyze.AnalyzeError, match="OpenFace only"):
            analyze.load_tables(plan, RUN_SECOND, face_table=face)

    def test_the_second_run_needs_the_first(self, tmp_path: Path) -> None:
        bundle, face, label_path = inputs(tmp_path)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, face)
        with pytest.raises(analyze.AnalyzeError, match="--prior"):
            analyze.run(
                plan,
                RUN_SECOND,
                labels_path=label_path,
                out_dir=tmp_path / "o",
                commit="c",
                face_table=face,
            )

    def test_results_are_never_overwritten(self, tmp_path: Path) -> None:
        bundle, _, label_path = inputs(tmp_path)
        plan = small_plan(bundle / analyze.BUNDLE_FEATURES, None)
        (tmp_path / "o").mkdir()
        (tmp_path / "o" / analyze.RUN_FILE).write_text("{}")
        with pytest.raises(analyze.AnalyzeError, match="already holds results"):
            analyze.run(
                plan,
                RUN_FIRST,
                labels_path=label_path,
                out_dir=tmp_path / "o",
                commit="c",
                bundle=bundle,
            )

    def test_correction_refuses_an_incomplete_family(self) -> None:
        plan = load_plan()
        univariate = pd.DataFrame(
            {"feature": ["prosody__hnr_db"], "target": ["K6"], "p": [0.1], "q": [np.nan]}
        )
        with pytest.raises(analyze.AnalyzeError, match="incomplete"):
            analyze.correct(plan, univariate, pd.DataFrame(columns=list(analyze.MODEL_COLUMNS)))
