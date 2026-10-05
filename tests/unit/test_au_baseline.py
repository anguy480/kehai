"""The exploratory AU-only baseline, on synthetic data only.

Tested hardest: that the plan resolves to exactly the columns aggregate names
(so a set cannot silently shrink), that the pooled window follows aggregate's
rules, that the search sees only training rows, that the max statistic is the
best of each permutation, and that no label value reaches any output file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vc_multimodal.config import AppConfig
from vc_multimodal.exploratory.au_baseline import analyze, cv, plan, pooled, search
from vc_multimodal.features.spans import Span
from vc_multimodal.io_utils import write_csv, write_parquet
from vc_multimodal.paths import DataRoots
from vc_multimodal.qc_notes import QcNote, QcNotes
from vc_multimodal.stages import aggregate as aggregate_stage
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import turns as turns_stage

REPO_ROOT = Path(__file__).resolve().parents[2]
BLUR = "camera out of focus throughout, confirmed by watching"


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    # The plan path is relative to the repository, like config/default.yaml.
    monkeypatch.chdir(REPO_ROOT)


def feature_table(config: AppConfig, n: int = 20, seed: int = 0) -> pd.DataFrame:
    """A synthetic table with every face, pooled and text column."""
    rng = np.random.default_rng(seed)
    columns = [
        *aggregate_stage.face_feature_names(config),
        *pooled.pooled_feature_names(config),
        "text__a",
        "text__b",
    ]
    table = pd.DataFrame(rng.random((n, len(columns))), columns=columns)
    table.insert(0, "session_id", range(1, n + 1))
    return table


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
class TestPlan:
    def test_the_committed_plan_resolves_against_aggregate_names(
        self, default_config: AppConfig
    ) -> None:
        sets = plan.resolve(plan.load_plan(), list(feature_table(default_config).columns))
        au = [s for s in sets if s.kind == plan.KIND_AU_SET]
        # 11 sets x 4 windows, less the 4 lower_face duplicates of au12.
        assert len(au) == 40
        au12 = next(s for s in au if s.name == "au12__pooled")
        assert au12.columns == ("face_pooled__au12_mean",)
        assert au12.aliases == ("lower_face__pooled",)
        both = next(s for s in au if s.name == "miyamoto__speaking_and_listening")
        assert len(both.columns) == 6
        assert next(s for s in au if s.name == "all_au_mean_sd__pooled").columns[:2] == (
            "face_pooled__au01_mean",
            "face_pooled__au01_sd",
        )
        refs = {s.name: s for s in sets if s.kind == plan.KIND_REFERENCE}
        assert len(refs["face_all_30"].columns) == 30
        assert refs["text"].columns == ("text__a", "text__b")
        searched = next(s for s in sets if s.is_search)
        assert len(searched.columns) == 15
        assert all(s.in_family for s in au)
        assert searched.in_family
        assert not any(s.in_family for s in refs.values())

    def test_a_missing_column_is_refused_not_dropped(self, default_config: AppConfig) -> None:
        table = feature_table(default_config).drop(columns=["face_pooled__au04_mean"])
        with pytest.raises(plan.PlanError, match="face_pooled__au04_mean"):
            plan.resolve(plan.load_plan(), list(table.columns))

    def test_a_reference_of_the_wrong_size_is_refused(self, default_config: AppConfig) -> None:
        table = feature_table(default_config).drop(columns=["face_speaking__head_yaw_sd"])
        with pytest.raises(plan.PlanError, match="29 column"):
            plan.resolve(plan.load_plan(), list(table.columns))


# ---------------------------------------------------------------------------
# The pooled window
# ---------------------------------------------------------------------------
def frames(config: AppConfig, values: np.ndarray, detected: np.ndarray) -> pd.DataFrame:
    n = len(values)
    data: dict[str, object] = {"timestamp_s": np.arange(n) * 0.2, "detected": detected}
    for key in config.face.unit_keys:
        data[key] = np.where(detected, values, np.nan)
    data["head_pitch"] = np.zeros(n)
    data["head_yaw"] = np.zeros(n)
    return pd.DataFrame(data)


class TestPooled:
    def test_pooled_is_the_union_of_both_windows_and_excludes_silence(
        self, default_config: AppConfig
    ) -> None:
        # 0-60 s speaking at 1.0, 60-80 s silence at 9.0, 80-140 s listening at 3.0.
        times = np.arange(700) * 0.2
        values = np.select([times < 60, times < 80], [1.0, 9.0], 3.0)
        timeline = aggregate_stage.Timeline(
            speaking=(Span(0.0, 60.0),), listening=(Span(80.0, 140.0),)
        )
        out = pooled.summarise_pooled(
            frames(default_config, values, np.ones(700, bool)), timeline, default_config
        )
        assert out["face_pooled__au12_mean"] == pytest.approx(2.0)
        assert set(out) == set(pooled.pooled_feature_names(default_config))

    def test_too_little_measured_time_yields_nothing(self, default_config: AppConfig) -> None:
        detected = np.zeros(700, bool)
        detected[:100] = True  # 20 s measured, under the 30 s minimum
        timeline = aggregate_stage.Timeline(speaking=(Span(0.0, 140.0),))
        out = pooled.summarise_pooled(
            frames(default_config, np.ones(700), detected), timeline, default_config
        )
        assert all(v is None for v in out.values())

    def test_a_face_note_blanks_pooled_and_both_windows(
        self, default_config: AppConfig, roots: DataRoots
    ) -> None:
        write_parquet(
            face_stage.face_path(roots, 7),
            frames(default_config, np.full(700, 0.5), np.ones(700, bool)),
        )
        timeline = pd.DataFrame(
            {"state": ["speaking", "listening"], "start_s": [0.0, 70.0], "end_s": [70.0, 140.0]}
        )
        write_parquet(turns_stage.timeline_path(roots, 7), timeline)

        clean = pooled.session_features(7, default_config, roots, QcNotes())
        assert clean.pooled["face_pooled__au12_mean"] == pytest.approx(0.5)
        assert clean.windows["face_speaking__au12_mean"] == pytest.approx(0.5)

        notes = QcNotes(notes=(QcNote(7, "face", "unavailable", BLUR, "tester", "2026-10-05"),))
        noted = pooled.session_features(7, default_config, roots, notes)
        assert all(v is None for v in noted.pooled.values())
        assert all(v is None for v in noted.windows.values())

    def test_the_reproduction_check_sees_values_and_missingness(self) -> None:
        a = pd.DataFrame({"session_id": [1, 2], "x": [0.5, np.nan]})
        assert pooled.compare(a, a.copy(), ["x"]).matches
        assert not pooled.compare(a, a.assign(x=[0.6, np.nan]), ["x"]).matches
        assert pooled.compare(a, a.assign(x=[0.5, 0.1]), ["x"]).n_missing_mismatches == 1


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------
class TestSearch:
    def test_it_finds_the_informative_column(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(60, 6))
        y = 2.0 * x[:, 3] + rng.normal(scale=0.3, size=60)
        spec = search.SearchSpec(max_features=3, inner_folds=5, seed=0)
        chosen = search.forward_select(x, y, list(range(60)), spec)
        assert chosen[0] == 3
        assert len(chosen) <= 3

    def test_selection_sees_only_the_training_rows(self) -> None:
        # Column 0 explains only the held-out rows; column 1 explains the rest.
        rng = np.random.default_rng(2)
        x = rng.normal(size=(40, 2))
        y = x[:, 1] + rng.normal(scale=0.1, size=40)
        held_out = np.zeros(40, bool)
        held_out[:8] = True
        y[held_out] = 50.0 * x[held_out, 0]
        task = cv.Task("s", x, search.SearchSpec(2, 5, 0))
        selections = cv.selections_for(task, y, list(range(40)), [held_out])
        assert selections is not None
        assert selections[0][0] == 1

    def test_missing_values_are_imputed_inside(self) -> None:
        x = np.array([[1.0], [np.nan], [3.0], [4.0]])
        predicted = search.ols_predict(x[:3], np.array([1.0, 2.0, 3.0]), x[3:])
        assert np.isfinite(predicted).all()


# ---------------------------------------------------------------------------
# Cross-validation and the nulls
# ---------------------------------------------------------------------------
class TestCv:
    def test_a_real_signal_scores(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(40, 2))
        y = x[:, 0] + rng.normal(scale=0.3, size=40)
        signal = cv.evaluate_set(
            cv.Task("s", x), y, list(range(40)), model="elastic_net", seed=0, folds=5, repeats=2
        )
        assert signal.loo_r2 > 0.5
        assert signal.kfold_spearman_mean > 0.5
        assert signal.n == 40

    def test_the_null_has_one_value_per_permutation_set_and_model(self) -> None:
        rng = np.random.default_rng(4)
        tasks = [cv.Task("a", rng.normal(size=(20, 1))), cv.Task("b", rng.normal(size=(20, 2)))]
        null = cv.null_chunk(
            tasks,
            rng.normal(size=20),
            list(range(20)),
            analyze.permutation_orders(20, 3, seed=0),
            models=("elastic_net",),
            seed=0,
            folds=5,
        )
        assert null.shape == (3, 2, 1)
        assert np.isfinite(null).all()

    def test_set_null_counts_the_observed_value(self) -> None:
        result = cv.set_null(0.5, np.array([0.1, 0.6, 0.2, 0.3]))
        assert result.p_value == pytest.approx(2 / 5)

    def test_the_max_statistic_takes_the_best_of_each_permutation(self) -> None:
        observed = np.array([[0.10, 0.05], [0.30, 0.20]])
        null = np.array(
            [
                [[0.0, 0.1], [0.2, 0.0]],  # best 0.2
                [[0.4, 0.0], [0.0, 0.0]],  # best 0.4
                [[0.0, 0.0], [0.1, 0.0]],  # best 0.1
            ]
        )
        result = cv.max_statistic(observed, null, ["a", "b"], ["en", "rf"])
        assert (result.best_set, result.best_model) == ("b", "en")
        assert result.observed == pytest.approx(0.30)
        assert list(result.null_max) == pytest.approx([0.2, 0.4, 0.1])
        assert result.p_value == pytest.approx(2 / 4)
        assert result.percentile == pytest.approx(200 / 3)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_end_to_end_writes_no_label_value(
    default_config: AppConfig,
    roots: DataRoots,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 16
    table = feature_table(default_config, n=n, seed=5)
    rng = np.random.default_rng(6)
    # Distinctive values, so any leak into an output file is findable.
    labels = pd.DataFrame(
        {
            "session_id": range(1, n + 1),
            "K6": np.round(rng.uniform(0, 24, n), 3) + 0.000123,
            "SRS2": np.round(rng.uniform(40, 90, n), 3) + 0.000456,
        }
    )
    labels_path = write_csv(tmp_path / "labels.csv", labels)
    monkeypatch.setattr(
        analyze, "load_inputs", lambda *_a, **_k: analyze.Inputs(table=table, digests={})
    )
    full = plan.load_plan()
    small = full.model_copy(
        update={
            "au_sets": full.au_sets[4:6],  # au12 and miyamoto
            "references": {"text": plan.Reference(prefixes=("text__",))},
            "evaluation": plan.Evaluation(
                targets=("K6", "SRS2"),
                primary_target="K6",
                models=("elastic_net",),
                kfold_folds=4,
                kfold_repeats=2,
            ),
            "permutation": plan.Permutation(n_permutations=3, max_statistic_family=("au_sets",)),
            "sensitivity": plan.Sensitivity(top_n=2),
        }
    )
    out = tmp_path / "results"
    result = analyze.run(
        default_config,
        roots,
        small,
        bundle=tmp_path,
        labels_path=labels_path,
        out_dir=out,
        commit="0" * 40,
        jobs=1,
        exclude_sessions=(3,),
    )
    assert result.n_sessions == n
    estimates = pd.read_csv(out / analyze.ESTIMATES_FILE)
    # 2 sets x 4 windows + search + text, one model, two targets.
    assert len(estimates) == 10 * 2
    assert set(estimates["tier"]) == {"exploratory"}
    assert estimates["perm_p_uncorrected"].between(0, 1).all()
    max_rows = pd.read_csv(out / analyze.MAX_FILE)
    assert set(max_rows["scope"]) == {analyze.BOTH_MODELS, "elastic_net"}
    assert len(pd.read_csv(out / analyze.SENSITIVITY_FILE)) == 2 * 2
    assert (out / analyze.SUMMARY_FILE).read_text().startswith("# AU-only baseline")

    written = "".join(
        p.read_text(encoding="utf-8") for p in out.iterdir() if p.suffix in {".csv", ".md", ".json"}
    )
    for value in (*labels["K6"], *labels["SRS2"]):
        assert f"{value:.6f}" not in written
