"""The fusion analysis, on synthetic tables and invented labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vc_multimodal.exploratory.fusion import analyze
from vc_multimodal.exploratory.fusion.__main__ import _parser
from vc_multimodal.exploratory.fusion.plan import (
    COMPARISONS,
    CONSTANT_TURNS,
    EXCLUDED_SESSIONS,
    FACE_LISTENING,
    FACE_SPEAKING,
    FEATURE_SETS,
    FLAGGED_SESSIONS,
    FULL,
    LLM,
    MODALITIES,
    PROSODY,
    TABLE_BUNDLE,
    TABLE_FACE,
    TEXT,
    TEXT_ONLY,
    TEXT_PLUS_NONVERBAL,
    TURNS,
    set_columns,
)
from vc_multimodal.exploratory.llm_ratings import analyze as language
from vc_multimodal.exploratory.llm_ratings.plan import ITEMS, REPO_ROOT, file_sha256
from vc_multimodal.exploratory.per_modality.analyze import BUNDLE_FEATURES, FACE_BACKEND_COLUMN
from vc_multimodal.exploratory.per_modality.plan import Inputs, Plan, load_plan
from vc_multimodal.modeling import evaluate
from vc_multimodal.modeling.evaluate import MODEL_ELASTIC_NET, MODEL_RANDOM_FOREST
from vc_multimodal.modeling.tiers import holm_adjust

PREREG = REPO_ROOT / "docs" / "prereg" / "fusion.md"
SESSIONS = [*range(1, 13), 43, 52]
ANALYSED = len(SESSIONS) - 1
#: What every MediaPipe face column of the synthetic bundle holds. No value of
#: the analysed table may be this.
BUNDLE_FACE_VALUE = 999.0
FACE_COLUMNS = (*MODALITIES[FACE_SPEAKING], *MODALITIES[FACE_LISTENING])
QUICK = analyze.Settings(
    stability_folds=2, stability_repeats=1, n_permutations=3, permutation_folds=3, n_resamples=200
)


class Tables:
    """Four synthetic tables, the plans that fix their digests, and invented labels."""

    def __init__(
        self,
        folder: Path,
        *,
        face_for_43: bool = False,
        drop_face_session: int | None = None,
        no_prosody_for: int | None = None,
    ) -> None:
        rng = np.random.default_rng(11)
        folder.mkdir(parents=True, exist_ok=True)
        n = len(SESSIONS)

        text = pd.DataFrame(rng.normal(size=(n, len(MODALITIES[TEXT]))), columns=MODALITIES[TEXT])
        text.insert(0, "session_id", SESSIONS)
        llm = pd.DataFrame(
            {item.feature: rng.integers(item.low, item.high + 1, n) for item in ITEMS}, dtype=float
        )
        llm.insert(0, "session_id", SESSIONS)

        bundle = pd.DataFrame({"session_id": SESSIONS})
        for column in (*MODALITIES[PROSODY], *MODALITIES[TURNS]):
            bundle[column] = rng.normal(size=n)
        for column in CONSTANT_TURNS:
            bundle[column] = 0.0
        # The bundle's MediaPipe face columns, under the same names as OpenFace's.
        for column in FACE_COLUMNS:
            bundle[column] = BUNDLE_FACE_VALUE
        if no_prosody_for is not None:
            bundle.loc[bundle["session_id"] == no_prosody_for, list(MODALITIES[PROSODY])] = np.nan

        face = pd.DataFrame({"session_id": SESSIONS})
        for column in FACE_COLUMNS:
            face[column] = rng.gamma(2.0, size=n)
        if not face_for_43:
            face.loc[face["session_id"] == 43, list(FACE_COLUMNS)] = np.nan
        if drop_face_session is not None:
            face = face[face["session_id"] != drop_face_session]
        face[FACE_BACKEND_COLUMN] = "openface"

        labels = pd.DataFrame({"session_id": [*SESSIONS, 99]})
        labels["K6"] = rng.integers(0, 24, len(labels)).astype(float)
        labels["SRS2"] = rng.integers(40, 90, len(labels)).astype(float)

        self.paths = analyze.Paths(
            text=folder / "text_features_original.csv",
            llm=folder / "llm_ratings_features.csv",
            bundle=folder / "bundle",
            face=folder / "face_openface_features.csv",
        )
        self.paths.bundle.mkdir(exist_ok=True)
        text.to_csv(self.paths.text, index=False)
        llm.to_csv(self.paths.llm, index=False)
        bundle.to_csv(self.paths.bundle / BUNDLE_FEATURES, index=False)
        face.to_csv(self.paths.face, index=False)
        self.labels = folder / "labels.csv"
        labels.to_csv(self.labels, index=False)
        self.face = face.set_index("session_id")

        self.language_plan: dict[str, tuple[str, str | None]] = {
            language.LLM_TABLE: (self.paths.llm.name, file_sha256(self.paths.llm)),
            language.TEXT_TABLE: (self.paths.text.name, file_sha256(self.paths.text)),
        }
        self.per_modality_plan: Plan = load_plan().model_copy(
            update={
                "inputs": Inputs(
                    base_features_sha256=file_sha256(self.paths.bundle / BUNDLE_FEATURES),
                    face_features_sha256=file_sha256(self.paths.face),
                )
            }
        )

    def load(self) -> analyze.Inputs:
        return analyze.load_inputs(
            self.paths, language_plan=self.language_plan, per_modality_plan=self.per_modality_plan
        )

    def run(self, out: Path) -> analyze.RunResult:
        return analyze.run(
            self.paths,
            self.labels,
            out,
            commit="c",
            jobs=1,
            settings=QUICK,
            language_plan=self.language_plan,
            per_modality_plan=self.per_modality_plan,
        )


@pytest.fixture
def few_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forest's size is not under test; its time is."""
    monkeypatch.setattr(evaluate, "_N_TREES", 5)


@pytest.fixture
def ran(tmp_path: Path, few_trees: None) -> Path:
    out = tmp_path / "out"
    result = Tables(tmp_path / "tables").run(out)
    assert result.n_by_target == {"K6": ANALYSED, "SRS2": ANALYSED}
    assert result.labels_only == 1
    assert (result.n_models, result.n_nulls, result.n_intervals) == (60, 4, 112)
    return out


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
def test_the_modalities_are_the_preregistered_ones() -> None:
    counts = {name: len(columns) for name, columns in MODALITIES.items()}
    assert counts == {
        TEXT: 20,
        LLM: 11,
        PROSODY: 12,
        TURNS: 10,
        FACE_SPEAKING: 9,
        FACE_LISTENING: 9,
    }
    assert len(set_columns(FULL)) == len(set(set_columns(FULL))) == 71

    per_modality = load_plan().modalities
    assert MODALITIES[PROSODY] == per_modality[PROSODY].features
    assert MODALITIES[TURNS] == tuple(
        c for c in per_modality[TURNS].features if c not in CONSTANT_TURNS
    )
    for window in (FACE_SPEAKING, FACE_LISTENING):
        assert MODALITIES[window] == per_modality[window].features
    assert MODALITIES[TEXT] == language.TEXT_COLUMNS

    prereg = PREREG.read_text(encoding="utf-8")
    for modality in (TEXT, LLM, PROSODY, TURNS):
        for column in MODALITIES[modality]:
            assert f"`{column}`" in prereg
    for column in CONSTANT_TURNS:
        assert f"`{column}`" in prereg


def test_the_feature_sets_and_their_sizes_are_the_preregistered_ones() -> None:
    sizes = {name: len(set_columns(name)) for name in FEATURE_SETS}
    assert sizes == {
        "full": 71,
        "text": 20,
        "full_minus_text": 51,
        "full_minus_llm": 60,
        "full_minus_prosody": 59,
        "full_minus_turns": 61,
        "full_minus_face_speaking": 62,
        "full_minus_face_listening": 62,
        "full_minus_face": 53,
        "full_minus_language": 40,
        "text_plus_llm": 31,
        "text_plus_prosody": 32,
        "text_plus_turns": 30,
        "text_plus_face_speaking": 29,
        "text_plus_face_listening": 29,
    }


def test_the_comparisons_are_the_plans_13_and_amendment_1() -> None:
    kinds = [c.kind for c in COMPARISONS]
    assert len(COMPARISONS) == 14
    assert {k: kinds.count(k) for k in kinds} == {
        "drop_one": 6,
        "grouped_drop": 2,
        "add_one": 5,
        "amendment_1": 1,
    }
    for comparison in COMPARISONS:
        larger, smaller = (set(FEATURE_SETS[s]) for s in (comparison.larger, comparison.smaller))
        assert smaller < larger, f"{comparison.name}: the smaller model must be nested"
    last = COMPARISONS[-1]
    assert (last.larger, last.smaller) == (TEXT_PLUS_NONVERBAL, TEXT_ONLY)
    assert FEATURE_SETS[TEXT_PLUS_NONVERBAL] == (
        TEXT,
        PROSODY,
        TURNS,
        FACE_SPEAKING,
        FACE_LISTENING,
    )

    amendment = PREREG.read_text(encoding="utf-8").split("## Amendment 1", 1)[1]
    assert "**112 intervals in all:** 14 comparisons" in amendment
    assert len(COMPARISONS) * 2 * 2 * 2 == 112


def test_the_settings_and_sessions_are_the_preregistered_ones() -> None:
    settings = analyze.Settings()
    assert settings.planned
    assert (settings.seed, settings.n_permutations, settings.n_resamples) == (
        20260923,
        1000,
        10_000,
    )
    assert (settings.stability_folds, settings.stability_repeats) == (5, 20)
    assert settings.permutation_folds == 5
    assert EXCLUDED_SESSIONS == load_plan().sessions.face_unavailable == (43,)
    assert FLAGGED_SESSIONS == load_plan().sessions.flagged == (52,)


def test_the_tables_are_read_by_the_digests_the_prereg_names() -> None:
    prereg = PREREG.read_text(encoding="utf-8")
    planned = language.load_inputs_plan()
    inputs = load_plan().inputs
    for digest in (
        planned[language.TEXT_TABLE][1],
        planned[language.LLM_TABLE][1],
        inputs.base_features_sha256,
        inputs.face_features_sha256,
    ):
        assert digest is not None
        assert f"`{digest}`" in prereg


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def test_no_face_column_comes_from_the_bundle(tmp_path: Path) -> None:
    tables = Tables(tmp_path)
    inputs = tables.load()
    assert all(inputs.sources[c] == TABLE_FACE for c in FACE_COLUMNS)
    values = inputs.frame.set_index("session_id")[list(FACE_COLUMNS)]
    assert not (values == BUNDLE_FACE_VALUE).to_numpy().any()
    pd.testing.assert_frame_equal(values, tables.face.loc[values.index, list(FACE_COLUMNS)])

    tampered = {**inputs.sources, FACE_COLUMNS[0]: TABLE_BUNDLE}
    with pytest.raises(analyze.AnalyzeError, match=f"{FACE_COLUMNS[0]} from the bundle"):
        analyze.check_sources(tampered)


def test_a_missing_or_unplanned_column_is_refused(tmp_path: Path) -> None:
    sources = Tables(tmp_path).load().sources
    missing = {c: t for c, t in sources.items() if c != MODALITIES[TURNS][0]}
    with pytest.raises(analyze.AnalyzeError, match="from the no table"):
        analyze.check_sources(missing)
    with pytest.raises(analyze.AnalyzeError, match="not a planned feature"):
        analyze.check_sources({**sources, CONSTANT_TURNS[0]: TABLE_BUNDLE})


def test_session_43_is_left_out_and_52_kept(tmp_path: Path) -> None:
    inputs = Tables(tmp_path).load()
    sessions = list(inputs.frame["session_id"])
    assert 43 not in sessions
    assert 52 in sessions
    assert len(sessions) == ANALYSED
    assert list(inputs.frame.columns) == ["session_id", *set_columns(FULL)]
    assert not set(CONSTANT_TURNS) & set(inputs.frame.columns)


def test_a_face_for_session_43_is_refused(tmp_path: Path) -> None:
    with pytest.raises(analyze.AnalyzeError, match="QC note was not applied"):
        Tables(tmp_path, face_for_43=True).load()


def test_tables_covering_other_sessions_are_refused(tmp_path: Path) -> None:
    with pytest.raises(analyze.AnalyzeError, match=re.escape("missing [5]")):
        Tables(tmp_path, drop_face_session=5).load()


def test_a_session_with_no_value_for_a_modality_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(analyze.AnalyzeError, match=re.escape("[7] have no prosody value")):
        Tables(tmp_path, no_prosody_for=7).load()


def test_a_table_that_is_not_the_planned_one_is_refused(tmp_path: Path) -> None:
    tables = Tables(tmp_path)
    tables.language_plan[language.TEXT_TABLE] = (tables.paths.text.name, "0" * 64)
    with pytest.raises(analyze.AnalyzeError, match="not the planned table"):
        tables.load()
    tables = Tables(tmp_path / "again")
    other = tables.per_modality_plan.inputs.model_copy(update={"face_features_sha256": "0" * 64})
    tables.per_modality_plan = tables.per_modality_plan.model_copy(update={"inputs": other})
    with pytest.raises(analyze.AnalyzeError, match="not the planned OpenFace face table"):
        tables.load()


# ---------------------------------------------------------------------------
# Tests and intervals, without fitting
# ---------------------------------------------------------------------------
def test_the_wilcoxon_difference_is_full_minus_text() -> None:
    rng = np.random.default_rng(2)
    text_errors = rng.gamma(2.0, size=30) + 1.0
    errors = {}
    rows = []
    for model, role in analyze.MODELS:
        for target in ("K6", "SRS2"):
            errors[(TEXT_ONLY, target, model)] = text_errors
            errors[(FULL, target, model)] = text_errors * 0.5
            rows.append(
                {
                    "feature_set": FULL,
                    "target": target,
                    "model": model,
                    "role": role,
                    "n": 30,
                    "perm_scheme": "5-fold",
                    "perm_observed_r2": 0.1,
                    "perm_p": 0.04,
                }
            )
    table = analyze.primary_tests(pd.DataFrame(rows), errors)
    wilcoxon = table[table["test"] == "full_vs_text"]
    assert (wilcoxon["statistic"] < 0).all(), "the full model was closer, so negative"
    primary = table["role"] == "primary"
    assert list(table.loc[primary, "model"]) == [MODEL_ELASTIC_NET] * 4
    np.testing.assert_allclose(table.loc[primary, "p_holm"], holm_adjust(table.loc[primary, "p"]))
    assert table.loc[~primary, "p_holm"].isna().all()
    assert list(table.loc[~primary, "model"]) == [MODEL_RANDOM_FOREST] * 4


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_the_run_writes_every_planned_summary(ran: Path) -> None:
    models = pd.read_csv(ran / analyze.MODELS_FILE)
    assert len(models) == 60
    assert (models["n"] == ANALYSED).all()
    for name in FEATURE_SETS:
        assert set(models.loc[models["feature_set"] == name, "n_features"]) == {
            len(set_columns(name))
        }
    stable = models["feature_set"].isin([FULL, TEXT_ONLY])
    assert models.loc[stable, "kfold_r2_mean"].notna().all()
    assert models.loc[~stable, "kfold_r2_mean"].isna().all()
    permuted = models["feature_set"] == FULL
    assert (models.loc[permuted, "n_permutations"] == QUICK.n_permutations).all()
    assert (models.loc[~permuted, "n_permutations"] == 0).all()

    primary = pd.read_csv(ran / analyze.PRIMARY_FILE)
    assert len(primary) == 8
    first = primary.iloc[:4]
    assert list(first["test"]) == ["full_vs_text"] * 2 + ["full_vs_chance"] * 2
    np.testing.assert_allclose(first["p_holm"], holm_adjust(first["p"].tolist()))
    assert primary.iloc[4:]["p_holm"].isna().all()

    table = pd.read_csv(ran / analyze.CONTRIBUTIONS_FILE)
    assert len(table) == 112
    assert set(table["comparison"]) == {c.name for c in COMPARISONS}
    assert (table["ci_low"] <= table["ci_high"]).all()
    assert (table["n_resamples_used"] == QUICK.n_resamples).all()
    indexed = models.set_index(["feature_set", "target", "model"])
    for row in table.itertuples():
        column = "loo_r2" if row.metric == "delta_r2" else "loo_mae"
        expected = (
            indexed.loc[(row.larger, row.target, row.model), column]
            - indexed.loc[(row.smaller, row.target, row.model), column]
        )
        assert row.estimate == pytest.approx(expected, abs=1e-9)

    record = json.loads((ran / analyze.RUN_FILE).read_text(encoding="utf-8"))
    assert record["planned_settings"] is False
    assert record["left_out"] == [43]
    assert record["n_intervals"] == 112
    summary = (ran / analyze.SUMMARY_FILE).read_text(encoding="utf-8")
    for text in ("a test run", "not comparable", "too narrow", "not a leave-one-out p"):
        assert text in summary


@pytest.mark.slow
def test_no_label_feature_or_per_session_value_is_written(ran: Path) -> None:
    written = sorted(p.name for p in ran.iterdir())
    assert written == sorted(
        [
            analyze.CONTRIBUTIONS_FILE,
            analyze.MODELS_FILE,
            analyze.PRIMARY_FILE,
            analyze.RUN_FILE,
            analyze.SUMMARY_FILE,
        ]
    )
    banned = {"session_id", "K6", "SRS2", "prediction", "error", "residual", *set_columns(FULL)}
    for name in (analyze.PRIMARY_FILE, analyze.MODELS_FILE, analyze.CONTRIBUTIONS_FILE):
        assert not set(pd.read_csv(ran / name).columns) & banned


def test_a_used_results_folder_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "models.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(analyze.AnalyzeError, match="already holds files"):
        Tables(tmp_path / "tables").run(out)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------
def test_analyze_needs_the_labels_and_estimate_reads_none() -> None:
    args = _parser().parse_args(["analyze", "--labels", "l.csv"])
    assert (args.labels, args.out, args.bundle, args.face_table) == (
        Path("l.csv"),
        None,
        None,
        None,
    )
    with pytest.raises(SystemExit):
        _parser().parse_args(["analyze"])
    estimate = _parser().parse_args(["estimate", "--probe", "2"])
    assert not hasattr(estimate, "labels")
    assert estimate.probe == 2
