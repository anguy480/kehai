"""Step 3 of the tokenization rerun, on synthetic tables and invented labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vc_multimodal.exploratory.tokenization import analyze
from vc_multimodal.exploratory.tokenization.run import NLP_COLUMNS, file_sha256
from vc_multimodal.exploratory.tokenization.token_features import FEATURES
from vc_multimodal.modeling.tiers import holm_adjust

SESSIONS = [1, 3, 6, 7, 8, 10, 12, 17, 18, 20, 22, 23]
QUICK = analyze.Settings(stability_folds=2, stability_repeats=1, n_permutations=3)

Setup = tuple[Path, dict[str, tuple[str, str | None]], Path]


def tables(folder: Path, *, change_unchanged: bool = False) -> dict[str, tuple[str, str | None]]:
    """Three step 2 tables, and a plan holding their digests."""
    rng = np.random.default_rng(5)
    folder.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame(rng.normal(size=(len(SESSIONS), len(NLP_COLUMNS))), columns=NLP_COLUMNS)
    base.insert(0, "session_id", SESSIONS)
    planned: dict[str, tuple[str, str | None]] = {}
    for name, filename in analyze.FEATURE_SET_FILES.items():
        frame = base.copy()
        if name != analyze.ORIGINAL:
            for column in FEATURES:
                frame[column] = rng.normal(size=len(SESSIONS))
            if change_unchanged:
                frame["Bert"] = frame["Bert"] + 1.0
        path = folder / filename
        frame.to_csv(path, index=False)
        planned[name] = (filename, file_sha256(path))
    return planned


def labels(path: Path, *, columns: tuple[str, ...] = ("K6", "SRS2")) -> Path:
    """Invented scores, not anyone's."""
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"session_id": [*SESSIONS, 99]})
    for column in columns:
        frame[column] = rng.integers(0, 24, size=len(frame)).astype(float)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    folder = tmp_path / "tokenization"
    return folder, tables(folder), labels(tmp_path / "labels.csv")


def run(setup: Setup, out: Path) -> analyze.RunResult:
    folder, planned, label_path = setup
    return analyze.run(folder, label_path, out, commit="c", jobs=2, settings=QUICK, planned=planned)


@pytest.mark.slow
def test_the_run_writes_every_planned_summary(setup: Setup, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run(setup, out)
    assert result.n_by_target == {"K6": len(SESSIONS), "SRS2": len(SESSIONS)}
    assert result.labels_only == 1

    models = pd.read_csv(out / analyze.MODELS_FILE)
    assert len(models) == 12
    assert set(models["feature_set"]) == {"original", "new_b", "new_c"}
    assert (models["n_permutations"] == QUICK.n_permutations).all()

    comparisons = pd.read_csv(out / analyze.COMPARISONS_FILE)
    assert len(comparisons) == 8
    primary = comparisons["role"] == "primary"
    assert primary.sum() == 4
    assert comparisons.loc[~primary, "p_holm"].isna().all()
    expected = holm_adjust(comparisons.loc[primary, "p"].tolist())
    np.testing.assert_allclose(comparisons.loc[primary, "p_holm"], expected)

    univariate = pd.read_csv(out / analyze.UNIVARIATE_FILE)
    assert len(univariate) == 3 * 2 * 20
    assert univariate.groupby(["feature_set", "target"]).size().eq(20).all()

    record = json.loads((out / analyze.RUN_FILE).read_text(encoding="utf-8"))
    assert record["planned_settings"] is False
    assert "a test run" in (out / analyze.SUMMARY_FILE).read_text(encoding="utf-8")


@pytest.mark.slow
def test_no_label_or_per_session_value_is_written(setup: Setup, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run(setup, out)
    for name in (analyze.MODELS_FILE, analyze.COMPARISONS_FILE, analyze.UNIVARIATE_FILE):
        columns = set(pd.read_csv(out / name).columns)
        assert not columns & {"session_id", "K6", "SRS2", "prediction", "error", "residual"}


def test_a_table_that_is_not_the_planned_one_is_refused(setup: Setup, tmp_path: Path) -> None:
    folder, planned, label_path = setup
    planned = {**planned, "new_b": (planned["new_b"][0], "0" * 64)}
    with pytest.raises(analyze.AnalyzeError, match="not the planned table"):
        analyze.run(folder, label_path, tmp_path / "out", commit="c", planned=planned)


def test_an_unfixed_digest_is_refused(setup: Setup, tmp_path: Path) -> None:
    folder, planned, label_path = setup
    planned = {**planned, "new_c": (planned["new_c"][0], None)}
    with pytest.raises(analyze.AnalyzeError, match="not fixed in the plan"):
        analyze.run(folder, label_path, tmp_path / "out", commit="c", planned=planned)


def test_the_shipped_plan_fixes_the_step_2_digests() -> None:
    plan = analyze.load_plan()
    assert {name: f for name, (f, _) in plan.items()} == analyze.FEATURE_SET_FILES
    for name in ("original", "new_b"):
        digest = plan[name][1]
        assert digest is not None
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_variants_must_share_the_unchanged_features(tmp_path: Path) -> None:
    folder = tmp_path / "tokenization"
    planned = tables(folder, change_unchanged=True)
    with pytest.raises(analyze.AnalyzeError, match="12 features"):
        analyze.load_feature_sets(folder, planned)


def test_labels_without_both_outcomes_are_refused(setup: Setup, tmp_path: Path) -> None:
    folder, planned, _ = setup
    label_path = labels(tmp_path / "k6_only.csv", columns=("K6",))
    with pytest.raises(analyze.AnalyzeError, match="SRS2"):
        analyze.run(folder, label_path, tmp_path / "out", commit="c", planned=planned)


def test_a_used_results_folder_is_refused_but_its_log_is_not(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / analyze.LOG_DIRNAME).mkdir(parents=True)
    analyze.refuse_used_folder(out)
    (out / "models.csv").write_text("x", encoding="utf-8")
    with pytest.raises(analyze.AnalyzeError, match="already holds files"):
        analyze.refuse_used_folder(out)


def test_benjamini_hochberg_matches_the_textbook_values() -> None:
    q = analyze.benjamini_hochberg([0.01, 0.04, 0.03, 0.20, float("nan")])
    np.testing.assert_allclose(q[:4], [0.04, 0.0533333, 0.0533333, 0.20], rtol=1e-5)
    assert np.isnan(q[4])
