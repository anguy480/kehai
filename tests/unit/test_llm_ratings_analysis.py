"""Step 3 of the LLM ratings, on synthetic tables and invented labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vc_multimodal.exploratory.llm_ratings import analyze
from vc_multimodal.exploratory.llm_ratings.plan import FEATURES, ITEMS, REPO_ROOT, file_sha256
from vc_multimodal.modeling.tiers import holm_adjust

SESSIONS = [1, 3, 6, 7, 8, 10, 12, 17, 18, 20, 22, 23]
NOT_RATED = 6
QUICK = analyze.Settings(stability_folds=2, stability_repeats=1, n_permutations=3)

Setup = tuple[Path, Path, Path, dict[str, tuple[str, str | None]]]


def tables(
    folder: Path, *, partly_rated: bool = False, drop_text_session: bool = False
) -> tuple[Path, Path, dict[str, tuple[str, str | None]]]:
    """A rater table (one session not rated), a text table, and a plan of their digests."""
    rng = np.random.default_rng(5)
    folder.mkdir(parents=True, exist_ok=True)
    llm = pd.DataFrame(
        {item.feature: rng.integers(item.low, item.high + 1, len(SESSIONS)) for item in ITEMS},
        dtype=float,
    )
    llm.insert(0, "session_id", SESSIONS)
    llm.loc[llm["session_id"] == NOT_RATED, list(FEATURES)] = np.nan
    if partly_rated:
        llm.loc[0, FEATURES[0]] = np.nan
    text = pd.DataFrame(
        rng.normal(size=(len(SESSIONS), len(analyze.TEXT_COLUMNS))), columns=analyze.TEXT_COLUMNS
    )
    text.insert(0, "session_id", SESSIONS)
    if drop_text_session:
        text = text.iloc[1:]
    llm_path, text_path = folder / "llm_ratings_features.csv", folder / "text_features_original.csv"
    llm.to_csv(llm_path, index=False)
    text.to_csv(text_path, index=False)
    planned: dict[str, tuple[str, str | None]] = {
        analyze.LLM_TABLE: (llm_path.name, file_sha256(llm_path)),
        analyze.TEXT_TABLE: (text_path.name, file_sha256(text_path)),
    }
    return llm_path, text_path, planned


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
    llm_path, text_path, planned = tables(tmp_path / "tables")
    return llm_path, text_path, labels(tmp_path / "labels.csv"), planned


def run(setup: Setup, out: Path) -> analyze.RunResult:
    llm_path, text_path, label_path, planned = setup
    return analyze.run(
        llm_path, text_path, label_path, out, commit="c", jobs=2, settings=QUICK, planned=planned
    )


@pytest.mark.slow
def test_the_run_writes_every_planned_summary(setup: Setup, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run(setup, out)
    rated = len(SESSIONS) - 1
    assert result.n_by_target == {"K6": rated, "SRS2": rated}
    assert result.failed == (NOT_RATED,)
    assert result.labels_only == 1

    primary = pd.read_csv(out / analyze.PRIMARY_FILE)
    assert list(primary["analysis"]) == ["zero_shot"] * 2 + ["llm_alone"] * 2 + ["added_value"] * 2
    np.testing.assert_allclose(primary["p_holm"], holm_adjust(primary["p"].tolist()))
    assert (primary["n"] == rated).all()

    models = pd.read_csv(out / analyze.MODELS_FILE)
    assert len(models) == 12
    assert set(models["feature_set"]) == {"llm", "text", "text_llm"}
    assert set(models.loc[models["feature_set"] == "text_llm", "n_features"]) == {31}
    assert (models["n"] == rated).all(), "the text arm leaves out the unrated session too"

    comparisons = pd.read_csv(out / analyze.COMPARISONS_FILE)
    assert len(comparisons) == 4
    assert "p_holm" not in comparisons.columns

    univariate = pd.read_csv(out / analyze.UNIVARIATE_FILE)
    assert len(univariate) == 2 * 11
    for _, block in univariate.groupby("target"):
        np.testing.assert_allclose(block["q"], analyze.benjamini_hochberg(block["p"].tolist()))

    record = json.loads((out / analyze.RUN_FILE).read_text(encoding="utf-8"))
    assert record["planned_settings"] is False
    assert record["not_rated"] == [NOT_RATED]
    assert "a test run" in (out / analyze.SUMMARY_FILE).read_text(encoding="utf-8")


@pytest.mark.slow
def test_no_label_rating_or_per_session_value_is_written(setup: Setup, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run(setup, out)
    banned = {"session_id", "K6", "SRS2", "prediction", "error", "residual", *FEATURES}
    for name in (
        analyze.PRIMARY_FILE,
        analyze.MODELS_FILE,
        analyze.COMPARISONS_FILE,
        analyze.UNIVARIATE_FILE,
    ):
        assert not set(pd.read_csv(out / name).columns) & banned


def test_zero_shot_correlates_each_total_with_its_outcome(setup: Setup) -> None:
    llm_path, text_path, label_path, planned = setup
    inputs = analyze.load_inputs(llm_path, text_path, planned)
    cohort = analyze.load_cohort(inputs, label_path)
    rated = inputs.frame.iloc[cohort.rows]
    rows = analyze.zero_shot(inputs.frame, cohort)
    for row, (target, scale) in zip(rows, analyze.ZERO_SHOT.items(), strict=True):
        expected = pd.Series(analyze.scale_total(rated, scale)).corr(
            pd.Series(cohort.truth[target]), method="spearman"
        )
        assert row["target"] == target
        assert row["statistic"] == pytest.approx(expected)


def test_the_totals_are_the_sums_of_their_sets() -> None:
    frame = pd.DataFrame([{item.feature: 1.0 for item in ITEMS}])
    assert analyze.scale_total(frame, "K")[0] == 6
    assert analyze.scale_total(frame, "S")[0] == 5


def test_a_table_that_is_not_the_planned_one_is_refused(setup: Setup, tmp_path: Path) -> None:
    llm_path, text_path, label_path, planned = setup
    planned = {**planned, analyze.TEXT_TABLE: (text_path.name, "0" * 64)}
    with pytest.raises(analyze.AnalyzeError, match="not the planned table"):
        analyze.run(llm_path, text_path, label_path, tmp_path / "out", commit="c", planned=planned)


def test_an_unfixed_digest_is_refused(setup: Setup, tmp_path: Path) -> None:
    llm_path, text_path, label_path, planned = setup
    planned = {**planned, analyze.LLM_TABLE: (llm_path.name, None)}
    with pytest.raises(analyze.AnalyzeError, match="not fixed in the plan"):
        analyze.run(llm_path, text_path, label_path, tmp_path / "out", commit="c", planned=planned)


def test_a_partly_rated_session_is_refused(tmp_path: Path) -> None:
    llm_path, text_path, planned = tables(tmp_path, partly_rated=True)
    with pytest.raises(analyze.AnalyzeError, match="partly rated"):
        analyze.load_inputs(llm_path, text_path, planned)


def test_tables_covering_different_sessions_are_refused(tmp_path: Path) -> None:
    llm_path, text_path, planned = tables(tmp_path, drop_text_session=True)
    with pytest.raises(analyze.AnalyzeError, match="different sessions"):
        analyze.load_inputs(llm_path, text_path, planned)


def test_labels_without_both_outcomes_are_refused(setup: Setup, tmp_path: Path) -> None:
    llm_path, text_path, _, planned = setup
    label_path = labels(tmp_path / "k6_only.csv", columns=("K6",))
    with pytest.raises(analyze.AnalyzeError, match="SRS2"):
        analyze.run(llm_path, text_path, label_path, tmp_path / "out", commit="c", planned=planned)


def test_the_shipped_plan_names_both_tables_and_fixes_the_text_digest() -> None:
    plan = analyze.load_inputs_plan()
    assert plan[analyze.LLM_TABLE][0] == "llm_ratings_features.csv"
    text_file, text_digest = plan[analyze.TEXT_TABLE]
    assert text_file == "text_features_original.csv"
    prereg = (REPO_ROOT / "docs" / "prereg" / "llm-ratings.md").read_text(encoding="utf-8")
    assert text_digest is not None
    assert re.fullmatch(r"[0-9a-f]{64}", text_digest)
    assert text_digest in prereg, "the plan reads the text table the pre-registration names"
    llm_digest = plan[analyze.LLM_TABLE][1]
    assert llm_digest is None or re.fullmatch(r"[0-9a-f]{64}", llm_digest)


def test_a_used_results_folder_is_refused_but_its_log_is_not(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / analyze.LOG_DIRNAME).mkdir(parents=True)
    analyze.refuse_used_folder(out)
    (out / "models.csv").write_text("x", encoding="utf-8")
    with pytest.raises(analyze.AnalyzeError, match="already holds files"):
        analyze.refuse_used_folder(out)


def test_holm_skips_a_missing_p_and_keeps_it_missing() -> None:
    adjusted = analyze.holm_finite([0.01, float("nan"), 0.04])
    assert np.isnan(adjusted[1])
    np.testing.assert_allclose([adjusted[0], adjusted[2]], holm_adjust([0.01, 0.04]))


def test_benjamini_hochberg_matches_the_textbook_values() -> None:
    q = analyze.benjamini_hochberg([0.01, 0.04, 0.03, 0.20, float("nan")])
    np.testing.assert_allclose(q[:4], [0.04, 0.0533333, 0.0533333, 0.20], rtol=1e-5)
    assert np.isnan(q[4])
