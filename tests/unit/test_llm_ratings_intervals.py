"""Amendment 1 of the LLM ratings, on synthetic tables and invented labels."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.unit.test_llm_ratings_analysis import QUICK, labels, tables
from vc_multimodal.exploratory.llm_ratings import analyze, intervals
from vc_multimodal.exploratory.llm_ratings.__main__ import _parser
from vc_multimodal.exploratory.llm_ratings.plan import FEATURES, REPO_ROOT
from vc_multimodal.modeling.evaluate import MODEL_ELASTIC_NET, score_predictions

RESAMPLES = 300

Primary = tuple[Path, Path, Path, dict[str, tuple[str, str | None]], Path]


@pytest.fixture(scope="module")
def primary(tmp_path_factory: pytest.TempPathFactory) -> Primary:
    """Synthetic inputs, and a primary analysis run on them."""
    root = tmp_path_factory.mktemp("primary")
    llm_path, text_path, planned = tables(root / "tables")
    label_path = labels(root / "labels.csv")
    results = root / "results"
    analyze.run(
        llm_path,
        text_path,
        label_path,
        results,
        commit="c",
        jobs=2,
        settings=QUICK,
        planned=planned,
    )
    return llm_path, text_path, label_path, planned, results


def run(primary: Primary, out: Path, results: Path | None = None) -> intervals.IntervalsResult:
    llm_path, text_path, label_path, planned, primary_results = primary
    return intervals.run(
        llm_path,
        text_path,
        label_path,
        results or primary_results,
        out,
        commit="c",
        jobs=2,
        settings=QUICK,
        n_resamples=RESAMPLES,
        planned=planned,
    )


@pytest.mark.slow
def test_the_intervals_reproduce_the_primary_run_and_summarise_it(
    primary: Primary, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = run(primary, out)
    assert result.n_gated == 8
    assert result.report_lines()[0].startswith("gate: 8 recomputed")

    table = pd.read_csv(out / intervals.INTERVALS_FILE)
    assert len(table) == 8, "4 comparisons, each with delta R² and delta MAE"
    assert set(table["metric"]) == {intervals.DELTA_R2, intervals.DELTA_MAE}
    assert (table["n_resamples_used"] == RESAMPLES).all()
    assert (table["ci_low"] <= table["ci_high"]).all()

    models = pd.read_csv(primary[4] / analyze.MODELS_FILE).set_index(
        ["feature_set", "target", "model"]
    )
    for row in table.itertuples():
        column = "loo_r2" if row.metric == intervals.DELTA_R2 else "loo_mae"
        expected = (
            models.loc[(analyze.TEXT_LLM, row.target, row.model), column]
            - models.loc[(analyze.TEXT, row.target, row.model), column]
        )
        assert row.estimate == pytest.approx(expected, abs=1e-9)

    summary = (out / intervals.SUMMARY_FILE).read_text(encoding="utf-8")
    assert "**Post hoc**" in summary
    assert "a test run" in summary
    assert "too narrow" in summary


@pytest.mark.slow
def test_no_label_prediction_or_per_session_value_is_written(
    primary: Primary, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run(primary, out)
    banned = {"session_id", "K6", "SRS2", "prediction", "error", "residual", *FEATURES}
    assert not set(pd.read_csv(out / intervals.INTERVALS_FILE).columns) & banned
    written = sorted(p.name for p in out.iterdir())
    assert written == [intervals.INTERVALS_FILE, intervals.SUMMARY_FILE]


@pytest.mark.slow
def test_a_primary_run_that_is_not_reproduced_stops_before_writing(
    primary: Primary, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    shutil.copytree(primary[4], results)
    models = pd.read_csv(results / analyze.MODELS_FILE)
    first = (models["feature_set"] == analyze.TEXT_LLM) & (models["model"] == MODEL_ELASTIC_NET)
    models.loc[first, "loo_r2"] += 1e-7
    models.to_csv(results / analyze.MODELS_FILE, index=False)
    out = tmp_path / "out"
    with pytest.raises(intervals.GateError, match=re.escape("loo_r2 differs by 1.0e-07")):
        run(primary, out, results)
    assert not out.exists()


# ---------------------------------------------------------------------------
# The gate, without fitting
# ---------------------------------------------------------------------------
def loo(r2: float = 0.2, mae: float = 1.5, n: int = 10) -> intervals.Loo:
    return intervals.Loo(analyze.TEXT, "K6", MODEL_ELASTIC_NET, n, r2, mae, np.zeros(n))


def models_table(r2: float = 0.2, mae: float = 1.5, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": analyze.TEXT,
                "target": "K6",
                "model": MODEL_ELASTIC_NET,
                "n": n,
                "loo_r2": r2,
                "loo_mae": mae,
            }
        ]
    )


def test_the_gate_passes_within_the_tolerance_and_stops_beyond_it() -> None:
    intervals.gate(models_table(r2=0.2 + 5e-10), [loo()])
    with pytest.raises(intervals.GateError, match="loo_r2 differs"):
        intervals.gate(models_table(r2=0.2 + 2e-9), [loo()])
    with pytest.raises(intervals.GateError, match="loo_mae differs"):
        intervals.gate(models_table(mae=1.6), [loo()])


def test_the_gate_checks_n_and_a_missing_model() -> None:
    with pytest.raises(intervals.GateError, match=re.escape("n 10 here, 9 in models.csv")):
        intervals.gate(models_table(n=9), [loo()])
    with pytest.raises(intervals.GateError, match="0 row"):
        intervals.gate(models_table().assign(target="SRS2"), [loo()])


def test_the_gate_accepts_a_missing_r2_only_on_both_sides() -> None:
    intervals.gate(models_table(r2=float("nan")), [loo(r2=float("nan"))])
    with pytest.raises(intervals.GateError, match="loo_r2"):
        intervals.gate(models_table(r2=float("nan")), [loo()])


def primary_folder(folder: Path, inputs: analyze.Inputs, settings: analyze.Settings) -> Path:
    folder.mkdir(parents=True)
    record = {"settings": asdict(settings), "inputs": inputs.digests}
    (folder / analyze.RUN_FILE).write_text(json.dumps(record), encoding="utf-8")
    models_table().to_csv(folder / analyze.MODELS_FILE, index=False)
    return folder


def test_a_primary_run_with_other_settings_or_tables_is_refused(tmp_path: Path) -> None:
    llm_path, text_path, planned = tables(tmp_path / "tables")
    inputs = analyze.load_inputs(llm_path, text_path, planned)
    folder = primary_folder(tmp_path / "results", inputs, QUICK)
    assert len(intervals.check_primary_run(folder, inputs, QUICK)) == 1
    with pytest.raises(intervals.GateError, match="other settings"):
        intervals.check_primary_run(folder, inputs, analyze.Settings())
    other = analyze.Inputs(frame=inputs.frame, digests={**inputs.digests, "x.csv": "0" * 64})
    with pytest.raises(intervals.GateError, match="other input tables"):
        intervals.check_primary_run(folder, other, QUICK)
    (folder / analyze.RUN_FILE).unlink()
    with pytest.raises(intervals.GateError, match="not found"):
        intervals.check_primary_run(folder, inputs, QUICK)


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------
def invented(n: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An outcome and two sets of predictions, all made up."""
    rng = np.random.default_rng(3)
    truth = rng.normal(10, 3, n)
    return truth, truth + rng.normal(0, 2, n), truth + rng.normal(0, 1, n)


def test_the_unresampled_sessions_give_the_full_sample_differences() -> None:
    truth, text, both = invented()
    identity = np.arange(truth.size)[None, :]
    delta_r2, delta_mae = intervals.bootstrap_differences(
        truth, np.abs(truth - text), np.abs(truth - both), identity
    )
    full_text, full_both = score_predictions(truth, text), score_predictions(truth, both)
    assert delta_r2[0] == pytest.approx(full_both.r2 - full_text.r2, abs=1e-12)
    assert delta_mae[0] == pytest.approx(full_both.mae - full_text.mae, abs=1e-12)


def test_each_resample_is_scored_as_score_predictions_scores_it() -> None:
    truth, text, both = invented()
    indices = intervals.resample_indices(truth.size, 5, seed=1)
    delta_r2, delta_mae = intervals.bootstrap_differences(
        truth, np.abs(truth - text), np.abs(truth - both), indices
    )
    for k, rows in enumerate(indices):
        a = score_predictions(truth[rows], text[rows])
        b = score_predictions(truth[rows], both[rows])
        assert delta_r2[k] == pytest.approx(b.r2 - a.r2, abs=1e-12)
        assert delta_mae[k] == pytest.approx(b.mae - a.mae, abs=1e-12)


def test_a_resample_with_a_constant_outcome_has_no_r2_and_is_counted() -> None:
    truth, text, both = invented()
    indices = np.array([np.arange(truth.size), np.zeros(truth.size, dtype=np.int64)])
    delta_r2, delta_mae = intervals.bootstrap_differences(
        truth, np.abs(truth - text), np.abs(truth - both), indices
    )
    assert np.isfinite(delta_r2[0])
    assert np.isnan(delta_r2[1])
    assert np.isfinite(delta_mae).all()
    assert intervals.percentile_interval(delta_r2)[2] == 1


def test_the_interval_is_the_2_5th_and_97_5th_percentiles() -> None:
    values = np.random.default_rng(4).normal(size=1000)
    low, high, used = intervals.percentile_interval(np.append(values, np.nan))
    np.testing.assert_allclose([low, high], np.quantile(values, [0.025, 0.975]))
    assert used == 1000


def test_the_resamples_are_fixed_by_the_seed() -> None:
    a = intervals.resample_indices(62, 100, seed=20260923)
    assert a.shape == (100, 62)
    assert a.min() >= 0
    assert a.max() < 62
    np.testing.assert_array_equal(a, intervals.resample_indices(62, 100, seed=20260923))
    assert not np.array_equal(a, intervals.resample_indices(62, 100, seed=1))


# ---------------------------------------------------------------------------
# The plan and the command
# ---------------------------------------------------------------------------
def test_the_settings_are_the_amendments() -> None:
    assert intervals.N_RESAMPLES == 10_000
    assert intervals.LEVEL == 0.95
    assert intervals.TOLERANCE == 1e-9
    assert analyze.Settings().seed == 20260923
    prereg = (REPO_ROOT / "docs" / "prereg" / "llm-ratings.md").read_text(encoding="utf-8")
    amendment = prereg.split("## Amendment 1", 1)[1]
    for text in ("**Post hoc.**", "10,000", "1e-9", "20260923", "95% percentile", "No p-value"):
        assert text in amendment


def test_the_command_needs_the_labels_and_the_primary_results() -> None:
    args = _parser().parse_args(["intervals", "--labels", "l.csv", "--results", "r"])
    assert (args.labels, args.results, args.out) == (Path("l.csv"), Path("r"), None)
    with pytest.raises(SystemExit):
        _parser().parse_args(["intervals", "--labels", "l.csv"])
