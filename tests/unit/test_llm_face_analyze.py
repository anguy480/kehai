"""The exploratory LLM face analysis, on synthetic sessions, ratings and labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vc_multimodal.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from vc_multimodal.exploratory.llm_face import analyze, describe, rate
from vc_multimodal.io_utils import read_csv, write_csv
from vc_multimodal.paths import DataRoots

N = 20


class TestIcc:
    def test_identical_runs_agree_perfectly(self) -> None:
        ratings = np.repeat(np.arange(1, 11, dtype=float)[:, None], 3, axis=1)
        assert analyze.icc_2_1(ratings) == pytest.approx(1.0)

    def test_unrelated_runs_agree_about_as_well_as_chance(self) -> None:
        rng = np.random.default_rng(0)
        icc = analyze.icc_2_1(rng.integers(1, 8, size=(200, 3)).astype(float))
        assert icc is not None
        assert abs(icc) < 0.15

    def test_sessions_that_never_vary_have_no_icc(self) -> None:
        assert analyze.icc_2_1(np.full((10, 3), 4.0)) is None


def runs_table(rng: np.random.Generator, sessions: list[int]) -> pd.DataFrame:
    rows = []
    for session_id in sessions:
        base = rng.integers(1, 8, size=len(rate.SCALES))
        rows.extend(
            {"session_id": session_id, "run": run, **dict(zip(rate.SCALES, base, strict=True))}
            for run in (1, 2, 3)
        )
    return pd.DataFrame(rows)


class TestScores:
    def test_runs_average_into_one_row_per_session(self) -> None:
        runs = runs_table(np.random.default_rng(1), [1, 2, 3])
        means = analyze.mean_scores(runs)
        assert list(means["session_id"]) == [1, 2, 3]
        assert all(c.startswith(analyze.LLM_PREFIX) for c in means.columns[1:])

    def test_identical_runs_are_counted_as_identical(self) -> None:
        table = analyze.stability_table(runs_table(np.random.default_rng(2), list(range(10))))
        assert (table["share_identical_across_runs"] == 1.0).all()


@pytest.fixture
def inputs(roots: DataRoots, tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(3)
    sessions = list(range(1, N + 1))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    features = pd.DataFrame({"session_id": sessions})
    for window in ("face_speaking", "face_listening"):
        for unit in ("au01_mean", "au12_mean"):
            features[f"{window}__{unit}"] = rng.uniform(0, 1, N)
    features.loc[features["session_id"] == 5, [c for c in features.columns if "face" in c]] = np.nan
    write_csv(bundle / "features.csv", features)
    write_csv(
        bundle / "text_features.csv",
        pd.DataFrame({"session_id": sessions, "text__jaccard": rng.uniform(0, 1, N)}),
    )
    runs = runs_table(rng, [s for s in sessions if s != 5])
    write_csv(describe.output_dir(roots) / rate.SCORES_FILE, runs)
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {"session_id": sessions, "K6": rng.integers(0, 25, N), "SRS2": rng.integers(15, 140, N)}
    ).to_csv(labels, index=False)
    return bundle, labels


class TestRun:
    def test_refuses_ratings_that_are_not_the_frozen_ones(
        self, default_config: AppConfig, roots: DataRoots, inputs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        bundle, labels = inputs
        with pytest.raises(analyze.AnalyzeError, match="frozen"):
            analyze.run(
                default_config,
                roots,
                bundle=bundle,
                labels_path=labels,
                out_dir=tmp_path / "out",
                commit="test",
            )

    def test_writes_every_table_and_no_per_session_row(
        self,
        default_config: AppConfig,
        roots: DataRoots,
        inputs: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bundle, labels = inputs
        scores = describe.output_dir(roots) / rate.SCORES_FILE
        monkeypatch.setattr(analyze, "FROZEN_SCORES_SHA256", describe.file_sha256(scores))
        out = tmp_path / "out"
        quick = load_config(DEFAULT_CONFIG_PATH, overrides={"model.tiers.stability_repeats": 2})
        result = analyze.run(
            quick,
            roots,
            bundle=bundle,
            labels_path=labels,
            out_dir=out,
            commit="test",
            n_permutations=3,
        )
        assert result.n_sessions == N
        for name in result.files:
            assert (out / name).exists()
        for name in result.files:
            if name.endswith(".csv"):
                frame = read_csv(out / name)
                assert "session_id" not in frame.columns
                assert (frame["tier"] == "exploratory").all()
        estimates = read_csv(out / analyze.ESTIMATES_FILE)
        assert set(estimates["feature_set"]) == set(analyze.FEATURE_SETS)
        assert set(estimates["target"]) == {"K6", "SRS2"}
        comparisons = read_csv(out / analyze.COMPARISONS_FILE)
        assert len(comparisons) == len(analyze.COMPARISONS) * 2
        summary = (out / analyze.SUMMARY_FILE).read_text()
        assert summary.count("EXPLORATORY") >= 1
        assert "Holm" not in summary
