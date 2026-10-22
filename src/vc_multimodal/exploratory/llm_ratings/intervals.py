"""Amendment 1 of docs/prereg/llm-ratings.md: intervals for the added value.

**Post hoc**: decided on 2026-10-22, after the primary results had been seen,
to quantify how large a gain from the 11 LLM features the data can rule out.
It changes no result of the primary analysis and adds no test.

Run by the label holder, from a clean checkout. For the 20 text features
against those 20 plus the 11 LLM features, for each outcome and both models,
it recomputes the leave-one-out predictions with `evaluate`, unchanged, and
stops unless every recomputed R² and MAE matches the primary run's
`models.csv` within 1e-9. It then resamples the sessions with replacement,
10,000 times, scoring both models on the same resampled sessions from their
fixed predictions, and reports the change in R² and in MAE with a 95%
percentile interval. No p-value and no correction: the intervals describe.

The bootstrap does not refit the models, so refitting variability is not in
the intervals, which are likely somewhat too narrow. What is written is
summary values only: no label, prediction or per-session error.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from vc_multimodal.exploratory.llm_ratings import analyze
from vc_multimodal.exploratory.llm_ratings.plan import file_sha256
from vc_multimodal.io_utils import read_csv, write_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.modeling.evaluate import evaluate

logger = get_logger(__name__)

STAGE: Final = "llm_ratings_intervals"
AMENDMENT: Final = "docs/prereg/llm-ratings.md, Amendment 1 (post hoc)"

N_RESAMPLES: Final = 10_000
LEVEL: Final = 0.95
#: Largest absolute difference allowed between a recomputed leave-one-out
#: estimate and the primary run's.
TOLERANCE: Final = 1e-9

#: The two arms of each comparison: first the text features, then text + LLM.
ARMS: Final = (analyze.TEXT, analyze.TEXT_LLM)
DELTA_R2: Final = "delta_r2"
DELTA_MAE: Final = "delta_mae"

INTERVALS_FILE: Final = "intervals.csv"
SUMMARY_FILE: Final = "intervals.md"
#: The default results folder, under the output root's `unblinded/`.
OUT_DIRNAME: Final = "llm_ratings_intervals"


class GateError(analyze.AnalyzeError):
    """Raised when the recomputation does not reproduce the primary run."""


# ---------------------------------------------------------------------------
# Recomputing the leave-one-out predictions
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Loo:
    """One model's leave-one-out estimate, and its per-session absolute errors.

    The errors stay in memory: only the bootstrap uses them.
    """

    feature_set: str
    target: str
    model: str
    n: int
    r2: float
    mae: float
    errors: np.ndarray


def recompute(
    feature_set: str,
    target: str,
    model: str,
    matrix: np.ndarray,
    truth: np.ndarray,
    groups: Sequence[object],
    seed: int,
) -> Loo:
    """The leave-one-out step of the primary analysis, and nothing else.

    `evaluate` computes the leave-one-out predictions before, and apart from,
    the repeated k-fold estimate, so leaving that out changes none of them.
    """
    evaluation = evaluate(matrix, truth, groups, model=model, seed=seed)
    return Loo(
        feature_set=feature_set,
        target=target,
        model=model,
        n=evaluation.loo.n,
        r2=evaluation.loo.r2,
        mae=evaluation.loo.mae,
        errors=evaluation.errors,
    )


def _same(a: float, b: float) -> bool:
    """Equal within the tolerance, or both missing."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= TOLERANCE


def gate(models: pd.DataFrame, recomputed: Sequence[Loo]) -> None:
    """Stop unless every recomputed estimate is the primary run's.

    A mismatch names the model and the size of the difference, never a value.

    Raises:
        GateError: if a model is missing from `models.csv`, or its n, R² or
            MAE differ.
    """
    problems = []
    for loo in recomputed:
        name = f"{loo.feature_set} {loo.target} {loo.model}"
        rows = models[
            (models["feature_set"] == loo.feature_set)
            & (models["target"] == loo.target)
            & (models["model"] == loo.model)
        ]
        if len(rows) != 1:
            problems.append(f"{name}: {len(rows)} row(s) in models.csv, expected 1")
            continue
        row = rows.iloc[0]
        if int(row["n"]) != loo.n:
            problems.append(f"{name}: n {loo.n} here, {int(row['n'])} in models.csv")
        for column, value in (("loo_r2", loo.r2), ("loo_mae", loo.mae)):
            primary = float(row[column])
            if not _same(value, primary):
                problems.append(f"{name}: {column} differs by {abs(value - primary):.1e}")
    if problems:
        msg = (
            f"the recomputed leave-one-out estimates do not reproduce the primary run "
            f"within {TOLERANCE:g}, so nothing was written:\n  " + "\n  ".join(problems)
        )
        raise GateError(msg)


def check_primary_run(
    results_dir: Path, inputs: analyze.Inputs, settings: analyze.Settings
) -> pd.DataFrame:
    """The primary run's `models.csv`, once its `run.json` shows the same inputs and settings.

    Raises:
        GateError: if a file is missing, or the run used other settings or
            other tables.
    """
    run_file, models_file = results_dir / analyze.RUN_FILE, results_dir / analyze.MODELS_FILE
    for path in (run_file, models_file):
        if not path.is_file():
            msg = f"{path} not found; --results must be the primary analysis's results folder"
            raise GateError(msg)
    record = json.loads(run_file.read_text(encoding="utf-8"))
    if record.get("settings") != asdict(settings):
        msg = f"{run_file} records other settings than these; the gate needs the same ones"
        raise GateError(msg)
    if record.get("inputs") != inputs.digests:
        msg = f"{run_file} records other input tables (by SHA-256) than these"
        raise GateError(msg)
    return read_csv(models_file)


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------
def resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """`n_resamples` draws of n session positions, with replacement."""
    indices: np.ndarray = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    return indices


def _r2(truth: np.ndarray, errors: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """1 - SSE/SST per resample, SST around the resample's own mean, as `score_predictions`."""
    residual = np.sum(errors**2, axis=1)
    total = np.sum((truth - truth.mean(axis=1, keepdims=True)) ** 2, axis=1)
    ratio = np.full(truth.shape[0], np.nan)
    np.divide(residual, total, out=ratio, where=usable)
    r2: np.ndarray = 1.0 - ratio
    return r2


def bootstrap_differences(
    truth: np.ndarray, text_errors: np.ndarray, both_errors: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ΔR² and ΔMAE, text + LLM minus text, in every resample.

    A resample whose outcome is constant has no R², so its ΔR² is NaN.
    """
    y = truth[indices]
    usable = np.ptp(y, axis=1) > 0
    first, second = text_errors[indices], both_errors[indices]
    delta_r2: np.ndarray = _r2(y, second, usable) - _r2(y, first, usable)
    delta_mae: np.ndarray = second.mean(axis=1) - first.mean(axis=1)
    return delta_r2, delta_mae


def percentile_interval(values: np.ndarray, level: float = LEVEL) -> tuple[float, float, int]:
    """The percentile interval over the resamples that have a value, and how many do."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return float(low), float(high), int(finite.size)


def intervals_table(
    cohort: analyze.Cohort, recomputed: Sequence[Loo], *, n_resamples: int, seed: int
) -> pd.DataFrame:
    """Two rows per comparison: the point estimate and interval of ΔR² and of ΔMAE."""
    by_key = {(loo.feature_set, loo.target, loo.model): loo for loo in recomputed}
    rows: list[dict[str, Any]] = []
    for target in analyze.TARGETS:
        truth = cohort.truth[target][cohort.mask(target)]
        indices = resample_indices(truth.size, n_resamples, seed)
        for model, role in analyze.MODELS:
            text, both = (by_key[(arm, target, model)] for arm in ARMS)
            delta_r2, delta_mae = bootstrap_differences(truth, text.errors, both.errors, indices)
            for metric, estimate, resampled in (
                (DELTA_R2, both.r2 - text.r2, delta_r2),
                (DELTA_MAE, both.mae - text.mae, delta_mae),
            ):
                low, high, used = percentile_interval(resampled)
                rows.append(
                    {
                        "target": target,
                        "model": model,
                        "role": role,
                        "n": both.n,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "level": LEVEL,
                        "n_resamples": n_resamples,
                        "n_resamples_used": used,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IntervalsResult:
    """What one run did, for the terminal. No estimate or interval is printed."""

    out_dir: Path
    n_by_target: Mapping[str, int]
    n_gated: int
    n_resamples: int
    seed: int
    written: tuple[Path, ...]

    def report_lines(self) -> list[str]:
        """The gate, coverage and the files written."""
        return [
            f"gate: {self.n_gated} recomputed leave-one-out estimates match models.csv "
            f"(n, R² and MAE within {TOLERANCE:g})",
            "sessions: " + ", ".join(f"{t} {n}" for t, n in self.n_by_target.items()),
            f"resamples: {self.n_resamples} per outcome, seed {self.seed}",
            *(f"wrote {path}" for path in self.written),
        ]


def run(
    llm_path: Path,
    text_path: Path,
    labels_path: Path,
    results_dir: Path,
    out_dir: Path,
    *,
    commit: str,
    jobs: int = 1,
    settings: analyze.Settings | None = None,
    n_resamples: int = N_RESAMPLES,
    planned: analyze.Planned | None = None,
) -> IntervalsResult:
    """Recompute, gate, resample and write the intervals.

    Raises:
        AnalyzeError: if the inputs are not the planned ones or `out_dir`
            already holds files.
        GateError: if the primary run cannot be reproduced; nothing is written.
    """
    settings = settings or analyze.Settings()
    analyze.refuse_used_folder(out_dir)
    started = datetime.now(UTC)
    inputs = analyze.load_inputs(llm_path, text_path, planned)
    models = check_primary_run(results_dir, inputs, settings)
    cohort = analyze.load_cohort(inputs, labels_path)

    calls = []
    for feature_set in ARMS:
        matrix = inputs.frame[list(analyze.FEATURE_SETS[feature_set])].to_numpy(dtype=np.float64)
        rated = matrix[cohort.rows]
        for target in analyze.TARGETS:
            mask = cohort.mask(target)
            for model, _ in analyze.MODELS:
                calls.append(
                    delayed(recompute)(
                        feature_set,
                        target,
                        model,
                        rated[mask],
                        cohort.truth[target][mask],
                        tuple(cohort.session_ids[mask]),
                        settings.seed,
                    )
                )
    logger.info("%s: recomputing %d model(s) on %d worker(s)", STAGE, len(calls), jobs)
    recomputed: list[Loo] = list(Parallel(n_jobs=jobs)(calls))
    gate(models, recomputed)
    logger.info("%s: the gate passed; resampling %d times per outcome", STAGE, n_resamples)
    table = intervals_table(cohort, recomputed, n_resamples=n_resamples, seed=settings.seed)

    record = {
        "commit": commit,
        "primary_models_sha256": file_sha256(results_dir / analyze.MODELS_FILE),
        "inputs": inputs.digests,
        "seed": settings.seed,
        "n_resamples": n_resamples,
        "planned": settings.planned and n_resamples == N_RESAMPLES,
        "sessions_analysed": cohort.counts(),
        "started": started.isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(out_dir / INTERVALS_FILE, table)
    summary_path = out_dir / SUMMARY_FILE
    summary_path.write_text(summary_markdown(table, record), encoding="utf-8")
    return IntervalsResult(
        out_dir=out_dir,
        n_by_target=cohort.counts(),
        n_gated=len(recomputed),
        n_resamples=n_resamples,
        seed=settings.seed,
        written=(csv_path, summary_path),
    )


def _table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """A plain markdown table, numbers to three decimals."""
    head = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in (row[c] for c in columns)
        )
        + " |"
        for row in frame.to_dict("records")
    ]
    return "\n".join([head, *body])


def summary_markdown(table: pd.DataFrame, record: Mapping[str, Any]) -> str:
    """A short readable summary that says plainly it is post hoc."""
    caveat = (
        ""
        if record["planned"]
        else "\n**Not the amendment's settings: a test run, not a result.**\n"
    )
    digests = ", ".join(f"`{name}` `{digest[:16]}...`" for name, digest in record["inputs"].items())
    columns = [
        "target",
        "model",
        "n",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_resamples_used",
    ]
    return "\n".join(
        [
            "# LLM ratings: intervals for the added value",
            "",
            "**Post hoc**, Amendment 1 of `docs/prereg/llm-ratings.md`: decided on 2026-10-22, "
            "after the primary results had been seen, to quantify how large a gain from the 11 "
            "LLM features the data can rule out. Exploratory; it changes no primary result.",
            caveat,
            "Each row compares the 20 text features with those 20 plus the 11 LLM features. "
            "`estimate` is (text + LLM) minus text over all sessions analysed, so a gain is a "
            "positive `delta_r2` and a negative `delta_mae`. The interval is the "
            f"{LEVEL:.0%} percentile interval of a {record['n_resamples']}-resample paired "
            "bootstrap over sessions. No p-value and no correction: the intervals describe, "
            "they do not test; the Wilcoxon tests of the primary analysis remain the tests.",
            "",
            _table(table, columns),
            "",
            "**Limitation.** The bootstrap resamples fixed leave-one-out predictions and does not "
            "refit the models, so refitting variability is not captured; the intervals are likely "
            "somewhat too narrow.",
            "",
            "## Provenance",
            "",
            f"- Commit `{record['commit']}`; seed {record['seed']}; sessions analysed "
            f"{record['sessions_analysed']}.",
            f"- Primary `models.csv` SHA-256 `{record['primary_models_sha256']}`; every "
            f"recomputed leave-one-out n, R² and MAE matched it within {TOLERANCE:g}.",
            f"- Inputs: {digests}.",
            f"- Started {record['started']}.",
            "",
        ]
    )
