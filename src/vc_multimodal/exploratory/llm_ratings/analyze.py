"""Step 3 of the LLM ratings: the analysis docs/prereg/llm-ratings.md fixes.

Run by the label holder, from a clean checkout. It reads the rater's feature
table and the manuscript's 20 text features, refusing either unless its
SHA-256 is the planned one, and the labels, and computes:

1. **Zero-shot**, no fitting: Spearman rho, two-sided, of the Set K total with
   K6 and of the Set S total with SRS-2.
2. **LLM features alone**: the 11 features, elastic net (primary) and random
   forest (secondary), with the confirmatory pipelines and functions unchanged
   (`modeling/evaluate.py`): leave-one-out, repeated 5-fold over 20 repeats,
   and a 1000-permutation null under 5-fold. The test is the elastic net's
   permutation p.
3. **Added value**: the 20 text features against those 20 plus the 11, by a
   paired Wilcoxon signed-rank test on per-session leave-one-out absolute
   errors.

Holm across the six primary tests (two of each). The random forest is
uncorrected. Spearman rho of each of the 11 features with each outcome, with
Benjamini-Hochberg q within each outcome.

A session the rater failed is left out of everything, both arms of the
comparison included. What is written is summary statistics only: no label
value, rating, prediction, residual or per-session error.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy import stats

from vc_multimodal.exploratory.llm_ratings.plan import (
    FEATURES,
    ITEMS,
    PLAN_PATH,
    file_sha256,
)
from vc_multimodal.io_utils import read_csv, write_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.modeling.evaluate import (
    MODEL_ELASTIC_NET,
    MODEL_RANDOM_FOREST,
    compare_errors,
    evaluate,
    permutation_baseline,
)
from vc_multimodal.modeling.tiers import holm_adjust
from vc_multimodal.provenance import environment_record
from vc_multimodal.stages.model import ModelError, load_labels

logger = get_logger(__name__)

STAGE: Final = "llm_ratings_analysis"

TARGETS: Final = ("K6", "SRS2")
#: Which total goes with which outcome in the zero-shot test.
ZERO_SHOT: Final = {"K6": "K", "SRS2": "S"}
MODELS: Final = ((MODEL_ELASTIC_NET, "primary"), (MODEL_RANDOM_FOREST, "secondary"))

LLM: Final = "llm"
TEXT: Final = "text"
TEXT_LLM: Final = "text_llm"
LLM_TABLE: Final = "llm_features"
TEXT_TABLE: Final = "text_features"

#: The manuscript's 20 text features, in `nlp_features.csv`'s order.
TEXT_COLUMNS: Final = (
    "Jaccard",
    "Cosine",
    "Bert",
    "Bert_turn",
    "Bert_turn_max",
    "Bert_turn_min",
    "Bert_turn_std",
    "Word_count",
    "Word_count_doc",
    "Word_ratio",
    "MTLD_patient",
    "MTLD_doctor",
    "Mean_words_per_turn",
    "Agenda_social",
    "Agenda_work",
    "Agenda_future",
    "Agenda_self",
    "Agenda_family",
    "Agenda_anxiety",
    "Agenda_depression",
)
FEATURE_SETS: Final = {
    LLM: tuple(FEATURES),
    TEXT: TEXT_COLUMNS,
    TEXT_LLM: (*TEXT_COLUMNS, *FEATURES),
}

PRIMARY_FILE: Final = "primary_tests.csv"
MODELS_FILE: Final = "models.csv"
COMPARISONS_FILE: Final = "comparisons.csv"
UNIVARIATE_FILE: Final = "univariate.csv"
SUMMARY_FILE: Final = "summary.md"
RUN_FILE: Final = "run.json"
LOG_DIRNAME: Final = "logs"


@dataclass(frozen=True, slots=True)
class Settings:
    """The pre-registered validation settings. Only tests change them."""

    seed: int = 20260923
    stability_folds: int = 5
    stability_repeats: int = 20
    n_permutations: int = 1000
    permutation_folds: int = 5

    @property
    def planned(self) -> bool:
        """Whether these are the pre-registered settings."""
        return self == Settings()


class AnalyzeError(RuntimeError):
    """Raised when the analysis cannot be run as pre-registered."""


Planned = Mapping[str, tuple[str, str | None]]


def load_inputs_plan(path: Path = PLAN_PATH) -> dict[str, tuple[str, str | None]]:
    """The two tables the analysis reads, and their planned SHA-256.

    Raises:
        AnalyzeError: if the plan does not name both.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tables = (raw.get("analysis") if isinstance(raw, dict) else None) or {}
    planned: dict[str, tuple[str, str | None]] = {}
    for name in (LLM_TABLE, TEXT_TABLE):
        entry = tables.get(name)
        if not isinstance(entry, dict) or not entry.get("file"):
            msg = f"{path.name} must name analysis.{name}.file"
            raise AnalyzeError(msg)
        digest = entry.get("sha256")
        planned[name] = (str(entry["file"]), None if digest is None else str(digest))
    return planned


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _checked_table(path: Path, expected: str | None, columns: Sequence[str]) -> pd.DataFrame:
    """A table, refused unless it is the planned one with the planned columns."""
    if not path.is_file():
        msg = f"no {path.name} at {path.parent}"
        raise AnalyzeError(msg)
    if expected is None:
        msg = (
            f"the SHA-256 of {path.name} is not fixed in the plan yet. It is fixed in a "
            f"commit of its own, pushed before the analysis is run."
        )
        raise AnalyzeError(msg)
    found = file_sha256(path)
    if found != expected:
        msg = (
            f"{path} is not the planned table: SHA-256 {found[:12]}..., planned {expected[:12]}..."
        )
        raise AnalyzeError(msg)
    frame = read_csv(path)
    if list(frame.columns) != ["session_id", *columns]:
        msg = f"{path.name} does not have session_id and its {len(columns)} features in order"
        raise AnalyzeError(msg)
    if frame["session_id"].duplicated().any():
        msg = f"{path.name} names a session twice"
        raise AnalyzeError(msg)
    return frame


@dataclass(frozen=True, slots=True)
class Inputs:
    """Both tables joined by session, and their digests."""

    frame: pd.DataFrame
    digests: dict[str, str]

    @property
    def rated(self) -> np.ndarray:
        """Sessions the rater rated: every LLM feature present."""
        mask: np.ndarray = self.frame[list(FEATURES)].notna().all(axis=1).to_numpy()
        return mask


def load_inputs(llm_path: Path, text_path: Path, planned: Planned | None = None) -> Inputs:
    """The rater's table and the 20 text features, joined by session ID.

    Raises:
        AnalyzeError: if a table is missing or unplanned, the two cover
            different sessions, or a session is partly rated.
    """
    planned = planned if planned is not None else load_inputs_plan()
    llm = _checked_table(llm_path, planned[LLM_TABLE][1], FEATURES)
    text = _checked_table(text_path, planned[TEXT_TABLE][1], TEXT_COLUMNS)
    if set(llm["session_id"]) != set(text["session_id"]):
        msg = "the LLM table and the text table cover different sessions"
        raise AnalyzeError(msg)
    present = llm[list(FEATURES)].notna()
    if (present.any(axis=1) & ~present.all(axis=1)).any():
        msg = "a session in the LLM table is partly rated; the rater writes all or nothing"
        raise AnalyzeError(msg)
    frame = text.merge(llm, on="session_id", how="inner", validate="one_to_one")
    frame = frame.sort_values("session_id", kind="mergesort").reset_index(drop=True)
    digests = {llm_path.name: file_sha256(llm_path), text_path.name: file_sha256(text_path)}
    return Inputs(frame=frame, digests=digests)


@dataclass(frozen=True, slots=True)
class Cohort:
    """The rated sessions with a label, per outcome."""

    session_ids: np.ndarray
    rows: np.ndarray
    truth: Mapping[str, np.ndarray]
    failed: tuple[int, ...]
    features_only: tuple[int, ...]
    labels_only: int

    def mask(self, target: str) -> np.ndarray:
        """Sessions whose label for this outcome is present."""
        mask: np.ndarray = ~np.isnan(self.truth[target])
        return mask

    def counts(self) -> dict[str, int]:
        """Sessions analysed per outcome."""
        return {target: int(self.mask(target).sum()) for target in TARGETS}


def load_cohort(inputs: Inputs, labels_path: Path) -> Cohort:
    """Match the labels to the rated sessions by session ID.

    The label values stay in memory: nothing here is written.

    Raises:
        AnalyzeError: if the labels lack an outcome or share no rated session.
    """
    try:
        labels = load_labels(labels_path, TARGETS)
    except ModelError as exc:
        raise AnalyzeError(str(exc)) from exc
    absent = [t for t in TARGETS if t not in labels.targets]
    if absent:
        msg = f"the labels file has no column {absent}; it needs {list(TARGETS)}"
        raise AnalyzeError(msg)
    frame = labels.frame.drop_duplicates("session_id")
    theirs = {int(s) for s in frame["session_id"]}
    ours = [int(s) for s in inputs.frame["session_id"]]
    rated = inputs.rated
    failed = tuple(s for s, ok in zip(ours, rated, strict=True) if not ok)
    rows = np.array(
        [i for i, (s, ok) in enumerate(zip(ours, rated, strict=True)) if ok and s in theirs],
        dtype=np.int64,
    )
    if rows.size == 0:
        msg = "no rated session appears in the labels file"
        raise AnalyzeError(msg)
    shared = [ours[i] for i in rows]
    indexed = frame.set_index("session_id")
    truth = {
        t: pd.to_numeric(indexed.loc[shared, t], errors="coerce").to_numpy(dtype=np.float64)
        for t in TARGETS
    }
    return Cohort(
        session_ids=np.array(shared, dtype=np.int64),
        rows=rows,
        truth=truth,
        failed=failed,
        features_only=tuple(s for s in ours if s not in theirs),
        labels_only=len(theirs - set(ours)),
    )


def scale_total(frame: pd.DataFrame, scale: str) -> np.ndarray:
    """The Set K or Set S total: the sum of that set's features."""
    columns = [item.feature for item in ITEMS if item.scale == scale]
    total: np.ndarray = frame[columns].to_numpy(dtype=np.float64).sum(axis=1)
    return total


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Task:
    """One model: a feature set, an outcome and a model family."""

    feature_set: str
    target: str
    model: str
    role: str


def make_tasks() -> list[Task]:
    """Every model the plan fits: 3 feature sets x 2 outcomes x 2 models = 12."""
    return [
        Task(feature_set, target, model, role)
        for feature_set in FEATURE_SETS
        for target in TARGETS
        for model, role in MODELS
    ]


def fit(
    task: Task, matrix: np.ndarray, truth: np.ndarray, groups: Sequence[object], settings: Settings
) -> tuple[Task, dict[str, Any], np.ndarray]:
    """Leave-one-out, the repeated k-fold estimate and the permutation null.

    Returns the task, its summary row, and its per-session absolute errors,
    which only the paired test uses and which are never written.
    """
    start = time.perf_counter()
    evaluation = evaluate(
        matrix,
        truth,
        groups,
        model=task.model,
        seed=settings.seed,
        stability_folds=settings.stability_folds,
        stability_repeats=settings.stability_repeats,
    )
    null = permutation_baseline(
        matrix,
        truth,
        groups,
        model=task.model,
        seed=settings.seed,
        n_permutations=settings.n_permutations,
        folds=settings.permutation_folds,
    )
    row: dict[str, Any] = {
        "feature_set": task.feature_set,
        "target": task.target,
        "model": task.model,
        "role": task.role,
        "n": evaluation.loo.n,
        "n_features": evaluation.n_features,
        "loo_r2": evaluation.loo.r2,
        "loo_mae": evaluation.loo.mae,
        "loo_spearman": evaluation.loo.spearman,
        "loo_spearman_p": evaluation.loo.spearman_p,
        "kfold_r2_mean": evaluation.stability,
        "kfold_r2_sd": evaluation.stability_sd,
        "perm_observed_r2": None if null is None else null.observed,
        "perm_null_mean": None if null is None else null.null_mean,
        "perm_null_p95": None if null is None else null.null_p95,
        "perm_p": None if null is None else null.p_value,
        "n_permutations": 0 if null is None else null.n_permutations,
        "n_fits_not_converged": evaluation.n_not_converged,
    }
    logger.info(
        "%s: %s %s %s done in %.0f s",
        STAGE,
        task.feature_set,
        task.target,
        task.model,
        time.perf_counter() - start,
    )
    return task, row, evaluation.errors


def compare(errors: Mapping[tuple[str, str, str], np.ndarray]) -> pd.DataFrame:
    """Analysis 3: the 20 text features against the 20 plus the 11, by outcome and model."""
    rows = []
    for target in TARGETS:
        for model, role in MODELS:
            result = compare_errors(
                errors[(TEXT, target, model)], errors[(TEXT_LLM, target, model)]
            )
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "role": role,
                    "n": result.n,
                    "median_error_difference": result.median_difference,
                    "p": result.p_value,
                }
            )
    return pd.DataFrame(rows)


def zero_shot(frame: pd.DataFrame, cohort: Cohort) -> list[dict[str, Any]]:
    """Analysis 1: each scale total against its outcome, no fitting."""
    rows = []
    rated = frame.iloc[cohort.rows]
    for target, scale in ZERO_SHOT.items():
        mask = cohort.mask(target)
        result = stats.spearmanr(scale_total(rated, scale)[mask], cohort.truth[target][mask])
        rows.append(
            {
                "analysis": "zero_shot",
                "test": f"Set {scale} total vs {target}, Spearman",
                "target": target,
                "n": int(mask.sum()),
                "statistic": float(result.statistic),
                "statistic_name": "rho",
                "p": float(result.pvalue),
            }
        )
    return rows


def primary_tests(
    zero: Sequence[Mapping[str, Any]], models: pd.DataFrame, comparisons: pd.DataFrame
) -> pd.DataFrame:
    """The six primary tests, with Holm across all six."""
    rows: list[dict[str, Any]] = [dict(row) for row in zero]
    for target in TARGETS:
        model = models[
            (models["feature_set"] == LLM)
            & (models["target"] == target)
            & (models["model"] == MODEL_ELASTIC_NET)
        ].iloc[0]
        rows.append(
            {
                "analysis": "llm_alone",
                "test": f"11 LLM features, elastic net vs permutation null, {target}",
                "target": target,
                "n": int(model["n"]),
                "statistic": model["perm_observed_r2"],
                "statistic_name": "kfold_r2",
                "p": model["perm_p"],
            }
        )
    for target in TARGETS:
        comparison = comparisons[
            (comparisons["target"] == target) & (comparisons["model"] == MODEL_ELASTIC_NET)
        ].iloc[0]
        rows.append(
            {
                "analysis": "added_value",
                "test": f"text 20 vs text 20 + LLM 11, elastic net, {target}",
                "target": target,
                "n": int(comparison["n"]),
                "statistic": comparison["median_error_difference"],
                "statistic_name": "median_error_difference",
                "p": comparison["p"],
            }
        )
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_finite([np.nan if v is None else float(v) for v in table["p"]])
    return table


def holm_finite(p_values: Sequence[float]) -> list[float]:
    """Holm over the p-values that exist; one that is missing stays missing."""
    finite = [i for i, v in enumerate(p_values) if np.isfinite(v)]
    adjusted = holm_adjust([p_values[i] for i in finite]) if finite else ()
    out = [float("nan")] * len(p_values)
    for i, value in zip(finite, adjusted, strict=True):
        out[i] = value
    return out


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH q-values, as statsmodels' `multipletests(method="fdr_bh")`.

    A NaN p (a constant feature) stays NaN and is not counted.
    """
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return q
    order = finite[np.argsort(p[finite], kind="mergesort")]
    ranked = p[order] * finite.size / np.arange(1, finite.size + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def correlate(frame: pd.DataFrame, cohort: Cohort) -> pd.DataFrame:
    """Spearman rho of each of the 11 features with each outcome, BH within each outcome."""
    rated = frame.iloc[cohort.rows]
    rows = []
    for target in TARGETS:
        mask = cohort.mask(target)
        y = cohort.truth[target][mask]
        block: list[dict[str, Any]] = []
        for item in ITEMS:
            x = rated[item.feature].to_numpy(dtype=np.float64)[mask]
            constant = np.unique(x).size < 2  # noqa: PLR2004 - a rank needs two values
            result = None if constant else stats.spearmanr(x, y)
            block.append(
                {
                    "target": target,
                    "feature": item.feature,
                    "set": item.scale,
                    "n": int(mask.sum()),
                    "rho": float("nan") if result is None else float(result.statistic),
                    "p": float("nan") if result is None else float(result.pvalue),
                }
            )
        q = benjamini_hochberg([float(r["p"]) for r in block])
        for row, value in zip(block, q, strict=True):
            row["q"] = float(value)
        rows.extend(block)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def refuse_used_folder(out_dir: Path) -> None:
    """Refuse a results folder that already holds anything but this run's log.

    Raises:
        AnalyzeError: if it does.
    """
    if out_dir.is_dir() and any(p.name != LOG_DIRNAME for p in out_dir.iterdir()):
        msg = f"{out_dir} already holds files; give a new folder"
        raise AnalyzeError(msg)


@dataclass(frozen=True, slots=True)
class RunResult:
    """What one run did, for the terminal. No result value is printed."""

    out_dir: Path
    n_by_target: Mapping[str, int]
    failed: tuple[int, ...]
    features_only: tuple[int, ...]
    labels_only: int
    written: tuple[Path, ...]

    def report_lines(self) -> list[str]:
        """Coverage and the files written."""
        lines = [
            "sessions analysed: " + ", ".join(f"{t} {n}" for t, n in self.n_by_target.items()),
        ]
        if self.failed:
            lines.append(f"  not rated, left out of everything: {list(self.failed)}")
        if self.features_only:
            lines.append(f"  features but no label, left out: {list(self.features_only)}")
        if self.labels_only:
            lines.append(f"  {self.labels_only} labelled session(s) have no features")
        lines.extend(f"wrote {path}" for path in self.written)
        return lines


def _matrix(inputs: Inputs, cohort: Cohort, feature_set: str) -> np.ndarray:
    matrix: np.ndarray = inputs.frame[list(FEATURE_SETS[feature_set])].to_numpy(dtype=np.float64)
    rated: np.ndarray = matrix[cohort.rows]
    return rated


def run(
    llm_path: Path,
    text_path: Path,
    labels_path: Path,
    out_dir: Path,
    *,
    commit: str,
    jobs: int = 1,
    settings: Settings | None = None,
    planned: Planned | None = None,
) -> RunResult:
    """Run every planned analysis and write the summaries.

    Raises:
        AnalyzeError: if the inputs are not the planned ones, or `out_dir`
            already holds files.
    """
    settings = settings or Settings()
    refuse_used_folder(out_dir)
    started = datetime.now(UTC)
    inputs = load_inputs(llm_path, text_path, planned)
    cohort = load_cohort(inputs, labels_path)

    calls = []
    for task in make_tasks():
        mask = cohort.mask(task.target)
        calls.append(
            delayed(fit)(
                task,
                _matrix(inputs, cohort, task.feature_set)[mask],
                cohort.truth[task.target][mask],
                tuple(cohort.session_ids[mask]),
                settings,
            )
        )
    logger.info("%s: fitting %d model(s) on %d worker(s)", STAGE, len(calls), jobs)
    fitted: list[tuple[Task, dict[str, Any], np.ndarray]] = list(Parallel(n_jobs=jobs)(calls))
    models = pd.DataFrame([row for _, row, _ in fitted])
    errors = {(t.feature_set, t.target, t.model): e for t, _, e in fitted}
    comparisons = compare(errors)
    primary = primary_tests(zero_shot(inputs.frame, cohort), models, comparisons)
    univariate = correlate(inputs.frame, cohort)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        write_csv(out_dir / PRIMARY_FILE, primary),
        write_csv(out_dir / MODELS_FILE, models),
        write_csv(out_dir / COMPARISONS_FILE, comparisons),
        write_csv(out_dir / UNIVARIATE_FILE, univariate),
    ]
    record = {
        "analysis": "exploratory/llm-ratings, step 3",
        "prereg": "docs/prereg/llm-ratings.md",
        "commit": commit,
        "inputs": inputs.digests,
        "settings": asdict(settings),
        "planned_settings": settings.planned,
        "sessions_analysed": cohort.counts(),
        "not_rated": list(cohort.failed),
        "features_only": list(cohort.features_only),
        "labels_only": cohort.labels_only,
        "started": started.isoformat(),
        "finished": datetime.now(UTC).isoformat(),
        "environment": environment_record(),
    }
    (out_dir / RUN_FILE).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    (out_dir / SUMMARY_FILE).write_text(
        summary_markdown(primary, models, comparisons, univariate, record), encoding="utf-8"
    )
    written += [out_dir / RUN_FILE, out_dir / SUMMARY_FILE]
    return RunResult(
        out_dir=out_dir,
        n_by_target=cohort.counts(),
        failed=cohort.failed,
        features_only=cohort.features_only,
        labels_only=cohort.labels_only,
        written=tuple(written),
    )


def _markdown(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """A plain markdown table, numbers to three decimals."""
    head = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
            for v in (row[c] for c in columns)
        )
        + " |"
        for row in frame.to_dict("records")
    ]
    return "\n".join([head, *body])


def summary_markdown(
    primary: pd.DataFrame,
    models: pd.DataFrame,
    comparisons: pd.DataFrame,
    univariate: pd.DataFrame,
    record: Mapping[str, Any],
) -> str:
    """A readable summary, for the person who ran it."""
    caveat = (
        ""
        if record["planned_settings"]
        else "\n**Not the pre-registered settings: a test run, not a result.**\n"
    )
    not_rated = record["not_rated"]
    return "\n".join(
        [
            "# LLM ratings of transcripts, analysis",
            "",
            "**Exploratory**, pre-registered in `docs/prereg/llm-ratings.md` after the labels "
            "were received. No result here is confirmatory.",
            caveat,
            f"Commit `{record['commit']}`. Sessions analysed: {record['sessions_analysed']}."
            + (f" Not rated, left out of everything: {not_rated}." if not_rated else ""),
            "",
            "## The six primary tests",
            "",
            "Holm across all six. Zero-shot: Spearman rho of the scale total with the outcome. "
            "LLM alone: the elastic net's repeated 5-fold R² against its 1000-permutation null. "
            "Added value: Wilcoxon signed-rank on per-session leave-one-out absolute errors; "
            "`median_error_difference` is (text + LLM) minus text, so negative means the LLM "
            "features helped.",
            "",
            _markdown(
                primary, ["analysis", "target", "n", "statistic_name", "statistic", "p", "p_holm"]
            ),
            "",
            "## Added value, both models",
            "",
            "The random forest row is uncorrected.",
            "",
            _markdown(comparisons, ["target", "model", "n", "median_error_difference", "p"]),
            "",
            "## Models",
            "",
            "The 20 text features already include seven LLM-rated `Agenda_*` scores.",
            "",
            _markdown(
                models,
                [
                    "feature_set",
                    "target",
                    "model",
                    "n",
                    "loo_r2",
                    "loo_mae",
                    "loo_spearman",
                    "kfold_r2_mean",
                    "perm_observed_r2",
                    "perm_p",
                    "n_fits_not_converged",
                ],
            ),
            "",
            "## Feature-level correlations",
            "",
            "Spearman rho; q is Benjamini-Hochberg within each outcome, across the 11 features.",
            "",
            _markdown(univariate, ["target", "feature", "set", "n", "rho", "p", "q"]),
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Timing, without labels
# ---------------------------------------------------------------------------
def estimate_runtime(
    llm_path: Path,
    text_path: Path,
    *,
    jobs: int = 1,
    n_probe: int = 4,
    planned: Planned | None = None,
) -> list[str]:
    """Time every model on random targets and extrapolate. Reads no labels."""
    settings = Settings()
    inputs = load_inputs(llm_path, text_path, planned)
    rows = np.flatnonzero(inputs.rated)
    rng = np.random.default_rng(settings.seed)
    fake = {t: rng.normal(size=rows.size) for t in TARGETS}
    groups = tuple(range(rows.size))
    probe = Settings(n_permutations=n_probe)
    tasks = make_tasks()
    calls = [
        delayed(_timed)(
            task,
            inputs.frame[list(FEATURE_SETS[task.feature_set])].to_numpy(dtype=np.float64)[rows],
            fake[task.target],
            groups,
            probe,
        )
        for task in tasks
    ]
    timings: list[tuple[float, float]] = list(Parallel(n_jobs=jobs)(calls))
    per_task = [
        observed + settings.n_permutations * per_permutation
        for observed, per_permutation in timings
    ]
    total = max(sum(per_task) / max(1, min(jobs, len(per_task))), *per_task)
    lines = [f"{jobs} worker(s), {rows.size} rated session(s), timed on random targets:"]
    lines.extend(
        f"  {t.feature_set} {t.target} {t.model}: {observed:.1f} s, "
        f"{per_permutation:.2f} s per permutation"
        for t, (observed, per_permutation) in zip(tasks, timings, strict=True)
    )
    lines.append(f"  about {total / 3600:.1f} h at {settings.n_permutations} permutations")
    return lines


def _timed(
    task: Task, matrix: np.ndarray, truth: np.ndarray, groups: Sequence[object], probe: Settings
) -> tuple[float, float]:
    """Seconds for the estimates without the null, and per permutation."""
    start = time.perf_counter()
    evaluate(
        matrix,
        truth,
        groups,
        model=task.model,
        seed=probe.seed,
        stability_folds=probe.stability_folds,
        stability_repeats=probe.stability_repeats,
    )
    observed = time.perf_counter() - start
    start = time.perf_counter()
    permutation_baseline(
        matrix,
        truth,
        groups,
        model=task.model,
        seed=probe.seed,
        n_permutations=probe.n_permutations,
        folds=probe.permutation_folds,
    )
    return observed, (time.perf_counter() - start) / max(1, probe.n_permutations)
