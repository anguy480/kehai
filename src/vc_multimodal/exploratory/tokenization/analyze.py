"""Step 3 of the tokenization rerun: the original 20 against new 20 (b) and (c).

Pre-registered in docs/prereg/tokenization.md, including its Amendment 1. Run
by the label holder, from a clean checkout. It reads the three 20-feature
tables that step 2 wrote, refusing any whose SHA-256 is not the planned one,
and the labels, and computes:

* **Models.** Elastic net (primary) and random forest (secondary) on each of
  the three feature sets and both outcomes, with the confirmatory pipelines
  and functions unchanged (`modeling/evaluate.py`): leave-one-out, repeated
  5-fold over 20 repeats, and a 1000-permutation null under 5-fold.
* **The test.** A paired Wilcoxon signed-rank test of the per-session
  leave-one-out absolute errors, original 20 against each variant, for each
  outcome and model. Holm across the four elastic-net tests; the four random
  forest tests are reported uncorrected.
* **Correlations.** Spearman rho of each feature with each outcome, as
  analysis.py computes it, and Benjamini-Hochberg q within each feature set
  and outcome, across its 20 features.

What it writes is summary statistics only: no label value, prediction,
residual or per-session error.
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

from vc_multimodal.exploratory.tokenization.run import (
    NLP_COLUMNS,
    TEXT_NEW,
    TEXT_NEW_C,
    TEXT_ORIGINAL,
    file_sha256,
)
from vc_multimodal.exploratory.tokenization.token_features import FEATURES
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

STAGE: Final = "tokenization_analysis"

TARGETS: Final = ("K6", "SRS2")
ORIGINAL: Final = "original"
VARIANTS: Final = ("new_b", "new_c")
#: The plan: each feature set's step 2 table and its planned SHA-256.
REPO_ROOT: Final = Path(__file__).resolve().parents[4]
PLAN_PATH: Final = REPO_ROOT / "config" / "exploratory" / "tokenization.yaml"
FEATURE_SET_FILES: Final = {ORIGINAL: TEXT_ORIGINAL, "new_b": TEXT_NEW, "new_c": TEXT_NEW_C}
MODELS: Final = ((MODEL_ELASTIC_NET, "primary"), (MODEL_RANDOM_FOREST, "secondary"))

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


def load_plan(path: Path = PLAN_PATH) -> dict[str, tuple[str, str | None]]:
    """Each feature set's table and planned SHA-256, as the plan fixes them.

    Raises:
        AnalyzeError: if the plan does not name exactly the planned tables.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tables = raw.get("feature_tables") if isinstance(raw, dict) else None
    if not isinstance(tables, dict) or set(tables) != set(FEATURE_SET_FILES):
        msg = f"{path.name} must name the tables {list(FEATURE_SET_FILES)}"
        raise AnalyzeError(msg)
    planned: dict[str, tuple[str, str | None]] = {}
    for name, filename in FEATURE_SET_FILES.items():
        entry = tables[name]
        if entry.get("file") != filename:
            msg = f"{path.name}: {name} must read {filename}"
            raise AnalyzeError(msg)
        digest = entry.get("sha256")
        planned[name] = (filename, None if digest is None else str(digest))
    return planned


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_feature_sets(
    features_dir: Path, planned: Mapping[str, tuple[str, str | None]] | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """The three tables, each checked against its planned SHA-256.

    Raises:
        AnalyzeError: if a table is missing, unplanned, malformed, or differs
            from the others where it should not.
    """
    planned = planned if planned is not None else load_plan()
    frames: dict[str, pd.DataFrame] = {}
    digests: dict[str, str] = {}
    for name, (filename, expected) in planned.items():
        path = features_dir / filename
        if not path.is_file():
            msg = f"no {filename} in {features_dir}: run step 2 first"
            raise AnalyzeError(msg)
        if expected is None:
            msg = (
                f"the SHA-256 of {filename} is not fixed in the plan yet. It is fixed in a "
                f"commit of its own, pushed before this step is run."
            )
            raise AnalyzeError(msg)
        found = file_sha256(path)
        if found != expected:
            msg = (
                f"{path} is not the planned table: SHA-256 {found[:12]}..., "
                f"planned {expected[:12]}..."
            )
            raise AnalyzeError(msg)
        frame = read_csv(path)
        if list(frame.columns) != ["session_id", *NLP_COLUMNS]:
            msg = f"{filename} does not have session_id and the 20 features in order"
            raise AnalyzeError(msg)
        frames[name] = frame
        digests[filename] = found

    first = next(iter(frames.values()))
    unchanged = [c for c in NLP_COLUMNS if c not in FEATURES]
    for name, frame in frames.items():
        if not frame["session_id"].equals(first["session_id"]):
            msg = f"{name} covers different sessions, or orders them differently"
            raise AnalyzeError(msg)
        if not frame[unchanged].equals(first[unchanged]):
            msg = f"{name}'s 12 features that do not use lemmas differ from the original's"
            raise AnalyzeError(msg)
    return frames, digests


@dataclass(frozen=True, slots=True)
class Cohort:
    """The sessions with features and a label, per outcome."""

    session_ids: np.ndarray
    truth: Mapping[str, np.ndarray]
    features_only: tuple[int, ...]
    labels_only: int

    def mask(self, target: str) -> np.ndarray:
        """Sessions whose label for this outcome is present."""
        mask: np.ndarray = ~np.isnan(self.truth[target])
        return mask

    def counts(self) -> dict[str, int]:
        """Sessions modelled per outcome."""
        return {target: int(self.mask(target).sum()) for target in TARGETS}


def load_cohort(session_ids: pd.Series, labels_path: Path) -> Cohort:
    """Match the labels to the feature tables by session.

    The label values stay in memory: nothing here is written.

    Raises:
        AnalyzeError: if the labels lack an outcome or share no session.
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
    ours = [int(s) for s in session_ids]
    theirs = {int(s) for s in frame["session_id"]}
    shared = [s for s in ours if s in theirs]
    if not shared:
        msg = "no session appears in both the feature tables and the labels file"
        raise AnalyzeError(msg)
    indexed = frame.set_index("session_id")
    truth = {
        t: pd.to_numeric(indexed.loc[shared, t], errors="coerce").to_numpy(dtype=np.float64)
        for t in TARGETS
    }
    return Cohort(
        session_ids=np.array(shared, dtype=np.int64),
        truth=truth,
        features_only=tuple(s for s in ours if s not in theirs),
        labels_only=len(theirs - set(ours)),
    )


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
    """Every model the plan fits, 3 x 2 x 2 = 12."""
    return [
        Task(feature_set, target, model, role)
        for feature_set in FEATURE_SET_FILES
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
    """Original against each variant, by outcome and model, with Holm for the primary."""
    rows = []
    for target in TARGETS:
        for model, role in MODELS:
            for variant in VARIANTS:
                result = compare_errors(
                    errors[(ORIGINAL, target, model)], errors[(variant, target, model)]
                )
                rows.append(
                    {
                        "variant": variant,
                        "target": target,
                        "model": model,
                        "role": role,
                        "n": result.n,
                        "median_error_difference": result.median_difference,
                        "p": result.p_value,
                    }
                )
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    primary = table["role"] == "primary"
    table.loc[primary, "p_holm"] = holm_adjust(table.loc[primary, "p"].tolist())
    return table


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH q-values, as statsmodels' `multipletests(method="fdr_bh")`.

    A NaN p (a constant or NaN feature) stays NaN and is not counted.
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


def correlate(
    frames: Mapping[str, pd.DataFrame], cohort: Cohort, positions: np.ndarray
) -> pd.DataFrame:
    """Spearman rho of every feature with every outcome, BH within each family.

    `spearmanr` with its defaults, as analysis.py calls it.
    """
    rows = []
    for feature_set, frame in frames.items():
        for target in TARGETS:
            mask = cohort.mask(target)
            y = cohort.truth[target][mask]
            block: list[dict[str, Any]] = []
            for feature in NLP_COLUMNS:
                x = frame[feature].to_numpy(dtype=np.float64)[positions][mask]
                result = stats.spearmanr(x, y)
                block.append(
                    {
                        "feature_set": feature_set,
                        "target": target,
                        "feature": feature,
                        "recomputed": feature in FEATURES,
                        "n": int(mask.sum()),
                        "rho": float(result.statistic),
                        "p": float(result.pvalue),
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
    features_only: tuple[int, ...]
    labels_only: int
    written: tuple[Path, ...]

    def report_lines(self) -> list[str]:
        """Coverage and the files written."""
        lines = [
            "sessions modelled: " + ", ".join(f"{t} {n}" for t, n in self.n_by_target.items()),
        ]
        if self.features_only:
            lines.append(f"  features but no label, excluded: {list(self.features_only)}")
        if self.labels_only:
            lines.append(f"  {self.labels_only} labelled session(s) have no text features")
        lines.extend(f"wrote {path}" for path in self.written)
        return lines


def run(
    features_dir: Path,
    labels_path: Path,
    out_dir: Path,
    *,
    commit: str,
    jobs: int = 1,
    settings: Settings | None = None,
    planned: Mapping[str, tuple[str, str | None]] | None = None,
) -> RunResult:
    """Fit every planned model, test and correlate, and write the summaries.

    Raises:
        AnalyzeError: if the inputs are not the planned ones, or `out_dir`
            already holds files.
    """
    settings = settings or Settings()
    refuse_used_folder(out_dir)
    started = datetime.now(UTC)
    frames, digests = load_feature_sets(features_dir, planned)
    session_ids = next(iter(frames.values()))["session_id"]
    cohort = load_cohort(session_ids, labels_path)
    index = {int(s): i for i, s in enumerate(session_ids)}
    positions = np.array([index[int(s)] for s in cohort.session_ids], dtype=np.int64)

    tasks = make_tasks()
    calls = []
    for task in tasks:
        mask = cohort.mask(task.target)
        matrix = frames[task.feature_set][list(NLP_COLUMNS)].to_numpy(dtype=np.float64)
        calls.append(
            delayed(fit)(
                task,
                matrix[positions][mask],
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
    univariate = correlate(frames, cohort, positions)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        write_csv(out_dir / MODELS_FILE, models),
        write_csv(out_dir / COMPARISONS_FILE, comparisons),
        write_csv(out_dir / UNIVARIATE_FILE, univariate),
    ]
    record = {
        "analysis": "exploratory/tokenization, step 3",
        "commit": commit,
        "inputs": digests,
        "settings": asdict(settings),
        "planned_settings": settings.planned,
        "sessions_modelled": cohort.counts(),
        "features_only": list(cohort.features_only),
        "labels_only": cohort.labels_only,
        "started": started.isoformat(),
        "finished": datetime.now(UTC).isoformat(),
        "environment": environment_record(),
    }
    (out_dir / RUN_FILE).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    (out_dir / SUMMARY_FILE).write_text(
        summary_markdown(models, comparisons, univariate, record), encoding="utf-8"
    )
    written += [out_dir / RUN_FILE, out_dir / SUMMARY_FILE]
    return RunResult(
        out_dir=out_dir,
        n_by_target=cohort.counts(),
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
    return "\n".join(
        [
            "# Tokenization rerun, step 3",
            "",
            "**Exploratory**, pre-registered in `docs/prereg/tokenization.md` (with Amendment 1) "
            "after the labels were received. No result here is confirmatory.",
            caveat,
            f"Commit `{record['commit']}`. Sessions modelled: {record['sessions_modelled']}.",
            "",
            "## The test: original 20 against each variant",
            "",
            "Wilcoxon signed-rank on per-session leave-one-out absolute errors. "
            "`median_error_difference` is variant minus original: positive means the original "
            "was closer. Holm across the four elastic-net tests; random forest uncorrected.",
            "",
            _markdown(
                comparisons,
                ["variant", "target", "model", "n", "median_error_difference", "p", "p_holm"],
            ),
            "",
            "## Models",
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
            "## Correlations",
            "",
            "Spearman rho; q is Benjamini-Hochberg within each feature set and outcome, "
            "across its 20 features.",
            "",
            _markdown(
                univariate,
                ["feature_set", "target", "feature", "recomputed", "n", "rho", "p", "q"],
            ),
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Timing, without labels
# ---------------------------------------------------------------------------
def estimate_runtime(
    features_dir: Path,
    *,
    jobs: int = 1,
    n_probe: int = 4,
    planned: Mapping[str, tuple[str, str | None]] | None = None,
) -> list[str]:
    """Time every model on random targets and extrapolate. Reads no labels."""
    settings = Settings()
    frames, _ = load_feature_sets(features_dir, planned)
    rng = np.random.default_rng(settings.seed)
    n = len(next(iter(frames.values())))
    fake = {t: rng.normal(size=n) for t in TARGETS}
    groups = tuple(range(n))
    probe = Settings(n_permutations=n_probe)
    tasks = make_tasks()
    calls = [
        delayed(_timed)(
            task,
            frames[task.feature_set][list(NLP_COLUMNS)].to_numpy(dtype=np.float64),
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
    lines = [f"{jobs} worker(s), timed on random targets:"]
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
    permutation_baseline(
        matrix,
        truth,
        groups,
        model=task.model,
        seed=probe.seed,
        n_permutations=probe.n_permutations,
        folds=probe.permutation_folds,
    )
    per_permutation = (time.perf_counter() - start) / (probe.n_permutations + 1)
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
    return time.perf_counter() - start + per_permutation, per_permutation
