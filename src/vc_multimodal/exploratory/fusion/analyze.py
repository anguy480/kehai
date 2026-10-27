"""The fusion analysis docs/prereg/fusion.md fixes, with its Amendment 1.

Run by the label holder, from a clean checkout. It reads four tables, each
refused unless its SHA-256 is the one already fixed for it: the text and LLM
tables through the LLM ratings loader, the bundle and the OpenFace face table
through the per-modality loader. Only prosody and turn-taking are taken from
the bundle, whose MediaPipe face columns share the OpenFace names, and every
column is checked against the table its modality is planned to come from.

Session 43 is left out of every model. On the other 61 it fits 15 feature sets
x 2 outcomes x 2 models with the confirmatory pipelines (`modeling/evaluate.py`,
unchanged): leave-one-out for all, repeated 5-fold for the full model and text,
and a 1000-permutation null under 5-fold for the full model. Four primary
tests, Holm across them: the full model against text by Wilcoxon on per-session
leave-one-out errors, and the full model against chance, for each outcome, with
the elastic net. Then 14 descriptive comparisons with paired bootstrap
intervals, as in Amendment 1 of the LLM ratings plan: 112 intervals.

What is written is summary statistics only: no label value, feature value,
prediction, residual or per-session error.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import reduce
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from vc_multimodal.exploratory.fusion.plan import (
    COMPARISONS,
    EXCLUDED_SESSIONS,
    FEATURE_SETS,
    FLAGGED_SESSIONS,
    FULL,
    MODALITIES,
    PERMUTED_SETS,
    PREREG,
    SOURCES,
    STABILITY_SETS,
    TABLE_BUNDLE,
    TABLE_FACE,
    TABLE_LLM,
    TABLE_TEXT,
    TARGETS,
    TEXT_ONLY,
    set_columns,
)
from vc_multimodal.exploratory.llm_ratings import analyze as language
from vc_multimodal.exploratory.llm_ratings import intervals
from vc_multimodal.exploratory.per_modality import analyze as per_modality
from vc_multimodal.exploratory.per_modality.plan import RUN_FIRST, RUN_SECOND, PlanError
from vc_multimodal.exploratory.per_modality.plan import Plan as PerModalityPlan
from vc_multimodal.exploratory.per_modality.plan import load_plan as load_per_modality_plan
from vc_multimodal.io_utils import write_csv
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.modeling.evaluate import (
    Permutation,
    compare_errors,
    evaluate,
    permutation_baseline,
)
from vc_multimodal.provenance import environment_record
from vc_multimodal.stages.model import ModelError, load_labels

logger = get_logger(__name__)

STAGE: Final = "fusion"
MODELS: Final = language.MODELS

PRIMARY_FILE: Final = "primary_tests.csv"
MODELS_FILE: Final = "models.csv"
CONTRIBUTIONS_FILE: Final = "contributions.csv"
SUMMARY_FILE: Final = "summary.md"
RUN_FILE: Final = "run.json"

#: Columns of `models.csv`, in order. None of them holds a label.
MODEL_COLUMNS: Final = (
    "feature_set",
    "target",
    "model",
    "role",
    "n",
    "n_features",
    "loo_r2",
    "loo_mae",
    "loo_spearman",
    "loo_spearman_p",
    "kfold_r2_mean",
    "kfold_r2_sd",
    "kfold_sign_disagrees",
    "perm_scheme",
    "perm_observed_r2",
    "perm_null_mean",
    "perm_null_p95",
    "perm_p",
    "n_permutations",
    "n_fits_not_converged",
)


class AnalyzeError(RuntimeError):
    """Raised when the analysis cannot be run as pre-registered."""


@dataclass(frozen=True, slots=True)
class Settings:
    """The pre-registered settings. Only tests change them."""

    seed: int = 20260923
    stability_folds: int = 5
    stability_repeats: int = 20
    n_permutations: int = 1000
    permutation_folds: int = 5
    n_resamples: int = intervals.N_RESAMPLES

    @property
    def planned(self) -> bool:
        """Whether these are the pre-registered settings."""
        return self == Settings()


@dataclass(frozen=True, slots=True)
class Paths:
    """The four tables."""

    text: Path
    llm: Path
    bundle: Path
    face: Path


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Inputs:
    """The 71 features of the analysed sessions, and where each came from."""

    frame: pd.DataFrame
    digests: dict[str, str]
    sources: dict[str, str]
    notes: tuple[str, ...]


def check_sources(sources: Mapping[str, str]) -> None:
    """Every planned column comes from its modality's table, and nothing else is used.

    Raises:
        AnalyzeError: if a column is missing, comes from another table (a face
            column from the bundle, above all), or is not planned.
    """
    planned = {c: SOURCES[m] for m, columns in MODALITIES.items() for c in columns}
    wrong = [
        f"{column} from the {sources.get(column, 'no table')}, planned from the {table}"
        for column, table in planned.items()
        if sources.get(column) != table
    ]
    extra = sorted(set(sources) - set(planned))
    if wrong or extra:
        msg = "the feature table is not the planned one: " + "; ".join(
            [*wrong, *(f"{c} is not a planned feature" for c in extra)]
        )
        raise AnalyzeError(msg)


def load_inputs(
    paths: Paths,
    *,
    language_plan: language.Planned | None = None,
    per_modality_plan: PerModalityPlan | None = None,
) -> Inputs:
    """The four tables, checked by SHA-256, joined by session, session 43 left out.

    Raises:
        AnalyzeError: if a table is missing or unplanned, the tables cover
            different sessions, or an analysed session has no value at all for
            a modality.
    """
    try:
        words = language.load_inputs(paths.llm, paths.text, language_plan)
        plan = per_modality_plan or load_per_modality_plan()
        bundle = per_modality.load_tables(plan, RUN_FIRST, bundle=paths.bundle)
        face = per_modality.load_tables(plan, RUN_SECOND, face_table=paths.face)
    except (language.AnalyzeError, per_modality.AnalyzeError, PlanError) as exc:
        raise AnalyzeError(str(exc)) from exc

    loaded = {
        TABLE_TEXT: words.frame,
        TABLE_LLM: words.frame,
        TABLE_BUNDLE: bundle.frame,
        TABLE_FACE: face.frame,
    }
    sources: dict[str, str] = {}
    pieces: dict[str, pd.DataFrame] = {}
    for table, frame in loaded.items():
        # Only the columns planned from this table are taken from it.
        columns = [c for m, cols in MODALITIES.items() if SOURCES[m] == table for c in cols]
        piece = frame[["session_id", *columns]].copy()
        piece["session_id"] = piece["session_id"].astype("int64")
        pieces[table] = piece
        sources.update(dict.fromkeys(columns, table))
    check_sources(sources)

    excluded = set(EXCLUDED_SESSIONS)
    cohort = set(pieces[TABLE_TEXT]["session_id"])
    absent = sorted(excluded - cohort)
    if absent:
        msg = f"session(s) {absent}, which the plan leaves out, are not in the text table"
        raise AnalyzeError(msg)
    for table, piece in pieces.items():
        found = set(piece["session_id"]) - excluded
        if found != cohort - excluded:
            msg = (
                f"the {table} covers other sessions than the text table: missing "
                f"{sorted(cohort - excluded - found)}, extra {sorted(found - cohort)}"
            )
            raise AnalyzeError(msg)

    joined = reduce(
        lambda left, right: left.merge(right, on="session_id", how="inner", validate="one_to_one"),
        pieces.values(),
    )
    joined = joined[~joined["session_id"].isin(excluded)]
    joined = joined.sort_values("session_id", kind="mergesort").reset_index(drop=True)
    for modality, features in MODALITIES.items():
        empty = joined.loc[joined[list(features)].isna().all(axis=1), "session_id"]
        if not empty.empty:
            msg = (
                f"session(s) {sorted(int(s) for s in empty)} have no {modality} value at all; "
                f"the plan fixes the sessions, so the run stops"
            )
            raise AnalyzeError(msg)

    digests = {
        **words.digests,
        f"{paths.bundle.name}/{per_modality.BUNDLE_FEATURES}": bundle.digests[
            "base_features_sha256"
        ],
        paths.face.name: face.digests["face_features_sha256"],
    }
    return Inputs(
        frame=joined[["session_id", *set_columns(FULL)]],
        digests=digests,
        sources=sources,
        notes=(*bundle.notes, *face.notes),
    )


@dataclass(frozen=True, slots=True)
class Cohort:
    """The analysed sessions with a label, per outcome."""

    session_ids: np.ndarray
    rows: np.ndarray
    truth: Mapping[str, np.ndarray]
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
    """Match the labels to the analysed sessions by session ID.

    The label values stay in memory: nothing here is written.

    Raises:
        AnalyzeError: if the labels lack an outcome or share no analysed session.
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
    rows = np.array([i for i, s in enumerate(ours) if s in theirs], dtype=np.int64)
    if rows.size == 0:
        msg = "no analysed session appears in the labels file"
        raise AnalyzeError(msg)
    shared = [ours[i] for i in rows]
    indexed = frame.assign(session_id=[int(s) for s in frame["session_id"]]).set_index("session_id")
    truth = {
        t: pd.to_numeric(indexed.loc[shared, t], errors="coerce").to_numpy(dtype=np.float64)
        for t in TARGETS
    }
    return Cohort(
        session_ids=np.array(shared, dtype=np.int64),
        rows=rows,
        truth=truth,
        features_only=tuple(s for s in ours if s not in theirs),
        labels_only=len(theirs - set(ours) - set(EXCLUDED_SESSIONS)),
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

    @property
    def key(self) -> tuple[str, str, str]:
        """How its errors and its null are looked up."""
        return self.feature_set, self.target, self.model


def make_tasks() -> list[Task]:
    """Every model the plan fits: 15 feature sets x 2 outcomes x 2 models = 60."""
    return [
        Task(feature_set, target, model, role)
        for feature_set in FEATURE_SETS
        for target in TARGETS
        for model, role in MODELS
    ]


@dataclass(frozen=True, slots=True)
class Fitted:
    """A model's summary row, and its per-session absolute errors, never written."""

    task: Task
    row: dict[str, Any]
    errors: np.ndarray


@dataclass(frozen=True, slots=True)
class Null:
    """A model's permutation null."""

    task: Task
    permutation: Permutation | None


def fit(
    task: Task, matrix: np.ndarray, truth: np.ndarray, groups: Sequence[object], settings: Settings
) -> Fitted:
    """Leave-one-out, and repeated 5-fold for the full model and text."""
    start = time.perf_counter()
    stable = task.feature_set in STABILITY_SETS
    evaluation = evaluate(
        matrix,
        truth,
        groups,
        model=task.model,
        seed=settings.seed,
        stability_folds=settings.stability_folds if stable else 0,
        stability_repeats=settings.stability_repeats if stable else 0,
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
        "kfold_sign_disagrees": evaluation.disagrees_with_stability if stable else None,
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
    return Fitted(task, row, evaluation.errors)


def null(
    task: Task, matrix: np.ndarray, truth: np.ndarray, groups: Sequence[object], settings: Settings
) -> Null:
    """The permutation null under 5-fold on the first split, not leave-one-out."""
    start = time.perf_counter()
    permutation = permutation_baseline(
        matrix,
        truth,
        groups,
        model=task.model,
        seed=settings.seed,
        n_permutations=settings.n_permutations,
        folds=settings.permutation_folds,
    )
    logger.info(
        "%s: %s %s %s null done in %.0f s",
        STAGE,
        task.feature_set,
        task.target,
        task.model,
        time.perf_counter() - start,
    )
    return Null(task, permutation)


def _with_null(row: Mapping[str, Any], permutation: Permutation | None) -> dict[str, Any]:
    return {
        **row,
        "perm_scheme": None if permutation is None else permutation.scheme,
        "perm_observed_r2": None if permutation is None else permutation.observed,
        "perm_null_mean": None if permutation is None else permutation.null_mean,
        "perm_null_p95": None if permutation is None else permutation.null_p95,
        "perm_p": None if permutation is None else permutation.p_value,
        "n_permutations": 0 if permutation is None else permutation.n_permutations,
    }


def _row(models: pd.DataFrame, feature_set: str, target: str, model: str) -> pd.Series:
    rows = models[
        (models["feature_set"] == feature_set)
        & (models["target"] == target)
        & (models["model"] == model)
    ]
    return rows.iloc[0]


def primary_tests(
    models: pd.DataFrame, errors: Mapping[tuple[str, str, str], np.ndarray]
) -> pd.DataFrame:
    """The four primary tests, Holm across them, then the four secondary ones uncorrected."""
    rows: list[dict[str, Any]] = []
    for model, role in MODELS:
        for target in TARGETS:
            result = compare_errors(
                errors[(TEXT_ONLY, target, model)], errors[(FULL, target, model)]
            )
            rows.append(
                {
                    "test": "full_vs_text",
                    "target": target,
                    "model": model,
                    "role": role,
                    "n": result.n,
                    "statistic_name": "median_error_difference",
                    "statistic": result.median_difference,
                    "scheme": "leave-one-out, paired by session",
                    "p": result.p_value,
                }
            )
        for target in TARGETS:
            full = _row(models, FULL, target, model)
            rows.append(
                {
                    "test": "full_vs_chance",
                    "target": target,
                    "model": model,
                    "role": role,
                    "n": int(full["n"]),
                    "statistic_name": "kfold_r2",
                    "statistic": full["perm_observed_r2"],
                    "scheme": f"{full['perm_scheme']} on the first split",
                    "p": full["perm_p"],
                }
            )
    table = pd.DataFrame(rows)
    primary = (table["role"] == "primary").to_numpy()
    p = [np.nan if v is None else float(v) for v in table.loc[primary, "p"]]
    table["p_holm"] = np.nan
    table.loc[primary, "p_holm"] = language.holm_finite(p)
    return table


def stability_checks(models: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out against repeated 5-fold: full, text, and full minus text (ADR 15)."""
    rows = []
    for model, _ in MODELS:
        for target in TARGETS:
            full, text = (_row(models, s, target, model) for s in (FULL, TEXT_ONLY))
            pairs = (
                (FULL, full["loo_r2"], full["kfold_r2_mean"]),
                (TEXT_ONLY, text["loo_r2"], text["kfold_r2_mean"]),
                (
                    "full minus text",
                    full["loo_r2"] - text["loo_r2"],
                    full["kfold_r2_mean"] - text["kfold_r2_mean"],
                ),
            )
            for what, loo, kfold in pairs:
                rows.append(
                    {
                        "what": what,
                        "target": target,
                        "model": model,
                        "loo_r2": float(loo),
                        "kfold_r2": float(kfold),
                        "sign_disagrees": bool((loo > 0) != (kfold > 0)),
                    }
                )
    return pd.DataFrame(rows)


def contributions(
    cohort: Cohort,
    models: pd.DataFrame,
    errors: Mapping[tuple[str, str, str], np.ndarray],
    settings: Settings,
) -> pd.DataFrame:
    """The 112 intervals: 14 comparisons x 2 outcomes x 2 models x (ΔR², ΔMAE)."""
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        truth = cohort.truth[target][cohort.mask(target)]
        # Drawn once per outcome and shared by every comparison and both models.
        indices = intervals.resample_indices(truth.size, settings.n_resamples, settings.seed)
        for model, role in MODELS:
            for comparison in COMPARISONS:
                larger = _row(models, comparison.larger, target, model)
                smaller = _row(models, comparison.smaller, target, model)
                delta_r2, delta_mae = intervals.bootstrap_differences(
                    truth,
                    errors[(comparison.smaller, target, model)],
                    errors[(comparison.larger, target, model)],
                    indices,
                )
                for metric, estimate, resampled in (
                    (intervals.DELTA_R2, larger["loo_r2"] - smaller["loo_r2"], delta_r2),
                    (intervals.DELTA_MAE, larger["loo_mae"] - smaller["loo_mae"], delta_mae),
                ):
                    low, high, used = intervals.percentile_interval(resampled)
                    rows.append(
                        {
                            "comparison": comparison.name,
                            "kind": comparison.kind,
                            "larger": comparison.larger,
                            "smaller": comparison.smaller,
                            "target": target,
                            "model": model,
                            "role": role,
                            "n": int(larger["n"]),
                            "metric": metric,
                            "estimate": float(estimate),
                            "ci_low": low,
                            "ci_high": high,
                            "level": intervals.LEVEL,
                            "n_resamples": settings.n_resamples,
                            "n_resamples_used": used,
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RunResult:
    """What one run did, for the terminal. No result value is printed."""

    out_dir: Path
    n_by_target: Mapping[str, int]
    features_only: tuple[int, ...]
    labels_only: int
    notes: tuple[str, ...]
    n_models: int
    n_nulls: int
    n_intervals: int
    written: tuple[Path, ...]

    def report_lines(self) -> list[str]:
        """Coverage, counts and the files written."""
        lines = [
            "sessions analysed: " + ", ".join(f"{t} {n}" for t, n in self.n_by_target.items()),
            f"  left out of every model: {list(EXCLUDED_SESSIONS)}; flagged, kept: "
            f"{list(FLAGGED_SESSIONS)}",
        ]
        if self.features_only:
            lines.append(f"  features but no label, left out: {list(self.features_only)}")
        if self.labels_only:
            lines.append(f"  {self.labels_only} labelled session(s) have no features")
        lines.extend(f"  note: {note}" for note in self.notes)
        lines.append(
            f"models: {self.n_models} leave-one-out estimates, {self.n_nulls} permutation "
            f"nulls; intervals: {self.n_intervals}"
        )
        lines.extend(f"wrote {path}" for path in self.written)
        return lines


def _matrix(inputs: Inputs, cohort: Cohort, feature_set: str) -> np.ndarray:
    matrix: np.ndarray = inputs.frame[list(set_columns(feature_set))].to_numpy(dtype=np.float64)
    analysed: np.ndarray = matrix[cohort.rows]
    return analysed


def run(
    paths: Paths,
    labels_path: Path,
    out_dir: Path,
    *,
    commit: str,
    jobs: int = 1,
    settings: Settings | None = None,
    language_plan: language.Planned | None = None,
    per_modality_plan: PerModalityPlan | None = None,
) -> RunResult:
    """Fit every planned model, test, resample, and write the summaries.

    Raises:
        AnalyzeError: if the inputs are not the planned ones, or `out_dir`
            already holds files.
    """
    settings = settings or Settings()
    try:
        language.refuse_used_folder(out_dir)
    except language.AnalyzeError as exc:
        raise AnalyzeError(str(exc)) from exc
    started = datetime.now(UTC)
    inputs = load_inputs(paths, language_plan=language_plan, per_modality_plan=per_modality_plan)
    cohort = load_cohort(inputs, labels_path)

    nulls, fits = [], []
    for task in make_tasks():
        mask = cohort.mask(task.target)
        arguments = (
            task,
            _matrix(inputs, cohort, task.feature_set)[mask],
            cohort.truth[task.target][mask],
            tuple(cohort.session_ids[mask]),
            settings,
        )
        fits.append(delayed(fit)(*arguments))
        if task.feature_set in PERMUTED_SETS:
            nulls.append(delayed(null)(*arguments))
    logger.info(
        "%s: %d model(s) and %d null(s) on %d worker(s)", STAGE, len(fits), len(nulls), jobs
    )
    # The nulls are the long calls, so they start first.
    done: list[Fitted | Null] = list(Parallel(n_jobs=jobs)([*nulls, *fits]))
    by_null = {d.task.key: d.permutation for d in done if isinstance(d, Null)}
    fitted = [d for d in done if isinstance(d, Fitted)]
    models = pd.DataFrame(
        [_with_null(f.row, by_null.get(f.task.key)) for f in fitted], columns=list(MODEL_COLUMNS)
    )
    errors = {f.task.key: f.errors for f in fitted}

    primary = primary_tests(models, errors)
    checks = stability_checks(models)
    logger.info("%s: resampling %d times per outcome", STAGE, settings.n_resamples)
    table = contributions(cohort, models, errors, settings)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        write_csv(out_dir / PRIMARY_FILE, primary),
        write_csv(out_dir / MODELS_FILE, models),
        write_csv(out_dir / CONTRIBUTIONS_FILE, table),
    ]
    record = {
        "analysis": "exploratory/fusion",
        "prereg": f"{PREREG}, with Amendment 1",
        "commit": commit,
        "inputs": inputs.digests,
        "settings": asdict(settings),
        "planned_settings": settings.planned,
        "sessions_analysed": cohort.counts(),
        "left_out": list(EXCLUDED_SESSIONS),
        "flagged": list(FLAGGED_SESSIONS),
        "features_only": list(cohort.features_only),
        "labels_only": cohort.labels_only,
        "notes": list(inputs.notes),
        "feature_sets": {name: len(set_columns(name)) for name in FEATURE_SETS},
        "n_comparisons": len(COMPARISONS),
        "n_intervals": len(table),
        "started": started.isoformat(),
        "finished": datetime.now(UTC).isoformat(),
        "environment": environment_record(),
    }
    (out_dir / RUN_FILE).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    (out_dir / SUMMARY_FILE).write_text(
        summary_markdown(primary, models, checks, table, record), encoding="utf-8"
    )
    written += [out_dir / RUN_FILE, out_dir / SUMMARY_FILE]
    return RunResult(
        out_dir=out_dir,
        n_by_target=cohort.counts(),
        features_only=cohort.features_only,
        labels_only=cohort.labels_only,
        notes=inputs.notes,
        n_models=len(models),
        n_nulls=len(by_null),
        n_intervals=len(table),
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
    checks: pd.DataFrame,
    table: pd.DataFrame,
    record: Mapping[str, Any],
) -> str:
    """A readable summary, for the person who ran it, with the plan's caveats."""
    caveat = (
        ""
        if record["planned_settings"]
        else "\n**Not the pre-registered settings: a test run, not a result.**\n"
    )
    disagreements = checks[checks["sign_disagrees"]]
    disagree_text = (
        "None: leave-one-out and repeated 5-fold agree on every sign."
        if disagreements.empty
        else "**They disagree** on the sign of: "
        + "; ".join(f"{r.what}, {r.target}, {r.model}" for r in disagreements.itertuples())
        + ". Where they disagree, the disagreement is the finding (ADR 15)."
    )
    return "\n".join(
        [
            "# All modalities combined, and what each one contributes",
            "",
            f"**Exploratory**, pre-registered in `{PREREG}` with Amendment 1, after the labels "
            "and the results of the earlier analyses had been seen. No result here is "
            "confirmatory.",
            caveat,
            f"Commit `{record['commit']}`. Sessions analysed: {record['sessions_analysed']}. "
            f"Left out of every model: {record['left_out']}. Flagged, kept: {record['flagged']}.",
            "",
            "These 61-session estimates are not comparable with the 62-session analyses "
            "(the confirmatory run, the tokenization rerun, the LLM ratings analysis). Compare "
            "models within this analysis only.",
            "",
            "## Primary tests",
            "",
            "Holm across the four elastic net tests; the random forest rows are uncorrected. "
            "`full_vs_text`: Wilcoxon signed-rank on per-session leave-one-out absolute errors; "
            "`median_error_difference` is full minus text, so negative means the full model "
            "was closer. `full_vs_chance`: the 5-fold R² on the first split against its "
            "1000-permutation null under the same split, not a leave-one-out p.",
            "",
            _markdown(
                primary,
                ["test", "target", "model", "n", "statistic_name", "statistic", "p", "p_holm"],
            ),
            "",
            "## Leave-one-out against repeated 5-fold",
            "",
            disagree_text,
            "",
            _markdown(checks, ["what", "target", "model", "loo_r2", "kfold_r2", "sign_disagrees"]),
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
                    "n_features",
                    "loo_r2",
                    "loo_mae",
                    "loo_spearman",
                    "kfold_r2_mean",
                    "perm_p",
                    "n_fits_not_converged",
                ],
            ),
            "",
            "## Contributions: descriptive",
            "",
            f"{record['n_intervals']} intervals, no test and no correction: whether one excludes "
            "zero is not a test, and reading one as a finding is the multiplicity problem ADR 12 "
            "exists to avoid. Each difference is the larger model minus the smaller, so a "
            "contribution is a positive `delta_r2` and a negative `delta_mae`. 95% percentile "
            f"intervals of a {record['settings']['n_resamples']}-resample paired bootstrap over "
            "sessions. The bootstrap does not refit the models, so the intervals are likely "
            "somewhat too narrow.",
            "",
            _markdown(
                table,
                [
                    "comparison",
                    "kind",
                    "target",
                    "model",
                    "metric",
                    "estimate",
                    "ci_low",
                    "ci_high",
                    "n_resamples_used",
                ],
            ),
            "",
            "## Reading the results",
            "",
            "- Drop-one undercounts modalities that overlap: the text features include seven "
            "LLM-rated `Agenda_*` scores, and the two face windows measure the same AUs. The "
            "grouped drops measure each pair together.",
            "- Drop-one asks what a modality adds given all the others; add-one, what it adds to "
            "text alone. Neither is a share of the full model's R², and the differences are not "
            "expected to sum to anything.",
            "- The elastic net gives more pull to modalities with more features, and the random "
            "forest has a similar tilt. Contributions are not adjusted for this.",
            "- 71 features on 60 training sessions per fold: the full model rests on "
            "regularisation; see `n_fits_not_converged`.",
            "- Amendment 1's `text_plus_nonverbal` is neither the confirmatory test nor a rerun of "
            "the confirmatory run's `text_plus_all`.",
            "",
            "## Provenance",
            "",
            "- Inputs: "
            + ", ".join(f"`{name}` `{digest[:16]}...`" for name, digest in record["inputs"].items())
            + ".",
            f"- Settings: {record['settings']}.",
            f"- Started {record['started']}, finished {record['finished']}.",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Timing, without labels
# ---------------------------------------------------------------------------
def estimate_runtime(
    paths: Paths,
    *,
    jobs: int = 1,
    n_probe: int = 4,
    language_plan: language.Planned | None = None,
    per_modality_plan: PerModalityPlan | None = None,
) -> list[str]:
    """Time each feature set on random targets and extrapolate. Reads no labels."""
    settings = Settings()
    inputs = load_inputs(paths, language_plan=language_plan, per_modality_plan=per_modality_plan)
    n = len(inputs.frame)
    fake = np.random.default_rng(settings.seed).normal(size=n)
    groups = tuple(range(n))
    probe = replace(settings, n_permutations=n_probe)
    pairs = [(feature_set, model) for feature_set in FEATURE_SETS for model, _ in MODELS]
    calls = [
        delayed(_timed)(
            feature_set,
            model,
            inputs.frame[list(set_columns(feature_set))].to_numpy(dtype=np.float64),
            fake,
            groups,
            probe,
        )
        for feature_set, model in pairs
    ]
    timings: list[tuple[float, float | None]] = list(Parallel(n_jobs=jobs)(calls))
    by_pair = dict(zip(pairs, timings, strict=True))

    n_targets = len(TARGETS)
    null_calls = [
        settings.n_permutations * per for _, per in timings if per is not None
    ] * n_targets
    work = n_targets * sum(fit_s for fit_s, _ in timings) + sum(null_calls)
    wall = max(work / max(1, jobs), *null_calls)
    lines = [
        f"{jobs} worker(s), {n} sessions, timed on random targets, one outcome each "
        f"(the run fits both):"
    ]
    for feature_set in FEATURE_SETS:
        (net, _), (forest, _) = (by_pair[(feature_set, model)] for model, _ in MODELS)
        lines.append(f"  {feature_set}: elastic net {net:.1f} s, random forest {forest:.1f} s")
    for (feature_set, model), (_, per) in by_pair.items():
        if per is not None:
            lines.append(
                f"  {feature_set} {model} null: {per:.2f} s per permutation, about "
                f"{settings.n_permutations * per / 60:.0f} min per outcome"
            )
    lines.append(f"  about {wall / 3600:.1f} h with {jobs} worker(s)")
    return lines


def _timed(
    feature_set: str,
    model: str,
    matrix: np.ndarray,
    truth: np.ndarray,
    groups: Sequence[object],
    probe: Settings,
) -> tuple[float, float | None]:
    """Seconds for one model's estimates, and per permutation where it has a null."""
    stable = feature_set in STABILITY_SETS
    start = time.perf_counter()
    evaluate(
        matrix,
        truth,
        groups,
        model=model,
        seed=probe.seed,
        stability_folds=probe.stability_folds if stable else 0,
        stability_repeats=probe.stability_repeats if stable else 0,
    )
    fit_s = time.perf_counter() - start
    if feature_set not in PERMUTED_SETS:
        return fit_s, None
    start = time.perf_counter()
    permutation_baseline(
        matrix,
        truth,
        groups,
        model=model,
        seed=probe.seed,
        n_permutations=probe.n_permutations,
        folds=probe.permutation_folds,
    )
    return fit_s, (time.perf_counter() - start) / max(1, probe.n_permutations)
