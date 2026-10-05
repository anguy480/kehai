"""The AU-only baseline: every planned set on K6 and SRS2, with its nulls.

Exploratory, requested after unblinding. Nothing here is corrected for
multiplicity and nothing supports a confirmatory claim. The labels are read
through `vc model`'s loader and go no further: no file written here holds a
label, a prediction, a residual or a per-session error.

The question is whether the action unit intensities carry K6 information on
their own. The per-set nulls say how each set does against shuffled labels;
the max-statistic null says whether the *best* of all the AU sets does better
than the best of them does on shuffled labels, which is the answer to whether
any combination beats chance once the search over combinations is counted.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats

from vc_multimodal.config import AppConfig
from vc_multimodal.exploratory.au_baseline import cv
from vc_multimodal.exploratory.au_baseline.plan import (
    PLAN_PATH,
    FeatureSet,
    Plan,
    column_names,
    resolve,
)
from vc_multimodal.exploratory.au_baseline.pooled import file_sha256, pooled_path
from vc_multimodal.exploratory.au_baseline.search import SearchSpec
from vc_multimodal.io_utils import read_csv, write_csv, write_json, write_text
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.modeling.text_features import BUNDLE_FILE, load_bundled
from vc_multimodal.modeling.text_features import join as join_text
from vc_multimodal.paths import DataRoots
from vc_multimodal.stages import model as model_stage

logger = get_logger(__name__)

STAGE: Final = "au-baseline-analyze"
TIER: Final = "exploratory"
FEATURES_FILE: Final = "features.csv"

ESTIMATES_FILE: Final = "estimates.csv"
SPEARMAN_FILE: Final = "single_au_spearman.csv"
MAX_FILE: Final = "max_statistic.csv"
MAX_NULL_FILE: Final = "max_statistic_null.csv"
SELECTION_FILE: Final = "search_selections.csv"
SENSITIVITY_FILE: Final = "sensitivity.csv"
SUMMARY_FILE: Final = "summary.md"
RUN_FILE: Final = "run.json"

#: Max-statistic scope covering both models.
BOTH_MODELS: Final = "both_models"

#: Permutation chunks per worker, for load balancing.
_CHUNKS_PER_JOB: Final = 4

#: Fewest pairs a Spearman correlation is computed from.
_MIN_SPEARMAN_PAIRS: Final = 3


class AnalyzeError(RuntimeError):
    """Raised when the analysis cannot be run on the inputs supplied."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _check_digest(path: Path, expected: str, what: str) -> str:
    if not expected:
        msg = f"no digest is frozen for the {what} in {PLAN_PATH}; run the pooled step first"
        raise AnalyzeError(msg)
    if not path.is_file():
        msg = f"the {what} is not at {path}"
        raise AnalyzeError(msg)
    digest = file_sha256(path)
    if digest != expected:
        msg = f"the {what} at {path} has digest {digest}, not the frozen {expected}"
        raise AnalyzeError(msg)
    return digest


@dataclass(frozen=True, slots=True)
class Inputs:
    """The joined feature table and the digests of what went into it."""

    table: pd.DataFrame
    digests: dict[str, str]


def load_inputs(config: AppConfig, roots: DataRoots, plan: Plan, bundle: Path) -> Inputs:
    """The frozen features, text features and pooled AUs, joined by session.

    Raises:
        AnalyzeError: if an input is missing or is not the frozen file.
    """
    if config.model.text_features is None:
        msg = "model.text_features is not configured, so there is no text reference"
        raise AnalyzeError(msg)
    features_path = bundle / FEATURES_FILE
    pooled = pooled_path(roots)
    digests = {
        "features_sha256": _check_digest(
            features_path, plan.inputs.bundle_features_sha256, "bundle features.csv"
        ),
        "pooled_sha256": _check_digest(
            pooled, plan.inputs.pooled_features_sha256, "pooled AU table"
        ),
        "text_features_sha256": file_sha256(bundle / BUNDLE_FILE),
        "plan_sha256": file_sha256(PLAN_PATH),
    }
    features = read_csv(features_path)
    text = load_bundled(bundle / BUNDLE_FILE, config.model.text_features)
    features, _ = join_text(features, text)
    table = features.merge(read_csv(pooled), on="session_id", how="left", validate="one_to_one")
    return Inputs(table=table, digests=digests)


def make_tasks(
    sets: Sequence[FeatureSet], table: pd.DataFrame, plan: Plan, seed: int
) -> list[cv.Task]:
    """A fit-ready task per set, rows in the table's order."""
    spec = SearchSpec(plan.search.max_features, plan.search.inner_folds, seed)
    return [
        cv.Task(
            name=s.name,
            matrix=table[list(s.columns)].to_numpy(dtype=np.float64),
            search=spec if s.is_search else None,
        )
        for s in sets
    ]


def target_vector(labels: pd.DataFrame, target: str) -> np.ndarray:
    """One outcome as floats.

    Raises:
        AnalyzeError: if any session lacks it. Dropping sessions per target
            would give the two targets different cohorts and nulls.
    """
    values = pd.to_numeric(labels[target], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        msg = f"{int(np.isnan(values).sum())} session(s) have no {target} value"
        raise AnalyzeError(msg)
    return values


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def _parallel(jobs: int, calls: Iterable[Any], what: str, total: int) -> list[Any]:
    """Run delayed calls on `jobs` processes, in order, logging progress."""
    results: list[Any] = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        runner = Parallel(n_jobs=jobs, return_as="generator")
        step = max(1, total // 10)
        for done, result in enumerate(runner(calls), start=1):
            results.append(result)
            if done % step == 0 or done == total:
                logger.info("%s: %s %d/%d", STAGE, what, done, total)
    return results


def observed_scores(
    tasks: Sequence[cv.Task],
    target: np.ndarray,
    groups: Sequence[object],
    plan: Plan,
    *,
    seed: int,
    jobs: int,
) -> dict[tuple[str, str], cv.SetScores]:
    """Every set under every model, keyed by (set, model)."""
    models = plan.evaluation.models
    pairs = [(task, model) for task in tasks for model in models]
    results = _parallel(
        jobs,
        (
            delayed(cv.evaluate_set)(
                task,
                target,
                groups,
                model=model,
                seed=seed,
                folds=plan.evaluation.kfold_folds,
                repeats=plan.evaluation.kfold_repeats,
            )
            for task, model in pairs
        ),
        "observed",
        len(pairs),
    )
    return {(task.name, model): r for (task, model), r in zip(pairs, results, strict=True)}


def permutation_orders(n_rows: int, n_permutations: int, seed: int) -> np.ndarray:
    """The shared label permutations, shape (permutations, rows)."""
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(n_rows) for _ in range(n_permutations)], dtype=np.int64)


def null_scores(
    tasks: Sequence[cv.Task],
    target: np.ndarray,
    groups: Sequence[object],
    orders: np.ndarray,
    plan: Plan,
    *,
    seed: int,
    jobs: int,
) -> np.ndarray:
    """Null R², shape (permutations, sets, models)."""
    n_chunks = min(len(orders), max(1, jobs * _CHUNKS_PER_JOB))
    chunks = [c for c in np.array_split(orders, n_chunks) if len(c)]
    results = _parallel(
        jobs,
        (
            delayed(cv.null_chunk)(
                tasks,
                target,
                groups,
                chunk,
                models=plan.evaluation.models,
                seed=seed,
                folds=plan.evaluation.kfold_folds,
            )
            for chunk in chunks
        ),
        "permutation chunks",
        len(chunks),
    )
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def estimate_rows(
    sets: Sequence[FeatureSet],
    target: str,
    observed: dict[tuple[str, str], cv.SetScores],
    null: np.ndarray | None,
    models: Sequence[str],
) -> list[dict[str, object]]:
    """One row per (set, model) for one target."""
    rows: list[dict[str, object]] = []
    for t, s in enumerate(sets):
        for m, model in enumerate(models):
            scores = observed[(s.name, model)]
            row: dict[str, object] = {
                "feature_set": s.name,
                "aliases": ";".join(s.aliases),
                "kind": s.kind,
                "group": s.group,
                "window": s.window,
                "in_max_family": s.in_family,
                "model": model,
                "target": target,
                "tier": TIER,
                "n_sessions": scores.n,
                "n_features": len(s.columns),
                "loo_r2": scores.loo_r2,
                "loo_spearman": scores.loo_spearman,
                "loo_spearman_p_uncorrected": scores.loo_spearman_p,
                "kfold_r2_mean": scores.kfold_r2_mean,
                "kfold_r2_sd": scores.kfold_r2_sd,
                "kfold_spearman_mean": scores.kfold_spearman_mean,
                "perm_split_r2": scores.split_r2,
                "n_fits_not_converged": scores.n_not_converged,
            }
            if null is not None and len(null):
                summary = cv.set_null(scores.split_r2, null[:, t, m])
                row.update(
                    {
                        "perm_null_mean_r2": summary.null_mean,
                        "perm_null_p95_r2": summary.null_p95,
                        "perm_p_uncorrected": summary.p_value,
                        "n_permutations": len(null),
                    }
                )
            rows.append(row)
    return rows


def max_statistic_rows(
    sets: Sequence[FeatureSet],
    target: str,
    observed: dict[tuple[str, str], cv.SetScores],
    null: np.ndarray,
    models: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    """The max-statistic test per scope: both models together, then each alone."""
    family = [t for t, s in enumerate(sets) if s.in_family]
    names = [sets[t].name for t in family]
    scopes = {BOTH_MODELS: list(range(len(models)))} | {m: [i] for i, m in enumerate(models)}
    rows: list[dict[str, object]] = []
    nulls: dict[str, np.ndarray] = {}
    for scope, model_index in scopes.items():
        scope_models = [models[i] for i in model_index]
        obs = np.array(
            [[observed[(sets[t].name, m)].split_r2 for m in scope_models] for t in family]
        )
        result = cv.max_statistic(obs, null[:, family][:, :, model_index], names, scope_models)
        nulls[scope] = result.null_max
        rows.append(
            {
                "target": target,
                "scope": scope,
                "tier": TIER,
                "n_sets": len(family),
                "n_models": len(scope_models),
                "best_set": result.best_set,
                "best_model": result.best_model,
                "observed_best_split_r2": result.observed,
                "null_max_mean": float(np.mean(result.null_max)),
                "null_max_p50": float(np.percentile(result.null_max, 50)),
                "null_max_p95": float(np.percentile(result.null_max, 95)),
                "null_max_p99": float(np.percentile(result.null_max, 99)),
                "percentile_of_observed": result.percentile,
                "max_stat_p": result.p_value,
                "n_permutations": len(result.null_max),
            }
        )
    return rows, nulls


def single_au_columns(plan: Plan) -> list[tuple[str, str, str]]:
    """(unit, window, column) for each single-unit mean, per single window."""
    out: list[tuple[str, str, str]] = []
    for au_set in plan.au_sets:
        if len(au_set.units) != 1 or au_set.name != au_set.units[0]:
            continue
        for window, prefixes in plan.windows.items():
            if len(prefixes) != 1:
                continue
            out.extend(
                (au_set.units[0], window, column)
                for column in column_names(prefixes, au_set.units, ("mean",))
            )
    return out


def single_au_spearman(
    features: pd.DataFrame, labels: pd.DataFrame, plan: Plan, targets: Sequence[str]
) -> pd.DataFrame:
    """Plain Spearman of each single-AU mean with each outcome, uncorrected."""
    rows: list[dict[str, object]] = []
    for unit, window, column in single_au_columns(plan):
        x = features[column].to_numpy(dtype=np.float64)
        usable = ~np.isnan(x)
        for target in targets:
            y = target_vector(labels, target)
            rho = p = float("nan")
            if usable.sum() >= _MIN_SPEARMAN_PAIRS and np.ptp(x[usable]) > 0:
                result = stats.spearmanr(x[usable], y[usable])
                rho, p = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "unit": unit,
                    "window": window,
                    "column": column,
                    "target": target,
                    "tier": TIER,
                    "n": int(usable.sum()),
                    "spearman": rho,
                    "p_uncorrected": p,
                }
            )
    return pd.DataFrame(rows)


def selection_rows(
    search: FeatureSet, target: str, scores: cv.SetScores
) -> list[dict[str, object]]:
    """How often each candidate was chosen across the leave-one-out folds."""
    folds = scores.loo_selections
    sizes = [len(chosen) for chosen in folds]
    return [
        {
            "target": target,
            "tier": TIER,
            "candidate": column,
            "n_outer_folds": len(folds),
            "times_selected": sum(1 for chosen in folds if index in chosen),
            "times_selected_first": sum(1 for chosen in folds if chosen[:1] == (index,)),
            "mean_subset_size": float(np.mean(sizes)) if sizes else float("nan"),
        }
        for index, column in enumerate(search.columns)
    ]


def top_pairs(estimates: pd.DataFrame, target: str, n: int) -> list[tuple[str, str]]:
    """The best (set, model) pairs of the max-statistic family by LOO R² on `target`."""
    pool = estimates[(estimates["target"] == target) & estimates["in_max_family"].astype(bool)]
    best = pool.sort_values("loo_r2", ascending=False).head(n)
    return [(str(r.feature_set), str(r.model)) for r in best.itertuples()]


def run_sensitivity(
    estimates: pd.DataFrame,
    sets: Sequence[FeatureSet],
    cohort: model_stage.Cohort,
    plan: Plan,
    *,
    seed: int,
    jobs: int,
    exclude: Sequence[int],
) -> pd.DataFrame:
    """The best pairs by primary LOO R², re-evaluated without `exclude`.

    Raises:
        AnalyzeError: if `exclude` names no session in the cohort.
    """
    if not exclude:
        return pd.DataFrame()
    keep = ~cohort.features["session_id"].isin(list(exclude)).to_numpy()
    if keep.all():
        msg = f"--exclude-session {list(exclude)} names no session in the cohort"
        raise AnalyzeError(msg)
    features = cohort.features[keep].reset_index(drop=True)
    labels = cohort.labels[keep].reset_index(drop=True)
    groups = [g for g, k in zip(cohort.groups, keep, strict=True) if k]
    by_name = {s.name: s for s in sets}
    pairs = top_pairs(estimates, plan.evaluation.primary_target, plan.sensitivity.top_n)
    tasks = {name: make_tasks([by_name[name]], features, plan, seed)[0] for name, _ in pairs}
    calls = [(name, model, target) for name, model in pairs for target in plan.evaluation.targets]
    results = _parallel(
        jobs,
        (
            delayed(cv.evaluate_set)(
                tasks[name],
                target_vector(labels, target),
                groups,
                model=model,
                seed=seed,
                folds=plan.evaluation.kfold_folds,
                repeats=plan.evaluation.kfold_repeats,
            )
            for name, model, target in calls
        ),
        "sensitivity",
        len(calls),
    )
    rows: list[dict[str, object]] = []
    for (name, model, target), scores in zip(calls, results, strict=True):
        full = estimates[
            (estimates["feature_set"] == name)
            & (estimates["model"] == model)
            & (estimates["target"] == target)
        ]
        rows.append(
            {
                "feature_set": name,
                "model": model,
                "target": target,
                "tier": TIER,
                "excluded_sessions": ";".join(str(s) for s in exclude),
                "n_sessions": scores.n,
                "loo_r2": scores.loo_r2,
                "kfold_r2_mean": scores.kfold_r2_mean,
                "loo_spearman": scores.loo_spearman,
                "kfold_spearman_mean": scores.kfold_spearman_mean,
                "full_cohort_loo_r2": float(full["loo_r2"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, bool | np.bool_):
        return "yes" if value else ""
    if isinstance(value, float | np.floating):
        return f"{float(value):.3f}"
    return str(value)


def _markdown(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    head = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join(
        "| " + " | ".join(_fmt(row[c]) for c in columns) + " |\n" for _, row in frame.iterrows()
    )
    return head + body


def wide_table(estimates: pd.DataFrame, primary: str, targets: Sequence[str]) -> pd.DataFrame:
    """One row per (set, model), targets side by side, sorted by primary LOO R²."""
    keys = ["feature_set", "model", "n_features", "in_max_family"]
    values = ["loo_r2", "kfold_r2_mean", "loo_spearman", "kfold_spearman_mean"]
    if "perm_p_uncorrected" in estimates.columns:
        values.append("perm_p_uncorrected")
    wide = estimates[estimates["target"] == primary][keys].reset_index(drop=True)
    for target in targets:
        part = estimates[estimates["target"] == target][["feature_set", "model", *values]]
        part = part.rename(columns={v: f"{target}_{v}" for v in values})
        wide = wide.merge(part, on=["feature_set", "model"], how="left")
    return wide.sort_values(f"{primary}_loo_r2", ascending=False, ignore_index=True)


def summary_markdown(
    *,
    commit: str,
    plan: Plan,
    estimates: pd.DataFrame,
    max_rows: pd.DataFrame,
    spearman: pd.DataFrame,
    selections: pd.DataFrame,
    sensitivity: pd.DataFrame,
    n_permutations: int,
    excluded: Sequence[int],
) -> str:
    """The human-readable report: every number exploratory and uncorrected."""
    primary = plan.evaluation.primary_target
    wide = wide_table(estimates, primary, plan.evaluation.targets)
    max_columns = [
        "target",
        "scope",
        "best_set",
        "best_model",
        "observed_best_split_r2",
        "null_max_p50",
        "null_max_p95",
        "null_max_p99",
        "percentile_of_observed",
        "max_stat_p",
    ]
    selection_columns = [
        "target",
        "candidate",
        "times_selected",
        "times_selected_first",
        "n_outer_folds",
        "mean_subset_size",
    ]
    sensitivity_columns = [
        "feature_set",
        "model",
        "target",
        "n_sessions",
        "loo_r2",
        "kfold_r2_mean",
        "loo_spearman",
        "full_cohort_loo_r2",
    ]
    sensitivity_text = (
        _markdown(sensitivity, sensitivity_columns)
        if len(sensitivity)
        else "Not run: no session was passed with --exclude-session.\n"
    )
    max_text = _markdown(max_rows, max_columns) if len(max_rows) else "Not run: no permutations.\n"
    folds, repeats = plan.evaluation.kfold_folds, plan.evaluation.kfold_repeats
    return "\n".join(
        [
            "# AU-only baseline: exploratory results",
            "",
            "**EXPLORATORY. Requested by Tanaka after unblinding. No multiplicity correction",
            "anywhere. Nothing here supports a confirmatory claim.**",
            f"Commit `{commit}`; every set fixed in `{PLAN_PATH}` at that commit.",
            "",
            f"Elastic net and random forest; leave-one-out, and {repeats}x repeated {folds}-fold",
            "beside it; median imputation and scaling inside every fold. Permutation p values",
            f"compare each set's R² on the first {folds}-fold split with the same statistic",
            f"over {n_permutations} shared label permutations.",
            "",
            f"## All sets, sorted by {primary} leave-one-out R²",
            "",
            _markdown(wide, list(wide.columns)),
            "## Max statistic",
            "",
            "Best split R² among the AU sets and the search, against the best of the same",
            "sets in each permutation. The references (all 30 face features, text) are not",
            "in the family.",
            "",
            max_text,
            "## Single AUs: plain Spearman correlation with the outcome",
            "",
            _markdown(spearman, ["column", "target", "n", "spearman", "p_uncorrected"]),
            "## The search: candidates chosen across the leave-one-out folds",
            "",
            _markdown(selections, selection_columns),
            f"## Sensitivity: best {plan.sensitivity.top_n} by {primary} LOO R², "
            f"without session(s) {list(excluded)}",
            "",
            sensitivity_text,
        ]
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AnalyzeResult:
    """Where everything went."""

    out_dir: Path
    files: tuple[str, ...]
    n_sessions: int
    elapsed_s: float

    def report_lines(self) -> list[str]:
        """File names and counts: nothing that could carry a label."""
        return [
            f"exploratory results for {self.n_sessions} session(s) -> {self.out_dir}",
            f"  elapsed {self.elapsed_s / 60:.1f} min",
            *(f"  {name}" for name in self.files),
        ]


def run(
    config: AppConfig,
    roots: DataRoots,
    plan: Plan,
    *,
    bundle: Path,
    labels_path: Path,
    out_dir: Path,
    commit: str,
    jobs: int,
    n_permutations: int | None = None,
    exclude_sessions: Sequence[int] = (),
) -> AnalyzeResult:
    """Run every part of the analysis and write its tables and summary."""
    started = time.monotonic()
    seed = config.runtime.seed
    models = plan.evaluation.models
    permutations = plan.permutation.n_permutations if n_permutations is None else n_permutations

    inputs = load_inputs(config, roots, plan, bundle)
    sets = resolve(plan, list(inputs.table.columns))
    labels = model_stage.load_labels(labels_path, plan.evaluation.targets)
    cohort = model_stage.build_cohort(inputs.table, labels, model_stage.load_groups(config, roots))
    for line in cohort.report_lines():
        logger.info("%s: %s", STAGE, line)
    tasks = make_tasks(sets, cohort.features, plan, seed)
    orders = permutation_orders(cohort.n, permutations, seed)
    search = next(s for s in sets if s.is_search)

    estimate_parts: list[dict[str, object]] = []
    max_parts: list[dict[str, object]] = []
    null_columns: dict[str, np.ndarray] = {}
    selection_parts: list[dict[str, object]] = []
    for target in plan.evaluation.targets:
        y = target_vector(cohort.labels, target)
        logger.info("%s: %s: %d set(s) x %d model(s)", STAGE, target, len(tasks), len(models))
        observed = observed_scores(tasks, y, cohort.groups, plan, seed=seed, jobs=jobs)
        null = (
            null_scores(tasks, y, cohort.groups, orders, plan, seed=seed, jobs=jobs)
            if permutations
            else None
        )
        estimate_parts.extend(estimate_rows(sets, target, observed, null, models))
        if null is not None:
            rows, nulls = max_statistic_rows(sets, target, observed, null, models)
            max_parts.extend(rows)
            null_columns.update({f"{target}_{scope}": v for scope, v in nulls.items()})
        selection_parts.extend(selection_rows(search, target, observed[(search.name, models[0])]))

    estimates = pd.DataFrame(estimate_parts)
    max_rows = pd.DataFrame(max_parts)
    spearman = single_au_spearman(cohort.features, cohort.labels, plan, plan.evaluation.targets)
    selections = pd.DataFrame(selection_parts)
    sensitivity = run_sensitivity(
        estimates, sets, cohort, plan, seed=seed, jobs=jobs, exclude=exclude_sessions
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        ESTIMATES_FILE: estimates,
        SPEARMAN_FILE: spearman,
        SELECTION_FILE: selections,
        SENSITIVITY_FILE: sensitivity,
    }
    if len(max_rows):
        tables[MAX_FILE] = max_rows
        tables[MAX_NULL_FILE] = pd.DataFrame(null_columns).assign(tier=TIER)
    for name, frame in tables.items():
        write_csv(out_dir / name, frame)
    write_text(
        out_dir / SUMMARY_FILE,
        summary_markdown(
            commit=commit,
            plan=plan,
            estimates=estimates,
            max_rows=max_rows,
            spearman=spearman,
            selections=selections,
            sensitivity=sensitivity,
            n_permutations=permutations,
            excluded=exclude_sessions,
        ),
    )
    elapsed = time.monotonic() - started
    write_json(
        out_dir / RUN_FILE,
        {
            "tier": TIER,
            "requested_by": "Tanaka",
            "designed_after_unblinding": True,
            "multiplicity_correction": None,
            "commit": commit,
            "run_at": datetime.now(UTC).isoformat(),
            **inputs.digests,
            "seed": seed,
            "models": list(models),
            "targets": list(plan.evaluation.targets),
            "kfold_folds": plan.evaluation.kfold_folds,
            "kfold_repeats": plan.evaluation.kfold_repeats,
            "n_permutations": permutations,
            "n_sets": len(sets),
            "n_sessions": cohort.n,
            "sensitivity_excluded_sessions": list(exclude_sessions),
            "jobs": jobs,
            "elapsed_s": round(elapsed, 1),
        },
    )
    return AnalyzeResult(
        out_dir=out_dir,
        files=tuple(sorted([*tables, SUMMARY_FILE, RUN_FILE])),
        n_sessions=cohort.n,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Runtime estimate
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RuntimeEstimate:
    """Measured timings, extrapolated to the planned run."""

    n_sets: int
    n_models: int
    n_targets: int
    jobs: int
    observed_s_per_target: float
    permutation_s_each: float
    n_probe_permutations: int
    sensitivity_s: float

    def total_s(self, n_permutations: int) -> float:
        """Projected wall time for `n_permutations`."""
        per_target = self.observed_s_per_target + n_permutations * self.permutation_s_each
        return self.n_targets * per_target + self.sensitivity_s

    def report_lines(self, planned: int) -> list[str]:
        """The measurement and the projection, with alternatives."""
        lines = [
            f"runtime estimate: {self.n_sets} set(s) x {self.n_models} model(s) x "
            f"{self.n_targets} target(s), {self.jobs} worker(s); random targets, no label read",
            f"  observed estimates (LOO + repeated k-fold): "
            f"{self.observed_s_per_target / 60:.1f} min per target, measured",
            f"  permutations: {self.permutation_s_each:.2f} s each per target, measured over "
            f"{self.n_probe_permutations}",
        ]
        for n in sorted({planned, 100, 200, 500, 1000}):
            mark = "  <- planned" if n == planned else ""
            lines.append(f"  total with {n:>4} permutations: {self.total_s(n) / 3600:.1f} h{mark}")
        return lines


def estimate_runtime(
    config: AppConfig,
    roots: DataRoots,
    plan: Plan,
    *,
    bundle: Path,
    jobs: int,
    n_probe: int,
) -> RuntimeEstimate:
    """Time the real code path on random targets, and extrapolate.

    Uses the real feature table, so the fits see its real shape and gaps, with
    targets drawn at random: no label is read. Observed estimates are run in
    full for one target; permutations for `n_probe` of them.
    """
    seed = config.runtime.seed
    inputs = load_inputs(config, roots, plan, bundle)
    sets = resolve(plan, list(inputs.table.columns))
    table = inputs.table.sort_values("session_id", ignore_index=True)
    tasks = make_tasks(sets, table, plan, seed)
    groups = tuple(int(v) for v in table["session_id"])
    y = np.random.default_rng(seed).normal(size=len(table))

    started = time.monotonic()
    observed_scores(tasks, y, groups, plan, seed=seed, jobs=jobs)
    observed_s = time.monotonic() - started

    orders = permutation_orders(len(table), n_probe, seed)
    started = time.monotonic()
    null_scores(tasks, y, groups, orders, plan, seed=seed, jobs=jobs)
    per_permutation = (time.monotonic() - started) / n_probe

    # The sensitivity rerun is top_n pairs x targets full evaluations, run in
    # parallel: about one observed pair's time per round of `jobs`.
    per_pair = observed_s * jobs / max(1, len(tasks) * len(plan.evaluation.models))
    n_calls = plan.sensitivity.top_n * len(plan.evaluation.targets)
    sensitivity_s = per_pair * -(-n_calls // jobs)

    return RuntimeEstimate(
        n_sets=len(tasks),
        n_models=len(plan.evaluation.models),
        n_targets=len(plan.evaluation.targets),
        jobs=jobs,
        observed_s_per_target=observed_s,
        permutation_s_each=per_permutation,
        n_probe_permutations=n_probe,
        sensitivity_s=sensitivity_s,
    )
