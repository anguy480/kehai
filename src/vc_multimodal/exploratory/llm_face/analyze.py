"""What the LLM face ratings measure, and whether they relate to the outcomes.

Exploratory, designed after unblinding. Every number this module writes is
exploratory and uncorrected, and none supports a confirmatory claim.

Three questions, in order:

1. **Stability.** Do the repeated ratings of one description agree? ICC(2,1)
   per scale across the runs, which are then averaged.
2. **Re-description.** Are the ratings more than the numeric face features in
   other words? A Spearman matrix against the 30 numeric face features, and the
   cross-validated R² of predicting each rating from them.
3. **Outcomes.** With `vc model`'s own machinery - elastic net, leave-one-out
   and repeated 5-fold, scaling and median imputation fitted inside each fold,
   permutation nulls, paired Wilcoxon on per-session absolute errors - for
   LLM face, numeric face, text, and text plus LLM face, on K6 and SRS2.

The labels are read through `vc model`'s loader and go no further: nothing
written here holds a label, a prediction, a residual or a per-session error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

from vc_multimodal.config import AppConfig
from vc_multimodal.exploratory.llm_face import rate
from vc_multimodal.exploratory.llm_face.describe import file_sha256, output_dir
from vc_multimodal.io_utils import read_csv, write_csv, write_json, write_text
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.modeling.evaluate import (
    MODEL_ELASTIC_NET,
    MODEL_RANDOM_FOREST,
    Evaluation,
    EvaluationError,
    compare_errors,
    evaluate,
    permutation_baseline,
)
from vc_multimodal.modeling.text_features import BUNDLE_FILE, load_bundled
from vc_multimodal.modeling.text_features import join as join_text
from vc_multimodal.paths import DataRoots
from vc_multimodal.stages import model as model_stage

logger = get_logger(__name__)

STAGE: Final = "llm-face-analyze"
TIER: Final = "exploratory"

#: The ratings this analysis is allowed to use: the ones frozen in
#: docs/exploratory/llm-face.md before any label was joined.
FROZEN_SCORES_SHA256: Final = (
    "3a868788a7c9bc816b23de298729626731289708b66203672fd102722e4dddb8"  # pragma: allowlist secret
)

LLM_PREFIX: Final = "llmface__"
FACE_PREFIXES: Final = ("face_speaking__", "face_listening__")
TEXT_PREFIX: Final = "text__"

LLM_FACE: Final = "llm_face"
NUMERIC_FACE: Final = "numeric_face"
TEXT: Final = "text"
TEXT_PLUS_LLM_FACE: Final = "text_plus_llm_face"
FEATURE_SETS: Final = (LLM_FACE, NUMERIC_FACE, TEXT, TEXT_PLUS_LLM_FACE)

#: (name, first, second): a positive median error difference means `first` was
#: closer, as in `compare_errors`.
COMPARISONS: Final = (
    ("llm_face_vs_numeric_face", LLM_FACE, NUMERIC_FACE),
    ("text_plus_llm_face_vs_text", TEXT_PLUS_LLM_FACE, TEXT),
)

_MIN_PAIRS: Final = 3
_MIN_RUNS: Final = 2

STABILITY_FILE: Final = "stability.csv"
SPEARMAN_FILE: Final = "llm_vs_numeric_spearman.csv"
REDESCRIPTION_FILE: Final = "llm_from_numeric_cv.csv"
ESTIMATES_FILE: Final = "outcome_estimates.csv"
COMPARISONS_FILE: Final = "outcome_comparisons.csv"
SCALE_OUTCOME_FILE: Final = "llm_scale_outcome_spearman.csv"
SUMMARY_FILE: Final = "summary.md"
RUN_FILE: Final = "run.json"


class AnalyzeError(RuntimeError):
    """Raised when the analysis cannot be run on the inputs supplied."""


# ---------------------------------------------------------------------------
# 1. Stability
# ---------------------------------------------------------------------------
def icc_2_1(ratings: np.ndarray) -> float | None:
    """ICC(2,1): two-way random effects, absolute agreement, single rating.

    `ratings` is sessions x runs. None when the sessions do not vary at all,
    where agreement is undefined rather than perfect or absent.
    """
    n, k = ratings.shape
    if n < _MIN_PAIRS or k < _MIN_RUNS:
        return None
    grand = float(ratings.mean())
    rows = ratings.mean(axis=1)
    cols = ratings.mean(axis=0)
    ms_rows = k * float(np.sum((rows - grand) ** 2)) / (n - 1)
    ms_cols = n * float(np.sum((cols - grand) ** 2)) / (k - 1)
    residual = ratings - rows[:, None] - cols[None, :] + grand
    ms_error = float(np.sum(residual**2)) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator <= 0:
        return None
    return float((ms_rows - ms_error) / denominator)


def stability_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Run-to-run agreement per scale."""
    rows: list[dict[str, object]] = []
    for scale in rate.SCALES:
        wide = runs.pivot(index="session_id", columns="run", values=scale).dropna()
        matrix = wide.to_numpy(dtype=np.float64)
        means = matrix.mean(axis=1)
        rows.append(
            {
                "scale": scale,
                "n_sessions": len(wide),
                "n_runs": wide.shape[1],
                "icc_2_1": icc_2_1(matrix),
                "share_identical_across_runs": float((matrix == matrix[:, :1]).all(axis=1).mean()),
                "mean_rating": float(means.mean()),
                "sd_rating": float(means.std(ddof=1)),
                "n_distinct_ratings": len(np.unique(means)),
            }
        )
    return pd.DataFrame(rows)


def mean_scores(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per session: each scale averaged over runs."""
    means = runs.groupby("session_id")[list(rate.SCALES)].mean()
    means.columns = [f"{LLM_PREFIX}{scale}" for scale in means.columns]
    return means.reset_index()


# ---------------------------------------------------------------------------
# 2. Re-description
# ---------------------------------------------------------------------------
def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, int]:
    usable = ~(np.isnan(x) | np.isnan(y))
    n = int(usable.sum())
    if n < _MIN_PAIRS or np.ptp(x[usable]) == 0 or np.ptp(y[usable]) == 0:
        return None, None, n
    result = stats.spearmanr(x[usable], y[usable])
    return float(result.statistic), float(result.pvalue), n


def spearman_matrix(table: pd.DataFrame, left: Sequence[str], right: Sequence[str]) -> pd.DataFrame:
    """Every pair of one column from `left` and one from `right`."""
    rows: list[dict[str, object]] = []
    for a in left:
        for b in right:
            rho, p, n = _spearman(
                table[a].to_numpy(dtype=np.float64), table[b].to_numpy(dtype=np.float64)
            )
            rows.append({"llm_scale": a, "other": b, "n": n, "rho": rho, "p_uncorrected": p})
    return pd.DataFrame(rows)


def _evaluation_row(
    name: str, target: str, model: str, evaluation: Evaluation
) -> dict[str, object]:
    return {
        "feature_set": name,
        "target": target,
        "model": model,
        "tier": TIER,
        "n_sessions": evaluation.loo.n,
        "n_features": evaluation.n_features,
        "loo_r2": evaluation.loo.r2,
        "kfold_r2_mean": evaluation.stability,
        "kfold_r2_sd": evaluation.stability_sd,
        "n_fits_not_converged": evaluation.n_not_converged,
    }


def redescription(
    table: pd.DataFrame,
    face_columns: Sequence[str],
    llm_columns: Sequence[str],
    *,
    seed: int,
    folds: int,
    repeats: int,
) -> pd.DataFrame:
    """Cross-validated R² of each LLM scale from the numeric face features."""
    rows: list[dict[str, object]] = []
    for column in llm_columns:
        keep = table[column].notna().to_numpy()
        target = table.loc[keep, column].to_numpy(dtype=np.float64)
        matrix = table.loc[keep, list(face_columns)].to_numpy(dtype=np.float64)
        groups = tuple(table.loc[keep, "session_id"])
        for model in (MODEL_ELASTIC_NET, MODEL_RANDOM_FOREST):
            try:
                evaluation = evaluate(
                    matrix,
                    target,
                    groups,
                    model=model,
                    seed=seed,
                    stability_folds=folds,
                    stability_repeats=repeats,
                )
            except EvaluationError as exc:
                logger.warning("%s: %s from numeric face: %s", STAGE, column, exc)
                continue
            rows.append(_evaluation_row(NUMERIC_FACE, column, model, evaluation))
            logger.info("%s: %s from numeric face (%s) done", STAGE, column, model)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Outcomes
# ---------------------------------------------------------------------------
def feature_columns(table: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """The columns of each feature set, in table order."""
    face = tuple(str(c) for c in table.columns if str(c).startswith(FACE_PREFIXES))
    text = tuple(str(c) for c in table.columns if str(c).startswith(TEXT_PREFIX))
    llm = tuple(str(c) for c in table.columns if str(c).startswith(LLM_PREFIX))
    return {
        LLM_FACE: llm,
        NUMERIC_FACE: face,
        TEXT: text,
        TEXT_PLUS_LLM_FACE: (*text, *llm),
    }


@dataclass(frozen=True, slots=True)
class OutcomeResults:
    """Estimates and paired comparisons for every target."""

    estimates: pd.DataFrame
    comparisons: pd.DataFrame


def outcomes(
    cohort: model_stage.Cohort,
    sets: Mapping[str, Sequence[str]],
    targets: Sequence[str],
    *,
    seed: int,
    folds: int,
    repeats: int,
    n_permutations: int,
) -> OutcomeResults:
    """Every feature set on every target, then the paired comparisons."""
    estimate_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for target in targets:
        truth = pd.to_numeric(cohort.labels[target], errors="coerce").to_numpy(np.float64)
        evaluations: dict[str, Evaluation] = {}
        for name in FEATURE_SETS:
            matrix = cohort.features[list(sets[name])].to_numpy(dtype=np.float64)
            evaluation = evaluate(
                matrix,
                truth,
                cohort.groups,
                model=MODEL_ELASTIC_NET,
                seed=seed,
                stability_folds=folds,
                stability_repeats=repeats,
            )
            permutation = permutation_baseline(
                matrix,
                truth,
                cohort.groups,
                model=MODEL_ELASTIC_NET,
                seed=seed,
                n_permutations=n_permutations,
                folds=folds,
            )
            evaluations[name] = evaluation
            estimate = model_stage.Estimate(
                feature_set=name,
                target=target,
                model=MODEL_ELASTIC_NET,
                tier=TIER,
                evaluation=evaluation,
                permutation=permutation,
            )
            estimate_rows.append(estimate.row())
            logger.info("%s: %s on %s done", STAGE, name, target)
        for label, first, second in COMPARISONS:
            result = compare_errors(evaluations[first].errors, evaluations[second].errors)
            comparison_rows.append(
                {
                    "comparison": label,
                    "target": target,
                    "tier": TIER,
                    "feature_set": first,
                    "compared_with": second,
                    "n_sessions": result.n,
                    "r2": evaluations[first].loo.r2,
                    "compared_r2": evaluations[second].loo.r2,
                    "median_error_difference": result.median_difference,
                    "p_value_uncorrected": result.p_value,
                }
            )
    return OutcomeResults(pd.DataFrame(estimate_rows), pd.DataFrame(comparison_rows))


def scale_outcome_spearman(
    cohort: model_stage.Cohort, llm_columns: Sequence[str], targets: Sequence[str]
) -> pd.DataFrame:
    """Each LLM scale against each outcome."""
    joined = cohort.features[["session_id", *llm_columns]].merge(cohort.labels, on="session_id")
    frame = spearman_matrix(joined, llm_columns, targets)
    frame.insert(len(frame.columns), "tier", TIER)
    return frame


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, float | np.floating):
        return f"{float(value):.3f}"
    return str(value)


def _markdown(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    head = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join(
        "| " + " | ".join(_fmt(row[c]) for c in columns) + " |\n" for _, row in frame.iterrows()
    )
    return head + body


def summary_markdown(
    *,
    commit: str,
    stability: pd.DataFrame,
    spearman: pd.DataFrame,
    redescribed: pd.DataFrame,
    results: OutcomeResults,
    scale_outcome: pd.DataFrame,
    n_permutations: int,
) -> str:
    """The human-readable report: every number exploratory and uncorrected."""
    strongest = (
        spearman.assign(abs_rho=spearman["rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .groupby("llm_scale", sort=False)
        .head(1)
    )
    estimate_columns = [
        "feature_set",
        "target",
        "n_features",
        "loo_r2",
        "kfold_r2_mean",
        "permutation_observed_r2",
        "permutation_null_p95_r2",
        "permutation_p",
        "n_fits_not_converged",
    ]
    comparison_columns = [
        "comparison",
        "target",
        "r2",
        "compared_r2",
        "median_error_difference",
        "p_value_uncorrected",
    ]
    return "\n".join(
        [
            "# LLM-rated face features: exploratory results",
            "",
            "**EXPLORATORY. Designed after unblinding (2026-10-02). No multiplicity",
            "correction anywhere. Nothing here supports a confirmatory claim.**",
            f"Commit `{commit}`; inputs frozen in `docs/exploratory/llm-face.md`.",
            "",
            "## 1. Stability across runs",
            "",
            "ICC(2,1), absolute agreement, single run, over identical repeated requests.",
            "",
            _markdown(stability, list(stability.columns)),
            "## 2. LLM ratings against the numeric face features",
            "",
            "Strongest Spearman correlation of each scale with any numeric face feature",
            f"(full matrix in `{SPEARMAN_FILE}`; p-values uncorrected):",
            "",
            _markdown(strongest, ["llm_scale", "other", "n", "rho", "p_uncorrected"]),
            "Cross-validated R² of each scale predicted from the numeric face features.",
            "A high value means the scale mostly re-describes those numbers.",
            "",
            _markdown(redescribed, ["target", "model", "n_sessions", "loo_r2", "kfold_r2_mean"]),
            "## 3. Outcomes",
            "",
            "Elastic net; leave-one-out with repeated 5-fold beside it; scaling and median",
            f"imputation inside each fold; {n_permutations}-permutation null under 5-fold.",
            "",
            _markdown(results.estimates, estimate_columns),
            "Paired Wilcoxon on per-session absolute errors, uncorrected. A positive",
            "median error difference means the first set was closer.",
            "",
            _markdown(results.comparisons, comparison_columns),
            "Spearman correlation of each scale with each outcome, uncorrected:",
            "",
            _markdown(scale_outcome, ["llm_scale", "other", "n", "rho", "p_uncorrected"]),
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

    def report_lines(self) -> list[str]:
        """File names and a count: nothing that could carry a label."""
        return [
            f"exploratory results for {self.n_sessions} session(s) -> {self.out_dir}",
            *(f"  {name}" for name in self.files),
        ]


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    bundle: Path,
    labels_path: Path,
    out_dir: Path,
    commit: str,
    n_permutations: int | None = None,
) -> AnalyzeResult:
    """Run all three parts and write their tables and summary."""
    scores_path = output_dir(roots) / rate.SCORES_FILE
    digest = file_sha256(scores_path)
    if digest != FROZEN_SCORES_SHA256:
        msg = (
            f"{scores_path.name} has digest {digest}, not the frozen "
            f"{FROZEN_SCORES_SHA256}: refusing to analyse ratings made after the freeze"
        )
        raise AnalyzeError(msg)
    if config.model.text_features is None:
        msg = "model.text_features is not configured, so there is no text baseline"
        raise AnalyzeError(msg)

    tiers = config.model.tiers
    seed = config.runtime.seed
    folds, repeats = tiers.stability_folds, tiers.stability_repeats
    permutations = config.model.n_permutations if n_permutations is None else n_permutations

    runs = read_csv(scores_path)
    stability = stability_table(runs)
    llm = mean_scores(runs)

    features = read_csv(bundle / "features.csv")
    text = load_bundled(bundle / BUNDLE_FILE, config.model.text_features)
    features, _ = join_text(features, text)
    table = features.merge(llm, on="session_id", how="left")
    sets = feature_columns(table)
    face_columns, llm_columns = sets[NUMERIC_FACE], sets[LLM_FACE]
    logger.info(
        "%s: %d numeric face, %d text, %d LLM column(s)",
        STAGE,
        len(face_columns),
        len(sets[TEXT]),
        len(llm_columns),
    )

    spearman = spearman_matrix(table, llm_columns, face_columns)
    spearman.insert(len(spearman.columns), "tier", TIER)
    redescribed = redescription(
        table, face_columns, llm_columns, seed=seed, folds=folds, repeats=repeats
    )

    labels = model_stage.load_labels(labels_path, config.model.targets)
    cohort = model_stage.build_cohort(table, labels, model_stage.load_groups(config, roots))
    for line in cohort.report_lines():
        logger.info("%s: %s", STAGE, line)
    results = outcomes(
        cohort,
        sets,
        labels.targets,
        seed=seed,
        folds=folds,
        repeats=repeats,
        n_permutations=permutations,
    )
    scale_outcome = scale_outcome_spearman(cohort, llm_columns, labels.targets)

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        STABILITY_FILE: stability.assign(tier=TIER),
        SPEARMAN_FILE: spearman,
        REDESCRIPTION_FILE: redescribed.assign(tier=TIER),
        ESTIMATES_FILE: results.estimates,
        COMPARISONS_FILE: results.comparisons,
        SCALE_OUTCOME_FILE: scale_outcome,
    }
    for name, frame in tables.items():
        write_csv(out_dir / name, frame)
    write_text(
        out_dir / SUMMARY_FILE,
        summary_markdown(
            commit=commit,
            stability=stability,
            spearman=spearman,
            redescribed=redescribed,
            results=results,
            scale_outcome=scale_outcome,
            n_permutations=permutations,
        ),
    )
    write_json(
        out_dir / RUN_FILE,
        {
            "tier": TIER,
            "designed_after_unblinding": True,
            "multiplicity_correction": None,
            "commit": commit,
            "run_at": datetime.now(UTC).isoformat(),
            "scores_sha256": digest,
            "features_sha256": file_sha256(bundle / "features.csv"),
            "text_features_sha256": file_sha256(bundle / BUNDLE_FILE),
            "seed": seed,
            "folds": folds,
            "repeats": repeats,
            "n_permutations": permutations,
            "model": MODEL_ELASTIC_NET,
        },
    )
    return AnalyzeResult(
        out_dir=out_dir,
        files=tuple(sorted([*tables, SUMMARY_FILE, RUN_FILE])),
        n_sessions=cohort.n,
    )
