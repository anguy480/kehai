"""Per-modality evaluation: Analysis A and B, run where the labels live.

Exploratory, pre-registered in docs/prereg/per-modality.md, with every choice
read from config/exploratory/per_modality.yaml.

The analysis is handed off in two runs, because the face features exist only
after OpenFace extraction. The first run (prosody, turn-taking) writes raw
values only. The second (face while speaking, face while listening) reads the
first run's output, checks it was made under the same analysis, and only then
computes the corrections, over the complete families. No q-value and no
Holm-adjusted p is ever computed over part of a family: it would be a
different number from the one that was planned.

The models are the confirmatory pipelines, called exactly as `vc model` calls
them. Nothing written or printed holds a label, a prediction or a residual:
only correlations, scores, p-values and counts.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from vc_multimodal.exploratory.per_modality.plan import (
    RUN_FIRST,
    RUN_SECOND,
    SOURCE_BASE,
    SOURCE_FACE,
    Plan,
    resolve_plan_path,
)
from vc_multimodal.io_utils import read_csv, write_csv, write_json, write_text
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.modeling import univariate
from vc_multimodal.modeling.evaluate import evaluate, permutation_baseline
from vc_multimodal.modeling.tiers import holm_adjust
from vc_multimodal.provenance import environment_record
from vc_multimodal.stages.model import build_cohort, load_labels

logger = get_logger(__name__)

STAGE: Final = "per_modality"

BUNDLE_FEATURES: Final = "features.csv"
UNIVARIATE_FILE: Final = "univariate.csv"
MODELS_FILE: Final = "models.csv"
RUN_FILE: Final = "run.json"
FINAL_UNIVARIATE_FILE: Final = "final_univariate.csv"
FINAL_MODELS_FILE: Final = "final_models.csv"
SUMMARY_FILE: Final = "summary.md"

FACE_BACKEND_COLUMN: Final = "qc__face_backend"
FACE_BACKEND: Final = "openface"

ROLE_PRIMARY: Final = "primary"
ROLE_SECONDARY: Final = "secondary"

#: Columns of the per-model table, in order. None of them holds a label.
MODEL_COLUMNS: Final = (
    "modality",
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
    "p_holm",
    "n_fits_not_converged",
)


class AnalyzeError(RuntimeError):
    """Raised when the analysis cannot be run as planned."""


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_digest(path: Path, expected: str | None, what: str) -> str:
    """Refuse a file whose digest is not the planned one.

    Raises:
        AnalyzeError: if the file is missing, or its digest is not fixed or
            does not match.
    """
    if not path.is_file():
        msg = f"no {what} at {path}"
        raise AnalyzeError(msg)
    if expected is None:
        msg = (
            f"the {what}'s SHA-256 is not fixed in the plan yet. It is fixed in a commit "
            f"of its own, pushed before this run is handed off."
        )
        raise AnalyzeError(msg)
    found = file_sha256(path)
    if found != expected:
        msg = (
            f"{path} is not the planned {what}: SHA-256 {found[:12]}..., planned {expected[:12]}..."
        )
        raise AnalyzeError(msg)
    return found


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Tables:
    """The feature columns one run uses, and the digests they were checked by."""

    frame: pd.DataFrame
    digests: Mapping[str, str]
    notes: tuple[str, ...] = ()


def load_tables(
    plan: Plan, run: str, *, bundle: Path | None = None, face_table: Path | None = None
) -> Tables:
    """Read and check the table one run needs.

    Raises:
        AnalyzeError: if a table is missing, is not the planned file, lacks a
            planned column, or breaks a planned session rule.
    """
    modalities = plan.run_modalities(run)
    sources = {plan.modalities[name].source for name in modalities}
    if len(sources) != 1:  # pragma: no cover - the shipped plan cannot reach this
        msg = f"run {run!r} mixes feature sources {sorted(sources)}"
        raise AnalyzeError(msg)
    source = sources.pop()

    notes: list[str] = []
    if source == SOURCE_BASE:
        if bundle is None:
            msg = f"run {run!r} reads the bundle's {BUNDLE_FEATURES}: pass --bundle"
            raise AnalyzeError(msg)
        path = bundle / BUNDLE_FEATURES
        digests = {
            "base_features_sha256": _check_digest(
                path, plan.inputs.base_features_sha256, f"bundle {BUNDLE_FEATURES}"
            )
        }
    else:
        if face_table is None:
            msg = f"run {run!r} reads the OpenFace face table: pass --face-table"
            raise AnalyzeError(msg)
        path = face_table
        digests = {
            "face_features_sha256": _check_digest(
                path, plan.inputs.face_features_sha256, "OpenFace face table"
            )
        }

    frame = read_csv(path)
    if "session_id" not in frame.columns:
        msg = f"{path.name} has no session_id column"
        raise AnalyzeError(msg)
    columns = [c for name in modalities for c in plan.modalities[name].features]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        msg = f"{path.name} lacks planned column(s) {missing}"
        raise AnalyzeError(msg)

    if source == SOURCE_FACE:
        notes.extend(_check_face_table(plan, frame, columns))

    kept = frame[["session_id", *columns]].copy()
    kept["session_id"] = pd.to_numeric(kept["session_id"], errors="raise").astype("int64")
    for column in columns:
        kept[column] = pd.to_numeric(kept[column], errors="coerce").astype("float64")
    return Tables(frame=kept, digests=digests, notes=tuple(notes))


def _check_face_table(plan: Plan, frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Check the face table is OpenFace throughout and the QC notes were applied.

    Raises:
        AnalyzeError: if any row comes from another backend, or a session whose
            face is unavailable still carries a facial value.
    """
    if FACE_BACKEND_COLUMN not in frame.columns:
        msg = f"the face table has no {FACE_BACKEND_COLUMN} column, so its backend is unknown"
        raise AnalyzeError(msg)
    backends = {str(v) for v in frame[FACE_BACKEND_COLUMN].dropna() if str(v)}
    if backends != {FACE_BACKEND}:
        msg = f"the face table must be OpenFace only; it records {sorted(backends)}"
        raise AnalyzeError(msg)

    notes: list[str] = []
    for session_id in plan.sessions.face_unavailable:
        row = frame.loc[frame["session_id"] == session_id, list(columns)]
        if row.empty:
            notes.append(f"session {session_id} (face unavailable) is not in the face table")
            continue
        if row.notna().to_numpy().any():
            msg = (
                f"session {session_id}'s face is marked unavailable, but the face table "
                f"carries values for it: the QC note was not applied"
            )
            raise AnalyzeError(msg)
    return notes


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Task:
    """One model: a modality, a target and a model family."""

    modality: str
    target: str
    model: str
    role: str


def make_tasks(plan: Plan, modalities: Sequence[str]) -> list[Task]:
    """Every model a run fits, in reporting order, primary before secondary."""
    primary, secondary = plan.model_names
    return [
        Task(modality, target, model, role)
        for modality in modalities
        for target in plan.targets
        for model, role in ((primary, ROLE_PRIMARY), (secondary, ROLE_SECONDARY))
    ]


def fit_task(
    task: Task,
    matrix: np.ndarray,
    truth: np.ndarray,
    groups: Sequence[object],
    plan: Plan,
    n_permutations: int,
) -> dict[str, Any]:
    """Cross-validate one model and compute its null. Returns metrics only.

    Called exactly as `vc model` calls `evaluate` and `permutation_baseline`
    for a confirmatory estimate, with the plan's seed, schemes and count.

    Raises:
        AnalyzeError: if no null could be computed.
    """
    validation = plan.validation
    evaluation = evaluate(
        matrix,
        truth,
        groups,
        model=task.model,
        seed=validation.seed,
        stability_folds=validation.stability_folds,
        stability_repeats=validation.stability_repeats,
    )
    null = permutation_baseline(
        matrix,
        truth,
        groups,
        model=task.model,
        seed=validation.seed,
        n_permutations=n_permutations,
        folds=validation.permutation_folds,
    )
    if null is None:  # pragma: no cover - n_permutations >= 1 by validation
        msg = "no permutation null was computed"
        raise AnalyzeError(msg)
    return {
        "modality": task.modality,
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
        "kfold_sign_disagrees": evaluation.disagrees_with_stability,
        "perm_scheme": null.scheme,
        "perm_observed_r2": null.observed,
        "perm_null_mean": null.null_mean,
        "perm_null_p95": null.null_p95,
        "perm_p": null.p_value,
        "n_permutations": null.n_permutations,
        "p_holm": np.nan,
        "n_fits_not_converged": evaluation.n_not_converged,
    }


@dataclass(frozen=True, slots=True)
class Prepared:
    """One modality and target, ready to fit, and which sessions it left out."""

    matrix: np.ndarray
    truth: np.ndarray
    groups: tuple[object, ...]
    features: pd.DataFrame
    outcome: pd.DataFrame
    excluded_no_features: tuple[int, ...]
    excluded_no_label: int


def prepare(
    features: pd.DataFrame, labels: pd.DataFrame, columns: Sequence[str], target: str
) -> Prepared:
    """The sessions one modality is analysed on for one target.

    A session with no value for any of the modality's features is left out,
    as is one with no value for the target. Partly missing features stay, to be
    imputed inside the folds.

    Args:
        features: Feature table, aligned row for row with `labels`.
        labels: Labels, aligned with `features`.
        columns: The modality's columns.
        target: The outcome.
    """
    has_feature = features[list(columns)].notna().any(axis=1).to_numpy()
    truth_all = pd.to_numeric(labels[target], errors="coerce").to_numpy(dtype=np.float64)
    has_label = np.isfinite(truth_all)
    keep = has_feature & has_label
    session_ids = features["session_id"].to_numpy()
    return Prepared(
        matrix=features.loc[keep, list(columns)].to_numpy(dtype=np.float64),
        truth=truth_all[keep],
        groups=tuple(int(s) for s in session_ids[keep]),
        features=features.loc[keep].reset_index(drop=True),
        outcome=labels.loc[keep].reset_index(drop=True),
        excluded_no_features=tuple(int(s) for s in session_ids[~has_feature]),
        excluded_no_label=int((has_feature & ~has_label).sum()),
    )


def _parallel(jobs: int, calls: Sequence[Any]) -> list[dict[str, Any]]:
    """Run the model fits in parallel, logging each as it lands."""
    results: list[dict[str, Any]] = []
    runner = Parallel(n_jobs=jobs, return_as="generator_unordered")
    for index, row in enumerate(runner(calls), start=1):
        results.append(row)
        logger.info(
            "%s: [%d/%d] %s %s on %s done",
            STAGE,
            index,
            len(calls),
            row["modality"],
            row["model"],
            row["target"],
        )
    return results


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------
def correct(
    plan: Plan, univariate_rows: pd.DataFrame, model_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BH within each target across every feature, Holm across the primary models.

    Raises:
        AnalyzeError: if either family is incomplete, which would make the
            corrected numbers different from the planned ones.
    """
    planned_features = {c for m in plan.modalities.values() for c in m.features}
    for target in plan.targets:
        seen = set(univariate_rows.loc[univariate_rows["target"] == target, "feature"])
        if seen != planned_features:
            msg = (
                f"Analysis A's family for {target} is incomplete: "
                f"{sorted(planned_features - seen)} missing"
            )
            raise AnalyzeError(msg)
    final_univariate = univariate.add_q(univariate_rows)

    primary = model_rows["role"] == ROLE_PRIMARY
    expected = len(plan.modalities) * len(plan.targets)
    p_values = model_rows.loc[primary, "perm_p"]
    if int(primary.sum()) != expected or p_values.isna().any():
        msg = (
            f"Holm needs {expected} {plan.models.primary} tests (every modality x every "
            f"target); found {int(primary.sum())} with {int(p_values.isna().sum())} missing"
        )
        raise AnalyzeError(msg)
    final_models = model_rows.copy()
    final_models["p_holm"] = np.nan
    final_models.loc[primary, "p_holm"] = holm_adjust([float(p) for p in p_values])
    return final_univariate, final_models


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RunResult:
    """What one run wrote."""

    run: str
    out_dir: Path
    n_labelled: int
    n_by_modality: Mapping[str, int]
    excluded: Mapping[str, tuple[int, ...]]
    corrected: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def report_lines(self) -> list[str]:
        """Counts and file names only."""
        lines = [
            f"per-modality run {self.run!r}: {self.n_labelled} labelled session(s) "
            f"with features -> {self.out_dir}"
        ]
        for modality, n in self.n_by_modality.items():
            gone = self.excluded.get(modality, ())
            suffix = f", left out for having no {modality} value: {list(gone)}" if gone else ""
            lines.append(f"  {modality}: {n} session(s){suffix}")
        if self.corrected:
            lines.append(
                f"corrections computed over the complete families: {FINAL_UNIVARIATE_FILE}, "
                f"{FINAL_MODELS_FILE}"
            )
        else:
            lines.append(
                "raw values only; BH q and Holm p are computed by the second run, over the "
                "complete families"
            )
        lines.extend(f"NOTE: {note}" for note in self.notes)
        lines.append(f"read {self.out_dir / SUMMARY_FILE}")
        return lines


def load_prior(plan: Plan, prior_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """The first run's output, checked against this plan.

    Raises:
        AnalyzeError: if it is missing, from another run, or from another
            analysis.
    """
    record_path = prior_dir / RUN_FILE
    if not record_path.is_file():
        msg = f"no {RUN_FILE} in {prior_dir}: pass the first run's results folder as --prior"
        raise AnalyzeError(msg)
    record: dict[str, Any] = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("run") != RUN_FIRST:
        msg = f"{prior_dir} holds run {record.get('run')!r}, not the first run"
        raise AnalyzeError(msg)
    if record.get("analysis_sha256") != plan.analysis_digest():
        msg = (
            f"{prior_dir} was made under a different analysis (digest "
            f"{str(record.get('analysis_sha256'))[:12]}..., this plan "
            f"{plan.analysis_digest()[:12]}...), so its results cannot share a correction"
        )
        raise AnalyzeError(msg)
    if record.get("inputs", {}).get("base_features_sha256") != plan.inputs.base_features_sha256:
        msg = f"{prior_dir} was made from a different bundle feature table"
        raise AnalyzeError(msg)
    if record.get("n_permutations") != plan.validation.n_permutations:
        msg = (
            f"{prior_dir} ran {record.get('n_permutations')} permutation(s); the plan "
            f"has {plan.validation.n_permutations}"
        )
        raise AnalyzeError(msg)
    return read_csv(prior_dir / UNIVARIATE_FILE), read_csv(prior_dir / MODELS_FILE), record


@dataclass(frozen=True, slots=True)
class Work:
    """Everything one run computes before the models are fitted."""

    calls: list[Any]
    univariate: pd.DataFrame
    n_by_modality: dict[str, int]
    excluded: dict[str, tuple[int, ...]]
    notes: list[str]


def _work(
    plan: Plan,
    modalities: Sequence[str],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    n_permutations: int,
) -> Work:
    """Prepare every model fit, and compute Analysis A, for one run's modalities."""
    work = Work(calls=[], univariate=pd.DataFrame(), n_by_modality={}, excluded={}, notes=[])
    correlations: list[pd.DataFrame] = []
    for task in make_tasks(plan, modalities):
        columns = plan.modalities[task.modality].features
        prepared = prepare(features, labels, columns, task.target)
        work.calls.append(
            delayed(fit_task)(
                task, prepared.matrix, prepared.truth, prepared.groups, plan, n_permutations
            )
        )
        if task.role != ROLE_PRIMARY:
            continue
        # Once per modality and target: the correlations, and who was left out.
        work.excluded[task.modality] = prepared.excluded_no_features
        work.n_by_modality[task.modality] = len(prepared.groups)
        if prepared.excluded_no_label:
            work.notes.append(
                f"{task.modality} on {task.target}: {prepared.excluded_no_label} session(s) "
                f"without a {task.target} value left out"
            )
        frame = univariate.table(
            univariate.correlate(prepared.features, columns, prepared.outcome, [task.target]),
            adjust=False,
        )
        frame.insert(0, "modality", task.modality)
        correlations.append(frame)
    return replace(work, univariate=pd.concat(correlations, ignore_index=True))


def _in_task_order(fitted: pd.DataFrame, plan: Plan, modalities: Sequence[str]) -> pd.DataFrame:
    """Model rows in the plan's order, whatever order the workers finished in."""
    position = {
        (t.modality, t.target, t.model): i for i, t in enumerate(make_tasks(plan, modalities))
    }
    ordered = fitted.assign(
        _order=[
            position[key]
            for key in zip(fitted["modality"], fitted["target"], fitted["model"], strict=True)
        ]
    )
    return ordered.sort_values("_order", ignore_index=True).drop(columns="_order")


def run(
    plan: Plan,
    run_name: str,
    *,
    labels_path: Path,
    out_dir: Path,
    commit: str,
    bundle: Path | None = None,
    face_table: Path | None = None,
    prior_dir: Path | None = None,
    jobs: int = 1,
    n_permutations: int | None = None,
    now: datetime | None = None,
) -> RunResult:
    """Run one of the two planned handoffs.

    Args:
        plan: The plan.
        run_name: `first` or `second`.
        labels_path: The questionnaire scores. Read here and nowhere else.
        out_dir: Results folder. Must not already hold results.
        commit: The clean commit this runs from, recorded with the results.
        bundle: The handoff bundle folder, for the first run.
        face_table: The OpenFace face table, for the second run.
        prior_dir: The first run's results folder, for the second run.
        jobs: Parallel workers.
        n_permutations: Override, for tests only. A run whose count differs
            from the plan cannot be combined with the other run.
        now: Timestamp, for deterministic tests.

    Raises:
        AnalyzeError: if anything planned cannot be honoured.
    """
    started = now or datetime.now(UTC)
    if (out_dir / RUN_FILE).exists():
        msg = f"{out_dir} already holds results; choose a new --out"
        raise AnalyzeError(msg)
    if run_name == RUN_SECOND and prior_dir is None:
        msg = "the second run needs the first run's results folder: pass --prior"
        raise AnalyzeError(msg)
    if run_name == RUN_FIRST and prior_dir is not None:
        msg = "the first run has no prior run; --prior belongs to the second"
        raise AnalyzeError(msg)
    count = plan.validation.n_permutations if n_permutations is None else n_permutations

    prior = load_prior(plan, prior_dir) if prior_dir is not None else None
    tables = load_tables(plan, run_name, bundle=bundle, face_table=face_table)
    labels = load_labels(labels_path, plan.targets)
    absent = [t for t in plan.targets if t not in labels.targets]
    if absent:
        msg = f"the labels file lacks planned target(s) {absent}"
        raise AnalyzeError(msg)
    cohort = build_cohort(tables.frame, labels, {})
    notes = list(tables.notes)
    if cohort.features_only:
        notes.append(
            f"{len(cohort.features_only)} session(s) with features but no label: "
            f"{list(cohort.features_only)}"
        )
    if cohort.labels_only:
        notes.append(f"{len(cohort.labels_only)} labelled session(s) have no row in the table")

    modalities = plan.run_modalities(run_name)
    work = _work(plan, modalities, cohort.features, cohort.labels, count)
    notes.extend(work.notes)
    n_by_modality, excluded, univariate_rows = work.n_by_modality, work.excluded, work.univariate

    present = set(cohort.features["session_id"].astype(int))
    present_flagged = [s for s in plan.sessions.flagged if s in present]
    logger.info("%s: fitting %d model(s), %d permutation(s) each", STAGE, len(work.calls), count)
    model_rows = _in_task_order(
        pd.DataFrame(_parallel(jobs, work.calls), columns=list(MODEL_COLUMNS)), plan, modalities
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / UNIVARIATE_FILE, univariate_rows)
    write_csv(out_dir / MODELS_FILE, model_rows)

    final: tuple[pd.DataFrame, pd.DataFrame] | None = None
    if prior is not None:
        prior_univariate, prior_models, _ = prior
        final = correct(
            plan,
            pd.concat([prior_univariate, univariate_rows], ignore_index=True),
            pd.concat([prior_models, model_rows], ignore_index=True),
        )
        write_csv(out_dir / FINAL_UNIVARIATE_FILE, final[0])
        write_csv(out_dir / FINAL_MODELS_FILE, final[1])

    plan_path = resolve_plan_path()
    record: dict[str, Any] = {
        "analysis": "per_modality",
        "status": "exploratory",
        "run": run_name,
        "commit": commit,
        "plan_path": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "analysis_sha256": plan.analysis_digest(),
        "inputs": dict(tables.digests),
        "prior": None
        if prior is None
        else {
            "dir": str(prior_dir),
            "commit": prior[2].get("commit"),
            "inputs": prior[2].get("inputs"),
        },
        "modalities": list(modalities),
        "n_sessions_labelled_with_features": cohort.n,
        "n_sessions_by_modality": n_by_modality,
        "excluded_no_features": {k: list(v) for k, v in excluded.items()},
        "flagged_sessions_present": present_flagged,
        "n_permutations": count,
        "corrected": final is not None,
        "notes": notes,
        "started": started.isoformat(),
        "finished": datetime.now(UTC).isoformat(),
        "environment": environment_record(),
    }
    write_json(out_dir / RUN_FILE, record)
    write_text(
        out_dir / SUMMARY_FILE,
        summary_markdown(plan, run_name, univariate_rows, model_rows, final, record),
    )
    return RunResult(
        run=run_name,
        out_dir=out_dir,
        n_labelled=cohort.n,
        n_by_modality=n_by_modality,
        excluded=excluded,
        corrected=final is not None,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _fmt(value: object, places: int = 3) -> str:
    """One table cell."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{places}f}"
    return str(value)


def _markdown(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """A markdown table; p-values and q to four places, the rest to three."""
    places = {c: 4 if c.startswith("p") or c == "q" else 3 for c in columns}
    head = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_fmt(row[c], places[c]) for c in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([head, rule, *body])


def summary_markdown(
    plan: Plan,
    run_name: str,
    univariate_rows: pd.DataFrame,
    model_rows: pd.DataFrame,
    final: tuple[pd.DataFrame, pd.DataFrame] | None,
    record: Mapping[str, Any],
) -> str:
    """A readable summary, for the person who ran it."""
    univariate_view, model_view = final if final is not None else (univariate_rows, model_rows)
    status = (
        "Corrections computed over the complete families."
        if final is not None
        else "**Raw values only.** BH q and Holm p are computed by the second run, over the "
        "complete families; none is computed here."
    )
    title = "final" if final is not None else f"{run_name} run"
    lines = [
        f"# Per-modality evaluation: {title}",
        "",
        "**Exploratory**, pre-registered in `docs/prereg/per-modality.md` after the labels "
        "were received. Not a confirmatory result.",
        "",
        f"Commit `{record['commit']}`, analysis digest `{str(record['analysis_sha256'])[:12]}`, "
        f"{record['n_permutations']} permutation(s) per model. {status}",
        "",
        "## Analysis B: one model per modality",
        "",
        f"`{plan.models.primary}` is primary, with Holm across its "
        f"{len(plan.modalities) * len(plan.targets)} permutation tests; "
        f"`{plan.models.secondary}` is secondary and uncorrected. R², MAE and rho are "
        "leave-one-out; the permutation p compares the 5-fold R² with its null.",
        "",
        _markdown(
            model_view,
            [
                "modality",
                "target",
                "model",
                "n",
                "loo_r2",
                "loo_mae",
                "loo_spearman",
                "kfold_r2_mean",
                "perm_observed_r2",
                "perm_p",
                "p_holm",
                "n_fits_not_converged",
            ],
        ),
        "",
        "## Analysis A: one feature at a time",
        "",
        "Spearman rho over the sessions with both values; q is Benjamini-Hochberg within each "
        "outcome across every testable feature of all four modalities.",
        "",
        _markdown(
            univariate_view, ["modality", "feature", "target", "n", "rho", "p", "q", "untestable"]
        ),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timing, without labels
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Estimate:
    """Measured cost of one run, on random targets."""

    run: str
    jobs: int
    observed_s: Mapping[str, float]
    per_permutation_s: Mapping[str, float]

    def total_s(self, n_permutations: int) -> float:
        """Wall time with the tasks spread over the workers.

        The longer of an even share of the total and the slowest single task,
        since one task runs on one worker.
        """
        per_task = [
            self.observed_s[key] + n_permutations * self.per_permutation_s[key]
            for key in self.observed_s
        ]
        return max(sum(per_task) / max(1, min(self.jobs, len(per_task))), *per_task)

    def report_lines(self, n_permutations: int) -> list[str]:
        """Timings only."""
        lines = [f"run {self.run!r}, {self.jobs} worker(s), timed on random targets:"]
        lines.extend(
            f"  {key}: {self.observed_s[key]:.1f} s observed, "
            f"{self.per_permutation_s[key]:.2f} s per permutation"
            for key in self.observed_s
        )
        lines.extend(
            f"  about {self.total_s(count) / 3600:.1f} h at {count} permutation(s)"
            for count in sorted({100, n_permutations})
        )
        return lines


def estimate_runtime(
    plan: Plan,
    run_name: str,
    *,
    bundle: Path | None = None,
    face_table: Path | None = None,
    jobs: int = 1,
    n_probe: int = 4,
) -> Estimate:
    """Time one run on random targets. Reads no labels."""
    tables = load_tables(plan, run_name, bundle=bundle, face_table=face_table)
    frame = tables.frame
    rng = np.random.default_rng(plan.validation.seed)
    fake = pd.DataFrame({t: rng.normal(size=len(frame)) for t in plan.targets})
    keys: list[str] = []
    calls: list[Any] = []
    for task in make_tasks(plan, plan.run_modalities(run_name)):
        prepared = prepare(frame, fake, plan.modalities[task.modality].features, task.target)
        keys.append(f"{task.modality} {task.model} {task.target}")
        calls.append(delayed(_timed)(task, prepared, plan, n_probe))
    timings: list[tuple[float, float]] = list(Parallel(n_jobs=jobs)(calls))
    return Estimate(
        run=run_name,
        jobs=jobs,
        observed_s={k: t[0] for k, t in zip(keys, timings, strict=True)},
        per_permutation_s={k: t[1] for k, t in zip(keys, timings, strict=True)},
    )


def _timed(task: Task, prepared: Prepared, plan: Plan, n_probe: int) -> tuple[float, float]:
    """Seconds for the observed estimates, and per permutation."""
    start = time.perf_counter()
    permutation_baseline(
        prepared.matrix,
        prepared.truth,
        prepared.groups,
        model=task.model,
        seed=plan.validation.seed,
        n_permutations=n_probe,
        folds=plan.validation.permutation_folds,
    )
    # n_probe permutations plus the observed split.
    per_permutation = (time.perf_counter() - start) / (n_probe + 1)
    start = time.perf_counter()
    evaluate(
        prepared.matrix,
        prepared.truth,
        prepared.groups,
        model=task.model,
        seed=plan.validation.seed,
        stability_folds=plan.validation.stability_folds,
        stability_repeats=plan.validation.stability_repeats,
    )
    return time.perf_counter() - start + per_permutation, per_permutation
