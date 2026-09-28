"""Fit the models. This runs where the questionnaire scores live.

The other half of the handoff (ADR 0001). Everything before this stage runs on
a machine that has never seen a label; this stage is the only one that reads
them, and it is written to be run once, by someone who did not write it, from a
bundle and a labels file.

Consequences that shape the code:

* **No label reaches any output.** Not the values, not per-session predictions,
  not a residual: the results are metrics, comparisons and counts. A prediction
  is a transformed label and would leak the thing the split exists to protect,
  so none are written.
* **The analysis is fixed before the labels are opened.** Which features are
  confirmatory, which comparisons are tested and how the correction is applied
  all come from the configuration, which is version-controlled and was written
  on a machine with no labels on it (docs/decisions/0012).
* **Everything says what it could not do.** A session missing from the labels,
  a feature set whose columns are absent, a text table that cannot be joined:
  each is reported by name and count rather than quietly reducing the sample.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

from vc_multimodal.config import AppConfig
from vc_multimodal.io_utils import read_csv, write_csv, write_text
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.modeling.evaluate import (
    MODEL_ELASTIC_NET,
    Comparison,
    Evaluation,
    EvaluationError,
    Permutation,
    compare_errors,
    evaluate,
    permutation_baseline,
)
from vc_multimodal.modeling.text_features import (
    TextFeatureError,
    TextFeatures,
    load_configured,
)
from vc_multimodal.modeling.text_features import join as join_text
from vc_multimodal.modeling.tiers import TierPlan, describe_plan, holm_adjust, resolve_tiers
from vc_multimodal.paths import DataRoots

logger = get_logger(__name__)

STAGE: Final = "model"
RESULTS_FILENAME: Final = "model_results.csv"
COMPARISONS_FILENAME: Final = "model_comparisons.csv"
SUMMARY_FILENAME: Final = "model_summary.md"

#: Column naming the session in a labels file, tried in order. Public because
#: the handoff README has to tell the label holder what is accepted.
LABEL_ID_COLUMNS: Final = ("session_id", "session", "id")


class ModelError(RuntimeError):
    """Raised when the analysis cannot be run as configured."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LabelTable:
    """The questionnaire scores, and what they cover.

    The values live here and go no further: nothing in this object is written
    to disk.
    """

    frame: pd.DataFrame
    targets: tuple[str, ...]
    n_rows: int


def load_labels(path: Path, targets: Sequence[str]) -> LabelTable:
    """Read the labels file.

    Raises:
        ModelError: if it cannot be read, has no identifier, or is missing
            every configured target.
    """
    try:
        frame = read_csv(path)
    except (OSError, ValueError) as exc:
        msg = f"could not read the labels file {path.name}: {exc}"
        raise ModelError(msg) from exc

    frame.columns = [str(name).strip() for name in frame.columns]
    lowered = {name.lower(): name for name in frame.columns}
    id_column = next((lowered[name] for name in LABEL_ID_COLUMNS if name in lowered), None)
    if id_column is None:
        msg = (
            f"{path.name} has no session identifier column. Tried "
            f"{list(LABEL_ID_COLUMNS)}, and found {list(frame.columns)}. The labels "
            f"are matched to features by session, never by row order."
        )
        raise ModelError(msg)

    present = [name for name in targets if name in frame.columns]
    absent = [name for name in targets if name not in frame.columns]
    if not present:
        msg = (
            f"{path.name} contains none of the configured targets {list(targets)}; it has "
            f"{list(frame.columns)}. Either rename the columns or set model.targets."
        )
        raise ModelError(msg)
    if absent:
        logger.warning(
            "%s: target(s) %s are configured but absent from %s, so they are not modelled",
            STAGE,
            absent,
            path.name,
        )

    kept = frame[[id_column, *present]].rename(columns={id_column: "session_id"})
    numeric = pd.to_numeric(kept["session_id"], errors="coerce")
    if numeric.isna().any():
        msg = f"{path.name} has non-numeric values in its {id_column!r} column"
        raise ModelError(msg)
    kept["session_id"] = numeric.astype("int64")
    if kept["session_id"].duplicated().any():
        repeated = sorted({int(v) for v in kept["session_id"][kept["session_id"].duplicated()]})
        msg = f"{path.name} has more than one row for session(s) {repeated}"
        raise ModelError(msg)

    # Log the shape and nothing else: a count is not a score.
    logger.info("%s: labels cover %d session(s) and %d target(s)", STAGE, len(kept), len(present))
    return LabelTable(frame=kept, targets=tuple(present), n_rows=len(kept))


@dataclass(frozen=True, slots=True)
class Cohort:
    """The sessions that will actually be modelled."""

    features: pd.DataFrame
    labels: pd.DataFrame
    groups: tuple[object, ...]
    features_only: tuple[int, ...]
    labels_only: tuple[int, ...]

    @property
    def n(self) -> int:
        """How many sessions both sides cover."""
        return len(self.features)

    def report_lines(self) -> list[str]:
        """Coverage, by session id."""
        lines = [f"modelling {self.n} session(s)"]
        if self.features_only:
            lines.append(
                f"  {len(self.features_only)} session(s) have features but no label and are "
                f"excluded: {list(self.features_only)}"
            )
        if self.labels_only:
            lines.append(
                f"  {len(self.labels_only)} session(s) have a label but no features: "
                f"{list(self.labels_only)}"
            )
        return lines


def build_cohort(
    features: pd.DataFrame, labels: LabelTable, groups: Mapping[int, object]
) -> Cohort:
    """Intersect features with labels, reporting what each side lost.

    Raises:
        ModelError: if nothing is left.
    """
    ours = {int(v) for v in features["session_id"]}
    theirs = {int(v) for v in labels.frame["session_id"]}
    shared = sorted(ours & theirs)
    if not shared:
        msg = (
            "no session appears in both the feature table and the labels file, so there "
            "is nothing to model. Check that both use the same session numbering."
        )
        raise ModelError(msg)

    kept_features = (
        features[features["session_id"].isin(shared)]
        .sort_values("session_id", ignore_index=True)
        .copy()
    )
    kept_labels = (
        labels.frame[labels.frame["session_id"].isin(shared)]
        .sort_values("session_id", ignore_index=True)
        .copy()
    )
    return Cohort(
        features=kept_features,
        labels=kept_labels,
        groups=tuple(groups.get(session_id, session_id) for session_id in shared),
        features_only=tuple(sorted(ours - theirs)),
        labels_only=tuple(sorted(theirs - ours)),
    )


def load_groups(config: AppConfig, roots: DataRoots) -> dict[int, object]:
    """The participant each session belongs to.

    With `grouping: session` a session *is* a participant, which is how this
    dataset is documented. A participant map overrides that, and is the only
    way two sessions of one person end up in the same fold.

    Raises:
        ModelError: if a map is configured but unusable.
    """
    if config.model.grouping == "session":
        return {}
    name = config.model.participant_map
    if not name:  # pragma: no cover - config validation forbids this
        msg = "model.grouping='participant_map' requires model.participant_map"
        raise ModelError(msg)
    path = roots.work / name
    if not path.is_file():
        msg = f"the participant map {name!r} is not under the work root"
        raise ModelError(msg)
    try:
        frame = read_csv(path)
    except (OSError, ValueError) as exc:
        msg = f"could not read {name}: {exc}"
        raise ModelError(msg) from exc
    missing = [c for c in ("session_id", "participant_id") if c not in frame.columns]
    if missing:
        msg = f"{name} is missing column(s) {missing}"
        raise ModelError(msg)
    mapping: dict[int, object] = {
        int(session): str(participant)
        for session, participant in zip(frame["session_id"], frame["participant_id"], strict=True)
    }
    logger.info("%s: %d session(s) grouped by participant", STAGE, len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
def family_of(column: str) -> str:
    """The feature family a column belongs to."""
    return column.split("__", 1)[0] if "__" in column else ""


def columns_for(frame: pd.DataFrame, families: Sequence[str]) -> tuple[str, ...]:
    """Feature columns belonging to any of `families`, in table order."""
    wanted = set(families)
    return tuple(
        str(column)
        for column in frame.columns
        if not str(column).startswith("qc__") and family_of(str(column)) in wanted
    )


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """One named set of features, resolved against the table."""

    name: str
    families: tuple[str, ...]
    columns: tuple[str, ...]
    absent_families: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        """Whether it has any columns at all."""
        return bool(self.columns)


def resolve_feature_sets(frame: pd.DataFrame, config: AppConfig) -> dict[str, FeatureSet]:
    """Work out which configured feature sets this table can supply."""
    available = {family_of(str(c)) for c in frame.columns} - {""}
    resolved: dict[str, FeatureSet] = {}
    for name, families in config.model.feature_sets.items():
        columns = columns_for(frame, families)
        resolved[name] = FeatureSet(
            name=name,
            families=tuple(families),
            columns=columns,
            absent_families=tuple(f for f in families if f not in available),
        )
    return resolved


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Estimate:
    """One cross-validated estimate, and how it was produced."""

    feature_set: str
    target: str
    model: str
    tier: str
    evaluation: Evaluation
    permutation: Permutation | None

    def row(self) -> dict[str, object]:
        """One row of the results table. Metrics only, never a label."""
        scores = self.evaluation.loo
        row: dict[str, object] = {
            "feature_set": self.feature_set,
            "target": self.target,
            "model": self.model,
            "tier": self.tier,
            "n_sessions": scores.n,
            "n_features": self.evaluation.n_features,
            "loo_r2": round(scores.r2, 6),
            "loo_mae": round(scores.mae, 6),
            "loo_spearman": round(scores.spearman, 6),
            "loo_spearman_p": round(scores.spearman_p, 6),
            "kfold_r2_mean": (
                round(self.evaluation.stability, 6)
                if self.evaluation.stability is not None
                else None
            ),
            "kfold_r2_sd": (
                round(self.evaluation.stability_sd, 6)
                if self.evaluation.stability_sd is not None
                else None
            ),
            "schemes_disagree_on_sign": self.evaluation.disagrees_with_stability,
        }
        if self.permutation is not None:
            row.update(
                {
                    "permutation_scheme": self.permutation.scheme,
                    "permutation_observed_r2": round(self.permutation.observed, 6),
                    "permutation_null_mean_r2": round(self.permutation.null_mean, 6),
                    "permutation_null_p95_r2": round(self.permutation.null_p95, 6),
                    "permutation_p": round(self.permutation.p_value, 6),
                    "n_permutations": self.permutation.n_permutations,
                }
            )
        return row


@dataclass(frozen=True, slots=True)
class ConfirmatoryTest:
    """One pre-registered comparison, after correction."""

    name: str
    target: str
    model: str
    first: str
    second: str
    comparison: Comparison
    first_r2: float
    second_r2: float

    def row(self) -> dict[str, object]:
        """One row of the comparisons table."""
        return {
            "test": self.name,
            "target": self.target,
            "model": self.model,
            "feature_set": self.first,
            "compared_with": self.second,
            "n_sessions": self.comparison.n,
            "r2": round(self.first_r2, 6),
            "compared_r2": round(self.second_r2, 6),
            "median_error_difference": round(self.comparison.median_difference, 6),
            "p_value": round(self.comparison.p_value, 6),
            "p_holm": (
                round(self.comparison.p_adjusted, 6)
                if self.comparison.p_adjusted is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Everything the stage produced."""

    estimates: tuple[Estimate, ...]
    tests: tuple[ConfirmatoryTest, ...]
    cohort: Cohort
    plan: TierPlan
    text: TextFeatures | None
    results_path: Path
    comparisons_path: Path
    summary_path: Path
    notes: tuple[str, ...] = field(default_factory=tuple)


TierName = Literal["confirmatory", "exploratory"]

TIER_CONFIRMATORY: TierName = "confirmatory"
TIER_EXPLORATORY: TierName = "exploratory"


@dataclass(frozen=True, slots=True)
class Variant:
    """One way of evaluating a feature set: which columns, in which tier."""

    feature_set: str
    tier: TierName
    columns: tuple[str, ...]


def confirmatory_columns(feature_set: FeatureSet, config: AppConfig) -> tuple[str, ...]:
    """The pre-registered columns of a feature set.

    Per family: where that family has pre-registered features, only those are
    used; where it has none, the family is used whole.

    The asymmetry is deliberate and it is about the text baseline. We
    pre-registered a small set of features from our own families (three each
    from turns, prosody and the two facial windows) on prior literature. We
    never pre-registered a subset of the manuscript's text features, and
    choosing one now would mean selecting the baseline we are measured against,
    on no prior basis. So the text baseline enters a confirmatory comparison
    whole.
    """
    primary = config.model.tiers.primary_features
    collected: list[str] = []
    for family in feature_set.families:
        pre_registered = tuple(primary.get(family, ()))
        if pre_registered:
            collected.extend(name for name in pre_registered if name in feature_set.columns)
        else:
            collected.extend(name for name in feature_set.columns if family_of(name) == family)
    return tuple(dict.fromkeys(collected))


def variants_for(feature_set: FeatureSet, config: AppConfig) -> list[Variant]:
    """How this feature set should be evaluated, in which tiers.

    A set named in a pre-registered comparison is evaluated twice: confirmatory
    on its pre-registered columns, and exploratory on all of them. Both are
    wanted - the first is the result, the second is the fuller picture reported
    without confirmatory claims - and computing them separately is what keeps
    the confirmatory tests from quietly becoming the exploratory ones.

    Where the two column sets are identical, as they are for the text baseline,
    one estimate serves and is reported as confirmatory.
    """
    named = {
        name for comparison in config.model.tiers.primary_comparisons for name in comparison.against
    }
    if feature_set.name not in named:
        return [Variant(feature_set.name, TIER_EXPLORATORY, feature_set.columns)]

    restricted = confirmatory_columns(feature_set, config)
    if not restricted:  # pragma: no cover - a named set with no columns is skipped
        return [Variant(feature_set.name, TIER_EXPLORATORY, feature_set.columns)]
    if set(restricted) == set(feature_set.columns):
        return [Variant(feature_set.name, TIER_CONFIRMATORY, restricted)]
    return [
        Variant(feature_set.name, TIER_CONFIRMATORY, restricted),
        Variant(feature_set.name, TIER_EXPLORATORY, feature_set.columns),
    ]


def run(
    config: AppConfig,
    roots: DataRoots,
    *,
    features_path: Path,
    labels_path: Path,
    out_dir: Path | None = None,
    now: datetime | None = None,
) -> ModelResult:
    """Run the configured analysis.

    Args:
        config: Resolved configuration, which fixes the analysis.
        roots: Data roots, for the text features and the participant map.
        features_path: The feature table from the handoff bundle.
        labels_path: The questionnaire scores. Read here and nowhere else.
        out_dir: Where results go. Defaults to `$VC_OUT_ROOT/model`.
        now: Timestamp, for deterministic tests.

    Returns:
        Every estimate, the corrected confirmatory tests, and where they were
        written.

    Raises:
        ModelError: if the analysis cannot be run as configured.
    """
    moment = now or datetime.now(UTC)
    destination = out_dir or roots.out_path("model")
    destination.mkdir(parents=True, exist_ok=True)

    if not features_path.is_file():
        msg = f"no feature table at {features_path}"
        raise ModelError(msg)
    try:
        features = read_csv(features_path)
    except (OSError, ValueError) as exc:
        msg = f"could not read the feature table: {exc}"
        raise ModelError(msg) from exc

    notes: list[str] = []
    text: TextFeatures | None = None
    if config.model.text_features is not None:
        try:
            text = load_configured(roots.work, config.model.text_features)
            features, join_report = join_text(features, text)
            notes.extend(join_report.report_lines())
        except TextFeatureError as exc:
            # Not fatal, but it removes the baseline every confirmatory test is
            # against, so it has to be loud.
            notes.append(f"text features unavailable, so no comparison against text: {exc}")
            logger.warning("%s: %s", STAGE, notes[-1])

    labels = load_labels(labels_path, config.model.targets)
    cohort = build_cohort(features, labels, load_groups(config, roots))
    for line in cohort.report_lines():
        logger.info("%s: %s", STAGE, line)

    plan = resolve_tiers([str(c) for c in cohort.features.columns], config.model)
    for line in describe_plan(plan, config.model):
        logger.info("%s: %s", STAGE, line)
    if plan.missing:
        notes.append(
            f"confirmatory feature(s) {list(plan.missing)} are named in the configuration "
            f"but absent from the table, so the pre-registered analysis is not the one "
            f"being run"
        )
        logger.warning("%s: %s", STAGE, notes[-1])

    sets = resolve_feature_sets(cohort.features, config)
    unusable = [name for name, fs in sorted(sets.items()) if not fs.is_usable]
    if unusable:
        notes.append(f"feature set(s) {unusable} have no columns in this table and are skipped")
        logger.warning("%s: %s", STAGE, notes[-1])

    estimates, tests = _evaluate_everything(cohort, sets, plan, config, notes=notes)
    results = pd.DataFrame([estimate.row() for estimate in estimates])
    comparisons = pd.DataFrame([test.row() for test in tests])

    results_path = destination / RESULTS_FILENAME
    comparisons_path = destination / COMPARISONS_FILENAME
    summary_path = destination / SUMMARY_FILENAME
    write_csv(results_path, results)
    write_csv(comparisons_path, comparisons)

    result = ModelResult(
        estimates=tuple(estimates),
        tests=tuple(tests),
        cohort=cohort,
        plan=plan,
        text=text,
        results_path=results_path,
        comparisons_path=comparisons_path,
        summary_path=summary_path,
        notes=tuple(notes),
    )
    write_text(summary_path, _summary_markdown(result, config, moment))
    logger.info("%s: wrote %d estimate(s) to %s", STAGE, len(estimates), results_path.name)
    return result


def _evaluate_everything(
    cohort: Cohort,
    sets: Mapping[str, FeatureSet],
    plan: TierPlan,
    config: AppConfig,
    *,
    notes: list[str],
) -> tuple[list[Estimate], list[ConfirmatoryTest]]:
    """Every estimate, then the pre-registered comparisons over the same folds."""
    del plan  # the tier comes from the variants, which read the config directly

    # A set named in a pre-registered comparison is evaluated twice, on its
    # pre-registered columns and on all of them, so the work is counted in
    # variants rather than in feature sets.
    plans = {
        name: variants_for(feature_set, config)
        for name, feature_set in sorted(sets.items())
        if feature_set.is_usable
    }
    targets = [name for name in cohort.labels.columns if name != "session_id"]
    n_variants = sum(len(items) for items in plans.values())
    n_confirmatory = sum(
        1 for items in plans.values() for item in items if item.tier == TIER_CONFIRMATORY
    )
    total = n_variants * len(targets) * len(config.model.models)

    # This runs for tens of minutes, mostly inside the permutation nulls, and it
    # is run by someone who did not write it. Silence for that long is
    # indistinguishable from a hang, so the work is counted up front and each
    # estimate is logged as it lands.
    logger.info(
        "%s: %d estimate(s) to compute: %d variant(s) of %d feature set(s) "
        "(%d confirmatory) x %d target(s) x %d model(s)",
        STAGE,
        total,
        n_variants,
        len(plans),
        n_confirmatory,
        len(targets),
        len(config.model.models),
    )
    if n_confirmatory == 0:
        notes.append(
            "no feature set is evaluated in the confirmatory tier, so the "
            "pre-registered tests cannot be the ones reported"
        )
        logger.warning("%s: %s", STAGE, notes[-1])

    estimates: list[Estimate] = []
    # Keyed by (feature set, tier, target, model) so the comparisons reuse the
    # per-session errors rather than recomputing them on different folds, and so
    # a confirmatory comparison cannot pick up the exploratory variant of the
    # same set.
    by_key: dict[tuple[str, str, str, str], Evaluation] = {}
    done = 0

    for target in targets:
        truth = pd.to_numeric(cohort.labels[target], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(truth).all():
            notes.append(f"target {target!r} has no usable values and is skipped")
            logger.warning("%s: %s", STAGE, notes[-1])
            continue

        for variant in (item for items in plans.values() for item in items):
            for model in config.model.models:
                done += 1
                estimate = _evaluate_variant(
                    variant,
                    cohort,
                    truth,
                    target=target,
                    model=model,
                    config=config,
                    notes=notes,
                    position=(done, total),
                )
                if estimate is None:
                    continue
                by_key[variant.feature_set, variant.tier, target, model] = estimate.evaluation
                estimates.append(estimate)

    tests = _confirmatory_tests(by_key, config, notes=notes)
    return estimates, tests


def _evaluate_variant(
    variant: Variant,
    cohort: Cohort,
    truth: np.ndarray,
    *,
    target: str,
    model: str,
    config: AppConfig,
    notes: list[str],
    position: tuple[int, int],
) -> Estimate | None:
    """Cross-validate one variant against one target with one model.

    Returns None where the fit could not be made, having recorded why: one
    feature set failing is not a reason to abandon the other thirty estimates.
    """
    tiers = config.model.tiers
    seed = config.runtime.seed
    matrix = cohort.features[list(variant.columns)].to_numpy(dtype=np.float64)

    try:
        evaluation = evaluate(
            matrix,
            truth,
            cohort.groups,
            model=model,
            seed=seed,
            stability_folds=tiers.stability_folds if tiers.stability_cv != "none" else 0,
            stability_repeats=tiers.stability_repeats if tiers.stability_cv != "none" else 0,
        )
    except EvaluationError as exc:
        notes.append(
            f"{variant.feature_set} ({variant.tier}) on {target} with {model} "
            f"could not be evaluated: {exc}"
        )
        logger.warning("%s: %s", STAGE, notes[-1])
        return None

    permutation = None
    if variant.tier == TIER_CONFIRMATORY and model == MODEL_ELASTIC_NET:
        if config.model.n_permutations > 0:
            logger.info(
                "%s: %s on %s: running %d permutation(s), the slow part",
                STAGE,
                variant.feature_set,
                target,
                config.model.n_permutations,
            )
        permutation = permutation_baseline(
            matrix,
            truth,
            cohort.groups,
            model=model,
            seed=seed,
            n_permutations=config.model.n_permutations,
            folds=max(2, tiers.stability_folds),
        )

    done, total = position
    logger.info(
        "%s: [%d/%d] %s (%s) on %s with %s: %d feature(s), LOO R2 %+.3f%s",
        STAGE,
        done,
        total,
        variant.feature_set,
        variant.tier,
        target,
        model,
        evaluation.n_features,
        evaluation.loo.r2,
        " (with permutation null)" if permutation is not None else "",
    )
    return Estimate(
        feature_set=variant.feature_set,
        target=target,
        model=model,
        tier=variant.tier,
        evaluation=evaluation,
        permutation=permutation,
    )


def _confirmatory_tests(
    by_key: Mapping[tuple[str, str, str, str], Evaluation],
    config: AppConfig,
    *,
    notes: list[str],
) -> list[ConfirmatoryTest]:
    """The pre-registered comparisons, corrected together.

    Correction is applied across every confirmatory test in one family, which
    is what makes the count stated in the summary the count that was actually
    corrected for.
    """
    tiers = config.model.tiers
    primary_model = config.model.models[0]
    collected: list[ConfirmatoryTest] = []

    for comparison in tiers.primary_comparisons:
        first_name, second_name = comparison.against
        for target in {key[2] for key in by_key}:
            # The confirmatory variant specifically: comparing the exploratory
            # ones would make the pre-registered test a different test.
            first = by_key.get((first_name, TIER_CONFIRMATORY, target, primary_model))
            second = by_key.get((second_name, TIER_CONFIRMATORY, target, primary_model))
            if first is None or second is None:
                if first is None and second is None:
                    which = f"neither {first_name} nor {second_name}"
                else:
                    which = first_name if first is None else second_name
                notes.append(
                    f"confirmatory test {comparison.name!r} on {target} could not be run: "
                    f"{which} produced no estimate"
                )
                logger.warning("%s: %s", STAGE, notes[-1])
                continue
            try:
                result = compare_errors(first.errors, second.errors)
            except EvaluationError as exc:  # pragma: no cover - defensive
                notes.append(f"confirmatory test {comparison.name!r} on {target} failed: {exc}")
                continue
            collected.append(
                ConfirmatoryTest(
                    name=comparison.name,
                    target=target,
                    model=primary_model,
                    first=first_name,
                    second=second_name,
                    comparison=result,
                    first_r2=first.loo.r2,
                    second_r2=second.loo.r2,
                )
            )

    if collected and tiers.multiplicity_correction == "holm":
        # `replace` rather than rebuilding from __dict__: these are slotted
        # dataclasses and have no __dict__ at all, which crashed here after
        # every estimate had been computed - the most expensive possible place
        # to fail, and it produced no confirmatory output whatsoever.
        adjusted = holm_adjust([test.comparison.p_value for test in collected])
        collected = [
            replace(test, comparison=replace(test.comparison, p_adjusted=value))
            for test, value in zip(collected, adjusted, strict=True)
        ]
    return sorted(collected, key=lambda test: (test.name, test.target))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(value: float | None, places: int = 3) -> str:
    """A number, or a dash where there is none."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:+.{places}f}"


def _confirmatory_table(result: ModelResult) -> list[str]:
    """The pre-registered tests, which are the headline."""
    if not result.tests:
        return ["No confirmatory test could be run. See the notes below."]
    header = "| test | target | feature set | R2 | vs | R2 | median error diff | p | p (Holm) |"
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for test in result.tests:
        adjusted = (
            f"{test.comparison.p_adjusted:.4f}" if test.comparison.p_adjusted is not None else "-"
        )
        lines.append(
            f"| {test.name} | {test.target} | {test.first} | {_fmt(test.first_r2)} "
            f"| {test.second} | {_fmt(test.second_r2)} "
            f"| {_fmt(test.comparison.median_difference)} "
            f"| {test.comparison.p_value:.4f} | {adjusted} |"
        )
    return lines


def _r2_for_sorting(estimate: Estimate) -> float:
    """R2 with NaN pushed to the bottom, so a failed fit does not head a table."""
    value = estimate.evaluation.loo.r2
    return -np.inf if np.isnan(value) else value


def _estimate_table(result: ModelResult, tier: str) -> list[str]:
    """Every estimate in one tier, best first within each target."""
    rows = [estimate for estimate in result.estimates if estimate.tier == tier]
    if not rows:
        return [f"No {tier} estimates."]
    header = (
        "| feature set | target | model | features | LOO R2 | LOO MAE | LOO rho "
        "| 5-fold R2 (sd) | signs differ |"
    )
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for estimate in sorted(rows, key=lambda e: (e.target, -_r2_for_sorting(e))):
        scores = estimate.evaluation.loo
        spread = (
            f"{_fmt(estimate.evaluation.stability)} ({_fmt(estimate.evaluation.stability_sd)})"
            if estimate.evaluation.stability is not None
            else "-"
        )
        lines.append(
            f"| {estimate.feature_set} | {estimate.target} | {estimate.model} "
            f"| {estimate.evaluation.n_features} | {_fmt(scores.r2)} | {scores.mae:.3f} "
            f"| {_fmt(scores.spearman)} | {spread} "
            f"| {'yes' if estimate.evaluation.disagrees_with_stability else 'no'} |"
        )
    return lines


def _permutation_lines(result: ModelResult) -> list[str]:
    """What the confirmatory feature sets scored against shuffled labels."""
    with_null = [e for e in result.estimates if e.permutation is not None]
    if not with_null:
        return []
    first = with_null[0].permutation
    scheme = first.scheme if first is not None else "k-fold"
    lines = [
        "",
        "## Against shuffled labels",
        "",
        f"Each confirmatory feature set was also run against permuted labels, under "
        f"{scheme} cross-validation. The observed value in this table is computed under "
        f"the same scheme, so the two are comparable; the leave-one-out figures above "
        f"are not directly comparable with this null.",
        "",
        "| feature set | target | observed R2 | null mean | null p95 | p | permutations |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for estimate in with_null:
        null = estimate.permutation
        assert null is not None
        lines.append(
            f"| {estimate.feature_set} | {estimate.target} | {_fmt(null.observed)} | "
            f"{_fmt(null.null_mean)} | {_fmt(null.null_p95)} | {null.p_value:.4f} | "
            f"{null.n_permutations} |"
        )
    return lines


def _summary_markdown(result: ModelResult, config: AppConfig, moment: datetime) -> str:
    """The document someone reads instead of the CSVs."""
    tiers = config.model.tiers
    planned = len(tiers.primary_comparisons) * len(config.model.targets)
    parts: list[str] = [
        f"""# Analysis results

Run {moment.strftime("%Y-%m-%d")} over **{result.cohort.n} sessions**.

The confirmatory tests below are the result. Everything under Exploratory is
exactly that: {len(result.plan.exploratory)} features across
{len([e for e in result.estimates if e.tier == "exploratory"])} estimates, which
is a large enough surface that the best of them would look good whether or not
anything is there.

## Confirmatory tests ({len(result.tests)} of {planned} planned)

Pre-registered before any label was seen, and corrected together by
{tiers.multiplicity_correction}. Each test compares two feature sets on the
same leave-one-participant-out folds, session by session: the p-value is a
paired Wilcoxon signed-rank test on per-session absolute errors, and a positive
median error difference means the named feature set was closer.
""",
    ]
    parts.append("\n".join(_confirmatory_table(result)))

    parts.append(
        f"""
## Coverage

{chr(10).join(result.cohort.report_lines())}

Grouping: `{config.model.grouping}`. Leave-one-participant-out is the primary
estimate, for comparability with the manuscript; repeated
{tiers.stability_folds}-fold over {tiers.stability_repeats} repeats is reported
beside it. **Where the two disagree about the sign, the disagreement is the
finding**: leave-one-out is high variance for comparing models, and a result
that changes sign between schemes is not a stable result.
"""
    )

    parts.append("## Confirmatory estimates\n")
    parts.append("\n".join(_estimate_table(result, "confirmatory")))
    parts.extend(_permutation_lines(result))
    parts.append("\n## Exploratory estimates\n")
    parts.append(
        "Reported for completeness, with the number of comparisons stated above. "
        "These are not corrected and should not be read as tests.\n"
    )
    parts.append("\n".join(_estimate_table(result, "exploratory")))

    if result.text is not None and result.text.plan is not None:
        parts.append(
            f"""
## How the text baseline was aligned

The text features carry no identifier of their own. Rows were matched to
sessions under the rule `{result.text.plan.rule}`, quoted from the code that
wrote the file:

> {result.text.plan.provenance}

{result.text.n_sessions} session(s), {len(result.text.feature_columns)} features.
"""
        )
    elif result.text is None:
        parts.append(
            """
## The text baseline is missing

Every confirmatory test compares against the manuscript's text features, and
they could not be loaded, so those tests could not run. See the notes.
"""
        )

    if result.plan.missing or result.plan.awaiting:
        parts.append(
            f"""
## The analysis that ran is not quite the one planned

Confirmatory feature(s) named in the configuration but absent from the table:
{list(result.plan.missing) or "none"}. Families with no primary features fixed:
{list(result.plan.awaiting) or "none"}. Anything listed here changes what the
confirmatory tier means and should be resolved rather than noted.
"""
        )

    if result.notes:
        parts.append("## Notes\n")
        parts.extend(f"* {note}" for note in result.notes)

    parts.append(
        """
## What this cannot tell you

* Sixty-two participants supports a modest, well-specified test and no more.
  Read the effect sizes, not the significance verdicts.
* One session per participant: nothing here separates a stable trait from how
  someone was on the day.
* No label appears in any file this stage wrote, including as a prediction or a
  residual. If you need per-session diagnostics, compute them here rather than
  sending these outputs on.
"""
    )
    return "\n".join(parts).strip() + "\n"


def summarise(result: ModelResult) -> list[str]:
    """What the CLI prints: counts and the confirmatory tests only."""
    lines = [
        f"sessions modelled: {result.cohort.n}",
        f"estimates: {len(result.estimates)} "
        f"({len([e for e in result.estimates if e.tier == 'confirmatory'])} confirmatory)",
        "",
    ]
    if result.tests:
        lines.append("confirmatory tests:")
        for test in result.tests:
            adjusted = (
                f"  p(Holm)={test.comparison.p_adjusted:.4f}"
                if test.comparison.p_adjusted is not None
                else ""
            )
            lines.append(
                f"  {test.name} [{test.target}]: {test.first} R²={_fmt(test.first_r2)} "
                f"vs {test.second} R²={_fmt(test.second_r2)}  p={test.comparison.p_value:.4f}"
                f"{adjusted}"
            )
    else:
        lines.append("no confirmatory test could be run")

    for note in result.notes:
        lines.append(f"NOTE: {note}")
    lines.extend(["", f"wrote {result.results_path}", f"read {result.summary_path}"])
    return lines
