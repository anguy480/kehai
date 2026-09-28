"""The confirmatory/exploratory split, and the multiplicity correction.

Implements the decision in docs/decisions/0012. Pure functions over column
names and p-values: no fitting, no data, so the arithmetic a reviewer will
question can be checked directly.

The point of this module is that the tiering is executable rather than a
paragraph in a methods section. `vc model` reports the confirmatory tests
first, with the correction applied and the family of tests stated, and labels
everything else as exploratory with its count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vc_multimodal.contracts import family_of

if TYPE_CHECKING:
    from vc_multimodal.config import ModelConfig


@dataclass(frozen=True, slots=True)
class TierPlan:
    """How an analysis divides into confirmatory and exploratory parts.

    Attributes:
        primary: Confirmatory feature columns, present in the table.
        exploratory: Every other feature column.
        missing: Confirmatory features named in configuration but absent from
            the table. Not silently dropped: a primary feature that vanished is
            a change to the pre-registered analysis and has to be visible.
        awaiting: Families whose primary features have not been fixed yet.
        n_primary_tests: How many confirmatory tests will be reported.
    """

    primary: tuple[str, ...]
    exploratory: tuple[str, ...]
    missing: tuple[str, ...]
    awaiting: tuple[str, ...]
    n_primary_tests: int

    @property
    def is_complete(self) -> bool:
        """Whether every configured primary feature was found."""
        return not self.missing and not self.awaiting

    @property
    def n_features(self) -> int:
        """Every feature column the plan covers, both tiers."""
        return len(self.primary) + len(self.exploratory)

    def counted_by_source(self) -> dict[str, int]:
        """Features by where they came from, for an honest denominator.

        The text baseline is counted separately because it is not measured by
        this pipeline, and a reader comparing "62 exploratory features" against
        a 54-feature extraction has no way to reconcile the two otherwise.
        """
        counts = {"measured here": 0, "text baseline": 0}
        for name in (*self.primary, *self.exploratory):
            key = "text baseline" if name.startswith("text__") else "measured here"
            counts[key] += 1
        return counts


def resolve_tiers(columns: Sequence[str], config: ModelConfig) -> TierPlan:
    """Work out the confirmatory and exploratory columns for a feature table.

    Args:
        columns: Every column in the feature table, identifiers included.
        config: The model configuration, carrying the tier definitions.

    Returns:
        The plan, including anything configured but missing.
    """
    features = tuple(name for name in columns if family_of(name) is not None)
    configured = config.tiers.primary_columns

    primary = tuple(name for name in configured if name in features)
    missing = tuple(name for name in configured if name not in features)
    exploratory = tuple(name for name in features if name not in set(primary))

    # The estimate count is deliberately not derived here. It used to be
    # len(feature_sets) x targets x models, which assumed one estimate per
    # feature set - false once a set named in a comparison is evaluated both
    # confirmatory and exploratory - and the "exploratory estimates" figure
    # subtracted a count of *tests* from a count of *estimates*, which are
    # different units. The stage that does the work supplies the real numbers
    # as `EstimateCounts`.
    n_primary_tests = len(config.tiers.primary_comparisons) * len(config.targets)

    return TierPlan(
        primary=primary,
        exploratory=exploratory,
        missing=missing,
        awaiting=config.tiers.families_awaiting_primaries,
        n_primary_tests=n_primary_tests,
    )


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm-Bonferroni step-down correction, in the input order.

    Chosen over Bonferroni because it is uniformly more powerful at the same
    familywise error rate, and over Benjamini-Hochberg because with four
    confirmatory tests controlling the familywise rate is the stricter and
    more conventional claim.

    Args:
        p_values: Unadjusted p-values.

    Returns:
        Adjusted p-values, each capped at 1 and monotone in the original
        ranking.

    Raises:
        ValueError: if any p-value is outside [0, 1].
    """
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        msg = f"p-values must lie in [0, 1], got {list(p_values)}"
        raise ValueError(msg)
    count = len(p_values)
    if count == 0:
        return ()

    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        # Step-down: the multiplier shrinks as we move to larger p-values, and
        # the running maximum keeps the result monotone.
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return tuple(adjusted)


@dataclass(frozen=True, slots=True)
class EstimateCounts:
    """How many cross-validated estimates a run will actually produce.

    Supplied by the stage that builds them rather than derived from a formula
    here: a formula drifts from the code the moment the code changes, which is
    exactly what happened.
    """

    confirmatory: int
    exploratory: int

    @property
    def total(self) -> int:
        """Every estimate."""
        return self.confirmatory + self.exploratory


def describe_plan(
    plan: TierPlan, config: ModelConfig, counts: EstimateCounts | None = None
) -> list[str]:
    """Explain the tier split, for a log or a stage summary.

    Args:
        plan: The resolved plan.
        config: The model configuration.
        counts: The estimates the run will produce, where the caller knows
            them. Omitted rather than guessed at: a wrong count is worse than
            no count, because it reads as information.
    """
    by_source = plan.counted_by_source()
    lines = [
        "analysis tiers (docs/decisions/0012)",
        f"  confirmatory: {len(plan.primary)} feature(s), "
        f"{plan.n_primary_tests} test(s), {config.tiers.multiplicity_correction} "
        f"correction across them",
    ]
    for comparison in config.tiers.primary_comparisons:
        lines.append(f"    {comparison.name}: {comparison.against[0]} vs {comparison.against[1]}")

    composition = ", ".join(f"{count} {source}" for source, count in by_source.items() if count)
    lines.append(
        f"  exploratory: {len(plan.exploratory)} of {plan.n_features} feature(s) "
        f"in the table ({composition}), reported without confirmatory claims"
    )
    if counts is not None:
        lines.append(
            f"  estimates: {counts.total} ({counts.confirmatory} confirmatory, "
            f"{counts.exploratory} exploratory)"
        )
    if plan.missing:
        lines.append(
            f"  MISSING: {list(plan.missing)} named as primary but absent from the "
            f"feature table. A pre-registered feature that vanished is a change to "
            f"the analysis, not a detail."
        )
    if plan.awaiting:
        lines.append(
            f"  NOT YET FIXED: primary features for {list(plan.awaiting)}. Those "
            f"families contribute to the exploratory tier only until they are named."
        )
    return lines


def primary_feature_report(
    columns: Sequence[str], config: ModelConfig
) -> Mapping[str, tuple[str, ...]]:
    """Confirmatory features per family, restricted to those present."""
    available = {name for name in columns if family_of(name) is not None}
    return {
        family: tuple(name for name in features if name in available)
        for family, features in config.tiers.primary_features.items()
    }
