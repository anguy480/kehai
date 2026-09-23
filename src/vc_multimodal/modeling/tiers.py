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
        n_estimates: How many cross-validated estimates the full run produces.
    """

    primary: tuple[str, ...]
    exploratory: tuple[str, ...]
    missing: tuple[str, ...]
    awaiting: tuple[str, ...]
    n_primary_tests: int
    n_estimates: int

    @property
    def is_complete(self) -> bool:
        """Whether every configured primary feature was found."""
        return not self.missing and not self.awaiting

    @property
    def n_exploratory_estimates(self) -> int:
        """Estimates reported as exploratory rather than confirmatory."""
        return max(0, self.n_estimates - self.n_primary_tests)


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

    n_targets = len(config.targets)
    n_primary_tests = len(config.tiers.primary_comparisons) * n_targets
    n_estimates = len(config.feature_sets) * n_targets * len(config.models)

    return TierPlan(
        primary=primary,
        exploratory=exploratory,
        missing=missing,
        awaiting=config.tiers.families_awaiting_primaries,
        n_primary_tests=n_primary_tests,
        n_estimates=n_estimates,
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


def describe_plan(plan: TierPlan, config: ModelConfig) -> list[str]:
    """Render the plan for the analysis report.

    States the number of confirmatory tests and the number of exploratory
    estimates explicitly, because the count is what makes the distinction
    meaningful to a reader.
    """
    lines = [
        "analysis tiers (docs/decisions/0012)",
        f"  confirmatory: {len(plan.primary)} feature(s), "
        f"{plan.n_primary_tests} test(s), "
        f"{config.tiers.multiplicity_correction} correction across them",
    ]
    for comparison in config.tiers.primary_comparisons:
        lines.append(f"    {comparison.name}: {comparison.against[0]} vs {comparison.against[1]}")
    lines.append(
        f"  exploratory: {len(plan.exploratory)} further feature(s), "
        f"{plan.n_exploratory_estimates} further estimate(s), reported without "
        f"confirmatory claims"
    )
    if plan.awaiting:
        lines.append(
            f"  NOT YET FIXED: primary features for {list(plan.awaiting)}. Those "
            f"families contribute to the exploratory tier only until they are named."
        )
    if plan.missing:
        lines.append(
            f"  MISSING: {list(plan.missing)} named as primary but absent from the "
            f"feature table. A pre-registered feature that vanished is a change to "
            f"the analysis, not a detail."
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
