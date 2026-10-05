"""The analysis plan: config/exploratory/au_baseline.yaml, validated and resolved.

Resolution turns each named set into the exact columns it uses, against the
table it will be fitted on, and refuses a set whose columns are missing rather
than fitting a smaller set under the same name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vc_multimodal.config import ConfigError, read_yaml

PLAN_PATH: Final = Path("config/exploratory/au_baseline.yaml")

KIND_AU_SET: Final = "au_sets"
KIND_SEARCH: Final = "search"
KIND_REFERENCE: Final = "reference"


class PlanError(ValueError):
    """Raised when the plan is invalid or does not fit the feature table."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Inputs(_Base):
    """Digests of the only inputs the analysis accepts."""

    bundle_features_sha256: str
    pooled_features_sha256: str = ""


class AuSet(_Base):
    """A named combination of units and statistics, crossed with every window."""

    name: str
    units: tuple[str, ...] = Field(min_length=1)
    stats: tuple[str, ...] = Field(min_length=1)


class Reference(_Base):
    """A reference feature set selected by column prefix."""

    prefixes: tuple[str, ...] = Field(min_length=1)
    expected_columns: int | None = None


class Search(_Base):
    """The nested-CV forward selection."""

    name: str
    units: tuple[str, ...] = Field(min_length=1)
    stats: tuple[str, ...] = Field(min_length=1)
    windows: tuple[str, ...] = Field(min_length=1)
    max_features: int = Field(ge=1)
    inner_folds: int = Field(ge=2)


class Evaluation(_Base):
    """Targets, models and cross-validation schemes."""

    targets: tuple[str, ...] = Field(min_length=1)
    primary_target: str
    models: tuple[Literal["elastic_net", "random_forest"], ...] = Field(min_length=1)
    kfold_folds: int = Field(ge=2)
    kfold_repeats: int = Field(ge=1)


class Permutation(_Base):
    """The shared permutation null."""

    n_permutations: int = Field(ge=0)
    max_statistic_family: tuple[Literal["au_sets", "search"], ...] = Field(min_length=1)


class Sensitivity(_Base):
    """The leave-sessions-out rerun of the best sets."""

    top_n: int = Field(ge=1)


class Plan(_Base):
    """The whole plan file."""

    inputs: Inputs
    windows: dict[str, tuple[str, ...]]
    au_sets: tuple[AuSet, ...] = Field(min_length=1)
    references: dict[str, Reference]
    search: Search
    evaluation: Evaluation
    permutation: Permutation
    sensitivity: Sensitivity

    @model_validator(mode="after")
    def _consistent(self) -> Plan:
        if self.evaluation.primary_target not in self.evaluation.targets:
            msg = f"primary_target {self.evaluation.primary_target!r} is not among the targets"
            raise ValueError(msg)
        unknown = [w for w in self.search.windows if w not in self.windows]
        if unknown:
            msg = f"search windows {unknown} are not defined under windows"
            raise ValueError(msg)
        names = [s.name for s in self.au_sets]
        if len(set(names)) != len(names):
            msg = "au_sets names must be unique"
            raise ValueError(msg)
        return self


def load_plan(path: Path = PLAN_PATH) -> Plan:
    """Read and validate the plan.

    Raises:
        PlanError: if the file is missing or invalid.
    """
    try:
        return Plan.model_validate(read_yaml(path))
    except (ConfigError, ValidationError) as exc:
        msg = f"invalid plan {path}: {exc}"
        raise PlanError(msg) from exc


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """One evaluated set of columns.

    Attributes:
        name: `<set>__<window>` for an AU set, else the configured name.
        kind: `au_sets`, `search` or `reference`.
        group: The AU set or reference it came from.
        window: The window, or "" for a reference or the search.
        columns: The columns fitted, or the candidates for the search.
        aliases: Other configured names with exactly these columns.
    """

    name: str
    kind: str
    group: str
    window: str
    columns: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    @property
    def in_family(self) -> bool:
        """Whether this set counts towards the max-statistic null."""
        return self.kind in (KIND_AU_SET, KIND_SEARCH)

    @property
    def is_search(self) -> bool:
        """Whether this is the nested forward selection."""
        return self.kind == KIND_SEARCH


def column_names(
    prefixes: Sequence[str], units: Sequence[str], stats: Sequence[str]
) -> tuple[str, ...]:
    """`<prefix>__<unit>_<stat>`, window-major, as aggregate names them."""
    return tuple(f"{p}__{u}_{s}" for p in prefixes for u in units for s in stats)


def _require(name: str, columns: Sequence[str], available: set[str]) -> None:
    missing = [c for c in columns if c not in available]
    if missing:
        msg = f"set {name!r} needs column(s) absent from the table: {missing}"
        raise PlanError(msg)


def resolve(plan: Plan, table_columns: Sequence[str]) -> tuple[FeatureSet, ...]:
    """Every set the analysis evaluates, in plan order.

    AU sets with identical columns are evaluated once: the first keeps its name
    and the rest are recorded as its aliases.

    Raises:
        PlanError: if a set's columns are absent, or a reference has the wrong
            number of columns.
    """
    available = {str(c) for c in table_columns}
    resolved: list[FeatureSet] = []
    by_columns: dict[tuple[str, ...], int] = {}

    for au_set in plan.au_sets:
        for window, prefixes in plan.windows.items():
            name = f"{au_set.name}__{window}"
            columns = column_names(prefixes, au_set.units, au_set.stats)
            _require(name, columns, available)
            if columns in by_columns:
                index = by_columns[columns]
                first = resolved[index]
                resolved[index] = replace(first, aliases=(*first.aliases, name))
                continue
            by_columns[columns] = len(resolved)
            resolved.append(FeatureSet(name, KIND_AU_SET, au_set.name, window, columns))

    search = plan.search
    candidates = tuple(
        column
        for window in search.windows
        for column in column_names(plan.windows[window], search.units, search.stats)
    )
    _require(search.name, candidates, available)
    resolved.append(FeatureSet(search.name, KIND_SEARCH, search.name, "", candidates))

    for name, reference in plan.references.items():
        columns = tuple(
            str(c) for c in table_columns if str(c).startswith(tuple(reference.prefixes))
        )
        if not columns:
            msg = f"reference {name!r} matched no column"
            raise PlanError(msg)
        if reference.expected_columns is not None and len(columns) != reference.expected_columns:
            msg = (
                f"reference {name!r} matched {len(columns)} column(s), "
                f"not the {reference.expected_columns} configured"
            )
            raise PlanError(msg)
        resolved.append(FeatureSet(name, KIND_REFERENCE, name, "", columns))

    return tuple(resolved)
