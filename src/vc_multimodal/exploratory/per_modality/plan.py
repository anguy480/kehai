"""The plan: config/exploratory/per_modality.yaml, validated.

Two digests are taken from it. The file digest records exactly what was read.
The analysis digest covers everything except the input digests, so the second
run can fix the face table's digest in the plan and still check that the first
run was made under the same analysis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vc_multimodal.config import ConfigError, read_yaml

PLAN_PATH: Final = Path("config/exploratory/per_modality.yaml")

RUN_FIRST: Final = "first"
RUN_SECOND: Final = "second"
RUNS: Final = (RUN_FIRST, RUN_SECOND)

SOURCE_BASE: Final = "base"
SOURCE_FACE: Final = "face"


class PlanError(ValueError):
    """Raised when the plan is missing or invalid."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Inputs(_Base):
    """Digests of the only tables the analysis accepts."""

    base_features_sha256: str = Field(min_length=64, max_length=64)
    face_features_sha256: str | None = Field(default=None, min_length=64, max_length=64)


class Modality(_Base):
    """One modality: where its columns come from, and which they are."""

    source: Literal["base", "face"]
    features: tuple[str, ...] = Field(min_length=1)


class Runs(_Base):
    """Which modalities each handoff covers."""

    first: tuple[str, ...] = Field(min_length=1)
    second: tuple[str, ...] = Field(min_length=1)


class Models(_Base):
    """The primary model is corrected; the secondary is not."""

    primary: Literal["elastic_net", "random_forest"]
    secondary: Literal["elastic_net", "random_forest"]


class Validation(_Base):
    """Seed, schemes and permutation count."""

    seed: int
    stability_folds: int = Field(ge=2)
    stability_repeats: int = Field(ge=1)
    permutation_folds: int = Field(ge=2)
    n_permutations: int = Field(ge=1)


class Correction(_Base):
    """How each family is corrected."""

    univariate: Literal["benjamini_hochberg"]
    primary_models: Literal["holm"]


class Sessions(_Base):
    """Sessions whose handling the analysis checks."""

    face_unavailable: tuple[int, ...] = ()
    flagged: tuple[int, ...] = ()


class Plan(_Base):
    """The whole plan."""

    prereg: str
    targets: tuple[str, ...] = Field(min_length=1)
    inputs: Inputs
    modalities: dict[str, Modality] = Field(min_length=1)
    runs: Runs
    models: Models
    validation: Validation
    correction: Correction
    sessions: Sessions

    @model_validator(mode="after")
    def _consistent(self) -> Plan:
        """Every modality runs exactly once, and no feature appears twice."""
        named = [*self.runs.first, *self.runs.second]
        if sorted(named) != sorted(self.modalities) or len(set(named)) != len(named):
            msg = (
                f"runs must name every modality exactly once: runs name {named}, "
                f"modalities are {sorted(self.modalities)}"
            )
            raise ValueError(msg)
        columns = [c for m in self.modalities.values() for c in m.features]
        if len(set(columns)) != len(columns):
            msg = "a feature is listed in more than one modality"
            raise ValueError(msg)
        if self.models.primary == self.models.secondary:
            msg = "the primary and secondary models must differ"
            raise ValueError(msg)
        return self

    def run_modalities(self, run: str) -> tuple[str, ...]:
        """The modalities one handoff covers.

        Raises:
            PlanError: if `run` is not a planned run.
        """
        if run == RUN_FIRST:
            return self.runs.first
        if run == RUN_SECOND:
            return self.runs.second
        msg = f"unknown run {run!r}; expected one of {list(RUNS)}"
        raise PlanError(msg)

    @property
    def model_names(self) -> tuple[str, str]:
        """Primary first."""
        return self.models.primary, self.models.secondary

    def analysis_digest(self) -> str:
        """SHA-256 of everything but the input digests, canonically serialised."""
        payload = self.model_dump(mode="json", exclude={"inputs"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_plan_path(path: Path = PLAN_PATH) -> Path:
    """The plan file: relative to the current folder, else beside the package.

    The label holder runs from the folder holding a bundle, where the relative
    path does not exist; the checkout this package was installed from is then
    used, as `config.load_config` does for the default configuration.
    """
    if path != PLAN_PATH or path.is_file():
        return path
    beside_package = Path(__file__).resolve().parents[4] / PLAN_PATH
    return beside_package if beside_package.is_file() else path


def load_plan(path: Path = PLAN_PATH) -> Plan:
    """Read and validate the plan.

    Raises:
        PlanError: if the file is missing or invalid.
    """
    resolved = resolve_plan_path(path)
    try:
        return Plan.model_validate(read_yaml(resolved))
    except (ConfigError, ValidationError) as exc:
        msg = f"invalid plan {resolved}: {exc}"
        raise PlanError(msg) from exc
