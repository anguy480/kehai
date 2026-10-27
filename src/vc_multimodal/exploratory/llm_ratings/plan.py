"""What docs/prereg/llm-ratings.md fixes: the items, the prompt, the sampling.

The sampling settings are constants here rather than config values, so they
cannot drift with an edit to a YAML file. The config,
`config/exploratory/llm_ratings.yaml`, holds only what the plan leaves to be
set later: the backend, its endpoint and model, and the prompt's SHA-256, which
the rater checks before anything else.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

REPO_ROOT: Final = Path(__file__).resolve().parents[4]
PLAN_PATH: Final = REPO_ROOT / "config" / "exploratory" / "llm_ratings.yaml"

#: Fixed by the plan.
TEMPERATURE: Final = 0.0
CALLS_PER_SESSION: Final = 3
MAX_RETRIES: Final = 3
PILOT_SESSIONS: Final = 3
EXPECTED_SESSIONS: Final = 62

BACKENDS: Final = ("openai", "local")
FEATURE_PREFIX: Final = "llm__"

SYSTEM_MARKER: Final = "=== system ===\n"
USER_MARKER: Final = "\n=== user ===\n"
DIALOGUE_PLACEHOLDER: Final = "{{DIALOGUE}}"


@dataclass(frozen=True, slots=True)
class Item:
    """One rated item and its integer range."""

    key: str
    scale: str
    low: int
    high: int

    @property
    def feature(self) -> str:
        """Its column in the feature table."""
        return f"{FEATURE_PREFIX}{self.key}"


#: Set K in K6 order, then Set S, as in the prompt.
ITEMS: Final = (
    Item("k_nervous", "K", 0, 4),
    Item("k_hopeless", "K", 0, 4),
    Item("k_restless", "K", 0, 4),
    Item("k_depressed", "K", 0, 4),
    Item("k_effort", "K", 0, 4),
    Item("k_worthless", "K", 0, 4),
    Item("s_awareness", "S", 1, 4),
    Item("s_cognition", "S", 1, 4),
    Item("s_communication", "S", 1, 4),
    Item("s_motivation", "S", 1, 4),
    Item("s_rrb", "S", 1, 4),
)
FEATURES: Final = tuple(item.feature for item in ITEMS)


class PlanError(RuntimeError):
    """Raised when the rater cannot run as pre-registered."""


class BackendNotSetError(PlanError):
    """Raised when no backend has been chosen, so no call may be made."""


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Endpoint:
    """Where the calls go. `api_key_env` names a variable; the key is never held here."""

    backend: str
    base_url: str
    model: str
    api_key_env: str | None


@dataclass(frozen=True, slots=True)
class RequestSettings:
    """How one call is made. Not part of the plan: none of it changes a rating."""

    timeout_s: float
    max_output_tokens: int
    transient_retries: int


@dataclass(frozen=True, slots=True)
class Plan:
    """The config, checked."""

    prompt_path: Path
    prompt_sha256: str
    backend: str | None
    backends: Mapping[str, Mapping[str, Any]]
    request: RequestSettings

    def endpoint(self) -> Endpoint:
        """The chosen backend's endpoint.

        Raises:
            BackendNotSetError: if no backend is chosen.
            PlanError: if the chosen one is not fully configured.
        """
        if self.backend is None:
            msg = (
                "no backend is set in config/exploratory/llm_ratings.yaml, so no call is made. "
                "Set `backend` to openai or local in a commit of its own, with a dated line in "
                "docs/analysis-log.md"
            )
            raise BackendNotSetError(msg)
        entry = self.backends.get(self.backend) or {}
        base_url, model = entry.get("base_url"), entry.get("model")
        if not base_url or not model:
            msg = f"backends.{self.backend} needs both base_url and model"
            raise PlanError(msg)
        key_env = entry.get("api_key_env")
        return Endpoint(
            backend=self.backend,
            base_url=str(base_url).rstrip("/"),
            model=str(model),
            api_key_env=None if key_env is None else str(key_env),
        )


def load_plan(path: Path = PLAN_PATH, *, repo_root: Path = REPO_ROOT) -> Plan:
    """The config at `path`, refused unless it has every planned part.

    Raises:
        PlanError: if a part is missing or the backend is not one we have.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{path.name} is not a mapping"
        raise PlanError(msg)
    prompt = raw.get("prompt") or {}
    if not prompt.get("file") or not prompt.get("sha256"):
        msg = f"{path.name} must name the prompt file and its sha256"
        raise PlanError(msg)
    backend = raw.get("backend")
    if backend is not None and backend not in BACKENDS:
        msg = f"backend must be one of {list(BACKENDS)} or null, not {backend!r}"
        raise PlanError(msg)
    backends = raw.get("backends") or {}
    if set(backends) != set(BACKENDS):
        msg = f"{path.name} must configure the backends {list(BACKENDS)}"
        raise PlanError(msg)
    request = raw.get("request") or {}
    try:
        settings = RequestSettings(
            timeout_s=float(request["timeout_s"]),
            max_output_tokens=int(request["max_output_tokens"]),
            transient_retries=int(request["transient_retries"]),
        )
    except (KeyError, TypeError, ValueError):
        msg = f"{path.name}: request needs timeout_s, max_output_tokens and transient_retries"
        raise PlanError(msg) from None
    return Plan(
        prompt_path=repo_root / str(prompt["file"]),
        prompt_sha256=str(prompt["sha256"]),
        backend=backend,
        backends=backends,
        request=settings,
    )


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Prompt:
    """The pre-registered prompt: a system message and a user template."""

    system: str
    user_template: str
    sha256: str

    def render(self, dialogue: str) -> tuple[str, str]:
        """The system and user messages for one session's dialogue."""
        return self.system, self.user_template.replace(DIALOGUE_PLACEHOLDER, dialogue)


def load_prompt(path: Path, expected_sha256: str) -> Prompt:
    """The prompt at `path`, refused unless it is the pre-registered one.

    Raises:
        PlanError: if the file is missing, its SHA-256 differs, or it is not a
            system message and a user template with one dialogue placeholder.
    """
    if not path.is_file():
        msg = f"no prompt file at {path}"
        raise PlanError(msg)
    digest = file_sha256(path)
    if digest != expected_sha256:
        msg = (
            f"{path.name} has SHA-256 {digest[:12]}..., not the pre-registered "
            f"{expected_sha256[:12]}...; the prompt may not change after the pre-registration"
        )
        raise PlanError(msg)
    text = path.read_text(encoding="utf-8")
    if not text.startswith(SYSTEM_MARKER):
        msg = f"{path.name} must start with {SYSTEM_MARKER.strip()!r}"
        raise PlanError(msg)
    parts = text[len(SYSTEM_MARKER) :].split(USER_MARKER)
    if len(parts) != 2:  # noqa: PLR2004 - a system part and a user part
        msg = f"{path.name} must hold one {USER_MARKER.strip()!r} section"
        raise PlanError(msg)
    system, user = parts[0].strip("\n"), parts[1].strip("\n")
    if user.count(DIALOGUE_PLACEHOLDER) != 1:
        msg = f"{path.name}: the user message must hold {DIALOGUE_PLACEHOLDER} exactly once"
        raise PlanError(msg)
    return Prompt(system=system, user_template=user, sha256=digest)


# ---------------------------------------------------------------------------
# A response
# ---------------------------------------------------------------------------
#: Why a response was malformed. Counted, never shown with the response.
NOT_JSON: Final = "not JSON"
NOT_OBJECT: Final = "not a JSON object"
WRONG_KEYS: Final = "not exactly the 11 keys"
NOT_INTEGER: Final = "a value that is not an integer"
OUT_OF_RANGE: Final = "a value out of range"
EMPTY: Final = "empty"


class _DuplicateKeyError(ValueError):
    """A JSON object names a key twice, which `json.loads` would hide."""


def _refuse_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKeyError
    return dict(pairs)


@dataclass(frozen=True, slots=True)
class Validation:
    """A response checked: its ratings if valid, else why not."""

    ratings: dict[str, int] | None
    problem: str | None

    @property
    def valid(self) -> bool:
        """Whether the response holds a rating for every item."""
        return self.ratings is not None


def validate_response(text: str | None) -> Validation:  # noqa: PLR0911 - one return per way to fail
    """Check one response against the plan's definition of a valid one.

    Valid is a JSON object with exactly the 11 keys, each value an integer in
    its item's range. `true`, `false` and `2.0` are not integers. Surrounding
    whitespace is allowed; anything else around the object is not.
    """
    if text is None or not text.strip():
        return Validation(None, EMPTY)
    try:
        parsed = json.loads(text, object_pairs_hook=_refuse_duplicates)
    except _DuplicateKeyError:
        return Validation(None, WRONG_KEYS)
    except json.JSONDecodeError:
        return Validation(None, NOT_JSON)
    if not isinstance(parsed, dict):
        return Validation(None, NOT_OBJECT)
    if set(parsed) != {item.key for item in ITEMS}:
        return Validation(None, WRONG_KEYS)
    ratings: dict[str, int] = {}
    for item in ITEMS:
        value = parsed[item.key]
        if isinstance(value, bool) or not isinstance(value, int):
            return Validation(None, NOT_INTEGER)
        if not item.low <= value <= item.high:
            return Validation(None, OUT_OF_RANGE)
        ratings[item.key] = value
    return Validation(ratings, None)
