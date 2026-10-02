"""Rate each description on five 1-7 scales with a pinned, local model.

The descriptions are derived from clinical recordings, so they go to one place
only: a model served on this machine. `assert_local` refuses any other host,
and the model is checked against a pinned digest before the first request, so a
re-pulled or substituted model cannot change the ratings silently.

Each description is rated `runs` times with identical requests at temperature 0
and a fixed seed. Agreement between the runs measures how deterministic the
setup is, not how reliable a human rater would be.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

import pandas as pd

from vc_multimodal.exploratory.llm_face.describe import (
    DESCRIPTIONS_FILE,
    file_sha256,
    output_dir,
    text_sha256,
)
from vc_multimodal.io_utils import read_csv, write_csv, write_json
from vc_multimodal.logging_setup import get_logger
from vc_multimodal.paths import DataRoots

logger = get_logger(__name__)

STAGE: Final = "llm-face-rate"

#: The model, by tag and by the digest of its weights as Ollama reports it. The
#: digest is a public checksum of open weights, not a credential.
MODEL: Final = "qwen2.5:3b-instruct-q4_K_M"
MODEL_DIGEST: Final = (
    "357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b"  # pragma: allowlist secret
)

#: Where the model is served. Loopback only; see `assert_local`.
HOST: Final = "http://127.0.0.1:11434"
LOCAL_HOSTS: Final = frozenset({"127.0.0.1", "localhost", "::1"})

SCALES: Final = (
    "positive_affect",
    "expressivity",
    "flat_affect",
    "tension_negative_affect",
    "listening_engagement",
)
SCALE_MIN: Final = 1
SCALE_MAX: Final = 7

#: Decoding is greedy and seeded; the context and output lengths are fixed.
OPTIONS: Final[Mapping[str, int | float]] = {
    "temperature": 0,
    "seed": 0,
    "num_ctx": 4096,
    "num_predict": 200,
}
RUNS: Final = 3
TIMEOUT_S: Final = 600

PROMPT_FILE: Final = Path(__file__).with_name("rating_prompt.txt")
SCORES_FILE: Final = "scores_runs.csv"
META_FILE: Final = "scores_meta.json"
SCORE_COLUMNS: Final = ("session_id", "run", *SCALES, "response_sha256")
_SYSTEM_MARK: Final = "### system"
_USER_MARK: Final = "### user"


class RatingError(RuntimeError):
    """Raised when a rating cannot be obtained safely or parsed."""


def assert_local(url: str) -> None:
    """Refuse any host that is not this machine."""
    host = urlparse(url).hostname
    if host not in LOCAL_HOSTS:
        msg = (
            f"refusing to send descriptions to {host!r}: they are derived from "
            f"clinical recordings and go only to a model on this machine"
        )
        raise RatingError(msg)


def prompt_sha256() -> str:
    """Digest of the prompt file, carried by every rating."""
    return file_sha256(PROMPT_FILE)


def build_messages(description: str, prompt: str | None = None) -> list[dict[str, str]]:
    """The chat messages for one description."""
    text = PROMPT_FILE.read_text(encoding="utf-8") if prompt is None else prompt
    if _SYSTEM_MARK not in text or _USER_MARK not in text:
        msg = f"{PROMPT_FILE.name} must have '{_SYSTEM_MARK}' and '{_USER_MARK}' sections"
        raise RatingError(msg)
    system, user = text.split(_USER_MARK, 1)
    system = system.replace(_SYSTEM_MARK, "", 1).strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip().replace("{description}", description)},
    ]


def response_schema() -> dict[str, Any]:
    """JSON schema the model's output is constrained to."""
    return {
        "type": "object",
        "properties": {
            name: {"type": "integer", "minimum": SCALE_MIN, "maximum": SCALE_MAX} for name in SCALES
        },
        "required": list(SCALES),
        "additionalProperties": False,
    }


def parse_scores(content: str) -> dict[str, int]:
    """Validate a model response into one integer per scale."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        msg = f"the model did not return JSON: {exc}"
        raise RatingError(msg) from exc
    if not isinstance(data, dict) or set(data) != set(SCALES):
        msg = f"expected exactly the keys {list(SCALES)}"
        raise RatingError(msg)
    scores: dict[str, int] = {}
    for name in SCALES:
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"{name} is not an integer"
            raise RatingError(msg)
        if not SCALE_MIN <= value <= SCALE_MAX:
            msg = f"{name}={value} is outside {SCALE_MIN}-{SCALE_MAX}"
            raise RatingError(msg)
        scores[name] = value
    return scores


def _request(url: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    assert_local(url)
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return body
    except (urllib.error.URLError, TimeoutError) as exc:
        msg = f"could not reach the local model server at {url}: {exc}"
        raise RatingError(msg) from exc


def check_model(host: str = HOST) -> None:
    """Confirm the pinned model is the one being served."""
    tags = _request(f"{host}/api/tags")
    found = {m.get("name"): m.get("digest") for m in tags.get("models", [])}
    if MODEL not in found:
        msg = f"{MODEL} is not available on the local server; `ollama pull {MODEL}`"
        raise RatingError(msg)
    if found[MODEL] != MODEL_DIGEST:
        msg = f"{MODEL} has digest {found[MODEL]}, not the pinned {MODEL_DIGEST}"
        raise RatingError(msg)


def rate_once(description: str, host: str = HOST) -> tuple[dict[str, int], str]:
    """One rating of one description, and the raw response text."""
    body = _request(
        f"{host}/api/chat",
        {
            "model": MODEL,
            "messages": build_messages(description),
            "stream": False,
            "format": response_schema(),
            "options": dict(OPTIONS),
        },
    )
    content = str(body.get("message", {}).get("content", ""))
    return parse_scores(content), content


@dataclass(frozen=True, slots=True)
class RateResult:
    """What `run` wrote, as counts and digests only."""

    path: Path
    n_sessions: int
    runs: int
    n_failed: int
    prompt_sha256: str
    file_sha256: str

    def report_lines(self) -> list[str]:
        """A summary that contains no description and no score."""
        return [
            f"ratings: {self.n_sessions} session(s) x {self.runs} run(s) -> {self.path}",
            f"  failed ratings: {self.n_failed}",
            f"  model: {MODEL} @ {MODEL_DIGEST}",
            f"  prompt sha256: {self.prompt_sha256}",
            f"  scores sha256: {self.file_sha256}",
        ]


def run(roots: DataRoots, *, runs: int = RUNS, host: str = HOST) -> RateResult:
    """Rate every described session `runs` times."""
    assert_local(host)
    check_model(host)
    folder = output_dir(roots)
    descriptions_path = folder / DESCRIPTIONS_FILE
    descriptions = read_csv(descriptions_path)
    texts = descriptions["description"].fillna("").astype(str)

    rows: list[dict[str, object]] = []
    n_sessions = 0
    for session_id, text in zip(descriptions["session_id"], texts, strict=True):
        if not text:
            continue
        n_sessions += 1
        for repeat in range(1, runs + 1):
            row: dict[str, object] = {"session_id": int(session_id), "run": repeat}
            try:
                scores, content = rate_once(text, host)
                row.update(scores)
                row["response_sha256"] = text_sha256(content)
            except RatingError as exc:
                row.update(dict.fromkeys(SCALES))
                row["response_sha256"] = ""
                logger.warning("%s: session %s run %d failed: %s", STAGE, session_id, repeat, exc)
            rows.append(row)
        logger.info("%s: session %s rated %d time(s)", STAGE, session_id, runs)

    table = pd.DataFrame(rows, columns=list(SCORE_COLUMNS))
    path = folder / SCORES_FILE
    write_csv(path, table)
    digest = file_sha256(path)
    write_json(
        folder / META_FILE,
        {
            "model": MODEL,
            "model_digest": MODEL_DIGEST,
            "options": dict(OPTIONS),
            "runs": runs,
            "prompt_sha256": prompt_sha256(),
            "descriptions_sha256": file_sha256(descriptions_path),
            "scores_sha256": digest,
        },
    )
    return RateResult(
        path=path,
        n_sessions=n_sessions,
        runs=runs,
        n_failed=int(table[list(SCALES)].isna().any(axis=1).sum()),
        prompt_sha256=prompt_sha256(),
        file_sha256=digest,
    )
