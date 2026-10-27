"""Rate every session three times, cache every response, and summarise.

As docs/prereg/llm-ratings.md fixes it:

* Each session's dialogue goes into the pre-registered prompt, and is sent in
  `CALLS_PER_SESSION` separate calls.
* A malformed response is retried at most `MAX_RETRIES` times. A call still
  malformed after that, or a request the backend rejects, is failed, and a
  session with a failed call is failed: all of its features are missing. No
  value is filled in, imputed or taken from fewer than every call.
* Each call's outcome is cached, keyed by session, call index, prompt SHA-256
  and model name, so a rerun makes no call already made. A malformed call is
  cached as malformed, so a rerun cannot draw again for it. A rejected request
  is not cached: nothing came back.
* The features are the mean of the calls, per item.

What reaches the terminal and the summary file is aggregate only: counts,
parse success, retries, token usage, time and, for the full run, the ICC of
each item across the calls. No per-session rating is shown. The pilot shows no
rating at all, not even its ICC.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from vc_multimodal.exploratory.llm_ratings.backend import Backend, RequestRejectedError
from vc_multimodal.exploratory.llm_ratings.dialogue import session_dialogue
from vc_multimodal.exploratory.llm_ratings.plan import (
    CALLS_PER_SESSION,
    EXPECTED_SESSIONS,
    FEATURES,
    ITEMS,
    MAX_RETRIES,
    PILOT_SESSIONS,
    Prompt,
    validate_response,
)
from vc_multimodal.io_utils import write_csv, write_json
from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

STAGE: Final = "llm_ratings"
OUT_DIRNAME: Final = ("exploratory", "llm_ratings")
CACHE_DIRNAME: Final = ("exploratory", "llm_ratings", "cache")
FEATURES_FILE: Final = "llm_ratings_features.csv"
SUMMARY_FILE: Final = "llm_ratings_summary.json"
PILOT_FILE: Final = "llm_ratings_pilot.json"

VALID: Final = "valid"
MALFORMED: Final = "malformed"
REJECTED: Final = "rejected"


class RaterError(RuntimeError):
    """Raised when the rater cannot run as pre-registered."""


# ---------------------------------------------------------------------------
# One call
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CallRecord:
    """One call's outcome, over all of its attempts."""

    session_id: int
    call: int
    status: str
    ratings: dict[str, int] | None
    #: Responses received, the valid one included.
    attempts: int
    #: Why each malformed response was malformed.
    problems: tuple[str, ...] = ()
    #: Per response; None where the backend did not say.
    prompt_tokens: tuple[int | None, ...] = ()
    completion_tokens: tuple[int | None, ...] = ()
    from_cache: bool = False


def rate_call(
    backend: Backend, system: str, user: str, session_id: int, call: int
) -> tuple[CallRecord, list[str | None]]:
    """One call: the first valid response of at most 1 + MAX_RETRIES.

    Returns:
        The outcome, and every response text received, for the cache.
    """
    responses: list[str | None] = []
    problems: list[str] = []
    prompt_tokens: list[int | None] = []
    completion_tokens: list[int | None] = []

    def record(status: str, ratings: dict[str, int] | None) -> CallRecord:
        return CallRecord(
            session_id=session_id,
            call=call,
            status=status,
            ratings=ratings,
            attempts=len(responses),
            problems=tuple(problems),
            prompt_tokens=tuple(prompt_tokens),
            completion_tokens=tuple(completion_tokens),
        )

    for _ in range(1 + MAX_RETRIES):
        try:
            completion = backend.complete(system, user)
        except RequestRejectedError:
            return record(REJECTED, None), responses
        responses.append(completion.text)
        prompt_tokens.append(completion.prompt_tokens)
        completion_tokens.append(completion.completion_tokens)
        checked = validate_response(completion.text)
        if checked.valid:
            return record(VALID, checked.ratings), responses
        problems.append(str(checked.problem))
    return record(MALFORMED, None), responses


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------
class Cache:
    """Every call's outcome, one JSON file per session and call.

    The folder is named by the prompt's SHA-256 and the model, and each file by
    the session and call, and every file repeats all four, so an entry can never
    be read under another key.
    """

    def __init__(self, root: Path, prompt_sha256: str, model: str) -> None:
        """The cache for one prompt and model, under `root`."""
        self.prompt_sha256 = prompt_sha256
        self.model = model
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)[:60]
        model_digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
        self.folder = root / prompt_sha256[:16] / f"{slug}-{model_digest}"

    def path(self, session_id: int, call: int) -> Path:
        """Where one call's outcome is kept."""
        return self.folder / f"{session_id}_call{call}.json"

    def get(self, session_id: int, call: int) -> CallRecord | None:
        """A cached outcome, or None if this call was never completed.

        Raises:
            RaterError: if the file is not the entry its name says it is.
        """
        path = self.path(session_id, call)
        if not path.is_file():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        key = (entry.get("session_id"), entry.get("call"), entry.get("prompt_sha256"))
        if key != (session_id, call, self.prompt_sha256) or entry.get("model") != self.model:
            msg = f"cache entry {path.name} does not match its key; move {self.folder} aside"
            raise RaterError(msg)
        if entry.get("status") not in (VALID, MALFORMED):
            msg = f"cache entry {path.name} has no valid status; move {self.folder} aside"
            raise RaterError(msg)
        return CallRecord(
            session_id=session_id,
            call=call,
            status=entry["status"],
            ratings=entry.get("ratings"),
            attempts=int(entry["attempts"]),
            problems=tuple(entry.get("problems", ())),
            prompt_tokens=tuple(entry.get("prompt_tokens", ())),
            completion_tokens=tuple(entry.get("completion_tokens", ())),
            from_cache=True,
        )

    def put(self, record: CallRecord, responses: Sequence[str | None], backend: str) -> None:
        """Keep a valid or malformed outcome. A rejected request is never kept."""
        if record.status == REJECTED:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        write_json(
            self.path(record.session_id, record.call),
            {
                "session_id": record.session_id,
                "call": record.call,
                "model": self.model,
                "backend": backend,
                "prompt_sha256": self.prompt_sha256,
                "status": record.status,
                "ratings": record.ratings,
                "attempts": record.attempts,
                "problems": list(record.problems),
                "prompt_tokens": list(record.prompt_tokens),
                "completion_tokens": list(record.completion_tokens),
                "responses": list(responses),
                "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SessionResult:
    """A session's calls, in order. It stops at the first failed call."""

    session_id: int
    calls: tuple[CallRecord, ...]

    @property
    def rated(self) -> bool:
        """Whether every planned call gave a valid rating."""
        return len(self.calls) == CALLS_PER_SESSION and all(
            record.status == VALID for record in self.calls
        )

    def ratings(self, key: str) -> list[int]:
        """One item's rating from each call. Only for a rated session."""
        return [record.ratings[key] for record in self.calls if record.ratings is not None]

    def features(self) -> dict[str, float]:
        """Each item's mean over the calls; every item missing if the session failed."""
        if not self.rated:
            return dict.fromkeys(FEATURES, float("nan"))
        return {item.feature: float(np.mean(self.ratings(item.key))) for item in ITEMS}


def rate_session(
    path: Path, prompt: Prompt, backend: Backend, cache: Cache, backend_name: str
) -> SessionResult:
    """Rate one transcript, from the cache where it can.

    A failed call ends the session: it is failed whatever the other calls give,
    so they are not made.
    """
    session_id = int(path.stem)
    system, user = prompt.render(session_dialogue(path))
    calls: list[CallRecord] = []
    for call in range(1, CALLS_PER_SESSION + 1):
        record = cache.get(session_id, call)
        if record is None:
            record, responses = rate_call(backend, system, user, session_id, call)
            cache.put(record, responses, backend_name)
        calls.append(record)
        if record.status != VALID:
            break
    return SessionResult(session_id, tuple(calls))


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------
def icc2(ratings: np.ndarray) -> tuple[float, float]:
    """ICC(2,1) and ICC(2,k) of an n sessions x k calls matrix (Shrout & Fleiss).

    Two-way random effects, absolute agreement, for a single call and for the
    mean of k calls. NaN where it is undefined: fewer than two sessions or
    calls, or no variance at all.
    """
    n, k = ratings.shape
    if n < 2 or k < 2:  # noqa: PLR2004 - a variance needs two
        return float("nan"), float("nan")
    grand = ratings.mean()
    ss_rows = k * float(np.sum((ratings.mean(axis=1) - grand) ** 2))
    ss_cols = n * float(np.sum((ratings.mean(axis=0) - grand) ** 2))
    ss_error = float(np.sum((ratings - grand) ** 2)) - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    single = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    average = ms_rows + (ms_cols - ms_error) / n
    return (
        (ms_rows - ms_error) / single if single > 0 else float("nan"),
        (ms_rows - ms_error) / average if average > 0 else float("nan"),
    )


def item_reliability(sessions: Sequence[SessionResult]) -> list[dict[str, Any]]:
    """Each item's ICCs across the calls, over the rated sessions."""
    rated = [session for session in sessions if session.rated]
    rows: list[dict[str, Any]] = []
    for item in ITEMS:
        matrix = np.array([s.ratings(item.key) for s in rated], dtype=float)
        single, average = icc2(matrix.reshape(len(rated), CALLS_PER_SESSION))
        rows.append(
            {"item": item.key, "n_sessions": len(rated), "icc_2_1": single, "icc_2_k": average}
        )
    return rows


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RaterResult:
    """A pilot or full run, and what it wrote."""

    pilot: bool
    sessions: tuple[SessionResult, ...]
    n_available: int
    elapsed_s: float
    run_info: dict[str, Any]
    written: tuple[Path, ...] = field(default=())

    @property
    def records(self) -> list[CallRecord]:
        """Every call, across sessions."""
        return [record for session in self.sessions for record in session.calls]

    def counts(self) -> dict[str, Any]:
        """Aggregate counts: calls, attempts, parse success, failures, tokens."""
        records = self.records
        responses = sum(record.attempts for record in records)
        valid = sum(record.status == VALID for record in records)
        with_response = [record for record in records if record.attempts > 0]
        first_try = sum(record.status == VALID and record.attempts == 1 for record in records)
        problems = Counter(problem for record in records for problem in record.problems)
        prompt_tokens = [t for record in records for t in record.prompt_tokens]
        known_prompt = [t for t in prompt_tokens if t is not None]
        known_completion = [
            t for record in records for t in record.completion_tokens if t is not None
        ]
        made = [record for record in records if not record.from_cache]
        failed = [session.session_id for session in self.sessions if not session.rated]
        return {
            "sessions_available": self.n_available,
            "sessions_attempted": len(self.sessions),
            "sessions_rated": len(self.sessions) - len(failed),
            "sessions_failed": failed,
            "calls": len(records),
            "calls_made_now": len(made),
            "calls_from_cache": len(records) - len(made),
            "calls_valid": valid,
            "calls_malformed": sum(record.status == MALFORMED for record in records),
            "calls_rejected": sum(record.status == REJECTED for record in records),
            "responses": responses,
            "responses_malformed": responses - valid,
            "malformed_by_reason": dict(problems),
            "parse_success_rate": valid / responses if responses else float("nan"),
            "first_attempt_success_rate": (
                first_try / len(with_response) if with_response else float("nan")
            ),
            "retries": responses - len(with_response),
            "prompt_tokens_total": sum(known_prompt),
            "prompt_tokens_mean_per_response": (
                float(np.mean(known_prompt)) if known_prompt else float("nan")
            ),
            "prompt_tokens_max_per_response": max(known_prompt, default=0),
            "completion_tokens_total": sum(known_completion),
            "responses_without_token_counts": len(prompt_tokens) - len(known_prompt),
            "elapsed_s": round(self.elapsed_s, 1),
            "seconds_per_call_made_now": (
                round(self.elapsed_s / len(made), 2) if made else float("nan")
            ),
        }

    def report_lines(self) -> list[str]:
        """What the terminal shows: aggregates only, never a session's rating."""
        info, counts = self.run_info, self.counts()
        mode = "pilot" if self.pilot else "full run"
        lines = [
            f"llm ratings, {mode}: backend {info['backend']}, model {info['model']}, "
            f"prompt {info['prompt_sha256'][:12]}, commit {info['commit'][:12]}",
            f"sessions: {counts['sessions_attempted']} of {counts['sessions_available']}"
            + (" (the three lowest session numbers)" if self.pilot else ""),
            f"calls: {counts['calls']} ({counts['calls_made_now']} made now, "
            f"{counts['calls_from_cache']} from the cache); valid {counts['calls_valid']}, "
            f"malformed after retries {counts['calls_malformed']}, "
            f"rejected {counts['calls_rejected']}",
            f"responses: {counts['responses']}, parse success "
            f"{_percent(counts['parse_success_rate'])}, valid at the first attempt "
            f"{_percent(counts['first_attempt_success_rate'])}, retries {counts['retries']}",
        ]
        if counts["malformed_by_reason"]:
            reasons = ", ".join(
                f"{reason}: {n}" for reason, n in sorted(counts["malformed_by_reason"].items())
            )
            lines.append(f"malformed responses by reason: {reasons}")
        failed = counts["sessions_failed"]
        lines.append(
            f"sessions rated: {counts['sessions_rated']}; failed: {len(failed)}"
            + (f" {failed}" if failed else "")
        )
        no_counts = counts["responses_without_token_counts"]
        lines.append(
            f"tokens: prompt {counts['prompt_tokens_total']} (mean "
            f"{counts['prompt_tokens_mean_per_response']:.0f}, max "
            f"{counts['prompt_tokens_max_per_response']} per response), completion "
            f"{counts['completion_tokens_total']}"
            + (f"; {no_counts} response(s) without counts" if no_counts else "")
        )
        lines.append(
            f"time: {counts['elapsed_s']:.0f} s, {counts['seconds_per_call_made_now']} s per call "
            "made now"
        )
        if not self.pilot:
            lines.append(
                f"ICC across the {CALLS_PER_SESSION} calls, per item: ICC(2,1), ICC(2,k), n"
            )
            lines.extend(
                f"  {row['item']:<16} {_icc(row['icc_2_1'])}  {_icc(row['icc_2_k'])}  "
                f"{row['n_sessions']}"
                for row in item_reliability(self.sessions)
            )
        lines.extend(f"wrote {path}" for path in self.written)
        return lines


def _percent(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{100 * value:.1f}%"


def _icc(value: float) -> str:
    return "    n/a" if np.isnan(value) else f"{value:7.3f}"


def feature_frame(sessions: Sequence[SessionResult]) -> pd.DataFrame:
    """`session_id` and the 11 features, in session order; a failed session is blank."""
    rows = [{"session_id": s.session_id, **s.features()} for s in sessions]
    return pd.DataFrame(rows, columns=["session_id", *FEATURES])


def run(
    files: Sequence[Path],
    prompt: Prompt,
    backend: Backend,
    cache: Cache,
    out_dir: Path,
    *,
    pilot: bool,
    run_info: dict[str, Any],
    jobs: int = 1,
    expected_sessions: int = EXPECTED_SESSIONS,
) -> RaterResult:
    """The pilot (the three lowest session numbers) or the full run.

    `files` must be sorted by session number, as `transcript_files` returns
    them. The pilot writes only its counts; the full run writes the feature
    table and the summary.

    Raises:
        RaterError: if the transcripts are not the planned sessions.
        BackendUnavailableError: if the backend stops answering. What was
            cached before it stays cached, so a rerun continues.
    """
    if len(files) != expected_sessions:
        msg = f"{len(files)} transcripts found; the plan rates the {expected_sessions} sessions"
        raise RaterError(msg)
    chosen = list(files[:PILOT_SESSIONS] if pilot else files)
    logger.info("%s: rating %d session(s), %d worker(s)", STAGE, len(chosen), jobs)
    started = time.perf_counter()
    results: list[SessionResult] = []
    backend_name = str(run_info["backend"])
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [
            pool.submit(rate_session, path, prompt, backend, cache, backend_name) for path in chosen
        ]
        try:
            for done, future in enumerate(futures, start=1):
                results.append(future.result())
                logger.info("%s: %d of %d session(s) done", STAGE, done, len(chosen))
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    elapsed = time.perf_counter() - started

    result = RaterResult(pilot, tuple(results), len(files), elapsed, run_info)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {**run_info, "pilot": pilot, "counts": result.counts()}
    written: tuple[Path, ...]
    if pilot:
        written = (write_json(out_dir / PILOT_FILE, payload),)
    else:
        payload["reliability"] = item_reliability(results)
        written = (
            write_csv(out_dir / FEATURES_FILE, feature_frame(results)),
            write_json(out_dir / SUMMARY_FILE, payload),
        )
    return RaterResult(pilot, tuple(results), len(files), elapsed, run_info, written)
