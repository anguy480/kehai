"""Per-session stage execution: parallelism, idempotency and failure isolation.

Every stage runs the same way, so the behaviour lives here rather than being
re-implemented eleven times:

* sessions are processed in parallel with a configurable worker count;
* a session whose output already exists is skipped unless `--force`;
* a failing session is logged and recorded, and the run continues;
* the stage ends with a summary of what succeeded, skipped and failed.

Combined with the atomic writes in `io_utils`, this makes an interrupted run
safe to restart: completed sessions are skipped and nothing half-written is
mistaken for a completed output.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, TypeVar, cast

import pandas as pd
from joblib import Parallel, delayed

from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

Status = Literal["ok", "skipped", "failed"]
Backend = Literal["threads", "processes"]

# Leave one core free so an interactive machine stays usable during a long run.
_RESERVED_CORES = 1


class HasSessionId(Protocol):
    """Anything a stage can process, identified by session ID."""

    @property
    def session_id(self) -> int:
        """The numeric session identifier."""


ItemT = TypeVar("ItemT", bound=HasSessionId)


@dataclass(frozen=True, slots=True)
class SessionOutcome:
    """What happened to one session in one stage."""

    session_id: int
    status: Status
    message: str = ""
    seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class StageReport:
    """The outcome of running a stage over a set of sessions."""

    stage: str
    outcomes: tuple[SessionOutcome, ...] = ()
    seconds: float = 0.0
    notes: tuple[str, ...] = field(default=())

    def _with_status(self, status: Status) -> tuple[SessionOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == status)

    @property
    def succeeded(self) -> tuple[SessionOutcome, ...]:
        """Sessions that completed in this run."""
        return self._with_status("ok")

    @property
    def skipped(self) -> tuple[SessionOutcome, ...]:
        """Sessions whose output already existed."""
        return self._with_status("skipped")

    @property
    def failed(self) -> tuple[SessionOutcome, ...]:
        """Sessions that raised."""
        return self._with_status("failed")

    @property
    def ok(self) -> bool:
        """Whether every session either completed or was skipped."""
        return not self.failed

    def with_notes(self, extra: Sequence[str]) -> StageReport:
        """The same report with more notes.

        Stages learn some things only after the per-session work is done - how
        many existing rows were kept, for instance - and those belong in the
        report the user reads rather than in the log alone.
        """
        if not extra:
            return self
        return replace(self, notes=(*self.notes, *extra))

    def to_frame(self) -> pd.DataFrame:
        """The per-session outcomes as a table, for QC reporting."""
        return pd.DataFrame(
            {
                "session_id": [o.session_id for o in self.outcomes],
                "stage": self.stage,
                "status": [o.status for o in self.outcomes],
                "seconds": [round(o.seconds, 3) for o in self.outcomes],
                "message": [o.message for o in self.outcomes],
            }
        )

    def summary_lines(self) -> list[str]:
        """Human-readable summary lines for the CLI to print."""
        lines = [
            f"{self.stage}: {len(self.succeeded)} ok, "
            f"{len(self.skipped)} skipped, {len(self.failed)} failed "
            f"in {self.seconds:.1f}s"
        ]
        lines.extend(f"  note: {note}" for note in self.notes)
        lines.extend(
            f"  FAILED session {o.session_id}: {o.message}"
            for o in sorted(self.failed, key=lambda o: o.session_id)
        )
        return lines


def resolve_workers(configured: int | None, n_items: int) -> int:
    """Choose a worker count.

    Args:
        configured: Explicit worker count, or None to choose from the CPU count.
        n_items: How many sessions will run; never spawn more workers than work.

    Returns:
        At least one worker.
    """
    if n_items <= 0:
        return 1
    if configured is not None:
        return max(1, min(configured, n_items))
    cpus = os.cpu_count() or 1
    return max(1, min(cpus - _RESERVED_CORES, n_items))


def _run_one(
    item: ItemT,
    task: Callable[[ItemT], str | None],
    is_done: Callable[[ItemT], bool] | None,
    force: bool,
) -> SessionOutcome:
    """Run `task` for one session, converting an exception into an outcome."""
    session_id = item.session_id
    if not force and is_done is not None and is_done(item):
        return SessionOutcome(session_id=session_id, status="skipped", message="output exists")

    started = time.perf_counter()
    try:
        message = task(item)
    # A broad catch is the point: one session must never stop the run.
    except Exception as exc:
        elapsed = time.perf_counter() - started
        # A one-line error at normal level, the traceback only at DEBUG. Most
        # per-session failures are expected conditions with a clear message -
        # a prerequisite stage has not run, a file is missing - and dumping a
        # traceback for each of 62 sessions buries the summary that matters.
        logger.error("session %s failed: %s: %s", session_id, type(exc).__name__, exc)
        logger.debug("session %s traceback", session_id, exc_info=True)
        return SessionOutcome(
            session_id=session_id,
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
            seconds=elapsed,
        )
    return SessionOutcome(
        session_id=session_id,
        status="ok",
        message=message or "",
        seconds=time.perf_counter() - started,
    )


def run_sessions(
    stage: str,
    items: Sequence[ItemT],
    task: Callable[[ItemT], str | None],
    *,
    workers: int | None = None,
    force: bool = False,
    is_done: Callable[[ItemT], bool] | None = None,
    backend: Backend = "threads",
    notes: Sequence[str] = (),
) -> StageReport:
    """Run `task` over `items`, one session at a time, in parallel.

    Args:
        stage: Stage name, used in logs and the report.
        items: Sessions to process.
        task: Called with one session; returns an optional note for the report.
            Raising marks that session failed without stopping the run.
        workers: Worker count, or None to choose automatically.
        force: Run sessions even when `is_done` says their output exists.
        is_done: Idempotency check for one session.
        backend: "threads" for stages dominated by subprocesses or I/O;
            "processes" for CPU-bound work. With "processes", `task` and `items`
            must be picklable.
        notes: Stage-level notes to carry into the report, e.g. requested
            session IDs that were not found.

    Returns:
        The stage report. Never raises on a per-session failure.
    """
    n_workers = resolve_workers(workers, len(items))
    logger.info(
        "%s: starting %d session(s) on %d worker(s)%s",
        stage,
        len(items),
        n_workers,
        " [force]" if force else "",
    )

    started = time.perf_counter()
    if not items:
        outcomes: list[SessionOutcome] = []
    elif n_workers == 1:
        # Run inline when there is nothing to gain from a pool: simpler
        # tracebacks, and tests stay deterministic.
        outcomes = [_run_one(item, task, is_done, force) for item in items]
    else:
        raw = Parallel(n_jobs=n_workers, prefer=backend)(
            delayed(_run_one)(item, task, is_done, force) for item in items
        )
        outcomes = cast("list[SessionOutcome]", list(raw))
    elapsed = time.perf_counter() - started

    report = StageReport(
        stage=stage,
        outcomes=tuple(sorted(outcomes, key=lambda o: o.session_id)),
        seconds=elapsed,
        notes=tuple(notes),
    )
    logger.info(
        "%s: %d ok, %d skipped, %d failed in %.1fs",
        stage,
        len(report.succeeded),
        len(report.skipped),
        len(report.failed),
        elapsed,
    )
    return report
