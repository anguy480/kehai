"""Stage execution: idempotency, failure isolation and parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.conftest import PackageLogCapture
from vc_multimodal.runner import (
    SessionOutcome,
    StageReport,
    resolve_workers,
    run_sessions,
)


@dataclass(frozen=True)
class Item:
    """Minimal stand-in for a session."""

    session_id: int


def items(*ids: int) -> list[Item]:
    return [Item(i) for i in ids]


# ---------------------------------------------------------------------------
# failure isolation
# ---------------------------------------------------------------------------
def test_one_failing_session_does_not_stop_the_others():
    def task(item: Item) -> str:
        if item.session_id == 2:
            msg = "ffprobe exited 1"
            raise RuntimeError(msg)
        return "done"

    report = run_sessions("demo", items(1, 2, 3), task, workers=1)

    assert [o.session_id for o in report.succeeded] == [1, 3]
    assert [o.session_id for o in report.failed] == [2]
    assert not report.ok


def test_the_failure_message_names_the_exception_type():
    def task(item: Item) -> str:
        msg = "no such file"
        raise FileNotFoundError(msg)

    report = run_sessions("demo", items(7), task, workers=1)
    assert report.failed[0].message == "FileNotFoundError: no such file"


def test_every_session_can_fail_without_raising():
    def task(item: Item) -> str:
        msg = "nope"
        raise ValueError(msg)

    report = run_sessions("demo", items(1, 2), task, workers=1)
    assert len(report.failed) == 2
    assert not report.ok


def test_a_keyboard_interrupt_is_not_swallowed():
    """Ctrl-C must stop the run, not be recorded as a failed session."""

    def task(item: Item) -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_sessions("demo", items(1), task, workers=1)


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------
def test_completed_sessions_are_skipped():
    ran: list[int] = []

    def task(item: Item) -> None:
        ran.append(item.session_id)

    report = run_sessions(
        "demo",
        items(1, 2, 3),
        task,
        workers=1,
        is_done=lambda item: item.session_id in {1, 3},
    )

    assert ran == [2]
    assert [o.session_id for o in report.skipped] == [1, 3]
    assert report.skipped[0].message == "output exists"
    assert report.ok


def test_force_reruns_completed_sessions():
    ran: list[int] = []

    def task(item: Item) -> None:
        ran.append(item.session_id)

    report = run_sessions(
        "demo", items(1, 2), task, workers=1, is_done=lambda item: True, force=True
    )
    assert ran == [1, 2]
    assert not report.skipped


def test_without_an_idempotency_check_everything_runs():
    ran: list[int] = []
    run_sessions("demo", items(1, 2), lambda item: ran.append(item.session_id), workers=1)
    assert ran == [1, 2]


# ---------------------------------------------------------------------------
# selection and reporting
# ---------------------------------------------------------------------------
def test_an_empty_session_list_is_not_an_error():
    report = run_sessions("demo", [], lambda item: None, workers=1)
    assert report.outcomes == ()
    assert report.ok


def test_outcomes_are_sorted_by_session_id():
    report = run_sessions("demo", items(28, 3, 17), lambda item: None, workers=1)
    assert [o.session_id for o in report.outcomes] == [3, 17, 28]


def test_report_frame_has_one_row_per_session():
    report = run_sessions("demo", items(1, 2), lambda item: "note", workers=1)
    frame = report.to_frame()
    assert list(frame.columns) == ["session_id", "stage", "status", "seconds", "message"]
    assert len(frame) == 2
    assert set(frame["stage"]) == {"demo"}


def test_summary_lines_report_counts_and_name_failures():
    def task(item: Item) -> None:
        if item.session_id == 9:
            msg = "bad stream"
            raise RuntimeError(msg)

    report = run_sessions(
        "diarize", items(1, 9), task, workers=1, is_done=lambda i: i.session_id == 1
    )
    lines = report.summary_lines()
    assert "1 skipped" in lines[0]
    assert "1 failed" in lines[0]
    assert any("FAILED session 9" in line and "bad stream" in line for line in lines)


def test_stage_notes_are_carried_into_the_summary():
    report = run_sessions(
        "demo", items(1), lambda item: None, workers=1, notes=["session 99 not found"]
    )
    assert any("session 99 not found" in line for line in report.summary_lines())


def test_a_returned_message_is_recorded():
    report = run_sessions("demo", items(1), lambda item: "2 audio streams", workers=1)
    assert report.succeeded[0].message == "2 audio streams"


def test_an_empty_report_is_ok_and_prints_zeroes():
    report = StageReport(stage="demo")
    assert report.ok
    assert "0 ok, 0 skipped, 0 failed" in report.summary_lines()[0]


def test_outcomes_record_elapsed_time():
    report = run_sessions("demo", items(1), lambda item: None, workers=1)
    assert report.succeeded[0].seconds >= 0.0
    assert isinstance(report.succeeded[0], SessionOutcome)


# ---------------------------------------------------------------------------
# parallelism
# ---------------------------------------------------------------------------
def test_parallel_execution_processes_every_session():
    report = run_sessions("demo", items(*range(1, 9)), lambda item: None, workers=4)
    assert len(report.succeeded) == 8


def test_parallel_execution_still_isolates_failures():
    def task(item: Item) -> None:
        if item.session_id % 2 == 0:
            msg = "even sessions fail"
            raise RuntimeError(msg)

    report = run_sessions("demo", items(*range(1, 9)), task, workers=4)
    assert len(report.failed) == 4
    assert len(report.succeeded) == 4


@pytest.mark.parametrize(
    ("configured", "n_items", "expected"),
    [
        (1, 10, 1),
        (4, 10, 4),
        (100, 3, 3),  # never more workers than sessions
        (4, 0, 1),
        (None, 0, 1),
    ],
)
def test_worker_count_resolution(configured: int | None, n_items: int, expected: int):
    assert resolve_workers(configured, n_items) == expected


def test_automatic_worker_count_is_at_least_one_and_bounded_by_the_work():
    assert resolve_workers(None, 1) == 1
    assert 1 <= resolve_workers(None, 1000) <= 1000


def test_a_failure_logs_one_line_and_keeps_the_traceback_for_debug(
    package_logs: PackageLogCapture,
):
    """62 tracebacks would bury the summary that actually matters."""

    def task(item: Item) -> None:
        msg = "run `vc diarize` first"
        raise FileNotFoundError(msg)

    run_sessions("demo", items(1), task, workers=1)

    errors = package_logs.messages_at("ERROR")
    assert any("vc diarize" in message for message in errors)
    assert all("Traceback" not in message for message in errors)
    # The traceback is kept, at DEBUG, for when it is actually wanted.
    traceback_records = [record for record in package_logs.records if record.exc_info is not None]
    assert traceback_records
    assert all(record.levelname == "DEBUG" for record in traceback_records)
