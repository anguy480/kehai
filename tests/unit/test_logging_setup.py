"""Logging configuration."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from vc_multimodal.logging_setup import (
    ROOT_LOGGER_NAME,
    configure_logging,
    get_logger,
    log_file_path,
    run_stamp,
)


def _package_logger() -> logging.Logger:
    return logging.getLogger(ROOT_LOGGER_NAME)


def test_configuring_twice_does_not_duplicate_handlers(tmp_path: Path):
    """`run-all` configures logging repeatedly; output must not double up."""
    configure_logging("INFO", log_file=tmp_path / "a.log")
    first = len(_package_logger().handlers)
    configure_logging("INFO", log_file=tmp_path / "b.log")
    assert len(_package_logger().handlers) == first


def test_third_party_handlers_are_left_alone(tmp_path: Path):
    logger = _package_logger()
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    try:
        configure_logging("INFO", log_file=tmp_path / "a.log")
        assert foreign in logger.handlers
    finally:
        logger.removeHandler(foreign)


def test_messages_reach_the_log_file(tmp_path: Path):
    target = tmp_path / "run.log"
    assert configure_logging("INFO", log_file=target) == target
    get_logger("test_module").info("session %s ok", 28)
    logging.shutdown()
    contents = target.read_text(encoding="utf-8")
    assert "session 28 ok" in contents
    assert "vc_multimodal.test_module" in contents


def test_the_level_is_respected(tmp_path: Path):
    target = tmp_path / "run.log"
    configure_logging("WARNING", log_file=target)
    logger = get_logger("test_module")
    logger.debug("invisible")
    logger.warning("visible")
    logging.shutdown()
    contents = target.read_text(encoding="utf-8")
    assert "invisible" not in contents
    assert "visible" in contents


def test_no_log_file_is_requested_returns_none():
    assert configure_logging("INFO") is None


def test_logger_names_are_namespaced():
    assert get_logger("stages.inventory").name == "vc_multimodal.stages.inventory"
    assert get_logger("vc_multimodal.already").name == "vc_multimodal.already"
    assert get_logger(ROOT_LOGGER_NAME).name == ROOT_LOGGER_NAME


def test_package_logs_do_not_propagate_to_the_root_logger(tmp_path: Path):
    """Otherwise a library's root handler could print transcript-bearing logs."""
    configure_logging("INFO", log_file=tmp_path / "a.log")
    assert _package_logger().propagate is False


def test_run_stamp_is_a_sortable_utc_timestamp():
    stamp = run_stamp(datetime(2026, 9, 23, 14, 25, 1, tzinfo=UTC))
    assert stamp == "20260923T142501Z"


def test_log_file_path_is_under_the_logs_directory(tmp_path: Path):
    path = log_file_path(tmp_path, "inventory", now=datetime(2026, 9, 23, 14, 25, 1, tzinfo=UTC))
    assert path == tmp_path / "logs" / "20260923T142501Z_inventory.log"


def test_the_log_directory_is_created(tmp_path: Path):
    target = log_file_path(tmp_path, "inventory")
    configure_logging("INFO", log_file=target)
    assert target.parent.is_dir()
