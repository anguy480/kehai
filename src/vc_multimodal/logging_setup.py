"""Logging configuration.

One format everywhere, two destinations: a stderr stream for the operator and a
per-run file under `$VC_OUT_ROOT/logs`. Stages log; only the CLI writes to
stdout, so stage output can be piped or redirected without mixing the two.

Log records describe *what happened to a session*, never what was said in it:
session IDs, counts, durations and error types only.
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
ROOT_LOGGER_NAME = "vc_multimodal"

_OUR_HANDLER_FLAG = "_vc_multimodal_handler"


def run_stamp(now: datetime | None = None) -> str:
    """Return a UTC timestamp suitable for a filename, e.g. `20260923T142501Z`."""
    moment = now or datetime.now(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def log_file_path(out_root: Path, stage: str, *, now: datetime | None = None) -> Path:
    """Path for this run's log file under `$VC_OUT_ROOT/logs`."""
    return out_root / "logs" / f"{run_stamp(now)}_{stage}.log"


def configure_logging(
    level: str = "INFO",
    *,
    log_file: Path | None = None,
) -> Path | None:
    """Attach our stream and file handlers to the package logger.

    Safe to call more than once: handlers added by previous calls are removed
    first, so repeated invocations (in tests, or `run-all`) do not duplicate
    output. Third-party handlers are left alone.

    Args:
        level: Minimum level to emit.
        log_file: File to also write to. Its parent is created.

    Returns:
        The log file path, or None if no file handler was requested.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in [h for h in logger.handlers if getattr(h, _OUR_HANDLER_FLAG, False)]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(formatter)
    setattr(stream, _OUR_HANDLER_FLAG, True)
    logger.addHandler(stream)

    if log_file is None:
        return None

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    setattr(file_handler, _OUR_HANDLER_FLAG, True)
    logger.addHandler(file_handler)
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the package namespace.

    Args:
        name: Usually `__name__`. A bare module name is prefixed automatically.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
