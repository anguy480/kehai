"""Atomic writes and small table helpers.

Every stage output is written through here. The contract is that a reader never
sees a partial file: content goes to a temporary file in the destination
directory and is then renamed, which is atomic within a filesystem. An
interrupted run therefore leaves either the previous output or none at all,
which is what makes `--force`-less reruns safe to skip completed work.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


@contextmanager
def atomic_path(target: Path, *, suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a temporary path that is renamed onto `target` on clean exit.

    The temporary file is created in `target`'s directory so the final rename
    stays on one filesystem. On any exception the temporary file is removed and
    `target` is left untouched.

    Args:
        target: Final destination path.
        suffix: Suffix for the temporary file.

    Yields:
        The path to write to.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=suffix, dir=target.parent)
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    if not tmp_path.exists():
        msg = f"nothing was written to the temporary file for {target}"
        raise OSError(msg)
    tmp_path.replace(target)


def write_text(target: Path, text: str) -> Path:
    """Atomically write `text` to `target` as UTF-8."""
    with atomic_path(target) as tmp:
        tmp.write_text(text, encoding="utf-8")
    return target


def write_json(target: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write `payload` to `target` as indented, sorted JSON."""
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    return write_text(target, text + "\n")


def write_csv(target: Path, frame: pd.DataFrame) -> Path:
    """Atomically write `frame` to `target` as CSV without the index."""
    with atomic_path(target) as tmp:
        frame.to_csv(tmp, index=False)
    return target


def write_parquet(target: Path, frame: pd.DataFrame) -> Path:
    """Atomically write `frame` to `target` as Parquet.

    Parquet is used for intermediates: it preserves dtypes across stages, which
    CSV does not, and keeps per-segment tables compact.
    """
    with atomic_path(target) as tmp:
        frame.to_parquet(tmp, index=False)
    return target


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV written by this pipeline."""
    return pd.read_csv(path)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet intermediate written by this pipeline."""
    return pd.read_parquet(path)
