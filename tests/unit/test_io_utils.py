"""Atomic write behaviour. An interrupted stage must never corrupt an output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vc_multimodal.io_utils import (
    atomic_path,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
    write_parquet,
    write_text,
)


def test_write_text_creates_parent_directories(tmp_path: Path):
    target = write_text(tmp_path / "a" / "b" / "note.txt", "hello")
    assert target.read_text(encoding="utf-8") == "hello"


def test_a_failed_write_leaves_the_previous_output_intact(tmp_path: Path):
    target = tmp_path / "features.txt"
    write_text(target, "the good version")

    with pytest.raises(RuntimeError, match="interrupted"), atomic_path(target) as tmp:
        tmp.write_text("half a file", encoding="utf-8")
        msg = "interrupted"
        raise RuntimeError(msg)

    assert target.read_text(encoding="utf-8") == "the good version"


def test_a_failed_write_leaves_no_temporary_files_behind(tmp_path: Path):
    target = tmp_path / "out.txt"
    with pytest.raises(RuntimeError), atomic_path(target):
        msg = "boom"
        raise RuntimeError(msg)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_does_not_create_the_target(tmp_path: Path):
    target = tmp_path / "out.txt"
    with pytest.raises(RuntimeError), atomic_path(target):
        msg = "boom"
        raise RuntimeError(msg)
    assert not target.exists()


def test_keyboard_interrupt_is_also_cleaned_up(tmp_path: Path):
    """Ctrl-C mid-run is the realistic interruption, and it is not an Exception."""
    target = tmp_path / "out.txt"
    with pytest.raises(KeyboardInterrupt), atomic_path(target) as tmp:
        tmp.write_text("partial", encoding="utf-8")
        raise KeyboardInterrupt
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_writing_nothing_is_an_error_rather_than_an_empty_output(tmp_path: Path):
    target = tmp_path / "out.txt"
    with pytest.raises(OSError, match="nothing was written"), atomic_path(target) as tmp:
        tmp.unlink()
    assert not target.exists()


def test_the_temporary_file_lives_beside_the_target(tmp_path: Path):
    """A rename is only atomic within one filesystem, so the temp must be local."""
    target = tmp_path / "nested" / "out.txt"
    with atomic_path(target) as tmp:
        assert tmp.parent == target.parent
        tmp.write_text("x", encoding="utf-8")


def test_an_existing_output_is_replaced(tmp_path: Path):
    target = tmp_path / "out.txt"
    write_text(target, "old")
    write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_json_is_written_sorted_and_readable(tmp_path: Path):
    target = write_json(tmp_path / "manifest.json", {"b": 2, "a": {"z": 1}})
    text = target.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"b"')
    assert json.loads(text) == {"b": 2, "a": {"z": 1}}


def test_json_falls_back_for_unserialisable_values(tmp_path: Path):
    target = write_json(tmp_path / "m.json", {"path": Path("/tmp/x")})
    assert json.loads(target.read_text(encoding="utf-8"))["path"] == "/tmp/x"


def test_csv_round_trips_without_an_index_column(tmp_path: Path):
    frame = pd.DataFrame({"session_id": [1, 2], "turns__latency_mean": [0.5, 1.5]})
    target = write_csv(tmp_path / "features.csv", frame)
    restored = read_csv(target)
    assert list(restored.columns) == ["session_id", "turns__latency_mean"]
    pd.testing.assert_frame_equal(restored, frame)


def test_parquet_round_trips_and_preserves_dtypes(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "session_id": pd.Series([1, 2], dtype="int64"),
            "speaker": ["SPEAKER_00", "SPEAKER_01"],
            "start_s": [0.5, 2.0],
        }
    )
    restored = read_parquet(write_parquet(tmp_path / "segments.parquet", frame))
    pd.testing.assert_frame_equal(restored, frame)
