"""The pre-commit media hook is a data-safety control, so it is tested."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / "scripts" / "block_media_files.py"
_spec = importlib.util.spec_from_file_location("block_media_files", _HOOK)
assert _spec is not None and _spec.loader is not None
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


@pytest.mark.parametrize(
    "path",
    [
        "28.mp4",
        "notes/session.MOV",
        "audio/12.wav",
        "transcripts/5.srt",
        "out/3.vtt",
        "dump/frames.npy",
        "features.csv",
        "labels.csv",
    ],
)
def test_data_shaped_paths_are_blocked(path):
    assert hook.is_blocked(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "src/vc_multimodal/cli.py",
        "README.md",
        "config/default.yaml",
        "tests/synth/make_video.py",
        # Synthetic fixtures and documented examples are explicitly allowed.
        "tests/fixtures/fake.srt",
        "docs/examples/features.csv",
    ],
)
def test_source_and_fixtures_are_allowed(path):
    assert hook.is_blocked(path) is None


def test_main_exits_nonzero_and_lists_every_violation(capsys):
    code = hook.main(["a.mp4", "src/ok.py", "b.wav"])
    captured = capsys.readouterr()
    assert code == 1
    assert "a.mp4" in captured.err
    assert "b.wav" in captured.err
    assert "src/ok.py" not in captured.err


def test_main_exits_zero_for_clean_input():
    assert hook.main(["src/ok.py", "README.md"]) == 0
