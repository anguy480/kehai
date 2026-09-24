"""Shared fixtures. Everything here is synthetic; no real data is ever used."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from tests.synth import generators as gen
from vc_multimodal import paths
from vc_multimodal.config import AppConfig, load_config
from vc_multimodal.logging_setup import ROOT_LOGGER_NAME

REPO_ROOT = Path(__file__).resolve().parents[1]


def _real_model_path() -> Path | None:
    """Locate the downloaded face landmarker, before any test moves the roots.

    Resolved at import time on purpose: the `roots` fixture repoints
    `$VC_WORK_ROOT` at a temporary directory, so by the time a test runs the
    real one is no longer in the environment.
    """
    paths.load_env(REPO_ROOT)
    configured = os.environ.get(paths.WORK_ROOT_ENV, "").strip()
    if not configured:
        return None
    candidate = Path(configured).expanduser() / "models" / "face_landmarker.task"
    return candidate if candidate.is_file() else None


#: The real landmarker model, or None if it has not been downloaded.
REAL_FACE_MODEL: Path | None = _real_model_path()

# Folder names and ID ranges mirror config/default.yaml.
WINTER_FOLDER = "December 21 2025"
SUMMER_FOLDER = "July 4 2026"


@pytest.fixture(scope="session")
def ffmpeg_bin() -> str:
    """Path to ffmpeg, skipping the test if it is unavailable."""
    found = shutil.which("ffmpeg")
    if not found:
        pytest.skip("ffmpeg not on PATH")
    return found


@pytest.fixture(scope="session")
def ffprobe_bin() -> str:
    """Path to ffprobe, skipping the test if it is unavailable."""
    found = shutil.which("ffprobe")
    if not found:
        pytest.skip("ffprobe not on PATH")
    return found


@pytest.fixture
def default_config() -> AppConfig:
    """The project's real default configuration, loaded from config/."""
    return load_config(REPO_ROOT / "config" / "default.yaml")


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> paths.DataRoots:
    """Three data roots under tmp_path, exported as the real environment vars."""
    data, work, out = tmp_path / "raw", tmp_path / "work", tmp_path / "out"
    for directory in (data, work, out):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(data))
    monkeypatch.setenv(paths.WORK_ROOT_ENV, str(work))
    monkeypatch.setenv(paths.OUT_ROOT_ENV, str(out))
    return paths.DataRoots(data=data, work=work, out=out)


@pytest.fixture
def raw_tree(roots: paths.DataRoots, default_config: AppConfig) -> Path:
    """An empty raw tree with every configured date folder present."""
    for folder in default_config.dataset.folders:
        (roots.data / folder).mkdir(parents=True, exist_ok=True)
    return roots.data


def place_fake_media(raw_root: Path, folder: str, session_ids: Sequence[int]) -> list[Path]:
    """Create placeholder mp4 files that are never decoded.

    For tests that only exercise discovery, the file contents are irrelevant, so
    this avoids the cost of rendering video.
    """
    created: list[Path] = []
    directory = raw_root / folder
    directory.mkdir(parents=True, exist_ok=True)
    for session_id in session_ids:
        path = directory / f"{session_id}.mp4"
        path.write_bytes(b"not a real recording")
        created.append(path)
    return created


@pytest.fixture
def make_real_media(raw_tree: Path, tmp_path: Path, ffmpeg_bin: str) -> Iterator[object]:
    """Factory writing decodable synthetic recordings into the raw tree."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def factory(
        session_id: int,
        *,
        folder: str = WINTER_FOLDER,
        per_speaker_audio: bool = False,
        duration: float = 8.0,
    ) -> tuple[Path, gen.SyntheticSession]:
        session = gen.alternating_session(
            session_id, n_turns=4, turn_s=1.5, gap_s=0.5, duration=duration
        )
        target = raw_tree / folder / f"{session_id}.mp4"
        gen.write_session_mp4(
            target,
            session,
            tmp_dir=scratch,
            per_speaker_audio=per_speaker_audio,
            ffmpeg=ffmpeg_bin,
        )
        return target, session

    yield factory


class PackageLogCapture:
    """Records emitted by the package logger during a test."""

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []

    @property
    def text(self) -> str:
        """Every captured message, formatted, one per line."""
        return "\n".join(
            f"{record.levelname} {record.name} {record.getMessage()}" for record in self.records
        )

    def messages_at(self, level: str) -> list[str]:
        """Messages captured at exactly `level`."""
        return [r.getMessage() for r in self.records if r.levelname == level]


@pytest.fixture
def package_logs() -> Iterator[PackageLogCapture]:
    """Capture log records from the package logger.

    pytest's own `caplog` cannot be used for this. `configure_logging` sets
    `propagate = False` on the package logger, so once any stage has configured
    logging nothing reaches the root logger, and `caplog.text` is empty. A test
    asserting that something is *not* logged would then pass for the wrong
    reason, which is exactly the kind of test that matters here.

    This attaches a handler to the package logger itself, so it captures
    whatever the code really emits either way.
    """
    capture = PackageLogCapture()

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            capture.records.append(record)

    handler = _Handler(level=logging.DEBUG)
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield capture
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
