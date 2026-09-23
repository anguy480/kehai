"""The external whisper-diarization backend.

The tool itself is deliberately not a dependency of this project, so what is
tested here is everything around invoking it: whether it is configured, what it
records, that existing output is reused rather than regenerated, and that a
missing command points at the setup instructions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.synth import generators as gen
from vc_multimodal.config import AppConfig, load_config
from vc_multimodal.diarization import (
    DiarizationError,
    ImportBackend,
    PyannoteBackend,
    WhisperDiarizationBackend,
    get_backend,
)
from vc_multimodal.paths import DataRoots, RawSession

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "config" / "default.yaml"
PATTERNS = ("{session_id}.srt",)


def _session(session_id: int = 28) -> RawSession:
    return RawSession(
        session_id=session_id,
        wave="winter",
        date_folder="December 21 2025",
        path=Path(f"/nowhere/{session_id}.mp4"),
    )


def _backend(
    output_dir: Path | None, *, command: tuple[str, ...] = ("python", "diarize.py")
) -> WhisperDiarizationBackend:
    config = load_config(
        DEFAULT, overrides={"diarization.whisper_diarization.command": list(command)}
    )
    return WhisperDiarizationBackend(
        config.diarization.whisper_diarization,
        output_dir=output_dir,
        patterns=PATTERNS,
    )


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_it_needs_an_output_directory(tmp_path: Path):
    backend = _backend(None)
    assert not backend.available()
    assert "output_dir" in backend.unavailable_reason()
    with pytest.raises(DiarizationError, match="output_dir"):
        backend.segments(_session())


def test_it_needs_a_command(tmp_path: Path):
    backend = _backend(tmp_path, command=())
    assert not backend.available()
    assert "command is empty" in backend.unavailable_reason()


def test_it_is_available_once_both_are_set(tmp_path: Path):
    backend = _backend(tmp_path)
    assert backend.available()
    assert backend.unavailable_reason() == ""


def test_the_recorded_version_names_the_external_command(tmp_path: Path):
    """The tool runs in its own environment, so its version is not visible."""
    version = _backend(tmp_path).version()
    assert "external" in version
    assert "diarize.py" in version


# ---------------------------------------------------------------------------
# reusing output
# ---------------------------------------------------------------------------
def test_existing_output_is_parsed_without_running_anything(tmp_path: Path):
    """A run takes minutes per session, so it must not be repeated."""
    session = gen.alternating_session(28, n_turns=4, turn_s=1.5, gap_s=0.5)
    gen.write_srt(tmp_path / "28.srt", session)

    # The command is nonsense: reaching it at all would raise.
    backend = _backend(tmp_path, command=("definitely-not-a-real-command",))
    segments = backend.segments(_session(28))

    assert len(segments) == 4


def test_a_missing_command_points_at_the_setup_instructions(tmp_path: Path):
    backend = _backend(tmp_path, command=("definitely-not-a-real-command",))
    with pytest.raises(DiarizationError, match=re.escape("whisper_diarization_setup.md")):
        backend.segments(_session(28))


def test_the_setup_instructions_exist():
    """The error message above would otherwise point at nothing."""
    assert (REPO_ROOT / "scripts" / "whisper_diarization_setup.md").is_file()


def test_a_failing_command_reports_its_exit_code(tmp_path: Path):
    backend = _backend(tmp_path, command=("sh", "-c", "exit 3", "_"))
    with pytest.raises(DiarizationError, match="exited 3"):
        backend.segments(_session(28))


def test_a_command_that_writes_nothing_is_reported(tmp_path: Path):
    backend = _backend(tmp_path, command=("sh", "-c", "true", "_"))
    with pytest.raises(DiarizationError, match="no output was found"):
        backend.segments(_session(28))


def test_output_written_by_the_command_is_parsed(tmp_path: Path):
    srt = "1\n00:00:00,000 --> 00:00:02,000\nSpeaker 0: hi\n"
    backend = _backend(tmp_path, command=("sh", "-c", f"printf '{srt}' > 28.srt; true", "_"))
    segments = backend.segments(_session(28))
    assert len(segments) == 1
    assert segments[0].speaker == "SPEAKER_00"


def test_a_timeout_is_reported(tmp_path: Path):
    config = load_config(
        DEFAULT,
        overrides={
            # The backend appends its own arguments, so the command must
            # tolerate extras: sh -c takes them as positional parameters.
            "diarization.whisper_diarization.command": ["sh", "-c", "sleep 5", "_"],
            "diarization.whisper_diarization.timeout_seconds": 0.2,
        },
    )
    backend = WhisperDiarizationBackend(
        config.diarization.whisper_diarization, output_dir=tmp_path, patterns=PATTERNS
    )
    with pytest.raises(DiarizationError, match="timed out"):
        backend.segments(_session(28))


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def test_every_configurable_backend_can_be_built(roots: DataRoots, default_config: AppConfig):
    (roots.work / "diarization").mkdir(parents=True, exist_ok=True)
    (roots.work / "wd-out").mkdir(parents=True, exist_ok=True)

    built = {
        "import": get_backend(
            load_config(DEFAULT, overrides={"diarization.import_dir": "diarization"}), roots
        ),
        "pyannote": get_backend(
            load_config(DEFAULT, overrides={"diarization.backend": "pyannote"}), roots
        ),
        "whisper_diarization": get_backend(
            load_config(
                DEFAULT,
                overrides={
                    "diarization.backend": "whisper_diarization",
                    "diarization.whisper_diarization.output_dir": "wd-out",
                },
            ),
            roots,
        ),
    }

    assert isinstance(built["import"], ImportBackend)
    assert isinstance(built["pyannote"], PyannoteBackend)
    assert isinstance(built["whisper_diarization"], WhisperDiarizationBackend)
    for name, backend in built.items():
        assert backend.name == name


def test_the_external_backends_output_dir_resolves_under_the_work_root(roots: DataRoots):
    config = load_config(
        DEFAULT,
        overrides={
            "diarization.backend": "whisper_diarization",
            "diarization.whisper_diarization.output_dir": "wd-out",
        },
    )
    backend = get_backend(config, roots)
    assert isinstance(backend, WhisperDiarizationBackend)
    assert backend.output_dir is not None
    assert backend.output_dir.is_relative_to(roots.work)


def test_an_absolute_configured_path_is_left_alone(roots: DataRoots, tmp_path: Path):
    """An operator who gives an absolute path means it."""
    config = load_config(DEFAULT, overrides={"diarization.import_dir": str(tmp_path / "elsewhere")})
    backend = get_backend(config, roots)
    assert isinstance(backend, ImportBackend)
    assert backend.import_dir == tmp_path / "elsewhere"


def test_an_unset_output_dir_leaves_the_backend_unavailable(roots: DataRoots):
    config = load_config(DEFAULT, overrides={"diarization.backend": "whisper_diarization"})
    backend = get_backend(config, roots)
    assert not backend.available()
    assert "output_dir" in backend.unavailable_reason()
