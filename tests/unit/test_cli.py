"""The CLI surface: option handling, exit codes and error messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import vc_multimodal
from tests.conftest import WINTER_FOLDER, place_fake_media
from vc_multimodal import ffmpeg as ffmpeg_module
from vc_multimodal import paths
from vc_multimodal.cli import EXIT_SETUP_ERROR, EXIT_STAGE_FAILED, app
from vc_multimodal.paths import DataRoots

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = str(REPO_ROOT / "config" / "default.yaml")


def _run(*args: str) -> Any:
    """Invoke the CLI with the shipped config."""
    return runner.invoke(app, ["--config", DEFAULT, *args])


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------
def test_version_subcommand_and_flag():
    for args in (["version"], ["--version"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert result.stdout.strip() == vc_multimodal.__version__


def test_no_arguments_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == EXIT_SETUP_ERROR
    assert "Usage" in result.stdout


def test_every_stage_is_listed_in_the_help():
    result = runner.invoke(app, ["--help"])
    for command in ("inventory", "preview", "doctor", "version"):
        assert command in result.stdout


def test_an_unknown_subcommand_fails():
    assert runner.invoke(app, ["not-a-stage"]).exit_code != 0


# ---------------------------------------------------------------------------
# setup errors are distinguishable from stage failures
# ---------------------------------------------------------------------------
def test_a_missing_data_root_is_a_setup_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(paths.DATA_ROOT_ENV, raising=False)
    monkeypatch.delenv(paths.WORK_ROOT_ENV, raising=False)
    monkeypatch.delenv(paths.OUT_ROOT_ENV, raising=False)
    monkeypatch.chdir(REPO_ROOT.parent)  # away from any .env

    result = _run("inventory")

    assert result.exit_code == EXIT_SETUP_ERROR
    assert "VC_DATA_ROOT" in result.output


def test_a_bad_config_path_is_a_setup_error(roots: DataRoots, tmp_path: Path):
    result = runner.invoke(app, ["--config", str(tmp_path / "absent.yaml"), "inventory"])
    assert result.exit_code == EXIT_SETUP_ERROR
    assert "not found" in result.output


def test_a_missing_ffmpeg_is_a_setup_error(roots: DataRoots, monkeypatch: pytest.MonkeyPatch):
    def missing(name: str, env_var: str) -> Path:
        msg = f"{name} was not found. This project does not install it."
        raise ffmpeg_module.FfmpegError(msg)

    monkeypatch.setattr(ffmpeg_module, "_resolve_binary", missing)
    result = _run("inventory")
    assert result.exit_code == EXIT_SETUP_ERROR
    assert "does not install it" in result.output


def test_an_invalid_session_spec_is_a_setup_error(roots: DataRoots):
    result = _run("--sessions", "abc", "inventory")
    assert result.exit_code == EXIT_SETUP_ERROR
    assert "invalid session" in result.output


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_doctor_reports_binaries_roots_and_layout_problems(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    result = _run("doctor")

    assert "ffmpeg version" in result.output
    assert str(roots.data) in result.output
    assert "1 session(s) found, expected 62" in result.output
    # An incomplete tree is a problem worth a non-zero exit.
    assert result.exit_code == EXIT_STAGE_FAILED


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_inventory_writes_the_table_and_prints_a_metadata_summary(
    roots: DataRoots, make_real_media: Any
):
    make_real_media(28)
    make_real_media(3, folder="January 17 2026")

    result = _run("inventory")

    assert (roots.out / "inventory.csv").exists()
    assert "sessions: 2 (expected 62)" in result.output
    assert "audio streams per file" in result.output
    assert "inventory: 2 ok" in result.output
    # A partial tree is reported, but probing succeeded, so the stage passes.
    assert result.exit_code == 0


@pytest.mark.slow
def test_inventory_prints_a_log_file_location(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    result = _run("inventory")
    assert "log:" in result.output
    assert list((roots.out / "logs").glob("*_inventory.log"))


@pytest.mark.slow
def test_inventory_exits_non_zero_when_a_session_fails(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    place_fake_media(roots.data, WINTER_FOLDER, [29])

    result = _run("inventory")

    assert result.exit_code == EXIT_STAGE_FAILED
    assert "FAILED session 29" in result.output


@pytest.mark.slow
def test_the_sessions_flag_limits_the_run(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    make_real_media(3, folder="January 17 2026")

    result = _run("--sessions", "28", "inventory")

    assert "inventory: 1 ok" in result.output


@pytest.mark.slow
def test_an_absent_requested_session_is_reported(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    result = _run("--sessions", "28,999", "inventory")
    assert "not found: [999]" in result.output


@pytest.mark.slow
def test_a_config_overlay_is_applied(roots: DataRoots, make_real_media: Any, tmp_path: Path):
    make_real_media(28)
    make_real_media(3, folder="January 17 2026")
    overlay = tmp_path / "pilot.yaml"
    overlay.write_text("runtime:\n  pilot_sessions: [28]\n", encoding="utf-8")

    result = _run("--overlay", str(overlay), "inventory")

    assert "inventory: 1 ok" in result.output


@pytest.mark.slow
def test_the_sessions_flag_beats_the_configured_pilot_set(
    roots: DataRoots, make_real_media: Any, tmp_path: Path
):
    make_real_media(28)
    make_real_media(3, folder="January 17 2026")
    overlay = tmp_path / "pilot.yaml"
    overlay.write_text("runtime:\n  pilot_sessions: [28]\n", encoding="utf-8")

    result = _run("--overlay", str(overlay), "--sessions", "3", "inventory")

    assert "inventory: 1 ok" in result.output
    assert "FAILED" not in result.output


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_preview_writes_sheets_and_asks_for_confirmation(roots: DataRoots, make_real_media: Any):
    make_real_media(28)

    result = _run("preview")

    assert result.exit_code == 0
    assert (roots.out / "previews" / "28.jpg").exists()
    # The stage exists to prompt a human check, so it says so.
    assert "gallery view" in result.output


@pytest.mark.slow
def test_preview_is_idempotent_across_runs(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    _run("preview")
    result = _run("preview")
    assert "0 ok, 1 skipped" in result.output


@pytest.mark.slow
def test_force_recomputes_previews(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    _run("preview")
    result = _run("--force", "preview")
    assert "1 ok, 0 skipped" in result.output


# ---------------------------------------------------------------------------
# a pre-existing file at the output path
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_an_unrecognised_inventory_file_is_a_clean_setup_error(
    roots: DataRoots, make_real_media: Any
):
    """Reported as a crash: KeyError from a headerless CSV left in $VC_OUT_ROOT."""
    make_real_media(28)
    planted = roots.out / "inventory.csv"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text("28,640.5,1920,1080\n", encoding="utf-8")

    result = _run("inventory")

    assert result.exit_code == EXIT_SETUP_ERROR
    assert "KeyError" not in result.output
    assert "Traceback" not in result.output
    assert "Move or delete" in result.output
    assert "--force" in result.output
    assert planted.read_text(encoding="utf-8") == "28,640.5,1920,1080\n"


@pytest.mark.slow
def test_force_moves_the_unrecognised_file_aside(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    planted = roots.out / "inventory.csv"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text("28,640.5,1920,1080\n", encoding="utf-8")

    result = _run("--force", "inventory")

    assert result.exit_code == 0
    assert "moved an unrecognised" in result.output
    backups = list(roots.out.glob("inventory.csv.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "28,640.5,1920,1080\n"


# ---------------------------------------------------------------------------
# verify-layout
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_verify_layout_reports_sides_without_printing_labels(
    roots: DataRoots, make_real_media: Any
):
    make_real_media(28, duration=12.0)
    make_real_media(3, folder="January 17 2026", duration=12.0)

    result = _run("verify-layout")

    assert (roots.out / "layout.csv").exists()
    assert "psychiatrist side, as found by label OCR" in result.output
    assert "assumed psychiatrist side: left" in result.output
    # Labels drawn into the synthetic video, which OCR may well have read.
    assert "SATO" not in result.output.upper()
    assert "GUEST" not in result.output.upper()


@pytest.mark.slow
def test_doctor_reports_whether_label_ocr_is_usable(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    result = _run("doctor")
    assert "label OCR:" in result.output
