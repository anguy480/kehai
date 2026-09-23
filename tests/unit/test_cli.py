"""The CLI surface: option handling, exit codes and error messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

import vc_multimodal
from tests.conftest import WINTER_FOLDER, place_fake_media
from tests.synth import generators as gen
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


# ---------------------------------------------------------------------------
# extract-audio
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_extract_audio_writes_audio_and_reports_the_channel_comparison(
    roots: DataRoots, make_real_media: Any
):
    make_real_media(28)

    result = _run("extract-audio")

    assert result.exit_code == 0
    assert (roots.work / "audio" / "28.wav").exists()
    assert (roots.out / "audio_qc.csv").exists()
    assert "audio extracted for 1 session(s)" in result.output
    assert "left/right" in result.output


@pytest.mark.slow
def test_extract_audio_exits_non_zero_when_a_session_fails(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    place_fake_media(roots.data, WINTER_FOLDER, [29])

    result = _run("extract-audio")

    assert result.exit_code == EXIT_STAGE_FAILED
    assert "FAILED session 29" in result.output


# ---------------------------------------------------------------------------
# diarize
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_diarize_without_an_import_dir_is_a_clean_setup_error(
    roots: DataRoots, make_real_media: Any
):
    """The shipped config leaves this unset, since the source is unresolved."""
    make_real_media(28)

    result = _run("diarize")

    assert result.exit_code == EXIT_SETUP_ERROR
    assert "import_dir" in result.output
    assert "Traceback" not in result.output


@pytest.mark.slow
def test_diarize_reads_imported_output_and_reports_without_transcripts(
    roots: DataRoots, make_real_media: Any, tmp_path: Path
):
    _, session = make_real_media(28, duration=18.0)
    import_dir = roots.work / "diarization"
    import_dir.mkdir(parents=True, exist_ok=True)
    gen.write_srt(import_dir / "28.srt", session)

    overlay = tmp_path / "diar.yaml"
    overlay.write_text('diarization:\n  import_dir: "diarization"\n', encoding="utf-8")

    result = runner.invoke(app, ["--config", DEFAULT, "--overlay", str(overlay), "diarize"])

    assert result.exit_code == 0
    assert (roots.work / "segments" / "28.parquet").exists()
    assert (roots.out / "diarization_qc.csv").exists()
    assert "speakers per session" in result.output
    # The synthetic transcript text must not reach stdout.
    assert "turn 0" not in result.output


# ---------------------------------------------------------------------------
# vad and turns
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_vad_requires_diarization_first(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    result = _run("vad")
    assert result.exit_code == EXIT_STAGE_FAILED
    assert "vc diarize" in result.output


@pytest.mark.slow
def test_turns_without_a_role_mapping_says_how_to_make_one(roots: DataRoots, make_real_media: Any):
    """The one thing this pipeline must never guess."""
    make_real_media(28)
    result = _run("turns")

    assert result.exit_code == EXIT_STAGE_FAILED
    assert "Traceback" not in result.output
    # The failure names both routes to a mapping.
    assert "vc assign-speakers" in result.output or "vc vad" in result.output


@pytest.mark.slow
def test_vad_and_turns_run_end_to_end(roots: DataRoots, make_real_media: Any, tmp_path: Path):
    """inventory -> extract-audio -> diarize -> vad -> turns, in one go."""
    _, session = make_real_media(28, duration=22.0)
    import_dir = roots.work / "diarization"
    import_dir.mkdir(parents=True, exist_ok=True)
    gen.write_srt(import_dir / "28.srt", session)
    pd.DataFrame(
        [(28, "SPEAKER_00", "psychiatrist"), (28, "SPEAKER_01", "participant")],
        columns=["session_id", "speaker", "role"],
    ).to_csv(roots.work / "roles.csv", index=False)

    overlay = tmp_path / "pipeline.yaml"
    overlay.write_text('diarization:\n  import_dir: "diarization"\n', encoding="utf-8")
    common = ["--config", DEFAULT, "--overlay", str(overlay)]

    assert runner.invoke(app, [*common, "extract-audio"]).exit_code == 0
    assert runner.invoke(app, [*common, "diarize"]).exit_code == 0
    assert runner.invoke(app, [*common, "vad"]).exit_code == 0

    turns_result = runner.invoke(app, [*common, "turns"])

    assert turns_result.exit_code == 0
    assert (roots.out / "turn_features.csv").exists()
    assert (roots.work / "turns" / "28.parquet").exists()
    assert (roots.work / "timeline" / "28.parquet").exists()
    assert "turn features for 1 session(s)" in turns_result.output


# ---------------------------------------------------------------------------
# the region diagnostic
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_debug_region_reports_geometry_without_names(roots: DataRoots, make_real_media: Any):
    make_real_media(28, duration=12.0)

    result = _run("verify-layout", "--debug-region")

    assert "region diagnostic" in result.output
    assert "letterbox" in result.output
    assert "observation(s)" in result.output
    # Fractional and pixel coordinates both present.
    assert "x=0." in result.output
    assert " at (" in result.output
    # The synthetic labels drawn into the video must not appear.
    assert "SATO" not in result.output.upper()


@pytest.mark.slow
def test_the_diagnostic_is_off_by_default(roots: DataRoots, make_real_media: Any):
    make_real_media(28, duration=12.0)
    result = _run("verify-layout")
    assert "region diagnostic" not in result.output


@pytest.mark.slow
def test_the_debug_table_is_written_and_reported(roots: DataRoots, make_real_media: Any):
    make_real_media(28, duration=12.0)
    result = _run("verify-layout")
    assert (roots.out / "layout_debug.csv").exists()
    assert "layout_debug.csv" in result.output


@pytest.mark.slow
def test_preview_explains_the_annotations(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    result = _run("preview")
    assert "green boxes are the label regions" in result.output


@pytest.mark.slow
def test_preview_annotation_can_be_disabled(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    result = _run("--force", "preview", "--no-label-regions")
    assert result.exit_code == 0
    assert "green boxes" not in result.output


# ---------------------------------------------------------------------------
# prosody
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_prosody_requires_vad_first(roots: DataRoots, make_real_media: Any):
    make_real_media(28)
    result = _run("prosody")
    assert result.exit_code == EXIT_STAGE_FAILED
    assert "vc vad" in result.output


@pytest.mark.slow
def test_prosody_measures_participant_speech(
    roots: DataRoots, make_real_media: Any, tmp_path: Path
):
    """Speech spans are written directly: Silero rejects synthetic tones."""
    session = gen.alternating_session(
        28, n_turns=8, turn_s=6.0, gap_s=1.0, lead_in_s=1.0, duration=58.0
    )
    place_fake_media(roots.data, WINTER_FOLDER, [28])
    gen.write_wav(
        roots.work_path("audio", "28.wav"),
        gen.voiced_session_waveform(session),
        session.sample_rate,
    )
    frame = pd.DataFrame(
        [(28, u.speaker, u.start, u.end) for u in session.utterances],
        columns=["session_id", "speaker", "start_s", "end_s"],
    )
    frame["session_id"] = frame["session_id"].astype("int64")
    frame["speaker"] = frame["speaker"].astype("string")
    for column in ("start_s", "end_s"):
        frame[column] = frame[column].astype("float64")
    frame.to_parquet(roots.work_path("speech", "28.parquet"), index=False)
    pd.DataFrame(
        [(28, "SPEAKER_00", "psychiatrist"), (28, "SPEAKER_01", "participant")],
        columns=["session_id", "speaker", "role"],
    ).to_csv(roots.work / "roles.csv", index=False)

    result = _run("prosody")

    assert result.exit_code == 0
    assert (roots.out / "prosody_features.csv").exists()
    assert "F0 variability (semitones)" in result.output
    assert "each speaker's own median" in result.output
