"""Root resolution and raw-session discovery."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import SUMMER_FOLDER, WINTER_FOLDER, place_fake_media
from vc_multimodal import paths
from vc_multimodal.config import AppConfig
from vc_multimodal.paths import (
    DataRoots,
    PathError,
    discover_sessions,
    parse_session_id,
    parse_session_spec,
    select_sessions,
)


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------
def test_resolve_roots_reads_the_environment(roots: DataRoots):
    resolved = paths.resolve_roots()
    assert resolved.data == roots.data
    assert resolved.work == roots.work
    assert resolved.out == roots.out


def test_unset_root_explains_how_to_fix_it(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(paths.DATA_ROOT_ENV, raising=False)
    with pytest.raises(PathError, match="VC_DATA_ROOT is not set"):
        paths.resolve_roots()


def test_blank_root_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch, roots: DataRoots):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, "   ")
    with pytest.raises(PathError, match="is not set"):
        paths.resolve_roots()


def test_missing_data_root_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, roots: DataRoots
):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "absent"))
    with pytest.raises(PathError, match="not an existing directory"):
        paths.resolve_roots()


def test_data_root_is_never_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, roots: DataRoots
):
    absent = tmp_path / "absent-raw"
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(absent))
    with pytest.raises(PathError):
        paths.resolve_roots()
    assert not absent.exists()


def test_work_and_out_roots_are_created_on_demand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, roots: DataRoots
):
    work, out = tmp_path / "new-work", tmp_path / "new-out"
    monkeypatch.setenv(paths.WORK_ROOT_ENV, str(work))
    monkeypatch.setenv(paths.OUT_ROOT_ENV, str(out))
    resolved = paths.resolve_roots()
    assert resolved.work.is_dir()
    assert resolved.out.is_dir()


def test_analysis_commands_can_run_without_a_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, roots: DataRoots
):
    """`vc model` runs on the professor's machine, which has no raw media."""
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "absent"))
    resolved = paths.resolve_roots(require_data=False)
    assert resolved.work.is_dir()


def test_tilde_in_a_root_is_expanded(monkeypatch: pytest.MonkeyPatch, roots: DataRoots):
    monkeypatch.setenv(paths.WORK_ROOT_ENV, "~/vc-test-work-should-not-be-created")
    resolved = paths.resolve_roots(create=False)
    assert "~" not in str(resolved.work)
    assert resolved.work.is_absolute()


def test_work_path_creates_parents_and_returns_the_file(roots: DataRoots):
    target = roots.work_path("audio", "28.wav")
    assert target.parent.is_dir()
    assert target == roots.work / "audio" / "28.wav"


def test_paths_cannot_escape_their_root(roots: DataRoots):
    with pytest.raises(PathError, match="outside"):
        roots.work_path("..", "..", "etc", "passwd")


def test_load_env_reads_a_dotenv_without_overriding_the_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("VC_TEST_EXISTING", "from-environment")
    (tmp_path / ".env").write_text(
        "VC_TEST_EXISTING=from-dotenv\nVC_TEST_NEW=from-dotenv\n", encoding="utf-8"
    )
    assert paths.load_env(tmp_path) == tmp_path / ".env"
    assert os.environ["VC_TEST_EXISTING"] == "from-environment"
    assert os.environ["VC_TEST_NEW"] == "from-dotenv"


def test_load_env_is_a_no_op_without_a_dotenv(tmp_path: Path):
    assert paths.load_env(tmp_path) is None


# ---------------------------------------------------------------------------
# filenames
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("name", "expected"), [("28.mp4", 28), ("1.mp4", 1), ("261.mp4", 261)])
def test_session_id_comes_from_the_filename(name: str, expected: int):
    assert parse_session_id(name) == expected


@pytest.mark.parametrize("name", ["28b.mp4", "session28.mp4", "28 copy.mp4", ".mp4", "-3.mp4"])
def test_non_numeric_filenames_are_reported_not_guessed(name: str):
    assert parse_session_id(name) is None


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("28", (28,)),
        ("3,17,28", (3, 17, 28)),
        (" 3 , 17 ", (3, 17)),
        ("1-5", (1, 2, 3, 4, 5)),
        ("1-3,210", (1, 2, 3, 210)),
        ("7-7", (7,)),
        ("3,3,17", (3, 17)),
        ("", ()),
    ],
)
def test_session_spec_parsing(spec: str, expected: tuple[int, ...]):
    assert parse_session_spec(spec) == expected


@pytest.mark.parametrize("spec", ["abc", "1-", "-5", "1-b", "5-1", "1..3"])
def test_bad_session_spec_is_rejected(spec: str):
    with pytest.raises(ValueError, match="invalid session"):
        parse_session_spec(spec)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def test_discovery_finds_sessions_and_assigns_waves(raw_tree: Path, default_config: AppConfig):
    place_fake_media(raw_tree, WINTER_FOLDER, [1, 28])
    place_fake_media(raw_tree, SUMMER_FOLDER, [210])
    found = discover_sessions(raw_tree, default_config.dataset)

    assert found.session_ids == (1, 28, 210)
    by_id = {s.session_id: s for s in found.sessions}
    assert by_id[28].wave == "winter"
    assert by_id[210].wave == "summer"
    assert by_id[28].relpath == f"{WINTER_FOLDER}/28.mp4"


def test_discovery_reports_the_expected_session_count(raw_tree: Path, default_config: AppConfig):
    place_fake_media(raw_tree, WINTER_FOLDER, [1])
    found = discover_sessions(raw_tree, default_config.dataset)
    assert any("expected 62 sessions, found 1" in p for p in found.problems)


def test_discovery_reports_a_missing_folder(roots: DataRoots, default_config: AppConfig):
    found = discover_sessions(roots.data, default_config.dataset)
    assert sum("missing folder" in p for p in found.problems) == 5


def test_discovery_reports_an_id_in_the_wrong_wave(raw_tree: Path, default_config: AppConfig):
    """A winter-range ID sitting in a summer folder is a real filing mistake."""
    place_fake_media(raw_tree, SUMMER_FOLDER, [28])
    found = discover_sessions(raw_tree, default_config.dataset)
    assert any("belongs to" in p and "28" in p for p in found.problems)
    # It is still returned, filed under the folder's wave, so later stages see it.
    assert found.sessions[0].wave == "summer"


def test_discovery_reports_an_id_outside_every_configured_range(
    raw_tree: Path, default_config: AppConfig
):
    place_fake_media(raw_tree, WINTER_FOLDER, [80])
    found = discover_sessions(raw_tree, default_config.dataset)
    assert any("no configured wave" in p for p in found.problems)


def test_discovery_reports_duplicate_ids_across_folders(raw_tree: Path, default_config: AppConfig):
    place_fake_media(raw_tree, WINTER_FOLDER, [28])
    place_fake_media(raw_tree, "January 17 2026", [28])
    found = discover_sessions(raw_tree, default_config.dataset)
    assert any("duplicate session ID 28" in p for p in found.problems)
    assert len(found.sessions) == 1


def test_discovery_reports_non_numeric_filenames(raw_tree: Path, default_config: AppConfig):
    (raw_tree / WINTER_FOLDER / "interview 28.mp4").write_bytes(b"x")
    found = discover_sessions(raw_tree, default_config.dataset)
    assert any("not a numeric session ID" in p for p in found.problems)


def test_discovery_reports_unexpected_extra_files(raw_tree: Path, default_config: AppConfig):
    place_fake_media(raw_tree, WINTER_FOLDER, [28])
    (raw_tree / WINTER_FOLDER / "notes.txt").write_text("x", encoding="utf-8")
    found = discover_sessions(raw_tree, default_config.dataset)
    assert any("unexpected file" in p and "notes.txt" in p for p in found.problems)


def test_discovery_ignores_dotfiles(raw_tree: Path, default_config: AppConfig):
    place_fake_media(raw_tree, WINTER_FOLDER, [28])
    (raw_tree / WINTER_FOLDER / ".DS_Store").write_bytes(b"x")
    found = discover_sessions(raw_tree, default_config.dataset)
    assert not any("unexpected file" in p for p in found.problems)


def test_discovery_reports_an_empty_folder(raw_tree: Path, default_config: AppConfig):
    found = discover_sessions(raw_tree, default_config.dataset)
    assert sum("no files matching" in p for p in found.problems) == 5


def test_discovery_sorts_sessions_by_id(raw_tree: Path, default_config: AppConfig):
    place_fake_media(raw_tree, WINTER_FOLDER, [28, 3, 17])
    found = discover_sessions(raw_tree, default_config.dataset)
    assert [s.session_id for s in found.sessions] == [3, 17, 28]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def test_selection_keeps_requested_order_and_reports_missing(
    raw_tree: Path, default_config: AppConfig
):
    place_fake_media(raw_tree, WINTER_FOLDER, [1, 2, 3])
    found = discover_sessions(raw_tree, default_config.dataset)
    selected, missing = select_sessions(found.sessions, [3, 1, 99])
    assert [s.session_id for s in selected] == [3, 1]
    assert missing == (99,)


def test_selection_of_none_returns_everything(raw_tree: Path, default_config: AppConfig):
    place_fake_media(raw_tree, WINTER_FOLDER, [1, 2])
    found = discover_sessions(raw_tree, default_config.dataset)
    selected, missing = select_sessions(found.sessions, None)
    assert len(selected) == 2
    assert missing == ()
