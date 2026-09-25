"""Tests for run provenance.

A real git repository is created in a temp directory; no network, no fixtures
from the project's own history.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from vc_multimodal.provenance import (
    TRACKED_PACKAGES,
    environment_record,
    git_state,
    package_versions,
)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Test")
    (path / "a.txt").write_text("one\n")
    git(path, "add", "a.txt")
    git(path, "commit", "-q", "-m", "first")
    return path


class TestGitState:
    def test_a_clean_repository_is_clean(self, tmp_path: Path) -> None:
        state = git_state(make_repo(tmp_path / "repo"))
        assert state is not None
        assert not state.is_dirty
        assert state.dirty_paths == ()
        assert len(state.commit) == 40
        assert state.short == state.commit[:12]

    def test_an_uncommitted_change_is_dirty_and_named(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        (repo / "a.txt").write_text("two\n")
        state = git_state(repo)
        assert state is not None
        assert state.is_dirty
        assert "a.txt" in state.dirty_paths

    def test_an_untracked_file_counts_as_dirty(self, tmp_path: Path) -> None:
        # It would be part of the next commit, so a bundle built now is not
        # described by the current one.
        repo = make_repo(tmp_path / "repo")
        (repo / "new.py").write_text("x = 1\n")
        state = git_state(repo)
        assert state is not None
        assert state.is_dirty
        assert "new.py" in state.dirty_paths

    def test_a_directory_that_is_not_a_repository_returns_none(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        assert git_state(plain) is None

    def test_the_record_is_serialisable(self, tmp_path: Path) -> None:
        state = git_state(make_repo(tmp_path / "repo"))
        assert state is not None
        record = state.record()
        assert record["commit"] == state.commit
        assert record["dirty"] is False
        assert isinstance(record["dirty_paths"], list)


class TestVersions:
    def test_every_tracked_package_is_reported(self) -> None:
        versions = package_versions()
        assert set(versions) == set(TRACKED_PACKAGES)

    def test_a_missing_package_is_recorded_rather_than_raising(self) -> None:
        versions = package_versions(("definitely-not-installed-xyz",))
        assert versions["definitely-not-installed-xyz"] == "not installed"

    def test_the_packages_whose_output_depends_on_them_are_tracked(self) -> None:
        # A different MediaPipe or Praat gives different numbers from the same
        # recording, so the bundle has to name the versions used.
        for name in ("mediapipe", "praat-parselmouth", "silero-vad"):
            assert name in TRACKED_PACKAGES

    def test_the_environment_names_the_interpreter_and_platform(self) -> None:
        record = environment_record()
        assert record["python"].startswith("3.")
        assert record["platform"]
        assert isinstance(record["packages"], dict)
