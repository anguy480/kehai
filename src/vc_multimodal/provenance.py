"""What produced a run: the commit, the tools, the versions.

A handoff bundle is only trustworthy if the person holding it can find out how
it was made. That means the exact commit, whether the tree was clean at the
time, and the versions of the tools whose output is not reproducible across
releases - a different MediaPipe or Praat gives different numbers from the same
video.

Nothing here reads the recordings, and nothing here can fail a run: an
unavailable version is recorded as unavailable rather than raising, because
refusing to build a bundle over an unreadable package version would be worse
than noting it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Final

from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

_GIT_TIMEOUT_S: Final = 10.0

#: Packages whose version changes the numbers, so the bundle records them.
TRACKED_PACKAGES: Final = (
    "mediapipe",
    "praat-parselmouth",
    "silero-vad",
    "onnxruntime",
    "opencv-contrib-python",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "pandera",
    "pyarrow",
)


@dataclass(frozen=True, slots=True)
class GitState:
    """The repository as it stood when the bundle was built."""

    commit: str
    short: str
    branch: str
    is_dirty: bool
    #: Paths with uncommitted changes, for the refusal message.
    dirty_paths: tuple[str, ...]

    def record(self) -> dict[str, object]:
        """What the manifest carries."""
        return {
            "commit": self.commit,
            "short": self.short,
            "branch": self.branch,
            "dirty": self.is_dirty,
            "dirty_paths": list(self.dirty_paths),
        }


def _git(repo: Path, *args: str) -> str | None:
    """Run one git command, returning None if git or the repo is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _porcelain_path(line: str) -> str:
    """The path from one `git status --porcelain` line.

    Parsed by splitting off the status code rather than by column offset: a
    line for an unstaged change begins with a space, and any leading whitespace
    may already have been stripped by the time it gets here. A fixed offset
    then eats the first character of the filename, which looks like a plausible
    path and is not.
    """
    parts = line.strip().split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else parts[0].strip()


def git_state(repo: Path) -> GitState | None:
    """The repository state, or None when `repo` is not a git checkout."""
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git(repo, "status", "--porcelain")
    dirty_paths = tuple(
        _porcelain_path(line) for line in (status or "").splitlines() if line.strip()
    )
    return GitState(
        commit=commit,
        short=commit[:12],
        branch=_git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        is_dirty=bool(dirty_paths),
        dirty_paths=dirty_paths,
    )


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of the packages whose output depends on them."""
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "not installed"
    return found


def environment_record() -> dict[str, object]:
    """The interpreter and platform, for the manifest."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
    }
