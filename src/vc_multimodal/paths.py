"""Filesystem roots and raw-session discovery.

Three roots, all outside the repository and all read from the environment (or a
gitignored `.env`), never hardcoded:

* `$VC_DATA_ROOT` - read-only raw media.
* `$VC_WORK_ROOT` - intermediates, including transcripts, which never leave it.
* `$VC_OUT_ROOT`  - inventory, previews, logs, features, handoff bundles.

Nothing here opens a media file. Discovery works from directory listings and
filenames only.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from vc_multimodal.config import DatasetConfig

DATA_ROOT_ENV = "VC_DATA_ROOT"
WORK_ROOT_ENV = "VC_WORK_ROOT"
OUT_ROOT_ENV = "VC_OUT_ROOT"

# How many unexpected filenames to name in a problem report before eliding.
_MAX_LISTED_PROBLEM_FILES = 5


class PathError(RuntimeError):
    """Raised when a required root is unset, missing, or escaped."""


def load_env(project_root: Path | None = None) -> Path | None:
    """Load `.env` from `project_root` without overriding the real environment.

    Args:
        project_root: Directory holding `.env`. Defaults to the current
            directory.

    Returns:
        The `.env` path if one was loaded, else None.
    """
    root = Path.cwd() if project_root is None else project_root
    env_file = root / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)
        return env_file
    return None


def _root_from_env(name: str, *, must_exist: bool, create: bool) -> Path:
    """Resolve one root from the environment."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        msg = (
            f"{name} is not set. Point it at a directory outside this repository "
            f"(see .env.example, or run `make env`)."
        )
        raise PathError(msg)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if create and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    if must_exist and not path.is_dir():
        msg = f"{name} points at {path}, which is not an existing directory."
        raise PathError(msg)
    return path


@dataclass(frozen=True, slots=True)
class DataRoots:
    """The three filesystem roots for one run."""

    data: Path
    work: Path
    out: Path

    def work_path(self, *parts: str, create_parent: bool = True) -> Path:
        """Path under `$VC_WORK_ROOT`, creating parent directories by default."""
        return self._child(self.work, parts, create_parent)

    def out_path(self, *parts: str, create_parent: bool = True) -> Path:
        """Path under `$VC_OUT_ROOT`, creating parent directories by default."""
        return self._child(self.out, parts, create_parent)

    @staticmethod
    def _child(root: Path, parts: Sequence[str], create_parent: bool) -> Path:
        """Join `parts` under `root`, refusing to escape it."""
        candidate = root.joinpath(*parts)
        resolved_root = root.resolve()
        resolved = Path(os.path.normpath(candidate))
        if resolved_root != resolved and resolved_root not in resolved.parents:
            msg = f"refusing to use {candidate}, which is outside {root}"
            raise PathError(msg)
        if create_parent:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved


def resolve_roots(*, require_data: bool = True, create: bool = True) -> DataRoots:
    """Resolve all three roots from the environment.

    Args:
        require_data: Whether `$VC_DATA_ROOT` must already exist. False for
            commands that touch only work and output trees, such as `vc model`.
        create: Create the work and output roots if absent. The data root is
            never created; a missing one is a configuration error.

    Returns:
        The resolved roots.

    Raises:
        PathError: if a root is unset or the data root is missing.
    """
    return DataRoots(
        data=_root_from_env(DATA_ROOT_ENV, must_exist=require_data, create=False),
        work=_root_from_env(WORK_ROOT_ENV, must_exist=False, create=create),
        out=_root_from_env(OUT_ROOT_ENV, must_exist=False, create=create),
    )


@dataclass(frozen=True, slots=True)
class RawSession:
    """One raw recording, identified by filename alone."""

    session_id: int
    wave: str
    date_folder: str
    path: Path

    @property
    def relpath(self) -> str:
        """Path relative to the data root, as recorded in the inventory."""
        return f"{self.date_folder}/{self.path.name}"


def parse_session_id(filename: str) -> int | None:
    """Extract the numeric session ID from a media filename.

    Files are named by session ID alone, e.g. `28.mp4`. Anything else returns
    None so the caller can report it rather than silently skipping it.
    """
    stem = Path(filename).stem
    if not stem.isdigit():
        return None
    return int(stem)


@dataclass(frozen=True, slots=True)
class Discovery:
    """The result of scanning the raw data tree."""

    sessions: tuple[RawSession, ...]
    problems: tuple[str, ...]

    @property
    def session_ids(self) -> tuple[int, ...]:
        """Discovered session IDs, ascending."""
        return tuple(sorted(s.session_id for s in self.sessions))


def discover_sessions(data_root: Path, dataset: DatasetConfig) -> Discovery:
    """Scan the configured date folders for raw recordings.

    Reports rather than raises, so that one malformed filename does not hide the
    rest of the tree. Detected problems: missing or empty folders, unexpected
    extra files, non-numeric filenames, duplicate session IDs, and IDs that fall
    outside the range declared for the folder's wave.

    Args:
        data_root: `$VC_DATA_ROOT`.
        dataset: Expected layout.

    Returns:
        Discovered sessions sorted by ID, plus a list of problems.
    """
    sessions: list[RawSession] = []
    problems: list[str] = []
    seen: dict[int, str] = {}

    for folder in dataset.folders:
        wave = dataset.wave_of_folder(folder)
        if wave is None:  # pragma: no cover - folders come from the wave map
            continue
        directory = data_root / folder
        if not directory.is_dir():
            problems.append(f"missing folder: {folder!r}")
            continue

        media = sorted(directory.glob(dataset.media_glob))
        others = sorted(
            entry.name
            for entry in directory.iterdir()
            if entry.is_file() and entry not in media and not entry.name.startswith(".")
        )
        if others:
            elided = len(others) > _MAX_LISTED_PROBLEM_FILES
            shown = ", ".join(others[:_MAX_LISTED_PROBLEM_FILES]) + (" ..." if elided else "")
            problems.append(f"{folder!r} contains {len(others)} unexpected file(s): {shown}")
        if not media:
            problems.append(f"{folder!r} contains no files matching {dataset.media_glob!r}")

        for path in media:
            session_id = parse_session_id(path.name)
            if session_id is None:
                problems.append(f"{folder}/{path.name}: filename is not a numeric session ID")
                continue
            if session_id in seen:
                problems.append(
                    f"duplicate session ID {session_id}: in {seen[session_id]!r} and {folder!r}"
                )
                continue
            seen[session_id] = folder
            id_wave = dataset.wave_of_id(session_id)
            if id_wave != wave:
                problems.append(
                    f"session {session_id} sits in {folder!r} (wave {wave!r}) but its ID "
                    f"belongs to {id_wave or 'no configured'} wave"
                )
            sessions.append(
                RawSession(session_id=session_id, wave=wave, date_folder=folder, path=path)
            )

    if len(sessions) != dataset.expected_sessions:
        problems.append(f"expected {dataset.expected_sessions} sessions, found {len(sessions)}")

    sessions.sort(key=lambda s: s.session_id)
    return Discovery(sessions=tuple(sessions), problems=tuple(problems))


def select_sessions(
    sessions: Sequence[RawSession], wanted: Sequence[int] | None
) -> tuple[tuple[RawSession, ...], tuple[int, ...]]:
    """Filter `sessions` to `wanted`, preserving order.

    Args:
        sessions: Discovered sessions.
        wanted: Session IDs to keep, or None for all.

    Returns:
        The selected sessions and any requested IDs that were not found.
    """
    if wanted is None:
        return tuple(sessions), ()
    requested = list(dict.fromkeys(wanted))
    by_id = {s.session_id: s for s in sessions}
    selected = tuple(by_id[i] for i in requested if i in by_id)
    missing = tuple(i for i in requested if i not in by_id)
    return selected, missing


def parse_session_spec(spec: str) -> tuple[int, ...]:
    """Parse a `--sessions` argument into session IDs.

    Accepts comma-separated IDs and inclusive ranges, e.g. `3,17,28` or
    `1-5,210`. Whitespace is ignored and duplicates are removed while order is
    preserved.

    Raises:
        ValueError: if a token is not an ID or a well-formed range.
    """
    ids: list[int] = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token.lstrip("-"):
            low, _, high = token.partition("-")
            if not low.strip().isdigit() or not high.strip().isdigit():
                msg = f"invalid session range {token!r}; expected e.g. 1-5"
                raise ValueError(msg)
            start, stop = int(low), int(high)
            if start > stop:
                msg = f"invalid session range {token!r}: {start} is above {stop}"
                raise ValueError(msg)
            ids.extend(range(start, stop + 1))
        elif token.isdigit():
            ids.append(int(token))
        else:
            msg = f"invalid session ID {token!r}; expected a number"
            raise ValueError(msg)
    return tuple(dict.fromkeys(ids))
