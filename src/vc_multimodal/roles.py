"""Mapping diarization's anonymous speakers to conversational roles.

Diarization returns `SPEAKER_00` and `SPEAKER_01`. Every feature after it
depends on knowing which is the participant: prosody is participant-only, and
the speaking/listening split for facial features inverts if the two are
swapped. Nothing here guesses. A role mapping either exists, as evidence
recorded by `vc assign-speakers`, or the caller is told to produce one.

Two sources, in order:

1. `$VC_WORK_ROOT/roles/<session_id>.json`, written by `vc assign-speakers`
   along with the evidence behind it.
2. `$VC_WORK_ROOT/roles.csv`, a table a human writes by hand. It exists so a
   few sessions can be piloted before the psychiatrist reference clip is
   available. It is an explicit human judgement, not an assumption, and which
   source was used is recorded in QC.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from vc_multimodal.contracts import ROLES, ROLES_SCHEMA, validate
from vc_multimodal.io_utils import read_csv, write_json
from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

ROLES_DIRNAME: Final = "roles"
MANUAL_ROLES_FILENAME: Final = "roles.csv"

ROLE_PARTICIPANT: Final = "participant"
ROLE_PSYCHIATRIST: Final = "psychiatrist"

SOURCE_ASSIGNED: Final = "assigned"
SOURCE_MANUAL: Final = "manual"


class RolesUnavailableError(RuntimeError):
    """Raised when no role mapping exists for a session."""


@dataclass(frozen=True, slots=True)
class RoleMapping:
    """Which speaker plays which role in one session."""

    session_id: int
    by_speaker: Mapping[str, str]
    source: str

    def role_of(self, speaker: str) -> str:
        """The role of `speaker`, or `unknown`."""
        return self.by_speaker.get(speaker, "unknown")

    def speakers_for(self, role: str) -> tuple[str, ...]:
        """Every speaker mapped to `role`.

        More than one is possible and is not an error: a diarizer that split a
        single voice produces two labels for one person.
        """
        return tuple(
            speaker for speaker, assigned in sorted(self.by_speaker.items()) if assigned == role
        )

    @property
    def has_participant(self) -> bool:
        """Whether any speaker is the participant."""
        return bool(self.speakers_for(ROLE_PARTICIPANT))

    @property
    def has_psychiatrist(self) -> bool:
        """Whether any speaker is the psychiatrist."""
        return bool(self.speakers_for(ROLE_PSYCHIATRIST))


def roles_dir(work_root: Path) -> Path:
    """Directory holding per-session role assignments."""
    return work_root / ROLES_DIRNAME


def roles_path(work_root: Path, session_id: int) -> Path:
    """Where one session's role assignment is written."""
    return roles_dir(work_root) / f"{session_id}.json"


def manual_roles_path(work_root: Path) -> Path:
    """Where a hand-written role table would live."""
    return work_root / MANUAL_ROLES_FILENAME


def write_role_mapping(
    work_root: Path,
    session_id: int,
    by_speaker: Mapping[str, str],
    *,
    evidence: Mapping[str, Any] | None = None,
) -> Path:
    """Record a role assignment, with whatever evidence produced it.

    Raises:
        ValueError: if a role is not one of the known roles.
    """
    unknown = sorted(set(by_speaker.values()) - set(ROLES))
    if unknown:
        msg = f"unknown role(s) {unknown}; expected any of {list(ROLES)}"
        raise ValueError(msg)

    payload: dict[str, Any] = {
        "session_id": session_id,
        "roles": dict(sorted(by_speaker.items())),
        "source": SOURCE_ASSIGNED,
    }
    if evidence:
        payload["evidence"] = dict(evidence)
    return write_json(roles_path(work_root, session_id), payload)


def load_manual_roles(path: Path) -> dict[int, dict[str, str]]:
    """Load a hand-written `session_id,speaker,role` table.

    Raises:
        RolesUnavailableError: if the file is unreadable or malformed.
    """
    try:
        frame = read_csv(path)
    except (OSError, ValueError) as exc:
        msg = f"could not read {path.name}: {exc}"
        raise RolesUnavailableError(msg) from exc

    missing = [column for column in ("session_id", "speaker", "role") if column not in frame]
    if missing:
        msg = (
            f"{path.name} is missing column(s) {missing}; it must have session_id, speaker and role"
        )
        raise RolesUnavailableError(msg)

    try:
        validate(frame, ROLES_SCHEMA, context=path.name)
    except Exception as exc:
        msg = f"{path.name} is not a valid role table: {exc}"
        raise RolesUnavailableError(msg) from exc

    mapping: dict[int, dict[str, str]] = {}
    for session_id, speaker, role in zip(
        frame["session_id"], frame["speaker"], frame["role"], strict=True
    ):
        mapping.setdefault(int(session_id), {})[str(speaker)] = str(role)
    return mapping


def load_role_mapping(work_root: Path, session_id: int) -> RoleMapping:
    """Load one session's role mapping from whichever source has it.

    Raises:
        RolesUnavailableError: if neither source covers this session, with
            instructions for producing one.
    """
    assigned = roles_path(work_root, session_id)
    if assigned.is_file():
        try:
            payload = json.loads(assigned.read_text(encoding="utf-8"))
            by_speaker = {str(k): str(v) for k, v in payload["roles"].items()}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            msg = f"could not read {assigned.name}: {exc}"
            raise RolesUnavailableError(msg) from exc
        return RoleMapping(
            session_id=session_id,
            by_speaker=by_speaker,
            source=str(payload.get("source", SOURCE_ASSIGNED)),
        )

    manual = manual_roles_path(work_root)
    if manual.is_file():
        table = load_manual_roles(manual)
        if session_id in table:
            logger.info("session %s: using the hand-written role table", session_id)
            return RoleMapping(
                session_id=session_id, by_speaker=table[session_id], source=SOURCE_MANUAL
            )

    msg = (
        f"no role mapping for session {session_id}. Run `vc assign-speakers`, which "
        f"needs a psychiatrist reference clip in $VC_WORK_ROOT/reference/. To pilot "
        f"a few sessions before that exists, write {MANUAL_ROLES_FILENAME} in "
        f"$VC_WORK_ROOT with columns session_id,speaker,role using the labels from "
        f"diarization_qc.csv."
    )
    raise RolesUnavailableError(msg)


def spans_by_role(
    frame: pd.DataFrame, mapping: RoleMapping
) -> dict[str, list[tuple[float, float]]]:
    """Group a speech table's spans by role rather than by speaker."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for speaker, start, end in zip(frame["speaker"], frame["start_s"], frame["end_s"], strict=True):
        role = mapping.role_of(str(speaker))
        grouped.setdefault(role, []).append((float(start), float(end)))
    return grouped
