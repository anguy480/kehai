"""Findings a person confirmed by looking, which no measurement can establish.

The pipeline can tell you that 99.2% of the frames in a session produced no
usable face. It cannot tell you *why*: a crop pointing at the wrong tile, a
detector failing on an unusual face, and a camera too out of focus to track all
look identical in the numbers, and they call for completely different responses
- fix the crop, change the backend, or accept that this session has no facial
features.

Only a person who watched the recording can settle that, and until it is
written down it lives in their head. Then the bundle goes to someone else, who
sees a flagged session with no explanation and has to either guess or ask.

So a confirmed finding is recorded here, as a row per session and modality, and
from then on it travels with everything: the feature table, the QC report, the
handoff README and the manifest.

Two properties make these notes safe to act on automatically:

* **They are scoped to a modality.** A blurry camera makes the facial features
  meaningless and leaves the audio untouched, so `face=unavailable` blanks the
  facial columns for that session and nothing else. Dropping the whole session
  would throw away good data.
* **They cannot be label-driven.** The half of the project that records them
  never holds a questionnaire score (docs/decisions/0001), so an exclusion
  written here cannot have been chosen to improve a result. That is what makes
  a human exclusion acceptable in a pre-registered analysis at all.

The reason text is a technical description of the recording - focus, framing,
audio quality - and never anything about the person in it.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd

from vc_multimodal.io_utils import read_csv, write_csv
from vc_multimodal.logging_setup import get_logger

logger = get_logger(__name__)

NOTES_FILENAME: Final = "qc_notes.csv"

COLUMNS: Final = (
    "session_id",
    "modality",
    "status",
    "reason",
    "recorded_by",
    "recorded_on",
)

#: What a note can say about a modality.
#:
#: `unavailable` is the only one that changes the feature table: those columns
#: are blanked, because a measurement nobody should use is worse than an absent
#: one - it will be modelled by whoever did not read the note.
STATUS_UNAVAILABLE: Final = "unavailable"
STATUS_DEGRADED: Final = "degraded"
STATUS_NOTE: Final = "note"
STATUSES: Final = (STATUS_UNAVAILABLE, STATUS_DEGRADED, STATUS_NOTE)

#: Modality names a note may use, and the feature families each covers.
#: `face` covers both windows, since a camera problem does not respect them.
MODALITY_FAMILIES: Final[Mapping[str, tuple[str, ...]]] = {
    "face": ("face_speaking", "face_listening"),
    "face_speaking": ("face_speaking",),
    "face_listening": ("face_listening",),
    "audio": ("turns", "prosody"),
    "turns": ("turns",),
    "prosody": ("prosody",),
    "text": ("text",),
    "all": ("face_speaking", "face_listening", "turns", "prosody", "text"),
}

#: Shortest reason that can say what was confirmed. "bad" is not a finding.
MIN_REASON_CHARS: Final = 15


class QcNoteError(ValueError):
    """Raised when a QC note cannot be recorded or used."""


@dataclass(frozen=True, slots=True)
class QcNote:
    """One confirmed finding about one session and one modality."""

    session_id: int
    modality: str
    status: str
    reason: str
    recorded_by: str
    recorded_on: str

    @property
    def families(self) -> tuple[str, ...]:
        """Feature families this note covers."""
        return MODALITY_FAMILIES[self.modality]

    @property
    def blanks_features(self) -> bool:
        """Whether the covered features must not be used."""
        return self.status == STATUS_UNAVAILABLE

    @property
    def label(self) -> str:
        """Compact form for a QC column, e.g. `face=unavailable`."""
        return f"{self.modality}={self.status}"

    def row(self) -> dict[str, object]:
        """As written to the notes file."""
        return {
            "session_id": self.session_id,
            "modality": self.modality,
            "status": self.status,
            "reason": self.reason,
            "recorded_by": self.recorded_by,
            "recorded_on": self.recorded_on,
        }


@dataclass(frozen=True, slots=True)
class QcNotes:
    """Every confirmed finding, indexed by session."""

    notes: tuple[QcNote, ...] = ()
    source: str = ""

    def __bool__(self) -> bool:
        """Whether there are any notes."""
        return bool(self.notes)

    def __len__(self) -> int:
        """How many notes there are."""
        return len(self.notes)

    def for_session(self, session_id: int) -> tuple[QcNote, ...]:
        """Notes about one session, in the order recorded."""
        return tuple(note for note in self.notes if note.session_id == session_id)

    def unavailable_families(self, session_id: int) -> tuple[str, ...]:
        """Families whose features must not be used for this session."""
        found: list[str] = []
        for note in self.for_session(session_id):
            if note.blanks_features:
                found.extend(note.families)
        return tuple(dict.fromkeys(found))

    def labels(self, session_id: int) -> str:
        """The compact QC column value, e.g. `face=unavailable`."""
        return ";".join(note.label for note in self.for_session(session_id))

    def reasons(self, session_id: int) -> str:
        """The reasons, joined, for the QC column."""
        return " | ".join(note.reason for note in self.for_session(session_id))

    @property
    def sessions(self) -> tuple[int, ...]:
        """Every session with a note."""
        return tuple(dict.fromkeys(note.session_id for note in self.notes))

    def report_lines(self) -> list[str]:
        """Human-readable, for a stage summary or the handoff README."""
        if not self.notes:
            return []
        lines = [f"human-confirmed QC notes ({len(self.notes)}):"]
        for note in sorted(self.notes, key=lambda n: (n.session_id, n.modality)):
            lines.append(
                f"  session {note.session_id}: {note.modality} {note.status.upper()}"
                f" - {note.reason} (confirmed by {note.recorded_by} on {note.recorded_on})"
            )
        return lines

    def markdown_rows(self) -> list[str]:
        """As a markdown table body, for the handoff README."""
        return [
            f"| {note.session_id} | {note.modality} | {note.status} | {note.reason} "
            f"| {note.recorded_by} | {note.recorded_on} |"
            for note in sorted(self.notes, key=lambda n: (n.session_id, n.modality))
        ]


def notes_path(work_root: Path, name: str = NOTES_FILENAME) -> Path:
    """Where the notes file lives."""
    return work_root / name


def _validate(
    session_id: object, modality: object, status: object, reason: object, *, where: str
) -> tuple[int, str, str, str]:
    """Check one note's fields, with messages that say what is allowed."""
    try:
        session = int(str(session_id).strip())
    except (TypeError, ValueError) as exc:
        msg = f"{where}: session_id {session_id!r} is not a session number"
        raise QcNoteError(msg) from exc

    name = str(modality).strip().lower()
    if name not in MODALITY_FAMILIES:
        msg = (
            f"{where}: modality {name!r} is not one of {sorted(MODALITY_FAMILIES)}. "
            f"A note has to say which features it is about, because a finding about "
            f"the camera says nothing about the audio."
        )
        raise QcNoteError(msg)

    state = str(status).strip().lower()
    if state not in STATUSES:
        msg = f"{where}: status {state!r} is not one of {list(STATUSES)}"
        raise QcNoteError(msg)

    text = " ".join(str(reason).split())
    if len(text) < MIN_REASON_CHARS:
        msg = (
            f"{where}: the reason {text!r} is too short to be a finding. Say what was "
            f"confirmed and how, in enough detail that someone who did not watch the "
            f"recording can act on it."
        )
        raise QcNoteError(msg)
    return session, name, state, text


def load(path: Path) -> QcNotes:
    """Read the notes file, or return nothing if there is none.

    Raises:
        QcNoteError: if the file exists but cannot be used. A malformed note is
            refused rather than skipped: a note nobody notices is dropping is
            worse than no note, because the finding it records is then invisible
            while appearing to be recorded.
    """
    if not path.exists():
        return QcNotes()
    try:
        frame = read_csv(path)
    except (OSError, ValueError) as exc:
        msg = f"could not read {path.name}: {exc}"
        raise QcNoteError(msg) from exc

    frame.columns = [str(name).strip().lower() for name in frame.columns]
    missing = [name for name in ("session_id", "modality", "status", "reason") if name not in frame]
    if missing:
        msg = (
            f"{path.name} is missing column(s) {missing}; it needs {list(COLUMNS)} "
            f"(recorded_by and recorded_on may be blank)"
        )
        raise QcNoteError(msg)

    collected: list[QcNote] = []
    for position, row in enumerate(frame.to_dict(orient="records"), start=2):
        session, modality, status, reason = _validate(
            row.get("session_id"),
            row.get("modality"),
            row.get("status"),
            row.get("reason"),
            where=f"{path.name} line {position}",
        )
        collected.append(
            QcNote(
                session_id=session,
                modality=modality,
                status=status,
                reason=reason,
                recorded_by=str(row.get("recorded_by") or "").strip() or "unknown",
                recorded_on=str(row.get("recorded_on") or "").strip() or "unknown",
            )
        )

    duplicated = sorted(
        {
            (note.session_id, note.modality)
            for note in collected
            if sum(
                1
                for other in collected
                if other.session_id == note.session_id and other.modality == note.modality
            )
            > 1
        }
    )
    if duplicated:
        msg = (
            f"{path.name} has more than one note for {duplicated}. Two notes about one "
            f"modality of one session can contradict each other, so update the row "
            f"rather than adding another."
        )
        raise QcNoteError(msg)

    logger.info("%d human-confirmed QC note(s) from %s", len(collected), path.name)
    return QcNotes(notes=tuple(collected), source=path.name)


def _default_author() -> str:
    """Who is recording this, from git or the environment."""
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - defensive
        pass
    return os.environ.get("USER", "unknown")


def record(
    path: Path,
    *,
    session_id: int,
    modality: str,
    status: str,
    reason: str,
    recorded_by: str | None = None,
    now: datetime | None = None,
    replace: bool = False,
) -> QcNote:
    """Add a note, keeping the file sorted and valid.

    Args:
        path: The notes file, created if absent.
        session_id: Session the finding is about.
        modality: Which features it concerns.
        status: One of `STATUSES`.
        reason: What was confirmed, and how.
        recorded_by: Who confirmed it; taken from git or `$USER` if omitted.
        now: For deterministic tests.
        replace: Overwrite an existing note for the same session and modality.

    Returns:
        The note as recorded.

    Raises:
        QcNoteError: if the note is invalid, or would duplicate one without
            `replace`.
    """
    session, name, state, text = _validate(session_id, modality, status, reason, where="the note")
    existing = load(path) if path.exists() else QcNotes()
    clash = [
        note for note in existing.notes if note.session_id == session and note.modality == name
    ]
    if clash and not replace:
        msg = (
            f"session {session} already has a {name} note, recorded by "
            f"{clash[0].recorded_by} on {clash[0].recorded_on}: {clash[0].reason!r}. "
            f"Pass --replace to change it."
        )
        raise QcNoteError(msg)

    note = QcNote(
        session_id=session,
        modality=name,
        status=state,
        reason=text,
        recorded_by=(recorded_by or _default_author()).strip() or "unknown",
        recorded_on=(now or datetime.now(UTC)).strftime("%Y-%m-%d"),
    )
    kept = [n for n in existing.notes if (n.session_id, n.modality) != (session, name)]
    rows = [n.row() for n in sorted([*kept, note], key=lambda n: (n.session_id, n.modality))]
    write_csv(path, pd.DataFrame(rows, columns=list(COLUMNS)))
    logger.info("recorded %s note for session %s in %s", name, session, path.name)
    return note


def families_of(modalities: Iterable[str]) -> tuple[str, ...]:
    """Every feature family covered by these modality names."""
    found: list[str] = []
    for name in modalities:
        found.extend(MODALITY_FAMILIES.get(name, ()))
    return tuple(dict.fromkeys(found))


def columns_for_families(columns: Sequence[str], families: Sequence[str]) -> tuple[str, ...]:
    """Feature columns belonging to any of `families`."""
    wanted = set(families)
    return tuple(
        str(column)
        for column in columns
        if "__" in str(column) and str(column).split("__", 1)[0] in wanted
    )
