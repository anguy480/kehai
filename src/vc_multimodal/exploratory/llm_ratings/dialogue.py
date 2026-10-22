# Parsing ported from analysis.py in
# https://github.com/hirokitan1155/virtual_client_SRS2_analysis at commit ec07010,
# under the following licence:
#
# MIT License
#
# Copyright (c) 2026 Hiroki Tanaka
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The session dialogue, with roles marked, as the model reads it.

The transcripts, their order, the speaker parsing and the role mapping are
those of `analysis.py` (its lines 157, 194-235): files sorted by
`int(stem)`, read as UTF-8 with BOM, only lines containing `Speaker 0` or
`Speaker 1`, the speaker the integer before the first `:`, the text everything
after it with half- and full-width spaces removed, and `opposite_files` naming
the sessions in which speaker 1 is the doctor.

Each used line becomes `医師: <text>` or `参加者: <text>`, in file order, joined
by line breaks. Nothing else about the session reaches the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

DOCTOR: Final = "医師"
PARTICIPANT: Final = "参加者"

#: Sessions where speaker 1, not speaker 0, is the doctor (`opposite_files`).
OPPOSITE_FILES: Final = frozenset(
    {
        6, 8, 10, 12, 18, 25, 30, 31, 32, 42, 44, 46, 50, 52, 57, 62,
        107, 113, 140, 144, 158, 194, 210, 212, 228, 245, 261,
    }
)  # fmt: skip


class TranscriptError(RuntimeError):
    """Raised when the transcripts are not as analysis.py reads them."""


def doctor_speaker(session_id: int) -> int:
    """Which diarized speaker is the doctor in this session."""
    return 1 if session_id in OPPOSITE_FILES else 0


def transcript_files(directory: Path) -> list[Path]:
    """The `.txt` transcripts, sorted by `int(stem)` as analysis.py sorts them.

    Raises:
        TranscriptError: if the folder is missing, empty, or holds a file whose
            stem is not a session number.
    """
    if not directory.is_dir():
        msg = f"no transcript folder at {directory}"
        raise TranscriptError(msg)
    files = [path for path in directory.glob("*.txt") if path.is_file()]
    if not files:
        msg = f"no .txt transcripts in {directory}"
        raise TranscriptError(msg)
    odd = [path for path in files if not path.stem.isdigit()]
    if odd:
        msg = f"{len(odd)} .txt file(s) in {directory} are not named by session number"
        raise TranscriptError(msg)
    return sorted(files, key=lambda path: int(path.stem))


def dialogue_lines(lines: Iterable[str], session_id: int) -> list[str]:
    """One transcript's used lines, each with its role.

    Raises:
        TranscriptError: if a used line has no speaker number before its `:`.
            The message names the line number only, never its text.
    """
    doctor_id = doctor_speaker(session_id)
    rendered: list[str] = []
    for number, line in enumerate(lines, start=1):
        if "Speaker 0" not in line and "Speaker 1" not in line:
            continue
        try:
            speaker_id = int(line.split(":")[0].replace("Speaker ", ""))
        except ValueError:
            msg = f"session {session_id}, line {number}: no speaker number before the ':'"
            raise TranscriptError(msg) from None
        text = line.split(":", 1)[1].replace(" ", "").replace("　", "").rstrip("\r\n")
        role = DOCTOR if speaker_id == doctor_id else PARTICIPANT
        rendered.append(f"{role}: {text}")
    return rendered


def session_dialogue(path: Path) -> str:
    """The dialogue the model reads for the transcript at `path`.

    Raises:
        TranscriptError: if the transcript has no used line.
    """
    session_id = int(path.stem)
    with path.open(encoding="utf-8-sig") as handle:
        lines = dialogue_lines(handle, session_id)
    if not lines:
        msg = f"session {session_id}: no line names Speaker 0 or Speaker 1"
        raise TranscriptError(msg)
    return "\n".join(lines)


def session_ids(files: Sequence[Path]) -> list[int]:
    """The session numbers of transcript files."""
    return [int(path.stem) for path in files]
