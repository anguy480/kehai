"""Parsers for diarization output produced elsewhere.

whisper-diarization writes an SRT whose cue text carries the speaker as a
prefix, like `Speaker 0: ...`. pyannote and several other tools write RTTM.
Both are supported, because the file the lab can supply is not yet known.

The parsers are deliberately forgiving about formatting and strict about
meaning: a malformed timestamp is an error, while a missing cue index, CRLF
line endings, a byte-order mark, blank blocks or multi-line cue text are all
tolerated. A file that arrives with none of the expected speaker prefixes is
reported rather than silently treated as a single speaker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from vc_multimodal.diarization.base import DiarizationError, Segment, canonical_speaker

# `HH:MM:SS,mmm` or `HH:MM:SS.mmm`, with hours optional and any digit count for
# the fractional part.
_TIMESTAMP: Final = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d+):(?P<seconds>\d+)(?:[,.](?P<fraction>\d+))?$"
)

_ARROW: Final = re.compile(r"\s*-->\s*")

# Speaker prefixes seen in the wild: "Speaker 0:", "SPEAKER_00:", "spk 1:",
# optionally bracketed. Matched case-insensitively at the start of a cue.
# The fullwidth colon (U+FF1A) is deliberate: Japanese transcripts use it.
_SPEAKER_PREFIX: Final = re.compile(
    r"^\s*[\[\(<]?\s*(?P<label>(?:speaker|spk)[\s_\-]*\d+)\s*[\]\)>]?\s*[:\uff1a]\s*",
    re.IGNORECASE,
)

_SECONDS_PER_MINUTE: Final = 60
_SECONDS_PER_HOUR: Final = 3600
_RTTM_MIN_FIELDS: Final = 8
# A timing line splits into exactly a start and an end around the arrow.
_TIMING_PARTS: Final = 2


def parse_timestamp(text: str) -> float:
    """Parse an SRT timestamp into seconds.

    Accepts a comma or a full stop as the decimal separator, and an optional
    hours field, because tools differ on both.

    Raises:
        DiarizationError: if the timestamp is not a timestamp.
    """
    match = _TIMESTAMP.match(text.strip())
    if match is None:
        msg = f"malformed timestamp {text.strip()!r}; expected HH:MM:SS,mmm"
        raise DiarizationError(msg)

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    fraction_text = match.group("fraction") or ""
    fraction = int(fraction_text) / (10 ** len(fraction_text)) if fraction_text else 0.0
    return hours * _SECONDS_PER_HOUR + minutes * _SECONDS_PER_MINUTE + seconds + fraction


def split_speaker(text: str) -> tuple[str | None, str]:
    """Split a speaker prefix off a cue's text.

    Returns:
        The raw speaker label, or None if the cue has no prefix, and the
        remaining text.
    """
    match = _SPEAKER_PREFIX.match(text)
    if match is None:
        return None, text.strip()
    return match.group("label"), text[match.end() :].strip()


@dataclass(frozen=True, slots=True)
class SrtCue:
    """One subtitle cue, before speaker attribution."""

    start: float
    end: float
    text: str


def parse_srt_cues(content: str) -> tuple[SrtCue, ...]:
    """Parse an SRT file into cues.

    Raises:
        DiarizationError: if a cue's timing line is malformed.
    """
    normalised = content.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[SrtCue] = []

    for block in normalised.split("\n\n"):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        # The index line is optional: some writers omit it, and nothing
        # downstream uses it.
        if "-->" not in lines[0] and len(lines) > 1:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue

        parts = _ARROW.split(lines[0], maxsplit=1)
        if len(parts) != _TIMING_PARTS:  # pragma: no cover - guarded by the "-->" check
            msg = f"malformed timing line {lines[0]!r}"
            raise DiarizationError(msg)

        start, end = parse_timestamp(parts[0]), parse_timestamp(parts[1])
        cues.append(SrtCue(start=start, end=end, text=" ".join(lines[1:]).strip()))

    return tuple(cues)


def segments_from_srt(content: str, *, keep_text: bool = True) -> tuple[Segment, ...]:
    """Parse whisper-diarization SRT output into segments.

    A cue with no speaker prefix inherits the previous cue's speaker, which is
    how continuation cues are written. A file whose cues carry no prefixes at
    all is an error: treating it as one speaker would silently destroy the
    distinction every later stage depends on.

    Args:
        content: The file's text.
        keep_text: Retain the transcript text. False drops it at the parse
            boundary, so it never enters memory downstream.

    Returns:
        Segments ordered as the file listed them.

    Raises:
        DiarizationError: if the file is unparseable, empty of cues, or carries
            no speaker labels at all.
    """
    cues = parse_srt_cues(content)
    if not cues:
        msg = "the SRT file contains no cues"
        raise DiarizationError(msg)

    segments: list[Segment] = []
    current_speaker: str | None = None
    labelled = 0

    for cue in cues:
        raw_label, text = split_speaker(cue.text)
        if raw_label is not None:
            current_speaker = canonical_speaker(raw_label)
            labelled += 1
        if current_speaker is None:
            # Cues before the first label cannot be attributed to anyone.
            continue
        if cue.end <= cue.start:
            continue
        segments.append(
            Segment(
                speaker=current_speaker,
                start=cue.start,
                end=cue.end,
                text=text if keep_text else None,
            )
        )

    if labelled == 0:
        msg = (
            "no speaker labels were found in the SRT file. Cue text should start "
            "with a prefix such as 'Speaker 0:'. Treating the file as a single "
            "speaker would destroy the distinction every later stage needs, so it "
            "is refused instead."
        )
        raise DiarizationError(msg)
    if not segments:
        msg = "the SRT file contained no cues with a positive duration"
        raise DiarizationError(msg)

    return tuple(segments)


def segments_from_rttm(content: str) -> tuple[Segment, ...]:
    """Parse RTTM output, as pyannote and similar tools write it.

    RTTM carries no transcript text, which makes it the safer format to be
    handed: there is nothing sensitive in it to begin with.

    Raises:
        DiarizationError: if no usable `SPEAKER` lines are present.
    """
    segments: list[Segment] = []

    for raw_line in content.splitlines():
        fields = raw_line.split()
        if not fields or fields[0].upper() != "SPEAKER":
            continue
        if len(fields) < _RTTM_MIN_FIELDS:
            msg = f"malformed RTTM line with {len(fields)} field(s): {raw_line.strip()[:60]!r}"
            raise DiarizationError(msg)
        try:
            start = float(fields[3])
            duration = float(fields[4])
        except ValueError as exc:
            msg = f"malformed RTTM timing in {raw_line.strip()[:60]!r}"
            raise DiarizationError(msg) from exc
        if duration <= 0:
            continue
        segments.append(
            Segment(
                speaker=canonical_speaker(fields[7]),
                start=start,
                end=start + duration,
                text=None,
            )
        )

    if not segments:
        msg = "the RTTM file contains no SPEAKER lines with a positive duration"
        raise DiarizationError(msg)
    return tuple(segments)
