"""Read diarization produced elsewhere, normally by whisper-diarization.

The preferred source. Reusing the lab manuscript's own transcripts keeps the
comparison against its text features apples-to-apples: re-diarizing locally
would confound "new modality" with "new transcripts", and a difference in
results could not be attributed to either.

Which files will arrive, and under what names, is not yet known, so matching is
pattern-driven and reports what it could not place rather than failing on the
first surprise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from vc_multimodal.diarization.base import (
    DiarizationBackend,
    DiarizationError,
    Segment,
    sort_segments,
)
from vc_multimodal.diarization.srt import segments_from_rttm, segments_from_srt
from vc_multimodal.logging_setup import get_logger

if TYPE_CHECKING:
    from vc_multimodal.paths import RawSession

logger = get_logger(__name__)

SRT_SUFFIXES: Final = (".srt", ".vtt")
RTTM_SUFFIXES: Final = (".rttm",)

# Extensions that carry no timing and so cannot be used on their own.
TIMELESS_SUFFIXES: Final = (".txt", ".json", ".tsv", ".csv")

# How many unmatched filenames to name before eliding the rest.
_MAX_LISTED_FILES: Final = 10


def resolve_patterns(patterns: Sequence[str], session_id: int) -> tuple[str, ...]:
    """Fill `{session_id}` into each configured filename pattern."""
    return tuple(pattern.format(session_id=session_id) for pattern in patterns)


def find_file(import_dir: Path, session_id: int, patterns: Sequence[str]) -> Path | None:
    """Find one session's diarization file, trying each pattern in order.

    Patterns are tried in the order configured, so a preferred format can be
    listed first.
    """
    for relative in resolve_patterns(patterns, session_id):
        candidate = import_dir / relative
        if candidate.is_file():
            return candidate
    return None


def parse_file(path: Path, *, keep_text: bool = True) -> tuple[Segment, ...]:
    """Parse one diarization file, choosing a parser by extension.

    Raises:
        DiarizationError: if the extension carries no timing, is unrecognised,
            or the file cannot be read or parsed.
    """
    suffix = path.suffix.lower()
    if suffix in TIMELESS_SUFFIXES:
        msg = (
            f"{path.name} carries no timestamps, so it cannot provide segment "
            f"boundaries. whisper-diarization also writes an .srt alongside it; "
            f"point diarization.import_dir at that instead."
        )
        raise DiarizationError(msg)
    if suffix not in (*SRT_SUFFIXES, *RTTM_SUFFIXES):
        msg = f"unrecognised diarization format {suffix!r} for {path.name}"
        raise DiarizationError(msg)

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Transcripts from Japanese tooling are occasionally not UTF-8.
        content = path.read_text(encoding="utf-8", errors="replace")
        logger.warning("%s is not valid UTF-8; undecodable characters were replaced", path.name)
    except OSError as exc:
        msg = f"could not read {path.name}: {exc}"
        raise DiarizationError(msg) from exc

    if suffix in RTTM_SUFFIXES:
        return sort_segments(segments_from_rttm(content))
    return sort_segments(segments_from_srt(content, keep_text=keep_text))


@dataclass(frozen=True, slots=True)
class ImportScan:
    """What a directory of diarization output contains.

    Attributes:
        matched: Session ID to the file that will be used for it.
        missing: Sessions with no file at all.
        unmatched: Files in the directory that no session claimed. Reported
            because a filename that does not match the configured patterns is
            far more likely to be a naming difference than a spare file.
        timeless: Files that carry no timestamps, which cannot be used alone.
    """

    matched: Mapping[int, Path] = field(default_factory=dict)
    missing: tuple[int, ...] = ()
    unmatched: tuple[str, ...] = ()
    timeless: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every session was matched and nothing was left over."""
        return not self.missing and not self.unmatched

    def report_lines(self) -> list[str]:
        """Human-readable summary of the match, filenames included.

        Filenames are session IDs, not content, so naming them is what makes a
        mismatch fixable.
        """
        lines = [f"matched {len(self.matched)} session(s) to diarization files"]
        if self.missing:
            lines.append(f"  no file for {len(self.missing)} session(s): {list(self.missing)}")
        if self.unmatched:
            shown = list(self.unmatched[:_MAX_LISTED_FILES])
            hidden = len(self.unmatched) - _MAX_LISTED_FILES
            more = "" if hidden <= 0 else f" (+{hidden} more)"
            lines.append(f"  {len(self.unmatched)} file(s) matched no session: {shown}{more}")
            lines.append("    check diarization.import_patterns against these names")
        if self.timeless:
            lines.append(
                f"  {len(self.timeless)} file(s) carry no timestamps and were ignored: "
                f"{list(self.timeless[:5])}"
            )
        return lines


def scan_import_dir(
    import_dir: Path, session_ids: Sequence[int], patterns: Sequence[str]
) -> ImportScan:
    """Match every session to a file and report whatever is left over.

    Raises:
        DiarizationError: if the directory does not exist.
    """
    if not import_dir.is_dir():
        msg = (
            f"diarization.import_dir points at {import_dir}, which is not a directory. "
            f"Keep imported diarization output under $VC_WORK_ROOT."
        )
        raise DiarizationError(msg)

    matched: dict[int, Path] = {}
    missing: list[int] = []
    for session_id in session_ids:
        found = find_file(import_dir, session_id, patterns)
        if found is None:
            missing.append(session_id)
        else:
            matched[session_id] = found

    claimed = {path.resolve() for path in matched.values()}
    unmatched: list[str] = []
    timeless: list[str] = []
    for path in sorted(import_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.resolve() in claimed:
            continue
        if path.suffix.lower() in TIMELESS_SUFFIXES:
            timeless.append(path.name)
        else:
            unmatched.append(path.name)

    return ImportScan(
        matched=matched,
        missing=tuple(missing),
        unmatched=tuple(unmatched),
        timeless=tuple(timeless),
    )


class ImportBackend(DiarizationBackend):
    """Reads diarization from files produced outside this project.

    Args:
        import_dir: Directory holding the files.
        patterns: Filename patterns to try, in order, with `{session_id}`.
        keep_text: Retain transcript text in the parsed segments.
    """

    name = "import"

    def __init__(
        self,
        import_dir: Path,
        patterns: Sequence[str],
        *,
        keep_text: bool = True,
    ) -> None:
        """Store where to read from and how to match filenames."""
        self.import_dir = Path(import_dir)
        self.patterns = tuple(patterns)
        self.keep_text = keep_text

    def available(self) -> bool:
        """Whether the configured directory exists."""
        return self.import_dir.is_dir()

    def unavailable_reason(self) -> str:
        """Why the backend cannot run."""
        if not self.available():
            return (
                f"{self.import_dir} is not a directory. Set diarization.import_dir to "
                f"the folder holding the whisper-diarization output, under $VC_WORK_ROOT."
            )
        return ""

    def version(self) -> str:
        """Identifier recorded in the manifest.

        The output was produced elsewhere, so this records only that it was
        imported and from where; the producing tool's own version is not
        recoverable from its output and has to be recorded by hand.
        """
        return f"import/{self.import_dir.name}"

    def scan(self, session_ids: Sequence[int]) -> ImportScan:
        """Match sessions to files without parsing any of them."""
        return scan_import_dir(self.import_dir, session_ids, self.patterns)

    def segments(self, session: RawSession) -> tuple[Segment, ...]:
        """Parse one session's diarization file.

        Raises:
            DiarizationError: if no file matches, or it cannot be parsed.
        """
        if not self.available():
            raise DiarizationError(self.unavailable_reason())

        path = find_file(self.import_dir, session.session_id, self.patterns)
        if path is None:
            tried = ", ".join(resolve_patterns(self.patterns, session.session_id))
            msg = (
                f"no diarization file for session {session.session_id} in "
                f"{self.import_dir}; tried: {tried}"
            )
            raise DiarizationError(msg)

        return parse_file(path, keep_text=self.keep_text)
