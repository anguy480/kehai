"""Run the upstream whisper-diarization tool as an external command.

Deliberately not a Python dependency of this project: the upstream repository
has heavy requirements that do not install cleanly on Apple silicon. It is
invoked as a subprocess in its own environment, and its output is read by the
same importer that handles output produced elsewhere, so there is one parser
rather than two.

See scripts/whisper_diarization_setup.md for setting it up separately.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from vc_multimodal.diarization.base import DiarizationBackend, DiarizationError, Segment
from vc_multimodal.diarization.import_backend import find_file, parse_file, resolve_patterns
from vc_multimodal.logging_setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vc_multimodal.config import WhisperDiarizationConfig
    from vc_multimodal.paths import RawSession

logger = get_logger(__name__)


class WhisperDiarizationBackend(DiarizationBackend):
    """Invokes whisper-diarization, then reads its output with the importer.

    Args:
        config: Command, language and timeout.
        output_dir: Where the tool writes, and where its output is read from.
        patterns: Filename patterns for locating that output.
        keep_text: Retain transcript text in the parsed segments.
    """

    name = "whisper_diarization"

    def __init__(
        self,
        config: WhisperDiarizationConfig,
        *,
        output_dir: Path | None,
        patterns: Sequence[str],
        keep_text: bool = True,
    ) -> None:
        """Store how to invoke the tool and where to read its output."""
        self.config = config
        self.output_dir = output_dir
        self.patterns = tuple(patterns)
        self.keep_text = keep_text

    def available(self) -> bool:
        """Whether a command and an output directory are configured."""
        return bool(self.config.command) and self.output_dir is not None

    def unavailable_reason(self) -> str:
        """Why the backend cannot run."""
        if not self.config.command:
            return "diarization.whisper_diarization.command is empty"
        if self.output_dir is None:
            return (
                "diarization.whisper_diarization.output_dir is not set; it says where "
                "the tool writes and where its output is read from"
            )
        return ""

    def version(self) -> str:
        """Identifier recorded in the manifest.

        The tool runs in its own environment, so its version is not visible
        from here; the command is recorded instead, and the tool's own commit
        has to be noted by hand.
        """
        return f"whisper-diarization/external:{' '.join(self.config.command)}"

    def segments(self, session: RawSession) -> tuple[Segment, ...]:
        """Run the tool for one session if needed, then parse its output.

        Existing output is reused rather than regenerated, since a run takes
        minutes per session.

        Raises:
            DiarizationError: if the backend is unavailable, the command fails,
                or its output cannot be found or parsed.
        """
        if not self.available() or self.output_dir is None:
            raise DiarizationError(self.unavailable_reason())

        existing = find_file(self.output_dir, session.session_id, self.patterns)
        if existing is not None:
            return parse_file(existing, keep_text=self.keep_text)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            *self.config.command,
            "-a",
            str(session.path),
            "--language",
            self.config.language,
        ]
        logger.info("session %s: running whisper-diarization externally", session.session_id)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config.timeout_seconds,
                cwd=self.output_dir,
            )
        except FileNotFoundError as exc:
            msg = (
                f"could not run {self.config.command[0]!r}; see "
                f"scripts/whisper_diarization_setup.md for setting the tool up "
                f"in its own environment"
            )
            raise DiarizationError(msg) from exc
        except subprocess.TimeoutExpired as exc:
            msg = (
                f"whisper-diarization timed out after {self.config.timeout_seconds:.0f}s "
                f"on session {session.session_id}"
            )
            raise DiarizationError(msg) from exc

        if result.returncode != 0:
            msg = (
                f"whisper-diarization exited {result.returncode} on session "
                f"{session.session_id}: {result.stderr.strip()[:300]}"
            )
            raise DiarizationError(msg)

        produced = find_file(self.output_dir, session.session_id, self.patterns)
        if produced is None:
            tried = ", ".join(resolve_patterns(self.patterns, session.session_id))
            msg = (
                f"whisper-diarization completed but no output was found for session "
                f"{session.session_id} in {self.output_dir}; tried: {tried}"
            )
            raise DiarizationError(msg)
        return parse_file(produced, keep_text=self.keep_text)
