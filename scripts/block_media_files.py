#!/usr/bin/env python3
"""Pre-commit hook: refuse to commit media, transcripts, or feature tables.

This repository sits next to clinical research data that must never leave the
lab. `.gitignore` is the first line of defence; this hook is the second, because
`git add -f` bypasses `.gitignore` but not a pre-commit hook.

Usage: block_media_files.py FILE [FILE ...]
Exits non-zero, listing every offending path, if any argument looks like data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Extensions that can carry recorded sessions or what was said in them.
BLOCKED_SUFFIXES: frozenset[str] = frozenset(
    {
        # video
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        # audio
        ".wav",
        ".m4a",
        ".mp3",
        ".flac",
        ".aac",
        ".ogg",
        ".opus",
        # transcripts / diarization
        ".srt",
        ".vtt",
        ".ass",
        ".rttm",
        # per-frame or feature dumps
        ".npy",
        ".npz",
    }
)

# Filenames that are data even though their extension is innocuous. Matched
# against the lowercased file name.
BLOCKED_NAMES: tuple[str, ...] = (
    "features.csv",
    "inventory.csv",
    "qc_report.csv",
    "labels.csv",
)

# Paths that are allowed to use a blocked name because they are synthetic
# fixtures or documentation examples.
ALLOWED_PREFIXES: tuple[str, ...] = ("tests/", "docs/")


def is_blocked(path: str) -> str | None:
    """Return a reason string if `path` must not be committed, else None."""
    p = Path(path)
    name = p.name.lower()

    if any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        return None

    if p.suffix.lower() in BLOCKED_SUFFIXES:
        return f"media/transcript extension '{p.suffix}'"

    if name in BLOCKED_NAMES:
        return f"looks like a data table ('{p.name}')"

    return None


def main(argv: list[str]) -> int:
    """Check each argument and report all violations at once."""
    violations = [(path, reason) for path in argv if (reason := is_blocked(path))]

    if not violations:
        return 0

    print("Refusing to commit files that may contain research data:\n", file=sys.stderr)
    for path, reason in violations:
        print(f"  {path}\n      -> {reason}", file=sys.stderr)
    print(
        "\nResearch data belongs under $VC_DATA_ROOT / $VC_WORK_ROOT / $VC_OUT_ROOT,\n"
        "never in this repository. If this file is genuinely a synthetic fixture,\n"
        "put it under tests/ or rename it.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
