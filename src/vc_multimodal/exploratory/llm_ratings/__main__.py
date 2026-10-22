"""`python -m vc_multimodal.exploratory.llm_ratings pilot | rate`.

Both read the transcripts and no label, so the project owner runs them, from a
clean checkout. Before anything else they check the prompt against the
pre-registered SHA-256 and refuse to go on with no backend set, so no
transcript is read and no call is made until `backend` is committed in
`config/exploratory/llm_ratings.yaml`.

`pilot` rates the three lowest session numbers and prints counts only: parse
success, retries, failures, tokens and time, and no rating. `rate` rates every
session, reusing the pilot's cached calls, and writes the feature table and an
aggregate summary; it prints the same counts and each item's ICC.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import typer

from vc_multimodal.config import ConfigError, load_config
from vc_multimodal.exploratory.llm_ratings import rater
from vc_multimodal.exploratory.llm_ratings.backend import BackendUnavailableError, make_backend
from vc_multimodal.exploratory.llm_ratings.dialogue import TranscriptError, transcript_files
from vc_multimodal.exploratory.llm_ratings.plan import (
    REPO_ROOT,
    PlanError,
    file_sha256,
    load_plan,
    load_prompt,
)
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.paths import PathError, load_env, resolve_roots
from vc_multimodal.provenance import git_state

EXIT_SETUP_ERROR: Final = 2
PLAN_FILE: Final = "config/exploratory/llm_ratings.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vc_multimodal.exploratory.llm_ratings")
    steps = parser.add_subparsers(dest="step", required=True)
    for name, text in (
        ("pilot", "Rate the three lowest session numbers; print counts only, no rating."),
        ("rate", "Rate every session; write the feature table and an aggregate summary."),
    ):
        step = steps.add_parser(name, help=text)
        step.add_argument(
            "--transcripts",
            type=Path,
            default=None,
            help="Folder of .txt transcripts. Default: diarization.import_dir under the work root.",
        )
        step.add_argument(
            "--out",
            type=Path,
            default=None,
            help="Where the outputs go. Default: exploratory/llm_ratings under the output root.",
        )
        step.add_argument(
            "--jobs", type=int, default=1, help="Sessions rated at once (default: 1)."
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pilot or the full run and print its counts."""
    args = _parser().parse_args(argv)
    load_env()
    try:
        lines = _rate(args, pilot=args.step == "pilot")
    except (
        PlanError,
        TranscriptError,
        ConfigError,
        PathError,
        rater.RaterError,
        BackendUnavailableError,
    ) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    for line in lines:
        typer.echo(line)
    return 0


def _rate(args: argparse.Namespace, *, pilot: bool) -> list[str]:
    """Check everything that needs no transcript, then rate.

    Raises:
        PlanError: if the prompt is not the pre-registered one, no backend is
            set, or the backend is not fully configured.
        RaterError: if the checkout is not clean.
    """
    plan = load_plan()
    prompt = load_prompt(plan.prompt_path, plan.prompt_sha256)
    endpoint = plan.endpoint()
    git = git_state(REPO_ROOT)
    if git is None or git.is_dirty:
        msg = (
            f"rate from a clean checkout of {REPO_ROOT} (no uncommitted or untracked files), "
            f"so the backend and prompt used are the committed ones"
        )
        raise rater.RaterError(msg)

    config = load_config()
    roots = resolve_roots(require_data=False)
    transcripts: Path | None = args.transcripts
    if transcripts is None:
        if not config.diarization.import_dir:
            msg = "diarization.import_dir is not set; pass --transcripts"
            raise rater.RaterError(msg)
        transcripts = roots.work / config.diarization.import_dir
    files = transcript_files(transcripts)
    backend = make_backend(endpoint, plan.request)
    out_dir: Path = args.out or roots.out_path(*rater.OUT_DIRNAME)

    log = configure_logging("INFO", log_file=log_file_path(out_dir, rater.STAGE))
    if log is not None:
        typer.echo(f"log: {log}")
    run_info = {
        "prereg": "docs/prereg/llm-ratings.md",
        "commit": git.commit,
        "plan_sha256": file_sha256(REPO_ROOT / PLAN_FILE),
        "prompt_sha256": prompt.sha256,
        "backend": endpoint.backend,
        "model": endpoint.model,
        "base_url": endpoint.base_url,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    cache = rater.Cache(roots.work_path(*rater.CACHE_DIRNAME), prompt.sha256, endpoint.model)
    result = rater.run(
        files, prompt, backend, cache, out_dir, pilot=pilot, run_info=run_info, jobs=args.jobs
    )
    return result.report_lines()


if __name__ == "__main__":
    raise SystemExit(main())
