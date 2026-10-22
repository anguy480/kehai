"""`python -m vc_multimodal.exploratory.llm_ratings pilot | rate | estimate | analyze`.

`pilot` and `rate` read the transcripts and no label, so the project owner runs
them, from a clean checkout. Before anything else they check the prompt against
the pre-registered SHA-256 and refuse to go on with no backend set, so no
transcript is read and no call is made until `backend` is committed in
`config/exploratory/llm_ratings.yaml`.

`pilot` rates the three lowest session numbers and prints counts only: parse
success, retries, failures, tokens and time, and no rating. `rate` rates every
session, reusing the pilot's cached calls, and writes the feature table and an
aggregate summary; it prints the same counts and each item's ICC.

`estimate` times the analysis on random targets and reads no label. `analyze`
(step 3) reads the labels, so only the label holder runs it, from a clean
checkout; it prints coverage and the files it wrote, not results.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import typer

from vc_multimodal.config import ConfigError, load_config
from vc_multimodal.exploratory.llm_ratings import analyze, rater
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
from vc_multimodal.paths import DataRoots, PathError, load_env, resolve_roots
from vc_multimodal.provenance import git_state

EXIT_SETUP_ERROR: Final = 2
PLAN_FILE: Final = "config/exploratory/llm_ratings.yaml"
UNBLINDED_DIR: Final = "unblinded"
#: Where step 2 of the tokenization rerun wrote the text table, under the output root.
TOKENIZATION_DIRNAME: Final = ("exploratory", "tokenization")


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
    for name, text in (
        ("estimate", "Time the analysis on random targets. Reads no label."),
        ("analyze", "Step 3: the pre-registered analysis. Reads the labels."),
    ):
        step = steps.add_parser(name, help=text)
        step.add_argument(
            "--features",
            type=Path,
            default=None,
            help="The rater's feature table. Default: exploratory/llm_ratings under the "
            "output root.",
        )
        step.add_argument(
            "--text-features",
            type=Path,
            default=None,
            help="The 20 text features from the tokenization rerun. Default: "
            "exploratory/tokenization under the output root.",
        )
        step.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
        if name == "estimate":
            step.add_argument("--probe", type=int, default=4, help="Permutations to time.")
        else:
            step.add_argument("--labels", type=Path, required=True, help="Labels CSV.")
            step.add_argument(
                "--out",
                type=Path,
                default=None,
                help="Results folder. Default: unblinded/<date>_llm_ratings_<commit> under "
                "the output root.",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step and print its summary."""
    args = _parser().parse_args(argv)
    load_env()
    try:
        if args.step in ("pilot", "rate"):
            lines = _rate(args, pilot=args.step == "pilot")
        else:
            lines = _analysis(args)
    except (
        PlanError,
        TranscriptError,
        ConfigError,
        PathError,
        rater.RaterError,
        BackendUnavailableError,
        analyze.AnalyzeError,
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


def _analysis(args: argparse.Namespace) -> list[str]:
    """`estimate`, or `analyze` from a clean checkout only.

    Raises:
        AnalyzeError: if an input cannot be found, the checkout is not clean,
            or there is nowhere to write.
    """
    roots: DataRoots | None
    try:
        roots = resolve_roots(require_data=False)
    except PathError:
        roots = None
    planned = analyze.load_inputs_plan()
    llm_path = args.features or _under_out(
        roots, rater.OUT_DIRNAME, planned[analyze.LLM_TABLE][0], "--features"
    )
    text_path = args.text_features or _under_out(
        roots, TOKENIZATION_DIRNAME, planned[analyze.TEXT_TABLE][0], "--text-features"
    )
    if args.step == "estimate":
        return analyze.estimate_runtime(llm_path, text_path, jobs=args.jobs, n_probe=args.probe)

    git = git_state(REPO_ROOT)
    if git is None or git.is_dirty:
        msg = (
            f"run the analysis from a clean checkout of {REPO_ROOT} "
            f"(no uncommitted or untracked files)"
        )
        raise analyze.AnalyzeError(msg)
    out_dir: Path | None = args.out
    if out_dir is None:
        if roots is None:
            msg = "VC_OUT_ROOT is not set, so say where results go with --out"
            raise analyze.AnalyzeError(msg)
        out_dir = roots.out / UNBLINDED_DIR / f"{datetime.now():%Y%m%d}_llm_ratings_{git.short}"
    analyze.refuse_used_folder(out_dir)
    log = configure_logging("INFO", log_file=log_file_path(out_dir, analyze.STAGE))
    lines = [f"log: {log}"] if log is not None else []
    result = analyze.run(
        llm_path, text_path, args.labels, out_dir, commit=git.commit, jobs=args.jobs
    )
    return lines + result.report_lines()


def _under_out(roots: DataRoots | None, folder: Sequence[str], name: str, option: str) -> Path:
    """A table's default place under the output root.

    Raises:
        AnalyzeError: if the output root is not set.
    """
    if roots is None:
        msg = f"VC_OUT_ROOT is not set, so say where {name} is with {option}"
        raise analyze.AnalyzeError(msg)
    return roots.out.joinpath(*folder, name)


if __name__ == "__main__":
    raise SystemExit(main())
