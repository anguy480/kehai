"""`python -m vc_multimodal.exploratory.tokenization features | estimate | analyze`.

`features` (step 2) reads the transcripts and no label, and prints aggregate
counts only. `estimate` times step 3 on random targets and reads no label.
`analyze` (step 3) reads the labels, so only the label holder runs it, from a
clean checkout; it prints coverage and the files it wrote, not results.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import typer
from janome.tokenizer import Tokenizer

from vc_multimodal.config import ConfigError, load_config
from vc_multimodal.exploratory.tokenization import analyze
from vc_multimodal.exploratory.tokenization import run as tokenization
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.paths import DataRoots, PathError, load_env, resolve_roots
from vc_multimodal.provenance import git_state

EXIT_REPRODUCTION_FAILED: Final = 1
EXIT_SETUP_ERROR: Final = 2
REPO_ROOT: Final = Path(__file__).resolve().parents[4]
UNBLINDED_DIR: Final = "unblinded"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vc_multimodal.exploratory.tokenization")
    steps = parser.add_subparsers(dest="step", required=True)
    features = steps.add_parser(
        "features", help="Check the port against nlp_features.csv, then compute both variants."
    )
    features.add_argument(
        "--transcripts",
        type=Path,
        default=None,
        help="Folder of .txt transcripts. Default: diarization.import_dir under the work root.",
    )
    features.add_argument(
        "--nlp-features",
        type=Path,
        default=None,
        help="The manuscript's table. Default: model.text_features.path under the work root.",
    )
    features.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where the tables go. Default: exploratory/tokenization under the output root.",
    )
    for name, text in (
        ("estimate", "Time step 3 on random targets. Reads no label."),
        ("analyze", "Step 3: the pre-registered analysis. Reads the labels."),
    ):
        step = steps.add_parser(name, help=text)
        step.add_argument(
            "--features-dir",
            type=Path,
            default=None,
            help="Step 2's tables. Default: exploratory/tokenization under the output root.",
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
                help="Results folder. Default: unblinded/<date>_tokenization_<commit> under "
                "the output root.",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step and print its summary."""
    args = _parser().parse_args(argv)
    load_env()
    if args.step == "features":
        return _features(args)
    roots: DataRoots | None
    try:
        roots = resolve_roots(require_data=False)
    except PathError:
        roots = None
    features_dir = args.features_dir or (
        roots.out_path(*tokenization.OUT_DIRNAME, create_parent=False) if roots else None
    )
    try:
        if features_dir is None:
            msg = "VC_OUT_ROOT is not set, so say where step 2's tables are with --features-dir"
            raise analyze.AnalyzeError(msg)
        lines = (
            analyze.estimate_runtime(features_dir, jobs=args.jobs, n_probe=args.probe)
            if args.step == "estimate"
            else _analyze(args, features_dir, roots)
        )
    except analyze.AnalyzeError as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    for line in lines:
        typer.echo(line)
    return 0


def _analyze(args: argparse.Namespace, features_dir: Path, roots: DataRoots | None) -> list[str]:
    """Step 3, from a clean checkout only.

    Raises:
        AnalyzeError: if the checkout is not clean, or there is nowhere to write.
    """
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
        out_dir = roots.out / UNBLINDED_DIR / f"{datetime.now():%Y%m%d}_tokenization_{git.short}"
    analyze.refuse_used_folder(out_dir)
    log = configure_logging("INFO", log_file=log_file_path(out_dir, analyze.STAGE))
    if log is not None:
        typer.echo(f"log: {log}")
    result = analyze.run(features_dir, args.labels, out_dir, commit=git.commit, jobs=args.jobs)
    return result.report_lines()


def _features(args: argparse.Namespace) -> int:
    """Step 2: check the port against the original, then compute every variant."""
    try:
        config = load_config()
        roots = resolve_roots(require_data=False)
    except (ConfigError, PathError) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    if config.model.text_features is None:
        typer.echo("error: model.text_features is not configured", err=True)
        return EXIT_SETUP_ERROR
    transcripts = args.transcripts
    if transcripts is None:
        if not config.diarization.import_dir:
            typer.echo("error: diarization.import_dir is not set; pass --transcripts", err=True)
            return EXIT_SETUP_ERROR
        transcripts = roots.work / config.diarization.import_dir

    log = configure_logging(
        config.runtime.log_level, log_file=log_file_path(roots.out, tokenization.STAGE)
    )
    if log is not None:
        typer.echo(f"log: {log}")

    try:
        result = tokenization.run(
            transcripts,
            args.nlp_features or roots.work / config.model.text_features.path,
            args.out or roots.out_path(*tokenization.OUT_DIRNAME),
            Tokenizer(),
        )
    except tokenization.ReproductionError as exc:
        for line in exc.lines:
            typer.echo(line, err=True)
        return EXIT_REPRODUCTION_FAILED
    except tokenization.TokenizationError as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    for line in result.report_lines(tokenization.file_sha256):
        typer.echo(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
