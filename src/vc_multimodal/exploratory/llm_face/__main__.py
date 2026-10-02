"""`python -m vc_multimodal.exploratory.llm_face describe|rate|analyze`.

A module entry point rather than a `vc` subcommand, so this exploratory
analysis adds files and changes none. Output is counts, digests and file names
only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import typer

from vc_multimodal.config import ConfigError, load_config
from vc_multimodal.exploratory.llm_face import analyze, describe, rate
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.paths import PathError, load_env, resolve_roots
from vc_multimodal.provenance import git_state
from vc_multimodal.stages.model import ModelError

EXIT_SETUP_ERROR = 2
STAGES = {"describe": describe.STAGE, "rate": rate.STAGE, "analyze": analyze.STAGE}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vc_multimodal.exploratory.llm_face")
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("describe", help="Write one description per session.")
    rating = steps.add_parser("rate", help="Rate every description with the local model.")
    rating.add_argument("--runs", type=int, default=rate.RUNS)
    analysis = steps.add_parser("analyze", help="Exploratory analysis of the frozen ratings.")
    analysis.add_argument("--bundle", type=Path, required=True, help="Handoff bundle folder.")
    analysis.add_argument("--labels", type=Path, required=True, help="Labels CSV.")
    analysis.add_argument("--out", type=Path, required=True, help="Results folder.")
    analysis.add_argument("--permutations", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step and print its summary."""
    args = _parser().parse_args(argv)

    load_env()
    try:
        config = load_config()
        roots = resolve_roots(require_data=False)
    except (ConfigError, PathError) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    log_path = log_file_path(roots.out, STAGES[args.step])
    log = configure_logging(config.runtime.log_level, log_file=log_path)
    if log is not None:
        typer.echo(f"log: {log}")

    try:
        if args.step == "describe":
            lines = describe.run(config, roots).report_lines()
        elif args.step == "rate":
            lines = rate.run(roots, runs=args.runs).report_lines()
        else:
            git = git_state(Path.cwd())
            if git is None or git.is_dirty:
                typer.echo("error: run the analysis from a clean checkout", err=True)
                return EXIT_SETUP_ERROR
            lines = analyze.run(
                config,
                roots,
                bundle=args.bundle,
                labels_path=args.labels,
                out_dir=args.out,
                commit=git.commit,
                n_permutations=args.permutations,
            ).report_lines()
    except (rate.RatingError, analyze.AnalyzeError, ModelError) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    for line in lines:
        typer.echo(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
