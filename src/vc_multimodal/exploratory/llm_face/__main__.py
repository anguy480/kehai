"""`python -m vc_multimodal.exploratory.llm_face describe|rate`.

A module entry point rather than a `vc` subcommand, so this exploratory
analysis adds files and changes none. Output is counts and digests only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import typer

from vc_multimodal.config import ConfigError, load_config
from vc_multimodal.exploratory.llm_face import describe, rate
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.paths import PathError, load_env, resolve_roots

EXIT_SETUP_ERROR = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step and print its summary."""
    parser = argparse.ArgumentParser(prog="python -m vc_multimodal.exploratory.llm_face")
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("describe", help="Write one description per session.")
    rating = steps.add_parser("rate", help="Rate every description with the local model.")
    rating.add_argument("--runs", type=int, default=rate.RUNS)
    args = parser.parse_args(argv)

    load_env()
    try:
        config = load_config()
        roots = resolve_roots(require_data=False)
    except (ConfigError, PathError) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    stage = describe.STAGE if args.step == "describe" else rate.STAGE
    log = configure_logging(config.runtime.log_level, log_file=log_file_path(roots.out, stage))
    if log is not None:
        typer.echo(f"log: {log}")

    try:
        if args.step == "describe":
            lines = describe.run(config, roots).report_lines()
        else:
            lines = rate.run(roots, runs=args.runs).report_lines()
    except rate.RatingError as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    for line in lines:
        typer.echo(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
