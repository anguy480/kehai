"""`python -m vc_multimodal.exploratory.tokenization features`.

Run by the person who may read the transcripts, on the machine that holds them.
Reads no labels. Prints aggregate counts only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import typer
from janome.tokenizer import Tokenizer

from vc_multimodal.config import ConfigError, load_config
from vc_multimodal.exploratory.tokenization import run as tokenization
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.paths import PathError, load_env, resolve_roots

EXIT_REPRODUCTION_FAILED: Final = 1
EXIT_SETUP_ERROR: Final = 2


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the port and print its aggregate summary."""
    args = _parser().parse_args(argv)
    load_env()
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
