"""Command line entry point for the `vc` tool.

One subcommand per pipeline stage, plus `run-all`. Stages are registered here as
they are implemented; this module stays thin, delegating to
`vc_multimodal.stages`. It is also the only module permitted to write to stdout:
everything else logs.
"""

from __future__ import annotations

from typing import Annotated

import typer

from vc_multimodal import __version__

app = typer.Typer(
    name="vc",
    help="Multimodal feature extraction and label-free handoff for clinical Zoom sessions.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print the version and exit, for `vc --version`."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the package version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Global options shared by every stage.

    Stage-independent settings (config file, log level, worker count) will be
    added here as the stages land, so that `vc --config ... <stage>` works
    uniformly.
    """


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover - module entry point
    app()
