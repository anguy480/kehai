"""Command line entry point for the `vc` tool.

One subcommand per pipeline stage, plus `run-all`. This module stays thin: it
resolves configuration and roots, calls a stage, and prints a summary. It is the
only module permitted to write to stdout, so stage logs (stderr, plus a per-run
file) and results can be redirected separately.

Nothing printed here is session content. Summaries are metadata: counts,
durations, codecs, session IDs and flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from vc_multimodal import __version__
from vc_multimodal.config import DEFAULT_CONFIG_PATH, AppConfig, ConfigError, load_config
from vc_multimodal.contracts import ContractError
from vc_multimodal.ffmpeg import FfmpegError, FfmpegTools
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.paths import (
    DataRoots,
    PathError,
    discover_sessions,
    load_env,
    parse_session_spec,
    resolve_roots,
)
from vc_multimodal.runner import StageReport
from vc_multimodal.stages import inventory as inventory_stage
from vc_multimodal.stages import preview as preview_stage

app = typer.Typer(
    name="vc",
    help="Multimodal feature extraction and label-free handoff for clinical Zoom sessions.",
    no_args_is_help=True,
    add_completion=False,
)

# Exit codes: 1 for a stage that reported failures, 2 for a setup problem
# (typer's own code for a usage error), so scripts can tell them apart.
EXIT_STAGE_FAILED = 1
EXIT_SETUP_ERROR = 2


@dataclass(frozen=True, slots=True)
class GlobalOptions:
    """Options accepted before the subcommand, resolved lazily.

    Resolution is deferred so that `vc --version` and `vc --help` work on a
    machine with no data roots configured.
    """

    config_path: Path
    overlays: tuple[Path, ...]
    sessions: str | None
    workers: int | None
    force: bool
    log_level: str | None

    def load(self, stage: str, *, require_data: bool = True) -> tuple[AppConfig, DataRoots]:
        """Load config, resolve roots and start logging for one stage."""
        load_env()
        config = load_config(self.config_path, overlays=self.overlays)
        roots = resolve_roots(require_data=require_data)
        level = self.log_level or config.runtime.log_level
        log_path = configure_logging(level, log_file=log_file_path(roots.out, stage))
        if log_path is not None:
            typer.echo(f"log: {log_path}")
        return config, roots

    def session_ids(self, config: AppConfig) -> tuple[int, ...] | None:
        """Sessions to run: `--sessions` if given, else the configured pilot set."""
        if self.sessions is not None:
            return parse_session_spec(self.sessions)
        return config.runtime.pilot_sessions or None

    def worker_count(self, config: AppConfig) -> int | None:
        """Worker count, preferring the CLI flag over the config file."""
        return self.workers if self.workers is not None else config.runtime.workers


def _options(ctx: typer.Context) -> GlobalOptions:
    """Retrieve the global options stored by the top-level callback."""
    if not isinstance(ctx.obj, GlobalOptions):  # pragma: no cover - defensive
        msg = "global options were not initialised; the callback did not run"
        raise RuntimeError(msg)
    return ctx.obj


def _fail(message: str) -> None:
    """Report a setup problem and exit."""
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(EXIT_SETUP_ERROR)


def _print_report(report: StageReport) -> None:
    """Print a stage summary and exit non-zero if any session failed."""
    for line in report.summary_lines():
        typer.echo(line)
    if not report.ok:
        raise typer.Exit(EXIT_STAGE_FAILED)


def _version_callback(value: bool) -> None:
    """Print the version and exit, for `vc --version`."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Base YAML configuration file."),
    ] = DEFAULT_CONFIG_PATH,
    overlay: Annotated[
        list[Path] | None,
        typer.Option("--overlay", help="Extra config file merged on top. Repeatable."),
    ] = None,
    sessions: Annotated[
        str | None,
        typer.Option(
            "--sessions",
            "-s",
            help="Sessions to process, e.g. '3,17,28' or '1-5'. Defaults to all.",
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option("--workers", "-w", min=1, help="Parallel workers. Overrides the config."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Recompute sessions whose output already exists."),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR."),
    ] = None,
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
    """Global options shared by every stage."""
    ctx.obj = GlobalOptions(
        config_path=config,
        overlays=tuple(overlay or ()),
        sessions=sessions,
        workers=workers,
        force=force,
        log_level=log_level,
    )


@dataclass(frozen=True, slots=True)
class StageSetup:
    """Everything a stage needs, with every setup error already reported."""

    config: AppConfig
    roots: DataRoots
    tools: FfmpegTools
    session_ids: tuple[int, ...] | None
    workers: int | None
    force: bool


def _setup(ctx: typer.Context, stage: str) -> StageSetup:
    """Resolve config, roots, logging, binaries and the session selection.

    Every foreseeable setup problem is turned into a one-line message and a
    distinct exit code here, so no stage ever greets the user with a traceback.
    """
    options = _options(ctx)
    try:
        config, roots = options.load(stage)
        tools = FfmpegTools.discover()
        session_ids = options.session_ids(config)
    except (ConfigError, PathError, FfmpegError, ValueError) as exc:
        _fail(str(exc))
        raise  # pragma: no cover - _fail always exits
    return StageSetup(
        config=config,
        roots=roots,
        tools=tools,
        session_ids=session_ids,
        workers=options.worker_count(config),
        force=options.force,
    )


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Check that the environment is ready: binaries, roots and raw layout."""
    options = _options(ctx)
    load_env()

    try:
        config = load_config(options.config_path, overlays=options.overlays)
        typer.echo(f"config: {options.config_path} ok")
    except ConfigError as exc:
        _fail(str(exc))
        return

    try:
        tools = FfmpegTools.discover()
        for name, line in tools.versions().items():
            typer.echo(f"{name}: {line}")
    except FfmpegError as exc:
        _fail(str(exc))
        return

    try:
        roots = resolve_roots()
    except PathError as exc:
        _fail(str(exc))
        return
    typer.echo(f"data root: {roots.data}")
    typer.echo(f"work root: {roots.work}")
    typer.echo(f"out root:  {roots.out}")

    discovery = discover_sessions(roots.data, config.dataset)
    typer.echo(
        f"raw layout: {len(discovery.sessions)} session(s) found, "
        f"expected {config.dataset.expected_sessions}"
    )
    for problem in discovery.problems:
        typer.secho(f"  problem: {problem}", fg=typer.colors.YELLOW)
    if discovery.problems:
        raise typer.Exit(EXIT_STAGE_FAILED)


@app.command()
def inventory(ctx: typer.Context) -> None:
    """Probe every recording with ffprobe and write inventory.csv."""
    setup = _setup(ctx, inventory_stage.STAGE)

    try:
        result = inventory_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
            workers=setup.workers,
            force=setup.force,
            tools=setup.tools,
        )
    except (inventory_stage.ExistingInventoryError, ContractError) as exc:
        _fail(str(exc))
        return

    typer.echo(f"\nwrote {result.path}")
    typer.echo("")
    for line in inventory_stage.summarise(result.frame, setup.config):
        typer.echo(line)

    if result.discovery.problems:
        typer.echo("")
        typer.secho("raw layout problems:", fg=typer.colors.YELLOW)
        for problem in result.discovery.problems:
            typer.secho(f"  {problem}", fg=typer.colors.YELLOW)

    typer.echo("")
    _print_report(result.report)


@app.command()
def preview(ctx: typer.Context) -> None:
    """Write one contact sheet per session to check the participant crop by eye."""
    setup = _setup(ctx, preview_stage.STAGE)

    report = preview_stage.run(
        setup.config,
        setup.roots,
        session_ids=setup.session_ids,
        workers=setup.workers,
        force=setup.force,
        tools=setup.tools,
    )

    typer.echo(f"\npreviews: {preview_stage.previews_dir(setup.roots)}")
    typer.echo(
        "Open them yourself and confirm: is this gallery view, and is each tile "
        "labelled with the right role?"
    )
    typer.echo("")
    _print_report(report)


if __name__ == "__main__":  # pragma: no cover - module entry point
    app()
