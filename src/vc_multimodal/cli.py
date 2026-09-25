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
from vc_multimodal.diarization import DiarizationError
from vc_multimodal.embeddings import EmbeddingError
from vc_multimodal.faces import FaceError
from vc_multimodal.ffmpeg import FfmpegError, FfmpegTools
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.ocr import OcrError, get_backend
from vc_multimodal.paths import (
    DataRoots,
    PathError,
    discover_sessions,
    load_env,
    parse_session_spec,
    resolve_roots,
)
from vc_multimodal.prosody import ProsodyError
from vc_multimodal.roles import RolesUnavailableError
from vc_multimodal.runner import StageReport
from vc_multimodal.stages import aggregate as aggregate_stage
from vc_multimodal.stages import assign_speakers as assign_stage
from vc_multimodal.stages import diarize as diarize_stage
from vc_multimodal.stages import extract_audio as extract_audio_stage
from vc_multimodal.stages import face as face_stage
from vc_multimodal.stages import handoff as handoff_stage
from vc_multimodal.stages import inventory as inventory_stage
from vc_multimodal.stages import preview as preview_stage
from vc_multimodal.stages import prosody as prosody_stage
from vc_multimodal.stages import turns as turns_stage
from vc_multimodal.stages import vad as vad_stage
from vc_multimodal.stages import verify_layout as verify_layout_stage
from vc_multimodal.stages.assign_speakers import AssignError
from vc_multimodal.stages.handoff import HandoffError

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

    ocr_config = config.speakers.label_ocr
    if ocr_config.enabled:
        engine = get_backend(ocr_config.backend)
        if engine.available():
            typer.echo(f"label OCR: {engine.name} ({engine.version()})")
        else:
            reason = getattr(engine, "unavailable_reason", lambda: "unavailable")()
            typer.secho(f"label OCR: {engine.name} unavailable - {reason}", fg=typer.colors.YELLOW)
            typer.echo(
                "  `vc verify-layout` will fall back to "
                f"speakers.assumed_psychiatrist_side ({config.speakers.assumed_psychiatrist_side})"
            )
    else:
        typer.echo("label OCR: disabled in config")

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
def preview(
    ctx: typer.Context,
    label_regions: Annotated[
        bool,
        typer.Option(
            "--label-regions/--no-label-regions",
            help=(
                "Draw the detected content area and the computed label regions "
                "on each sheet, showing exactly what label OCR reads."
            ),
        ),
    ] = True,
) -> None:
    """Write one contact sheet per session to check the participant crop by eye."""
    setup = _setup(ctx, preview_stage.STAGE)

    report = preview_stage.run(
        setup.config,
        setup.roots,
        session_ids=setup.session_ids,
        workers=setup.workers,
        force=setup.force,
        tools=setup.tools,
        label_regions=label_regions,
    )

    typer.echo(f"\npreviews: {preview_stage.previews_dir(setup.roots)}")
    typer.echo(
        "Open them yourself and confirm: is this gallery view, and is each tile "
        "labelled with the right role?"
    )
    if label_regions:
        typer.echo(
            "The orange box is the detected content area; the green boxes are the "
            "label regions label OCR reads. If a green box is not over a name, "
            "that is why `vc verify-layout` finds nothing."
        )
    typer.echo("")
    _print_report(report)


@app.command(name="extract-audio")
def extract_audio(ctx: typer.Context) -> None:
    """Extract mono 16 kHz audio, and compare each recording's stereo channels."""
    setup = _setup(ctx, extract_audio_stage.STAGE)

    try:
        result = extract_audio_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
            workers=setup.workers,
            force=setup.force,
            tools=setup.tools,
        )
    except ContractError as exc:
        _fail(str(exc))
        return

    typer.echo(f"\naudio: {extract_audio_stage.audio_dir(setup.roots)}")
    typer.echo(f"wrote {result.path}")
    typer.echo("")
    for line in extract_audio_stage.summarise(result.frame):
        typer.echo(line)
    typer.echo("")
    _print_report(result.report)


@app.command()
def diarize(ctx: typer.Context) -> None:
    """Work out who spoke when, using the configured diarization backend."""
    setup = _setup(ctx, diarize_stage.STAGE)

    try:
        result = diarize_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
            workers=setup.workers,
            force=setup.force,
            tools=setup.tools,
        )
    except (DiarizationError, ConfigError, ContractError) as exc:
        _fail(str(exc))
        return

    typer.echo(f"\nsegments: {diarize_stage.segments_dir(setup.roots)}")
    typer.echo(f"wrote {result.path}")
    typer.echo("")
    for line in diarize_stage.summarise(result.frame, setup.config):
        typer.echo(line)
    typer.echo("")
    _print_report(result.report)


@app.command()
def vad(ctx: typer.Context) -> None:
    """Refine the diarized segments into the spans that actually contain speech."""
    setup = _setup(ctx, vad_stage.STAGE)

    try:
        result = vad_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
            workers=setup.workers,
            force=setup.force,
        )
    except ContractError as exc:
        _fail(str(exc))
        return

    typer.echo(f"\nspeech spans: {vad_stage.speech_dir(setup.roots)}")
    typer.echo(f"wrote {result.path}")
    typer.echo("")
    for line in vad_stage.summarise(result.frame):
        typer.echo(line)
    typer.echo("")
    _print_report(result.report)


@app.command()
def turns(ctx: typer.Context) -> None:
    """Derive turns, response latency, pauses and the speaking/listening timeline."""
    setup = _setup(ctx, turns_stage.STAGE)

    try:
        result = turns_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
            workers=setup.workers,
            force=setup.force,
        )
    except (RolesUnavailableError, ContractError) as exc:
        _fail(str(exc))
        return

    typer.echo(f"\nwrote {result.path}")
    typer.echo("")
    for line in turns_stage.summarise(result.frame):
        typer.echo(line)
    typer.echo("")
    _print_report(result.report)


@app.command()
def prosody(ctx: typer.Context) -> None:
    """Measure the participant's prosody, with overlapping speech excluded."""
    setup = _setup(ctx, prosody_stage.STAGE)

    try:
        result = prosody_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
            workers=setup.workers,
            force=setup.force,
        )
    except (ProsodyError, RolesUnavailableError, ContractError) as exc:
        _fail(str(exc))
        return

    typer.echo(f"\nwrote {result.path}")
    typer.echo("")
    for line in prosody_stage.summarise(result.frame):
        typer.echo(line)
    typer.echo("")
    _print_report(result.report)


@app.command()
def face(ctx: typer.Context) -> None:
    """Measure facial action units in the participant's video tile."""
    setup = _setup(ctx, face_stage.STAGE)

    try:
        result = face_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
            workers=setup.workers,
            force=setup.force,
            tools=setup.tools,
        )
    except (FaceError, ContractError) as exc:
        _fail(str(exc))
        return

    typer.echo(f"\nper-frame measures: {face_stage.face_dir(setup.roots)}")
    typer.echo(f"wrote {result.path}")
    typer.echo("")
    for line in face_stage.summarise(result.frame, setup.config):
        typer.echo(line)
    typer.echo("")
    _print_report(result.report)


@app.command()
def aggregate(ctx: typer.Context) -> None:
    """Join every stage into one row per session: the table the handoff carries."""
    setup = _setup(ctx, aggregate_stage.STAGE)

    try:
        result = aggregate_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
        )
    except (FaceError, ContractError) as exc:
        _fail(str(exc))
        return

    typer.echo(f"\nwrote {result.path}")
    typer.echo("")
    for line in aggregate_stage.summarise(result, setup.config):
        typer.echo(line)
    typer.echo("")
    _print_report(result.report)


@app.command(name="verify-layout")
def verify_layout(
    ctx: typer.Context,
    debug_region: Annotated[
        bool,
        typer.Option(
            "--debug-region",
            help=(
                "Report where each tile and label region was looked for, in "
                "fractional and pixel coordinates, with how much OCR saw there."
            ),
        ),
    ] = False,
) -> None:
    """Check which side the psychiatrist is on, by reading Zoom name labels.

    Reports counts and session IDs only. Recognised text is never printed,
    logged or written: the labels are people's names.
    """
    setup = _setup(ctx, verify_layout_stage.STAGE)

    try:
        result = verify_layout_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
            workers=setup.workers,
            tools=setup.tools,
        )
    except (OcrError, ContractError) as exc:
        _fail(str(exc))
        return

    typer.echo(f"\nwrote {result.path}")
    if result.debug_path is not None:
        typer.echo(f"wrote {result.debug_path}")

    if debug_region:
        typer.echo("")
        for line in verify_layout_stage.debug_report(result.observations, setup.config):
            typer.echo(line)

    typer.echo("")
    for line in verify_layout_stage.summarise(result.frame, setup.config):
        typer.echo(line)
    typer.echo("")
    _print_report(result.report)


@app.command()
def handoff(
    ctx: typer.Context,
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty",
            help=(
                "Build even though the working tree has uncommitted changes. The "
                "bundle then records that its commit does not describe the code that "
                "produced it."
            ),
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace an existing bundle with the same name."),
    ] = False,
) -> None:
    """Build the bundle the label holder receives: features, dictionary, QC, manifest.

    Carries no labels, no transcripts, no audio and no frames. Refuses to build
    from a modified working tree unless --allow-dirty, because the commit
    recorded in the bundle is its main provenance.
    """
    setup = _setup(ctx, handoff_stage.STAGE)

    try:
        result = handoff_stage.run(
            setup.config,
            setup.roots,
            allow_dirty=allow_dirty,
            force=force or setup.force,
        )
    except (HandoffError, ContractError) as exc:
        _fail(str(exc))
        return

    typer.echo("")
    for line in handoff_stage.summarise(result):
        typer.echo(line)
    typer.echo("")
    typer.echo(f"read {result.path / handoff_stage.README_FILE} before sending it on")


@app.command(name="assign-speakers")
def assign_speakers(ctx: typer.Context) -> None:
    """Work out which diarized speaker is the participant.

    Speaker embeddings against the psychiatrist reference clips decide it.
    Mouth movement per tile and the Zoom name labels corroborate it where they
    can, and any disagreement is flagged rather than used to break a tie.

    Prints similarities, margins and session IDs. Nothing derived from what was
    said, and no recognised label text, is printed or written.
    """
    setup = _setup(ctx, assign_stage.STAGE)

    try:
        result = assign_stage.run(
            setup.config,
            setup.roots,
            session_ids=setup.session_ids,
            workers=setup.workers,
            force=setup.force,
            tools=setup.tools,
        )
    except (AssignError, EmbeddingError, ContractError) as exc:
        _fail(str(exc))
        return

    typer.echo(f"\nwrote {result.path}")
    typer.echo("")
    for line in assign_stage.summarise(result, setup.config):
        typer.echo(line)
    typer.echo("")
    _print_report(result.report)


if __name__ == "__main__":  # pragma: no cover - module entry point
    app()
