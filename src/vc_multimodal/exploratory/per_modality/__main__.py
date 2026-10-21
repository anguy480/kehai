"""`python -m vc_multimodal.exploratory.per_modality face-table|estimate|analyze`.

A module entry point rather than a `vc` subcommand, so this exploratory
analysis adds files and changes no stage. Runs with or without data roots: on
the label holder's machine there are none, and `--out` says where results go.
`face-table` is the exception: it runs where the recordings are, reads no
labels, and needs the data roots.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import typer

from vc_multimodal.config import AppConfig, ConfigError, load_config
from vc_multimodal.exploratory.per_modality import analyze, face_table
from vc_multimodal.exploratory.per_modality.plan import RUN_FIRST, RUNS, Plan, PlanError, load_plan
from vc_multimodal.faces import FaceError
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.paths import PathError, load_env, resolve_roots
from vc_multimodal.provenance import git_state
from vc_multimodal.qc_notes import QcNoteError
from vc_multimodal.stages.model import ModelError

EXIT_SETUP_ERROR: Final = 2
UNBLINDED_DIR: Final = "unblinded"

#: The checkout this package runs from, whatever the current folder.
REPO_ROOT: Final = Path(__file__).resolve().parents[4]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vc_multimodal.exploratory.per_modality")
    steps = parser.add_subparsers(dest="step", required=True)

    def inputs(step: argparse.ArgumentParser) -> None:
        step.add_argument("--run", choices=RUNS, required=True, help="Which planned handoff.")
        step.add_argument("--bundle", type=Path, default=None, help="Bundle folder (first run).")
        step.add_argument(
            "--face-table", type=Path, default=None, help="OpenFace face table (second run)."
        )
        step.add_argument("--jobs", type=int, default=os.cpu_count() or 1)

    steps.add_parser(
        "face-table", help="Build the OpenFace face table from `vc face` output (no labels)."
    )

    timing = steps.add_parser("estimate", help="Time a run on random targets (no labels).")
    inputs(timing)
    timing.add_argument("--probe", type=int, default=4, help="Permutations to time.")

    analysis = steps.add_parser("analyze", help="Run the planned analysis on the labels.")
    inputs(analysis)
    analysis.add_argument("--labels", type=Path, required=True, help="Labels CSV.")
    analysis.add_argument("--out", type=Path, default=None, help="Results folder.")
    analysis.add_argument(
        "--prior", type=Path, default=None, help="The first run's results folder (second run)."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step and print its summary."""
    args = _parser().parse_args(argv)

    load_env()
    try:
        config = load_config()
        plan = load_plan()
    except (ConfigError, PlanError) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    if args.step == "face-table":
        return _face_table(config, plan)
    try:
        if args.step == "estimate":
            estimate = analyze.estimate_runtime(
                plan,
                args.run,
                bundle=args.bundle,
                face_table=args.face_table,
                jobs=args.jobs,
                n_probe=args.probe,
            )
            lines = estimate.report_lines(plan.validation.n_permutations)
        else:
            lines = _analyze(args, config, plan)
    except (analyze.AnalyzeError, PlanError, ModelError) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    for line in lines:
        typer.echo(line)
    return 0


def _analyze(args: argparse.Namespace, config: AppConfig, plan: Plan) -> list[str]:
    """Run one planned handoff from a clean checkout.

    Raises:
        AnalyzeError: if the request or the checkout cannot be honoured.
    """
    if args.run == RUN_FIRST and args.face_table is not None:
        msg = "the first run reads the bundle, not a face table"
        raise analyze.AnalyzeError(msg)
    git = git_state(REPO_ROOT)
    if git is None or git.is_dirty:
        msg = f"run the analysis from a clean checkout of {REPO_ROOT}"
        raise analyze.AnalyzeError(msg)
    out_dir = args.out
    if out_dir is None:
        try:
            out_root = resolve_roots(require_data=False).out
        except PathError as exc:
            msg = "VC_OUT_ROOT is not set, so say where results go with --out"
            raise analyze.AnalyzeError(msg) from exc
        out_dir = (
            out_root
            / UNBLINDED_DIR
            / f"{datetime.now():%Y%m%d}_per_modality_{args.run}_{git.short}"
        )
    log = configure_logging(
        config.runtime.log_level, log_file=log_file_path(out_dir, analyze.STAGE)
    )
    if log is not None:
        typer.echo(f"log: {log}")
    return analyze.run(
        plan,
        args.run,
        labels_path=args.labels,
        out_dir=out_dir,
        commit=git.commit,
        bundle=args.bundle,
        face_table=args.face_table,
        prior_dir=args.prior,
        jobs=args.jobs,
    ).report_lines()


def _face_table(config: AppConfig, plan: Plan) -> int:
    """Build the face table where the recordings are, and print its summary."""
    try:
        roots = resolve_roots(require_data=True)
    except PathError as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    log = configure_logging(
        config.runtime.log_level, log_file=log_file_path(roots.out, face_table.STAGE)
    )
    if log is not None:
        typer.echo(f"log: {log}")
    try:
        table = face_table.build(config, roots, plan)
    except (face_table.FaceTableError, FaceError, QcNoteError) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    for line in [
        *table.report_lines(),
        "",
        *table.valid_fraction_lines(),
        "",
        *table.distribution_lines(),
    ]:
        typer.echo(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
