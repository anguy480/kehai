"""`python -m vc_multimodal.exploratory.au_baseline pooled|estimate|analyze`.

A module entry point rather than a `vc` subcommand, so this exploratory
analysis adds files and changes none. Output is counts, digests, timings and
file names; `analyze` additionally writes its results under $VC_OUT_ROOT.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import typer

from vc_multimodal.config import ConfigError, load_config
from vc_multimodal.exploratory.au_baseline import analyze, pooled
from vc_multimodal.exploratory.au_baseline.plan import PlanError, load_plan
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.modeling.text_features import TextFeatureError
from vc_multimodal.paths import PathError, load_env, resolve_roots
from vc_multimodal.provenance import git_state
from vc_multimodal.stages.model import ModelError

EXIT_SETUP_ERROR = 2
STAGES = {"pooled": pooled.STAGE, "estimate": analyze.STAGE, "analyze": analyze.STAGE}
UNBLINDED_DIR = "unblinded"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vc_multimodal.exploratory.au_baseline")
    steps = parser.add_subparsers(dest="step", required=True)

    pooling = steps.add_parser("pooled", help="Compute the whole-session AU table (no labels).")
    pooling.add_argument("--bundle", type=Path, required=True, help="Handoff bundle folder.")

    timing = steps.add_parser("estimate", help="Time Part B on random targets (no labels).")
    timing.add_argument("--bundle", type=Path, required=True, help="Handoff bundle folder.")
    timing.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    timing.add_argument("--probe", type=int, default=16, help="Permutations to time.")

    analysis = steps.add_parser("analyze", help="Run the planned analysis on the labels.")
    analysis.add_argument("--bundle", type=Path, required=True, help="Handoff bundle folder.")
    analysis.add_argument("--labels", type=Path, required=True, help="Labels CSV.")
    analysis.add_argument("--out", type=Path, default=None, help="Results folder.")
    analysis.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    analysis.add_argument("--permutations", type=int, default=None)
    analysis.add_argument(
        "--exclude-session",
        type=int,
        action="append",
        default=[],
        help="Session left out of the sensitivity rerun. Repeatable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step and print its summary."""
    args = _parser().parse_args(argv)

    load_env()
    try:
        config = load_config()
        plan = load_plan()
        roots = resolve_roots(require_data=False)
    except (ConfigError, PlanError, PathError) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    log = configure_logging(
        config.runtime.log_level, log_file=log_file_path(roots.out, STAGES[args.step])
    )
    if log is not None:
        typer.echo(f"log: {log}")

    try:
        if args.step == "pooled":
            features = args.bundle / analyze.FEATURES_FILE
            if pooled.file_sha256(features) != plan.inputs.bundle_features_sha256:
                typer.echo(f"error: {features} is not the frozen features.csv", err=True)
                return EXIT_SETUP_ERROR
            lines = pooled.run(config, roots, bundle_features=features).report_lines()
        elif args.step == "estimate":
            estimate = analyze.estimate_runtime(
                config, roots, plan, bundle=args.bundle, jobs=args.jobs, n_probe=args.probe
            )
            lines = estimate.report_lines(plan.permutation.n_permutations)
        else:
            git = git_state(Path.cwd())
            if git is None or git.is_dirty:
                typer.echo("error: run the analysis from a clean checkout", err=True)
                return EXIT_SETUP_ERROR
            out_dir = args.out or roots.out_path(
                UNBLINDED_DIR, f"{datetime.now():%Y%m%d}_au_baseline_{git.short}"
            )
            lines = analyze.run(
                config,
                roots,
                plan,
                bundle=args.bundle,
                labels_path=args.labels,
                out_dir=out_dir,
                commit=git.commit,
                jobs=args.jobs,
                n_permutations=args.permutations,
                exclude_sessions=tuple(args.exclude_session),
            ).report_lines()
    except (
        analyze.AnalyzeError,
        pooled.PooledError,
        PlanError,
        ModelError,
        TextFeatureError,
    ) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    for line in lines:
        typer.echo(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
