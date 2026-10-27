"""`python -m vc_multimodal.exploratory.fusion estimate | analyze`.

`estimate` times the analysis on random targets: it reads the four feature
tables, each checked by SHA-256, and no label. `analyze` reads the labels, so
only the label holder runs it, from a clean checkout; it prints coverage, counts
and the files it wrote, not results.

Every table defaults to where the step that made it wrote it, under the output
root: the text table from the tokenization rerun, the LLM table from the
rater, the bundle under `handoff/`, the face table from `face-table`.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import typer

from vc_multimodal.exploratory.fusion import analyze
from vc_multimodal.exploratory.fusion.plan import BUNDLE_NAME
from vc_multimodal.exploratory.llm_ratings import analyze as language
from vc_multimodal.exploratory.llm_ratings import rater
from vc_multimodal.exploratory.llm_ratings.plan import REPO_ROOT
from vc_multimodal.exploratory.per_modality.face_table import FACE_TABLE_FILENAME
from vc_multimodal.logging_setup import configure_logging, log_file_path
from vc_multimodal.paths import DataRoots, PathError, load_env, resolve_roots
from vc_multimodal.provenance import git_state
from vc_multimodal.stages.handoff import BUNDLE_DIRNAME

EXIT_SETUP_ERROR: Final = 2
UNBLINDED_DIR: Final = "unblinded"
#: Where step 2 of the tokenization rerun wrote the text table, under the output root.
TOKENIZATION_DIRNAME: Final = ("exploratory", "tokenization")
#: Where `face-table` wrote the OpenFace face table, under the output root.
PER_MODALITY_DIRNAME: Final = ("exploratory", "per_modality")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vc_multimodal.exploratory.fusion")
    steps = parser.add_subparsers(dest="step", required=True)
    for name, text in (
        ("estimate", "Time the analysis on random targets. Reads no label."),
        ("analyze", "The pre-registered analysis. Reads the labels."),
    ):
        step = steps.add_parser(name, help=text)
        step.add_argument(
            "--text-features",
            type=Path,
            default=None,
            help="The 20 text features. Default: exploratory/tokenization under the output root.",
        )
        step.add_argument(
            "--llm-features",
            type=Path,
            default=None,
            help="The 11 LLM ratings. Default: exploratory/llm_ratings under the output root.",
        )
        step.add_argument(
            "--bundle",
            type=Path,
            default=None,
            help=f"The bundle folder. Default: {BUNDLE_DIRNAME}/{BUNDLE_NAME} under the output "
            "root.",
        )
        step.add_argument(
            "--face-table",
            type=Path,
            default=None,
            help="The OpenFace face table. Default: exploratory/per_modality under the output "
            "root.",
        )
        step.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
        if name == "estimate":
            step.add_argument("--probe", type=int, default=4, help="Permutations to time.")
            continue
        step.add_argument("--labels", type=Path, required=True, help="Labels CSV.")
        step.add_argument(
            "--out",
            type=Path,
            default=None,
            help="Results folder, which must be new. Default: unblinded/<date>_fusion_<commit> "
            "under the output root.",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step and print its summary."""
    args = _parser().parse_args(argv)
    load_env()
    try:
        lines = _run(args)
    except (analyze.AnalyzeError, language.AnalyzeError, PathError) as exc:
        typer.echo(f"error: {exc}", err=True)
        return EXIT_SETUP_ERROR
    for line in lines:
        typer.echo(line)
    return 0


def _run(args: argparse.Namespace) -> list[str]:
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
    planned = language.load_inputs_plan()
    paths = analyze.Paths(
        text=args.text_features
        or _under_out(
            roots, (*TOKENIZATION_DIRNAME, planned[language.TEXT_TABLE][0]), "--text-features"
        ),
        llm=args.llm_features
        or _under_out(
            roots, (*rater.OUT_DIRNAME, planned[language.LLM_TABLE][0]), "--llm-features"
        ),
        bundle=args.bundle or _under_out(roots, (BUNDLE_DIRNAME, BUNDLE_NAME), "--bundle"),
        face=args.face_table
        or _under_out(roots, (*PER_MODALITY_DIRNAME, FACE_TABLE_FILENAME), "--face-table"),
    )
    if args.step == "estimate":
        return analyze.estimate_runtime(paths, jobs=args.jobs, n_probe=args.probe)

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
        out_dir = roots.out / UNBLINDED_DIR / f"{datetime.now():%Y%m%d}_fusion_{git.short}"
    language.refuse_used_folder(out_dir)
    log = configure_logging("INFO", log_file=log_file_path(out_dir, analyze.STAGE))
    lines = [f"log: {log}"] if log is not None else []
    result = analyze.run(paths, args.labels, out_dir, commit=git.commit, jobs=args.jobs)
    return lines + result.report_lines()


def _under_out(roots: DataRoots | None, parts: Sequence[str], option: str) -> Path:
    """A table's default place under the output root.

    Raises:
        AnalyzeError: if the output root is not set.
    """
    if roots is None:
        msg = f"VC_OUT_ROOT is not set, so say where {parts[-1]} is with {option}"
        raise analyze.AnalyzeError(msg)
    return roots.out.joinpath(*parts)


if __name__ == "__main__":
    raise SystemExit(main())
