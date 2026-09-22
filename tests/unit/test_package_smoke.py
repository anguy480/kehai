"""Skeleton-stage checks: the package imports and the CLI is wired up."""

from __future__ import annotations

from typer.testing import CliRunner

import vc_multimodal
from vc_multimodal.cli import app

runner = CliRunner()


def test_version_is_a_dotted_string():
    assert vc_multimodal.__version__.count(".") == 2


def test_version_subcommand():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == vc_multimodal.__version__


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == vc_multimodal.__version__


def test_no_args_shows_help_rather_than_running_a_stage():
    """Guards against typer collapsing a single-command app into that command."""
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage" in result.stdout


def test_unknown_subcommand_is_an_error():
    result = runner.invoke(app, ["not-a-stage"])
    assert result.exit_code != 0
