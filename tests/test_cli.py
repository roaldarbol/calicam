"""Tests for the calicam CLI (__main__.py)."""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from calicam.__main__ import cli
from calicam.io import save_json, save_yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Root / help
# ---------------------------------------------------------------------------


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "calicam" in result.output


def test_cli_no_args_shows_usage(runner):
    # Click groups print usage and exit with code 2 when no subcommand is given
    result = runner.invoke(cli, [])
    assert "Usage" in result.output or "calicam" in result.output


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_yaml(runner, tmp_path, sample_result):
    path = tmp_path / "cal.yaml"
    save_yaml(sample_result, path)

    result = runner.invoke(cli, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert "640" in result.output
    assert "480" in result.output
    assert "800" in result.output   # fx, fy


def test_inspect_json(runner, tmp_path, sample_result):
    path = tmp_path / "cal.json"
    save_json(sample_result, path)

    result = runner.invoke(cli, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert "640" in result.output


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def test_convert_yaml_to_json(runner, tmp_path, sample_result):
    src = tmp_path / "cal.yaml"
    dst = tmp_path / "cal.json"
    save_yaml(sample_result, src)

    result = runner.invoke(cli, ["convert", str(src), "--output", str(dst)])
    assert result.exit_code == 0, result.output
    assert dst.exists()

    data = json.loads(dst.read_text())
    assert data["image_width"] == 640
    assert data["image_height"] == 480


def test_convert_json_to_yaml(runner, tmp_path, sample_result):
    src = tmp_path / "cal.json"
    dst = tmp_path / "cal.yaml"
    save_json(sample_result, src)

    result = runner.invoke(cli, ["convert", str(src), "--output", str(dst)])
    assert result.exit_code == 0, result.output
    assert dst.exists()


def test_convert_auto_extension(runner, tmp_path, sample_result):
    """When --output is omitted the extension is flipped automatically."""
    src = tmp_path / "cal.yaml"
    save_yaml(sample_result, src)

    result = runner.invoke(cli, ["convert", str(src)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cal.json").exists()


def test_convert_shows_reprojection_error(runner, tmp_path, sample_result):
    src = tmp_path / "cal.yaml"
    dst = tmp_path / "cal.json"
    save_yaml(sample_result, src)

    result = runner.invoke(cli, ["convert", str(src), "--output", str(dst)])
    assert "0.42" in result.output


# ---------------------------------------------------------------------------
# generate checkerboard
# ---------------------------------------------------------------------------


def test_generate_checkerboard(runner, tmp_path):
    out = tmp_path / "board.png"
    result = runner.invoke(
        cli,
        ["generate", "checkerboard", "--cols", "4", "--rows", "3",
         "--square-size", "20", "--dpi", "72", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_generate_checkerboard_default_name(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["generate", "checkerboard", "--cols", "4", "--rows", "3", "--dpi", "72"],
        )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# generate circles
# ---------------------------------------------------------------------------


def test_generate_circles(runner, tmp_path):
    out = tmp_path / "circles.png"
    result = runner.invoke(
        cli,
        ["generate", "circles", "--cols", "3", "--rows", "5",
         "--spacing", "15", "--radius", "4", "--dpi", "72", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


# ---------------------------------------------------------------------------
# generate charuco
# ---------------------------------------------------------------------------


def test_generate_charuco(runner, tmp_path):
    out = tmp_path / "charuco.png"
    result = runner.invoke(
        cli,
        ["generate", "charuco", "--cols", "4", "--rows", "4",
         "--square-size", "20", "--marker-size", "14", "--dpi", "72", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_generate_help(runner):
    result = runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "checkerboard" in result.output
    assert "charuco" in result.output
    assert "circles" in result.output
