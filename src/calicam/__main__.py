"""CLI entry point for calicam."""

from __future__ import annotations

from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """calicam - camera calibration tool."""


# ---------------------------------------------------------------------------
# gui
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("source", default="0")
@click.option("--cols", default=9, show_default=True, help="Inner corner columns.")
@click.option("--rows", default=6, show_default=True, help="Inner corner rows.")
@click.option("--square-size", default=25.0, show_default=True, help="Square size in mm.")
@click.option("--min-frames", default=20, show_default=True, help="Target capture frame count.")
@click.option("--no-auto", is_flag=True, default=False, help="Disable auto-capture mode.")
@click.option(
    "--output-dir",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory for saved calibration files.",
)
def gui(
    source: str,
    cols: int,
    rows: int,
    square_size: float,
    min_frames: int,
    no_auto: bool,
    output_dir: str,
) -> None:
    """Open the calibration GUI.

    SOURCE is a camera index (e.g. 0) or a path to a video file.
    """
    from .gui.app import CaliCamApp

    try:
        src: int | str = int(source)
    except ValueError:
        src = source

    CaliCamApp(
        source=src,
        board_cols=cols,
        board_rows=rows,
        square_size_mm=square_size,
        min_frames=min_frames,
        auto_capture=not no_auto,
        output_dir=output_dir,
    ).run()


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output", "-o",
    default=None,
    help="Output path (defaults to same stem with swapped extension).",
)
def convert(input_file: str, output: str | None) -> None:
    """Convert a calibration file between YAML and JSON.

    INPUT_FILE may be .yaml/.yml or .json; the format is inferred from the extension.
    """
    from . import load, save

    src = Path(input_file)
    result = load(src)

    if output is None:
        new_ext = ".json" if src.suffix.lower() in {".yaml", ".yml"} else ".yaml"
        dst = src.with_suffix(new_ext)
    else:
        dst = Path(output)

    save(result, dst)
    click.echo(f"Converted  {src}  ->  {dst}")
    click.echo(f"  Reprojection error: {result.reprojection_error:.6f} px")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("calibration_file", type=click.Path(exists=True, dir_okay=False))
def inspect(calibration_file: str) -> None:
    """Print a summary of a calibration file."""
    from . import load

    r = load(Path(calibration_file))
    click.echo(f"Image size        : {r.image_size[0]} x {r.image_size[1]} px")
    click.echo(f"Focal length      : fx={r.camera_matrix[0, 0]:.2f}  fy={r.camera_matrix[1, 1]:.2f}")
    click.echo(f"Principal point   : cx={r.camera_matrix[0, 2]:.2f}  cy={r.camera_matrix[1, 2]:.2f}")
    click.echo(f"Distortion coeffs : {r.dist_coeffs.flatten().tolist()}")
    click.echo(f"Reprojection error: {r.reprojection_error:.6f} px")


# ---------------------------------------------------------------------------
# generate  (subcommand group)
# ---------------------------------------------------------------------------


@cli.group()
def generate() -> None:
    """Generate a printable calibration pattern image.

    Choose a pattern type as the subcommand:

    \b
      checkerboard   Classic black/white grid
      charuco        Checkerboard with embedded ArUco markers
      circles        Asymmetric circle grid
    """


_dpi_option = click.option(
    "--dpi", default=300.0, show_default=True,
    help="Output resolution in dots per inch.",
)
_margin_option = click.option(
    "--margin", default=10.0, show_default=True,
    help="White border around the pattern in mm.",
)
_output_option = click.option(
    "--output", "-o", default=None,
    help="Output file path. Extension sets the image format (png, tiff, pdf, …).",
)


# ---- checkerboard ----


@generate.command("checkerboard")
@click.option("--cols", default=9, show_default=True, help="Inner corner columns.")
@click.option("--rows", default=6, show_default=True, help="Inner corner rows.")
@click.option("--square-size", default=25.0, show_default=True, help="Square side length in mm.")
@_margin_option
@_dpi_option
@_output_option
def gen_checkerboard(
    cols: int,
    rows: int,
    square_size: float,
    margin: float,
    dpi: float,
    output: str | None,
) -> None:
    """Generate a classic checkerboard pattern.

    \b
    The board has (COLS+1) x (ROWS+1) squares. Specify the number of *inner
    corners* to match what your calibration software expects (calicam gui
    defaults to 9 x 6 inner corners).

    \b
    Example:
      calicam generate checkerboard --cols 9 --rows 6 --square-size 25 -o board.png
    """
    from .board import generate_checkerboard

    dst = output or f"checkerboard_{cols}x{rows}.png"
    path = generate_checkerboard(
        cols=cols,
        rows=rows,
        square_size_mm=square_size,
        margin_mm=margin,
        dpi=dpi,
        output=dst,
    )
    _print_result(path, dpi)


# ---- charuco ----


@generate.command("charuco")
@click.option("--cols", default=5, show_default=True, help="Number of squares along the width.")
@click.option("--rows", default=7, show_default=True, help="Number of squares along the height.")
@click.option("--square-size", default=30.0, show_default=True, help="Square side length in mm.")
@click.option(
    "--marker-size", default=22.0, show_default=True,
    help="ArUco marker side length in mm (must be < square-size).",
)
@click.option(
    "--dict", "aruco_dict", default="5x5_100", show_default=True,
    help=(
        "ArUco dictionary. Options: 4x4_50, 4x4_100, 4x4_250, 4x4_1000, "
        "5x5_50, 5x5_100, 5x5_250, 5x5_1000, 6x6_250, 7x7_1000."
    ),
)
@_margin_option
@_dpi_option
@_output_option
def gen_charuco(
    cols: int,
    rows: int,
    square_size: float,
    marker_size: float,
    aruco_dict: str,
    margin: float,
    dpi: float,
    output: str | None,
) -> None:
    """Generate a ChArUco (checkerboard + ArUco markers) pattern.

    \b
    ChArUco boards work well under partial occlusion and at longer range.
    The ArUco dictionary must be large enough to assign a unique marker to
    every white square: a 5x7 board has 17 white squares, so use at least
    a _50 dictionary.

    \b
    Example:
      calicam generate charuco --cols 5 --rows 7 --square-size 30 --marker-size 22
    """
    from .board import generate_charuco

    dst = output or f"charuco_{cols}x{rows}.png"
    path = generate_charuco(
        cols=cols,
        rows=rows,
        square_size_mm=square_size,
        marker_size_mm=marker_size,
        margin_mm=margin,
        dpi=dpi,
        aruco_dict=aruco_dict,
        output=dst,
    )
    _print_result(path, dpi)


# ---- circles ----


@generate.command("circles")
@click.option("--cols", default=4, show_default=True, help="Number of circle columns.")
@click.option("--rows", default=11, show_default=True, help="Number of circle rows.")
@click.option(
    "--spacing", default=20.0, show_default=True,
    help="Centre-to-centre spacing between circles in mm.",
)
@click.option("--radius", default=5.0, show_default=True, help="Circle radius in mm.")
@_margin_option
@_dpi_option
@_output_option
def gen_circles(
    cols: int,
    rows: int,
    spacing: float,
    radius: float,
    margin: float,
    dpi: float,
    output: str | None,
) -> None:
    """Generate an asymmetric circle grid pattern.

    \b
    Odd rows are offset by half the column spacing, making orientation
    unambiguous. Detected with OpenCV's findCirclesGrid (CALIB_CB_ASYMMETRIC_GRID).

    \b
    Example:
      calicam generate circles --cols 4 --rows 11 --spacing 20 --radius 5
    """
    from .board import generate_circles

    dst = output or f"circles_{cols}x{rows}.png"
    path = generate_circles(
        cols=cols,
        rows=rows,
        spacing_mm=spacing,
        radius_mm=radius,
        margin_mm=margin,
        dpi=dpi,
        output=dst,
    )
    _print_result(path, dpi)


# ---------------------------------------------------------------------------
# Shared output summary
# ---------------------------------------------------------------------------


def _print_result(path: Path, dpi: float) -> None:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        h, w = img.shape
        w_mm = w / dpi * 25.4
        h_mm = h / dpi * 25.4
        click.echo(f"Saved  {path}")
        click.echo(f"  {w} x {h} px  ({w_mm:.1f} x {h_mm:.1f} mm at {dpi:.0f} dpi)")
    else:
        click.echo(f"Saved  {path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
