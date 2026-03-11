"""Generate printable calibration pattern images.

Three pattern types are supported:

- ``checkerboard`` -- classic black/white grid (use with :class:`~callycam.calibration.Calibrator`)
- ``charuco``      -- checkerboard with embedded ArUco markers; more robust under
                      partial occlusion and works at longer range
- ``circles``      -- asymmetric circle grid; sub-pixel accuracy, good for low-texture scenes

All generators work in physical units (mm) and convert to pixels via *dpi* so
the output can be printed at the correct size.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mm_to_px(mm: float, dpi: float) -> int:
    """Convert millimetres to pixels at the given DPI."""
    return round(mm * dpi / 25.4)


def _make_canvas(
    width_mm: float,
    height_mm: float,
    dpi: float,
    background: int = 255,
) -> np.ndarray:
    w = _mm_to_px(width_mm, dpi)
    h = _mm_to_px(height_mm, dpi)
    return np.full((h, w), background, dtype=np.uint8)


def _save(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise OSError(f"cv2.imwrite failed for {path!r} — check the extension is supported.")


# ---------------------------------------------------------------------------
# Checkerboard
# ---------------------------------------------------------------------------


def generate_checkerboard(
    cols: int = 9,
    rows: int = 6,
    square_size_mm: float = 25.0,
    margin_mm: float = 10.0,
    dpi: float = 300.0,
    output: str | Path = "checkerboard.png",
) -> Path:
    """Generate a checkerboard calibration pattern and save it to *output*.

    Parameters
    ----------
    cols:
        Number of inner corners along the width (squares = cols + 1).
    rows:
        Number of inner corners along the height (squares = rows + 1).
    square_size_mm:
        Side length of each square in millimetres.
    margin_mm:
        White border around the pattern in millimetres.
    dpi:
        Output resolution in dots per inch.
    output:
        Destination file path (extension determines format, e.g. ``.png``, ``.pdf``).

    Returns
    -------
    Path
        Resolved path to the saved file.
    """
    n_cols = cols + 1  # number of squares
    n_rows = rows + 1

    board_w_mm = n_cols * square_size_mm
    board_h_mm = n_rows * square_size_mm
    total_w_mm = board_w_mm + 2 * margin_mm
    total_h_mm = board_h_mm + 2 * margin_mm

    canvas = _make_canvas(total_w_mm, total_h_mm, dpi, background=255)

    sq = _mm_to_px(square_size_mm, dpi)
    ox = _mm_to_px(margin_mm, dpi)
    oy = _mm_to_px(margin_mm, dpi)

    for r in range(n_rows):
        for c in range(n_cols):
            if (r + c) % 2 == 0:
                x0 = ox + c * sq
                y0 = oy + r * sq
                canvas[y0 : y0 + sq, x0 : x0 + sq] = 0

    path = Path(output).resolve()
    _save(canvas, path)
    return path


# ---------------------------------------------------------------------------
# ChArUco
# ---------------------------------------------------------------------------


def _get_aruco_dict(name: str) -> cv2.aruco.Dictionary:
    """Return the cv2.aruco.Dictionary for a short name string."""
    mapping: dict[str, int] = {
        "4x4_50":   cv2.aruco.DICT_4X4_50,
        "4x4_100":  cv2.aruco.DICT_4X4_100,
        "4x4_250":  cv2.aruco.DICT_4X4_250,
        "4x4_1000": cv2.aruco.DICT_4X4_1000,
        "5x5_50":   cv2.aruco.DICT_5X5_50,
        "5x5_100":  cv2.aruco.DICT_5X5_100,
        "5x5_250":  cv2.aruco.DICT_5X5_250,
        "5x5_1000": cv2.aruco.DICT_5X5_1000,
        "6x6_250":  cv2.aruco.DICT_6X6_250,
        "7x7_1000": cv2.aruco.DICT_7X7_1000,
    }
    key = name.lower().replace(" ", "_")
    if key not in mapping:
        raise ValueError(
            f"Unknown ArUco dictionary {name!r}. "
            f"Valid options: {', '.join(mapping)}"
        )
    return cv2.aruco.getPredefinedDictionary(mapping[key])


def generate_charuco(
    cols: int = 5,
    rows: int = 7,
    square_size_mm: float = 30.0,
    marker_size_mm: float = 22.0,
    margin_mm: float = 10.0,
    dpi: float = 300.0,
    aruco_dict: str = "5x5_100",
    output: str | Path = "charuco.png",
) -> Path:
    """Generate a ChArUco calibration pattern and save it to *output*.

    ChArUco boards combine a checkerboard with ArUco markers inside the white
    squares, enabling robust detection even when parts of the board are occluded.

    Parameters
    ----------
    cols:
        Number of squares along the width.
    rows:
        Number of squares along the height.
    square_size_mm:
        Side length of each checkerboard square in millimetres.
    marker_size_mm:
        Side length of the ArUco marker inside each white square in millimetres.
        Must be smaller than *square_size_mm*.
    margin_mm:
        White border around the pattern in millimetres.
    dpi:
        Output resolution in dots per inch.
    aruco_dict:
        ArUco dictionary name, e.g. ``"5x5_100"``. The dictionary must be large
        enough to provide a unique marker for every white square on the board.
    output:
        Destination file path.

    Returns
    -------
    Path
        Resolved path to the saved file.
    """
    if marker_size_mm >= square_size_mm:
        raise ValueError(
            f"marker_size_mm ({marker_size_mm}) must be smaller than "
            f"square_size_mm ({square_size_mm})."
        )

    sq_px = _mm_to_px(square_size_mm, dpi)
    mk_px = _mm_to_px(marker_size_mm, dpi)
    margin_px = _mm_to_px(margin_mm, dpi)

    board_w_px = cols * sq_px
    board_h_px = rows * sq_px
    total_w_px = board_w_px + 2 * margin_px
    total_h_px = board_h_px + 2 * margin_px

    dictionary = _get_aruco_dict(aruco_dict)
    board = cv2.aruco.CharucoBoard(
        (cols, rows),
        float(sq_px),
        float(mk_px),
        dictionary,
    )

    board_img = board.generateImage((board_w_px, board_h_px))

    canvas = np.full((total_h_px, total_w_px), 255, dtype=np.uint8)
    canvas[margin_px : margin_px + board_h_px, margin_px : margin_px + board_w_px] = board_img

    path = Path(output).resolve()
    _save(canvas, path)
    return path


# ---------------------------------------------------------------------------
# Asymmetric circle grid
# ---------------------------------------------------------------------------


def generate_circles(
    cols: int = 4,
    rows: int = 11,
    spacing_mm: float = 20.0,
    radius_mm: float = 5.0,
    margin_mm: float = 15.0,
    dpi: float = 300.0,
    output: str | Path = "circles.png",
) -> Path:
    """Generate an asymmetric circle grid calibration pattern and save it to *output*.

    In an asymmetric grid, odd rows are offset by half the column spacing,
    breaking the grid symmetry so that orientation can be determined unambiguously.

    Parameters
    ----------
    cols:
        Number of circle columns.
    rows:
        Number of circle rows.
    spacing_mm:
        Centre-to-centre distance between adjacent circles in millimetres.
    radius_mm:
        Radius of each circle in millimetres. Should be less than
        ``spacing_mm / 2``.
    margin_mm:
        White border around the pattern in millimetres.
    dpi:
        Output resolution in dots per inch.
    output:
        Destination file path.

    Returns
    -------
    Path
        Resolved path to the saved file.
    """
    if radius_mm >= spacing_mm / 2:
        raise ValueError(
            f"radius_mm ({radius_mm}) must be less than spacing_mm/2 ({spacing_mm/2:.1f})."
        )

    sp = _mm_to_px(spacing_mm, dpi)
    r = _mm_to_px(radius_mm, dpi)
    mg = _mm_to_px(margin_mm, dpi)

    # Asymmetric grid: odd rows are shifted right by sp/2, and vertically
    # the row pitch is sp/2
    grid_w = (cols - 1) * sp + sp // 2  # extra half-step for offset rows
    grid_h = (rows - 1) * (sp // 2)

    total_w = grid_w + 2 * mg
    total_h = grid_h + 2 * mg

    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)

    for row in range(rows):
        y = mg + row * (sp // 2)
        x_offset = (sp // 2) if row % 2 == 1 else 0
        for col in range(cols):
            x = mg + col * sp + x_offset
            cv2.circle(canvas, (x, y), r, 0, thickness=-1, lineType=cv2.LINE_AA)

    path = Path(output).resolve()
    _save(canvas, path)
    return path


# ---------------------------------------------------------------------------
# Registry — maps pattern name -> (generate_fn, description)
# ---------------------------------------------------------------------------

PATTERNS: dict[str, tuple] = {
    "checkerboard": (generate_checkerboard, "Classic black/white checkerboard grid"),
    "charuco":      (generate_charuco,      "Checkerboard with embedded ArUco markers"),
    "circles":      (generate_circles,      "Asymmetric circle grid"),
}
