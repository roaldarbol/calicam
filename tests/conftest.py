"""Shared fixtures for calicam tests."""

import numpy as np
import pytest

from calicam.calibration import CalibrationResult


@pytest.fixture
def sample_result():
    """A CalibrationResult with known, arbitrary values."""
    return CalibrationResult(
        camera_matrix=np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist_coeffs=np.array([[0.1, -0.2, 0.001, 0.002, 0.05]], dtype=np.float64),
        reprojection_error=0.42,
        image_size=(640, 480),
    )


@pytest.fixture
def checkerboard_frame():
    """BGR image of a 4×3 inner-corner checkerboard with one-square border.

    Returns (image, inner_cols, inner_rows).
    """
    inner_cols, inner_rows, sq = 4, 3, 60
    n_sq_x = inner_cols + 1   # 5 squares wide
    n_sq_y = inner_rows + 1   # 4 squares tall
    pad = sq                   # one-square white border

    h = n_sq_y * sq + 2 * pad
    w = n_sq_x * sq + 2 * pad
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    for r in range(n_sq_y):
        for c in range(n_sq_x):
            if (r + c) % 2 != 0:   # black squares
                y0 = pad + r * sq
                x0 = pad + c * sq
                img[y0 : y0 + sq, x0 : x0 + sq] = 0

    return img, inner_cols, inner_rows
