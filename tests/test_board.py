"""Tests for calicam.board."""

import numpy as np
import pytest

from calicam.board import (
    PATTERNS,
    _get_aruco_dict,
    _mm_to_px,
    generate_charuco,
    generate_checkerboard,
    generate_circles,
)


# ---------------------------------------------------------------------------
# _mm_to_px
# ---------------------------------------------------------------------------


def test_mm_to_px_known_value():
    # 25.4 mm at 100 dpi == exactly 100 px
    assert _mm_to_px(25.4, 100.0) == 100


def test_mm_to_px_rounds():
    # 10 mm at 72 dpi = 10 * 72 / 25.4 ≈ 28.35 → rounds to 28
    assert _mm_to_px(10.0, 72.0) == 28


def test_mm_to_px_zero():
    assert _mm_to_px(0.0, 300.0) == 0


# ---------------------------------------------------------------------------
# generate_checkerboard
# ---------------------------------------------------------------------------


def test_generate_checkerboard_creates_file(tmp_path):
    out = tmp_path / "board.png"
    result = generate_checkerboard(cols=4, rows=3, dpi=72.0, output=out)
    assert result == out
    assert out.exists()


def test_generate_checkerboard_dimensions(tmp_path):
    import cv2

    cols, rows, sq_mm, margin_mm, dpi = 4, 3, 20.0, 5.0, 72.0
    out = tmp_path / "board.png"
    generate_checkerboard(
        cols=cols, rows=rows, square_size_mm=sq_mm,
        margin_mm=margin_mm, dpi=dpi, output=out,
    )
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert img is not None

    n_sq_x = cols + 1
    n_sq_y = rows + 1
    expected_w = _mm_to_px(n_sq_x * sq_mm + 2 * margin_mm, dpi)
    expected_h = _mm_to_px(n_sq_y * sq_mm + 2 * margin_mm, dpi)
    assert img.shape == (expected_h, expected_w)


def test_generate_checkerboard_has_black_and_white(tmp_path):
    import cv2

    out = tmp_path / "board.png"
    generate_checkerboard(cols=4, rows=3, dpi=72.0, output=out)
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert img.min() == 0    # has black pixels
    assert img.max() == 255  # has white pixels


def test_generate_checkerboard_creates_parent_dirs(tmp_path):
    out = tmp_path / "deep" / "nested" / "board.png"
    generate_checkerboard(cols=3, rows=2, dpi=72.0, output=out)
    assert out.exists()


# ---------------------------------------------------------------------------
# generate_charuco
# ---------------------------------------------------------------------------


def test_generate_charuco_creates_file(tmp_path):
    out = tmp_path / "charuco.png"
    result = generate_charuco(
        cols=4, rows=4, square_size_mm=20.0, marker_size_mm=14.0,
        dpi=72.0, output=out,
    )
    assert result == out
    assert out.exists()


def test_generate_charuco_marker_too_large_raises(tmp_path):
    with pytest.raises(ValueError, match="marker_size_mm"):
        generate_charuco(
            square_size_mm=20.0, marker_size_mm=20.0,
            output=tmp_path / "x.png",
        )


def test_generate_charuco_invalid_dict_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown ArUco"):
        generate_charuco(aruco_dict="bad_dict", output=tmp_path / "x.png")


def test_get_aruco_dict_valid():
    import cv2
    d = _get_aruco_dict("5x5_100")
    assert isinstance(d, cv2.aruco.Dictionary)


def test_get_aruco_dict_case_insensitive():
    import cv2
    d = _get_aruco_dict("4X4_50")
    assert isinstance(d, cv2.aruco.Dictionary)


def test_get_aruco_dict_unknown_raises():
    with pytest.raises(ValueError, match="Unknown ArUco"):
        _get_aruco_dict("99x99_999")


# ---------------------------------------------------------------------------
# generate_circles
# ---------------------------------------------------------------------------


def test_generate_circles_creates_file(tmp_path):
    out = tmp_path / "circles.png"
    result = generate_circles(
        cols=3, rows=5, spacing_mm=15.0, radius_mm=4.0,
        dpi=72.0, output=out,
    )
    assert result == out
    assert out.exists()


def test_generate_circles_radius_too_large_raises(tmp_path):
    with pytest.raises(ValueError, match="radius_mm"):
        generate_circles(
            spacing_mm=10.0, radius_mm=5.0,  # radius == spacing/2 → invalid
            output=tmp_path / "x.png",
        )


def test_generate_circles_has_black_circles(tmp_path):
    import cv2

    out = tmp_path / "circles.png"
    generate_circles(cols=3, rows=5, dpi=72.0, output=out)
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert img.min() == 0    # circles are black
    assert img.max() == 255  # background is white


# ---------------------------------------------------------------------------
# PATTERNS registry
# ---------------------------------------------------------------------------


def test_patterns_registry_keys():
    assert set(PATTERNS.keys()) == {"checkerboard", "charuco", "circles"}


def test_patterns_registry_callables():
    for name, (fn, desc) in PATTERNS.items():
        assert callable(fn), f"{name} generator is not callable"
        assert isinstance(desc, str)
