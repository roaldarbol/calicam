"""Tests for calicam.calibration."""

import cv2
import numpy as np
import pytest

from calicam.calibration import CalibrationResult, Calibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_SIZE = (640, 480)

# Known camera intrinsics used to generate synthetic observations
_K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
_DIST = np.zeros(5, dtype=np.float64)


def _synthetic_corners(
    board_cols: int,
    board_rows: int,
    square_size: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Project 3-D board points into 2-D image corners using the known camera."""
    objp = np.zeros((board_cols * board_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size
    pts, _ = cv2.projectPoints(objp, rvec, tvec, _K, _DIST)
    return pts.astype(np.float32)


def _make_frames(board_cols=4, board_rows=3, square_size=25.0, n=5):
    """Return *n* synthetic corner arrays from distinct viewpoints."""
    frames = []
    for i in range(n):
        rvec = np.array([0.05 * i, 0.05 * (i % 3), 0.02 * i], dtype=np.float64)
        tvec = np.array([5.0 * i, 3.0 * i, 800.0 + 20.0 * i], dtype=np.float64)
        frames.append(_synthetic_corners(board_cols, board_rows, square_size, rvec, tvec))
    return frames


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------


def test_calibration_result_n_frames_empty(sample_result):
    assert sample_result.n_frames == 0


def test_calibration_result_n_frames_with_vecs(sample_result):
    sample_result.rvecs = [np.zeros(3)] * 3
    sample_result.tvecs = [np.zeros(3)] * 3
    assert sample_result.n_frames == 3


def test_calibration_result_image_size(sample_result):
    assert sample_result.image_size == (640, 480)


def test_calibration_result_reprojection_error(sample_result):
    assert sample_result.reprojection_error == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Calibrator — initialisation
# ---------------------------------------------------------------------------


def test_calibrator_defaults():
    cal = Calibrator()
    assert cal.board_cols == 9
    assert cal.board_rows == 6
    assert cal.board_size == (9, 6)
    assert cal.min_frames == 20


def test_calibrator_initial_state():
    cal = Calibrator()
    assert cal.n_frames == 0
    assert cal.result is None
    assert cal.best_result is None
    assert cal.reprojection_errors == []


def test_calibrator_can_calibrate_false_initially():
    cal = Calibrator(min_frames=20)
    assert not cal.can_calibrate


def test_calibrator_can_calibrate_true_at_min_frames():
    cal = Calibrator(board_cols=4, board_rows=3, min_frames=3)
    frames = _make_frames(n=3)
    for corners in frames:
        cal.add_frame(corners, IMAGE_SIZE)
    assert cal.can_calibrate


# ---------------------------------------------------------------------------
# Calibrator — should_auto_capture
# ---------------------------------------------------------------------------


def test_should_auto_capture_first_frame():
    cal = Calibrator(board_cols=4, board_rows=3)
    corners = _make_frames(n=1)[0]
    assert cal.should_auto_capture(corners)


def test_should_auto_capture_with_large_motion():
    cal = Calibrator(board_cols=4, board_rows=3, auto_min_motion=5.0)
    # Build two corner arrays that are 50 px apart
    base = _synthetic_corners(4, 3, 25.0, np.zeros(3), np.array([0.0, 0.0, 800.0]))
    shifted = base + 50.0  # move all corners 50 px
    cal.add_frame(base, IMAGE_SIZE)
    assert cal.should_auto_capture(shifted)


def test_should_auto_capture_without_motion():
    cal = Calibrator(board_cols=4, board_rows=3, auto_min_motion=100.0)
    corners = _make_frames(n=1)[0]
    cal.add_frame(corners, IMAGE_SIZE)
    # Same corners again — displacement is zero
    assert not cal.should_auto_capture(corners)


# ---------------------------------------------------------------------------
# Calibrator — add_frame / calibrate
# ---------------------------------------------------------------------------


def test_add_frame_returns_none_for_first_two():
    cal = Calibrator(board_cols=4, board_rows=3)
    frames = _make_frames(n=2)
    for corners in frames:
        result = cal.add_frame(corners, IMAGE_SIZE)
        assert result is None


def test_add_frame_returns_error_from_third():
    cal = Calibrator(board_cols=4, board_rows=3)
    frames = _make_frames(n=3)
    errors = [cal.add_frame(c, IMAGE_SIZE) for c in frames]
    assert errors[0] is None
    assert errors[1] is None
    assert isinstance(errors[2], float)
    assert errors[2] > 0


def test_calibrate_raises_with_fewer_than_3_frames():
    cal = Calibrator(board_cols=4, board_rows=3)
    with pytest.raises(ValueError, match="3 frames"):
        cal.calibrate(IMAGE_SIZE)


def test_calibrate_succeeds():
    cal = Calibrator(board_cols=4, board_rows=3, square_size_mm=25.0, min_frames=5)
    frames = _make_frames(board_cols=4, board_rows=3, n=5)
    for corners in frames:
        cal.add_frame(corners, IMAGE_SIZE)

    result = cal.calibrate(IMAGE_SIZE)

    assert isinstance(result, CalibrationResult)
    assert result.camera_matrix.shape == (3, 3)
    assert result.reprojection_error > 0
    assert result.image_size == IMAGE_SIZE
    assert result.n_frames == 5


def test_calibrate_best_result_tracking():
    cal = Calibrator(board_cols=4, board_rows=3, min_frames=3)
    frames = _make_frames(n=5)
    for corners in frames:
        cal.add_frame(corners, IMAGE_SIZE)

    # best_result is set once we have >= min_frames
    assert cal.best_result is not None
    assert cal.best_result.reprojection_error <= cal.result.reprojection_error


# ---------------------------------------------------------------------------
# Calibrator — reset
# ---------------------------------------------------------------------------


def test_reset_clears_all_state():
    cal = Calibrator(board_cols=4, board_rows=3)
    frames = _make_frames(n=4)
    for corners in frames:
        cal.add_frame(corners, IMAGE_SIZE)

    assert cal.n_frames == 4
    assert cal.result is not None

    cal.reset()

    assert cal.n_frames == 0
    assert cal.result is None
    assert cal.best_result is None
    assert cal.reprojection_errors == []


# ---------------------------------------------------------------------------
# Calibrator — detect (requires a real image)
# ---------------------------------------------------------------------------


def test_detect_returns_not_found_on_blank_image():
    cal = Calibrator(board_cols=4, board_rows=3)
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    found, corners, annotated = cal.detect(blank)
    assert not found
    assert corners is None
    assert annotated.shape == blank.shape


def test_detect_finds_board(checkerboard_frame):
    img, inner_cols, inner_rows = checkerboard_frame
    cal = Calibrator(board_cols=inner_cols, board_rows=inner_rows)
    found, corners, annotated = cal.detect(img)
    assert found, "Expected corners to be detected in the synthetic checkerboard image"
    assert corners is not None
    assert corners.shape == (inner_cols * inner_rows, 1, 2)
    assert annotated.shape == img.shape
