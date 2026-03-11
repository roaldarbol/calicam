"""Checkerboard detection and camera calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    """Intrinsic calibration result for a single camera.

    Parameters
    ----------
    camera_matrix:
        3x3 intrinsic matrix (fx, fy, cx, cy).
    dist_coeffs:
        Distortion coefficients (k1, k2, p1, p2[, k3[, ...]]).
    reprojection_error:
        RMS reprojection error in pixels.
    image_size:
        (width, height) of the calibration images.
    rvecs:
        Per-image rotation vectors (may be empty after loading from file).
    tvecs:
        Per-image translation vectors (may be empty after loading from file).
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_error: float
    image_size: tuple[int, int]
    rvecs: list[np.ndarray] = field(default_factory=list, repr=False)
    tvecs: list[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def n_frames(self) -> int:
        """Number of frames used for this calibration."""
        return len(self.rvecs)


class Calibrator:
    """Incrementally captures checkerboard frames and calibrates a camera.

    Parameters
    ----------
    board_cols:
        Number of inner corners along the board width.
    board_rows:
        Number of inner corners along the board height.
    square_size_mm:
        Physical size of one square in millimetres.
    min_frames:
        Minimum number of captured frames before calibration is allowed.
    auto_min_motion:
        Minimum mean corner displacement (px) required before a frame is
        accepted in auto-capture mode.
    """

    def __init__(
        self,
        board_cols: int = 9,
        board_rows: int = 6,
        square_size_mm: float = 25.0,
        min_frames: int = 20,
        auto_min_motion: float = 20.0,
    ) -> None:
        self.board_cols = board_cols
        self.board_rows = board_rows
        self.board_size = (board_cols, board_rows)
        self.square_size_mm = square_size_mm
        self.min_frames = min_frames
        self.auto_min_motion = auto_min_motion

        self._object_points: list[np.ndarray] = []
        self._image_points: list[np.ndarray] = []
        self._reprojection_errors: list[float] = []
        self._result: CalibrationResult | None = None
        self._best_result: CalibrationResult | None = None

        # 3-D reference points for the board (z = 0 plane)
        objp = np.zeros((board_cols * board_rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
        objp *= square_size_mm
        self._objp = objp

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of successfully captured frames."""
        return len(self._image_points)

    @property
    def reprojection_errors(self) -> list[float]:
        """Running reprojection errors after each captured frame (>= 3 frames)."""
        return list(self._reprojection_errors)

    @property
    def result(self) -> CalibrationResult | None:
        """Most recent calibration result, or None if not yet calibrated."""
        return self._result

    @property
    def best_result(self) -> CalibrationResult | None:
        """Best calibration result seen so far (lowest RMS), or None."""
        return self._best_result

    @property
    def can_calibrate(self) -> bool:
        return self.n_frames >= max(3, self.min_frames)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self, frame: np.ndarray
    ) -> tuple[bool, np.ndarray | None, np.ndarray]:
        """Detect checkerboard corners in *frame*.

        Returns
        -------
        found:
            Whether the full board was detected.
        corners:
            Sub-pixel corner array of shape (N, 1, 2), or None.
        annotated:
            Copy of *frame* with corners drawn (if found).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)

        annotated = frame.copy()
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(annotated, self.board_size, corners, found)

        return found, (corners if found else None), annotated

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def should_auto_capture(self, corners: np.ndarray) -> bool:
        """Return True if *corners* is sufficiently different from the last capture."""
        if not self._image_points:
            return True
        last = self._image_points[-1]
        mean_dist = float(np.mean(np.linalg.norm(corners - last, axis=2)))
        return mean_dist >= self.auto_min_motion

    def add_frame(
        self, corners: np.ndarray, image_size: tuple[int, int]
    ) -> float | None:
        """Record *corners* from a detected board.

        Runs a fresh calibration after every capture (once >= 3 frames) so that
        the running reprojection error is always up to date.

        Returns the current reprojection error, or None if fewer than 3 frames
        have been captured yet.
        """
        self._object_points.append(self._objp.copy())
        self._image_points.append(corners.copy())

        if self.n_frames >= 3:
            error = self._run_calibration(image_size)
            self._reprojection_errors.append(error)
            return error
        return None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, image_size: tuple[int, int]) -> CalibrationResult:
        """Run calibration with all captured frames and return the result."""
        if self.n_frames < 3:
            raise ValueError(
                f"At least 3 frames required, but only {self.n_frames} captured."
            )
        self._run_calibration(image_size)
        assert self._result is not None
        return self._result

    def _run_calibration(self, image_size: tuple[int, int]) -> float:
        w, h = image_size
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._object_points,
            self._image_points,
            (w, h),
            None,
            None,
        )
        self._result = CalibrationResult(
            camera_matrix=mtx,
            dist_coeffs=dist,
            reprojection_error=float(ret),
            image_size=(w, h),
            rvecs=list(rvecs),
            tvecs=list(tvecs),
        )
        if (
            self.n_frames >= self.min_frames
            and (
                self._best_result is None
                or float(ret) < self._best_result.reprojection_error
            )
        ):
            self._best_result = self._result
        return float(ret)

    def reset(self) -> None:
        """Discard all captured frames and results."""
        self._object_points.clear()
        self._image_points.clear()
        self._reprojection_errors.clear()
        self._result = None
        self._best_result = None
