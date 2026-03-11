"""Load and save calibration results in OpenCV YAML and JSON formats."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import CalibrationResult


# ---------------------------------------------------------------------------
# OpenCV YAML  (matches the format produced by cv::FileStorage)
# ---------------------------------------------------------------------------


def save_yaml(result: CalibrationResult, path: str | Path) -> None:
    """Write *result* to an OpenCV-compatible YAML file at *path*.

    The output format mirrors what OpenCV's ``cv::FileStorage`` produces and
    is compatible with the Bonsai / FreeMoCap ecosystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", result.image_size[0])
    fs.write("image_height", result.image_size[1])
    fs.write("camera_matrix", result.camera_matrix)
    fs.write("distortion_coefficients", result.dist_coeffs)
    fs.write("reprojection_error", result.reprojection_error)
    fs.release()


def load_yaml(path: str | Path) -> CalibrationResult:
    """Read a calibration result from an OpenCV YAML file."""
    path = Path(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {path}")

    width = int(fs.getNode("image_width").real())
    height = int(fs.getNode("image_height").real())
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    reprojection_error = float(fs.getNode("reprojection_error").real())
    fs.release()

    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        reprojection_error=reprojection_error,
        image_size=(width, height),
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def save_json(result: CalibrationResult, path: str | Path) -> None:
    """Write *result* to a JSON file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_width": result.image_size[0],
        "image_height": result.image_size[1],
        "camera_matrix": result.camera_matrix.tolist(),
        "distortion_coefficients": result.dist_coeffs.tolist(),
        "reprojection_error": result.reprojection_error,
    }
    path.write_text(json.dumps(data, indent=2))


def load_json(path: str | Path) -> CalibrationResult:
    """Read a calibration result from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return CalibrationResult(
        camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.array(data["distortion_coefficients"], dtype=np.float64),
        reprojection_error=float(data["reprojection_error"]),
        image_size=(int(data["image_width"]), int(data["image_height"])),
    )


# ---------------------------------------------------------------------------
# Auto-dispatch by extension
# ---------------------------------------------------------------------------


def save(result: CalibrationResult, path: str | Path) -> None:
    """Save *result* to *path*, inferring format from the file extension.

    Supported extensions: ``.yaml``, ``.yml``, ``.json``.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        save_yaml(result, path)
    elif ext == ".json":
        save_json(result, path)
    else:
        raise ValueError(f"Unsupported calibration format: {ext!r}")


def load(path: str | Path) -> CalibrationResult:
    """Load a calibration result from *path*, inferring format from the extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        return load_yaml(path)
    elif ext == ".json":
        return load_json(path)
    else:
        raise ValueError(f"Unsupported calibration format: {ext!r}")
