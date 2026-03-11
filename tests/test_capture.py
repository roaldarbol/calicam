"""Tests for calicam.capture.CameraSource."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from calicam.capture import CameraSource


# ---------------------------------------------------------------------------
# Properties before opening
# ---------------------------------------------------------------------------


def test_repr_camera():
    src = CameraSource(0)
    assert "camera" in repr(src)
    assert "0" in repr(src)


def test_repr_file():
    src = CameraSource("/tmp/video.mp4")
    assert "file" in repr(src)


def test_source_property_int():
    src = CameraSource(2)
    assert src.source == 2


def test_source_property_string():
    src = CameraSource("video.mp4")
    assert src.source == "video.mp4"


def test_source_property_path_converted_to_str():
    src = CameraSource(Path("/tmp/video.mp4"))
    assert isinstance(src.source, str)


def test_is_file_true_for_path_string():
    assert CameraSource("video.mp4").is_file is True
    assert CameraSource("/tmp/test.avi").is_file is True


def test_is_file_false_for_int():
    assert CameraSource(0).is_file is False
    assert CameraSource(1).is_file is False


def test_is_file_false_for_digit_string():
    # String "0" should be treated as a camera index
    assert CameraSource("0").is_file is False


# ---------------------------------------------------------------------------
# State before open
# ---------------------------------------------------------------------------


def test_is_open_false_before_open():
    assert not CameraSource(0).is_open


def test_read_before_open_returns_failure():
    src = CameraSource(0)
    ok, frame = src.read()
    assert not ok
    assert frame is None


def test_width_before_open_is_zero():
    assert CameraSource(0).width == 0


def test_height_before_open_is_zero():
    assert CameraSource(0).height == 0


def test_fps_before_open_is_default():
    assert CameraSource(0).fps == 30.0


# ---------------------------------------------------------------------------
# close() is safe to call multiple times
# ---------------------------------------------------------------------------


def test_close_without_open_is_safe():
    src = CameraSource(0)
    src.close()  # should not raise
    src.close()  # idempotent


# ---------------------------------------------------------------------------
# open() raises on invalid source
# ---------------------------------------------------------------------------


def test_open_invalid_source_raises():
    src = CameraSource("/nonexistent/path/video.mp4")
    with pytest.raises(RuntimeError, match="Cannot open"):
        src.open()


# ---------------------------------------------------------------------------
# Context manager (mocked VideoCapture)
# ---------------------------------------------------------------------------


def _make_mock_cap(is_opened=True, width=1280, height=720, fps=30.0):
    cap = MagicMock()
    cap.isOpened.return_value = is_opened
    cap.get.side_effect = lambda prop: {
        3: width,   # CAP_PROP_FRAME_WIDTH  == 3
        4: height,  # CAP_PROP_FRAME_HEIGHT == 4
        5: fps,     # CAP_PROP_FPS          == 5
    }.get(prop, 0)
    return cap


@patch("calicam.capture.cv2.VideoCapture")
def test_context_manager_opens_and_closes(mock_vc):
    mock_cap = _make_mock_cap()
    mock_vc.return_value = mock_cap

    with CameraSource(0) as src:
        assert src.is_open

    mock_cap.release.assert_called_once()
    assert not src.is_open


@patch("calicam.capture.cv2.VideoCapture")
def test_width_height_after_open(mock_vc):
    mock_cap = _make_mock_cap(width=1280, height=720)
    mock_vc.return_value = mock_cap

    src = CameraSource(0)
    src.open()
    assert src.width == 1280
    assert src.height == 720
    src.close()


@patch("calicam.capture.cv2.VideoCapture")
def test_fps_after_open(mock_vc):
    mock_cap = _make_mock_cap(fps=60.0)
    mock_vc.return_value = mock_cap

    src = CameraSource(0)
    src.open()
    assert src.fps == 60.0
    src.close()


@patch("calicam.capture.cv2.VideoCapture")
def test_read_success(mock_vc):
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = _make_mock_cap()
    mock_cap.read.return_value = (True, frame)
    mock_vc.return_value = mock_cap

    src = CameraSource(0)
    src.open()
    ok, f = src.read()
    assert ok
    assert f is not None
    src.close()


@patch("calicam.capture.cv2.VideoCapture")
def test_set_format_calls_cap_set(mock_vc):
    import cv2

    mock_cap = _make_mock_cap()
    mock_vc.return_value = mock_cap

    src = CameraSource(0)
    src.open()
    src.set_format(1920, 1080)

    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    src.close()


@patch("calicam.capture.cv2.VideoCapture")
def test_set_fps_calls_cap_set(mock_vc):
    import cv2

    mock_cap = _make_mock_cap()
    mock_vc.return_value = mock_cap

    src = CameraSource(0)
    src.open()
    src.set_fps(60)

    mock_cap.set.assert_called_with(cv2.CAP_PROP_FPS, 60)
    src.close()
