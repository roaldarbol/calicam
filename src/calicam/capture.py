"""Camera source abstraction for USB cameras and video files."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2


class CameraSource:
    """Unified source for USB cameras (by index) and video files (by path).

    Parameters
    ----------
    source:
        Integer camera index for USB cameras, or a path to a video file.
    """

    def __init__(self, source: int | str | Path = 0) -> None:
        if isinstance(source, Path):
            source = str(source)
        self._source = source
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera or video file."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self._source!r}")

    def close(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Format control
    # ------------------------------------------------------------------

    def set_format(self, width: int, height: int) -> None:
        """Request a specific resolution from the camera.

        The driver may choose the closest supported resolution;
        read back ``width``/``height`` after opening to confirm.
        """
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_fps(self, fps: int) -> None:
        """Request a specific frame rate from the camera."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_FPS, fps)

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, cv2.Mat | None]:
        """Read the next frame. Returns (success, frame)."""
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret and self.is_file:
            # Loop video files back to the start
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return ret, frame

    def frames(self) -> Generator[cv2.Mat, None, None]:
        """Yield frames indefinitely (loops video files)."""
        while True:
            ret, frame = self.read()
            if ret and frame is not None:
                yield frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_file(self) -> bool:
        return isinstance(self._source, str) and not str(self._source).isdigit()

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 30.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    @property
    def source(self) -> int | str:
        return self._source

    def __repr__(self) -> str:
        kind = "file" if self.is_file else "camera"
        return f"CameraSource({kind}={self._source!r})"
