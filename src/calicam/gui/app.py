"""Dear PyGui application for interactive camera calibration."""

from __future__ import annotations

import contextlib
import os
import queue
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from ..calibration import Calibrator
from ..capture import CameraSource
from ..io import save_json, save_yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISPLAY_W = 640
_DISPLAY_H = 480
_CORRECTED_H = 240  # undistorted preview shown below main feed
_SIDEBAR_W = 320
_PLOT_H = 160
_ERROR_GOOD = 0.5
_ERROR_OK = 1.0

_PROBE_RESOLUTIONS = [
    (320, 240),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 1024),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]


# ---------------------------------------------------------------------------
# Startup helpers — run once before the window opens
# ---------------------------------------------------------------------------


def _query_win32_camera_names() -> list[str]:
    """Return camera friendly names on Windows in device-index order.

    Uses PowerShell ``Get-PnpDevice`` with a wmic fallback.
    Returns an empty list on non-Windows or if the query fails.
    """
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "(Get-PnpDevice -Class Camera -Status OK | "
                "Sort-Object InstanceId).FriendlyName",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        names = [s.strip() for s in out.splitlines() if s.strip()]
        if names:
            return names
    except Exception:
        pass
    # Fallback: wmic
    try:
        out = subprocess.check_output(
            [
                "wmic", "path", "Win32_PnPEntity",
                "where", "PNPClass='Camera'",
                "get", "Name", "/value",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return [
            line[5:].strip()
            for line in out.splitlines()
            if line.startswith("Name=") and line[5:].strip()
        ]
    except Exception:
        return []


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect C-level stderr to /dev/null to silence OpenCV backend noise."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _enumerate_all(
    max_index: int = 6,
) -> tuple[list[str], dict[int, list[str]]]:
    """Probe all cameras once at startup.

    Opens each camera index exactly once and collects supported resolutions
    and frame rates in the same session (avoiding a second open for format probing).

    Returns
    -------
    labels
        Combo-box strings such as ``["0 (HD USB Camera)", "1"]``.
    formats_by_index
        ``{0: ["640x480 (30 fps)", "1280x720 (60 fps)"], 1: [...]}``
    """
    available: list[int] = []
    formats_by_index: dict[int, list[str]] = {}

    for i in range(max_index):
        with _suppress_c_stderr():
            cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        available.append(i)
        seen: set[tuple[int, int]] = set()
        for w, h in _PROBE_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (aw, ah) in seen:
                continue
            seen.add((aw, ah))
        cap.release()
        pairs = sorted(seen, key=lambda t: t[0] * t[1], reverse=True)
        fmts = [f"{aw}x{ah}" for aw, ah in pairs]
        formats_by_index[i] = fmts or ["640x480"]

    if not available:
        return ["0"], {0: ["640x480"]}

    win_names = _query_win32_camera_names()
    labels: list[str] = []
    for pos, idx in enumerate(available):
        if pos < len(win_names):
            labels.append(f"{idx} ({win_names[pos]})")
        else:
            labels.append(str(idx))

    return labels, formats_by_index


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def _frame_to_rgba_flat(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert a 24bpp frame to a flat float32 RGBA array for Dear PyGui.

    Letterboxes the frame to fit (w, h) while preserving aspect ratio.
    Black bars are added on whichever axis has excess space.
    """
    fh, fw = frame.shape[:2]
    if fw != w or fh != h:
        scale = min(w / fw, h / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        frame = canvas
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    return (rgba.astype(np.float32) / 255.0).flatten()


def _error_colour(error: float | None) -> tuple[int, int, int, int]:
    if error is None:
        return (180, 180, 180, 255)
    if error <= _ERROR_GOOD:
        return (80, 200, 100, 255)
    if error <= _ERROR_OK:
        return (230, 180, 50, 255)
    return (220, 70, 70, 255)


# ---------------------------------------------------------------------------
# Worker thread result
# ---------------------------------------------------------------------------


class _FrameResult:
    __slots__ = ("raw", "annotated", "found", "corners", "image_size")

    def __init__(
        self,
        raw: np.ndarray,
        annotated: np.ndarray,
        found: bool,
        corners: np.ndarray | None,
        image_size: tuple[int, int],
    ) -> None:
        self.raw = raw
        self.annotated = annotated
        self.found = found
        self.corners = corners
        self.image_size = image_size


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class CaliCamApp:
    """Main Dear PyGui calibration application."""

    def __init__(
        self,
        source: int | str | Path = 0,
        board_cols: int = 9,
        board_rows: int = 6,
        square_size_mm: float = 25.0,
        min_frames: int = 20,
        auto_capture: bool = True,
        output_dir: str | Path = ".",
    ) -> None:
        self._source_val: int | str | Path = source
        self._board_cols = board_cols
        self._board_rows = board_rows
        self._square_size_mm = square_size_mm
        self._min_frames = min_frames
        self._auto_capture = auto_capture
        self._output_dir = Path(output_dir)

        self._desired_w: int | None = None
        self._desired_h: int | None = None

        # Populated in run() before the window opens
        self._camera_labels: list[str] = []
        self._camera_formats: dict[int, list[str]] = {}

        self._calibrator = Calibrator(
            board_cols=board_cols,
            board_rows=board_rows,
            square_size_mm=square_size_mm,
            min_frames=min_frames,
        )

        self._camera: CameraSource | None = None
        self._capture_running = False

        self._latest_raw: np.ndarray | None = None
        self._latest_raw_lock = threading.Lock()
        self._new_raw_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._detect_thread: threading.Thread | None = None
        self._frame_queue: queue.Queue[_FrameResult] = queue.Queue(maxsize=4)

        self._last_result: _FrameResult | None = None
        self._corrected_visible = False

        # DPG tag constants
        self._texture_tag = "feed_texture"
        self._corrected_texture_tag = "corrected_texture"
        self._image_tag = "feed_image"
        self._corrected_image_tag = "corrected_image"
        self._corrected_label_tag = "corrected_label"
        self._status_tag = "status_text"
        self._frames_tag = "frames_count"
        self._error_tag = "repr_error"
        self._best_error_tag = "best_repr_error"
        self._plot_x_tag = "plot_x"
        self._capture_btn_tag = "btn_capture"
        self._calibrate_btn_tag = "btn_calibrate"
        self._save_yaml_tag = "btn_save_yaml"
        self._save_json_tag = "btn_save_json"
        self._format_combo_tag = "combo_format"
        self._camera_combo_tag = "combo_camera"
        self._startstop_btn_tag = "btn_startstop"
        self._apply_board_tag = "btn_apply_board"

        # Tags that are locked while the camera is streaming
        self._source_tags = [
            self._camera_combo_tag,
            self._format_combo_tag,
        ]
        self._board_tags = [
            "input_cols",
            "input_rows",
            "input_square",
            self._apply_board_tag,
        ]

    # ------------------------------------------------------------------
    # Worker threads
    # ------------------------------------------------------------------

    def _capture_worker(self) -> None:
        assert self._camera is not None
        for frame in self._camera.frames():
            if not self._capture_running:
                break
            with self._latest_raw_lock:
                self._latest_raw = frame
            self._new_raw_event.set()

    def _detect_worker(self) -> None:
        assert self._camera is not None
        while self._capture_running:
            if not self._new_raw_event.wait(timeout=0.1):
                continue
            self._new_raw_event.clear()
            with self._latest_raw_lock:
                frame = self._latest_raw
            if frame is None:
                continue
            image_size = (self._camera.width, self._camera.height)
            found, corners, annotated = self._calibrator.detect(frame)
            result = _FrameResult(
                raw=frame,
                annotated=annotated,
                found=found,
                corners=corners,
                image_size=image_size,
            )
            try:
                self._frame_queue.put_nowait(result)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._frame_queue.put_nowait(result)

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def _start_camera(self) -> None:
        if self._capture_running:
            return
        source = self._source_val
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass
        self._camera = CameraSource(source)
        try:
            self._camera.open()
        except RuntimeError as exc:
            self._set_status(str(exc), error=True)
            self._camera = None
            return

        if self._desired_w is not None and self._desired_h is not None:
            self._camera.set_format(self._desired_w, self._desired_h)

        self._capture_running = True
        self._latest_raw = None
        self._new_raw_event.clear()

        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._detect_thread = threading.Thread(target=self._detect_worker, daemon=True)
        self._capture_thread.start()
        self._detect_thread.start()
        self._set_status("Camera running")

    def _stop_camera(self) -> None:
        self._capture_running = False
        self._new_raw_event.set()
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        self._set_status("Camera stopped")

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------

    def _set_streaming_ui(self, streaming: bool) -> None:
        """Enable/disable controls that must be locked while streaming."""
        for tag in self._source_tags + self._board_tags:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=not streaming)

    def _set_status(self, text: str, *, error: bool = False) -> None:
        if dpg.does_item_exist(self._status_tag):
            colour = (220, 70, 70, 255) if error else (180, 180, 180, 255)
            dpg.set_value(self._status_tag, text)
            dpg.configure_item(self._status_tag, color=colour)

    def _update_texture(self, frame: np.ndarray) -> None:
        if dpg.does_item_exist(self._texture_tag):
            dpg.set_value(self._texture_tag, _frame_to_rgba_flat(frame, _DISPLAY_W, _DISPLAY_H))

    def _update_corrected_texture(self, frame: np.ndarray) -> None:
        if dpg.does_item_exist(self._corrected_texture_tag):
            dpg.set_value(
                self._corrected_texture_tag,
                _frame_to_rgba_flat(frame, _DISPLAY_W, _CORRECTED_H),
            )

    def _update_error_display(self, error: float | None) -> None:
        if not dpg.does_item_exist(self._error_tag):
            return
        dpg.set_value(self._error_tag, "---" if error is None else f"{error:.4f} px")
        dpg.configure_item(self._error_tag, color=_error_colour(error))

    def _update_best_error_display(self) -> None:
        if not dpg.does_item_exist(self._best_error_tag):
            return
        best = self._calibrator.best_result
        if best is None:
            dpg.set_value(self._best_error_tag, "---")
            dpg.configure_item(self._best_error_tag, color=_error_colour(None))
        else:
            dpg.set_value(
                self._best_error_tag,
                f"{best.reprojection_error:.4f} px  ({best.n_frames} frames)",
            )
            dpg.configure_item(
                self._best_error_tag, color=_error_colour(best.reprojection_error)
            )

    def _update_plot(self) -> None:
        errors = self._calibrator.reprojection_errors
        if not errors or not dpg.does_item_exist(self._plot_x_tag):
            return
        xs = list(range(1, len(errors) + 1))
        dpg.set_value(self._plot_x_tag, [xs, errors])
        dpg.fit_axis_data("plot_x_axis")
        dpg.fit_axis_data("plot_y_axis")

    def _show_corrected_feed(self, visible: bool) -> None:
        self._corrected_visible = visible
        for tag in (self._corrected_label_tag, self._corrected_image_tag):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=visible)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cb_start_stop(self, sender, _app_data) -> None:
        label = dpg.get_item_label(sender)
        if label == "Start":
            raw = dpg.get_value(self._camera_combo_tag)
            # Label may be "0 (Name)" — extract the leading index
            try:
                self._source_val = int(raw.split()[0])
            except (ValueError, IndexError):
                self._source_val = raw

            fmt = dpg.get_value(self._format_combo_tag)
            if fmt and "x" in fmt:
                try:
                    w_str, h_str = fmt.split("x")
                    self._desired_w, self._desired_h = int(w_str), int(h_str)
                except ValueError:
                    self._desired_w = self._desired_h = None

            self._start_camera()
            if self._capture_running:
                dpg.configure_item(sender, label="Stop")
                self._set_streaming_ui(True)
        else:
            self._stop_camera()
            dpg.configure_item(sender, label="Start")
            self._set_streaming_ui(False)

    def _cb_camera_changed(self, _sender, app_data) -> None:
        """Update format combo instantly from the pre-built cache."""
        try:
            idx = int(str(app_data).split()[0])
        except (ValueError, IndexError):
            return
        formats = self._camera_formats.get(idx, ["640x480"])
        if dpg.does_item_exist(self._format_combo_tag):
            dpg.configure_item(self._format_combo_tag, items=formats)
            dpg.set_value(self._format_combo_tag, formats[0])

    def _cb_auto_toggle(self, _sender, app_data) -> None:
        self._auto_capture = bool(app_data)

    def _cb_manual_capture(self, _sender, _app_data) -> None:
        if self._last_result and self._last_result.found:
            self._do_capture(self._last_result)

    def _cb_calibrate(self, _sender, _app_data) -> None:
        if self._last_result is None:
            return
        try:
            result = self._calibrator.calibrate(self._last_result.image_size)
            self._update_error_display(result.reprojection_error)
            self._update_best_error_display()
            self._update_plot()
            best = self._calibrator.best_result
            assert best is not None
            self._set_status(
                f"Calibrated  ({result.n_frames} frames, RMS={result.reprojection_error:.4f})"
                f"  |  Best: {best.reprojection_error:.4f} ({best.n_frames} frames)"
            )
            dpg.configure_item(self._save_yaml_tag, enabled=True)
            dpg.configure_item(self._save_json_tag, enabled=True)
            self._show_corrected_feed(True)
        except ValueError as exc:
            self._set_status(str(exc), error=True)

    def _cb_reset(self, _sender, _app_data) -> None:
        self._calibrator.reset()
        if dpg.does_item_exist(self._frames_tag):
            dpg.set_value(self._frames_tag, f"0 / {self._min_frames}")
        self._update_error_display(None)
        self._update_best_error_display()
        if dpg.does_item_exist(self._plot_x_tag):
            dpg.set_value(self._plot_x_tag, [[], []])
        dpg.configure_item(self._save_yaml_tag, enabled=False)
        dpg.configure_item(self._save_json_tag, enabled=False)
        dpg.configure_item(self._calibrate_btn_tag, enabled=False)
        self._show_corrected_feed(False)
        blank = np.zeros((_CORRECTED_H, _DISPLAY_W, 4), dtype=np.float32).flatten()
        if dpg.does_item_exist(self._corrected_texture_tag):
            dpg.set_value(self._corrected_texture_tag, blank)
        self._set_status("Reset")

    def _make_output_stem(self) -> str:
        """Build a descriptive filename stem: calibration_cam{idx}_{name}_{WxH}_{ts}.

        The resolution is taken from the calibration result's stored image_size
        (i.e. what the camera driver actually delivered), not the requested format,
        so the filename always matches the YAML/JSON contents exactly.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        idx = str(self._source_val)
        label = dpg.get_value(self._camera_combo_tag) if dpg.does_item_exist(self._camera_combo_tag) else ""
        name_part = ""
        if "(" in label and label.endswith(")"):
            raw = label[label.index("(") + 1:-1]
            slug = re.sub(r"[^\w]+", "_", raw).strip("_")
            if slug:
                name_part = f"_{slug}"
        result = self._calibrator.best_result
        if result is not None:
            w, h = result.image_size
        else:
            w = self._desired_w or (self._camera.width if self._camera else 0)
            h = self._desired_h or (self._camera.height if self._camera else 0)
        fmt_part = f"_{w}x{h}" if w and h else ""
        return f"calibration_cam{idx}{name_part}{fmt_part}_{ts}"

    def _cb_save_yaml(self, _sender, _app_data) -> None:
        result = self._calibrator.best_result
        if result is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out = self._output_dir / f"{self._make_output_stem()}.yaml"
        save_yaml(result, out)
        self._set_status(
            f"Saved {out.name}  ({result.reprojection_error:.4f} px, {result.n_frames} frames)"
        )

    def _cb_save_json(self, _sender, _app_data) -> None:
        result = self._calibrator.best_result
        if result is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out = self._output_dir / f"{self._make_output_stem()}.json"
        save_json(result, out)
        self._set_status(
            f"Saved {out.name}  ({result.reprojection_error:.4f} px, {result.n_frames} frames)"
        )

    def _cb_apply_board(self, _sender, _app_data) -> None:
        cols = int(dpg.get_value("input_cols"))
        rows = int(dpg.get_value("input_rows"))
        size = float(dpg.get_value("input_square"))
        self._calibrator.reset()
        self._calibrator.board_cols = cols
        self._calibrator.board_rows = rows
        self._calibrator.board_size = (cols, rows)
        self._calibrator.square_size_mm = size
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= size
        self._calibrator._objp = objp
        self._set_status(f"Board updated: {cols}×{rows}, {size}mm — captures reset")
        if dpg.does_item_exist(self._frames_tag):
            dpg.set_value(self._frames_tag, f"0 / {self._min_frames}")
        self._update_error_display(None)
        self._update_best_error_display()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _do_capture(self, result: _FrameResult) -> None:
        error = self._calibrator.add_frame(result.corners, result.image_size)
        n = self._calibrator.n_frames
        if dpg.does_item_exist(self._frames_tag):
            dpg.set_value(self._frames_tag, f"{n} / {self._min_frames}")
        if error is not None:
            self._update_error_display(error)
            self._update_best_error_display()
            self._update_plot()
        dpg.configure_item(self._calibrate_btn_tag, enabled=(n >= self._min_frames))

    def _process_frame(self, result: _FrameResult) -> None:
        self._last_result = result
        self._update_texture(result.annotated if result.found else result.raw)

        if self._corrected_visible:
            best = self._calibrator.best_result
            if best is not None:
                corrected = cv2.undistort(result.raw, best.camera_matrix, best.dist_coeffs)
                self._update_corrected_texture(corrected)

        if self._auto_capture and result.found and result.corners is not None:
            if self._calibrator.should_auto_capture(result.corners):
                self._do_capture(result)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    @staticmethod
    def _tip(text: str) -> None:
        """Attach a tooltip to the most recently added item."""
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text(text)

    def _build_ui(self) -> None:
        total_w = _DISPLAY_W + _SIDEBAR_W + 24
        total_h = _DISPLAY_H + 120

        with dpg.texture_registry():
            blank_main = np.zeros((_DISPLAY_H, _DISPLAY_W, 4), dtype=np.float32).flatten()
            dpg.add_raw_texture(
                width=_DISPLAY_W, height=_DISPLAY_H,
                default_value=blank_main,
                format=dpg.mvFormat_Float_rgba,
                tag=self._texture_tag,
            )
            blank_corr = np.zeros((_CORRECTED_H, _DISPLAY_W, 4), dtype=np.float32).flatten()
            dpg.add_raw_texture(
                width=_DISPLAY_W, height=_CORRECTED_H,
                default_value=blank_corr,
                format=dpg.mvFormat_Float_rgba,
                tag=self._corrected_texture_tag,
            )

        with dpg.window(
            label="calicam", tag="main_window",
            width=total_w, height=total_h,
            no_close=True, no_resize=False,
        ):
            with dpg.table(header_row=False, borders_innerV=True):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=_DISPLAY_W + 4)
                dpg.add_table_column()

                with dpg.table_row():
                    # ---- LEFT: camera feeds ----
                    with dpg.table_cell():
                        dpg.add_image(
                            self._texture_tag, width=_DISPLAY_W, height=_DISPLAY_H,
                            tag=self._image_tag,
                        )
                        self._tip("Live camera feed. Detected checkerboard corners are drawn when the board is visible.")
                        dpg.add_text("", tag=self._status_tag, color=(180, 180, 180, 255))
                        dpg.add_text(
                            "Undistorted (best calibration)",
                            tag=self._corrected_label_tag,
                            color=(140, 140, 140, 255),
                            show=False,
                        )
                        dpg.add_image(
                            self._corrected_texture_tag,
                            width=_DISPLAY_W, height=_CORRECTED_H,
                            tag=self._corrected_image_tag,
                            show=False,
                        )
                        self._tip(
                            "Raw frame corrected with the best calibration found so far.\n"
                            "Straight lines in the scene should appear straight here.\n"
                            "Black borders are expected — they are areas with no image data after undistortion."
                        )

                    # ---- RIGHT: controls ----
                    with dpg.table_cell():
                        # -- Board settings --
                        with dpg.collapsing_header(label="Board", default_open=True):
                            dpg.add_input_int(
                                label="Cols (inner)", default_value=self._board_cols,
                                min_value=3, max_value=20, tag="input_cols", width=80,
                            )
                            self._tip(
                                "Number of inner corners along the board WIDTH.\n"
                                "This is one less than the number of squares per row.\n"
                                "e.g. a 10x7 square board has 9 inner corner columns."
                            )
                            dpg.add_input_int(
                                label="Rows (inner)", default_value=self._board_rows,
                                min_value=3, max_value=20, tag="input_rows", width=80,
                            )
                            self._tip(
                                "Number of inner corners along the board HEIGHT.\n"
                                "This is one less than the number of squares per column."
                            )
                            dpg.add_input_float(
                                label="Square (mm)", default_value=self._square_size_mm,
                                min_value=1.0, tag="input_square", width=80,
                            )
                            self._tip(
                                "Physical size of one square in millimetres.\n"
                                "Measure carefully — this determines the real-world scale of the calibration."
                            )
                            dpg.add_button(
                                label="Apply",
                                tag=self._apply_board_tag,
                                callback=self._cb_apply_board,
                                width=80,
                            )
                            self._tip("Apply board settings. This resets all captured frames.")

                        dpg.add_spacer(height=4)

                        # -- Source --
                        with dpg.collapsing_header(label="Source", default_open=True):
                            labels = self._camera_labels
                            src_str = str(self._source_val)
                            default_label = next(
                                (lb for lb in labels if lb.split()[0] == src_str),
                                labels[0],
                            )
                            dpg.add_combo(
                                label="Camera",
                                items=labels,
                                default_value=default_label,
                                tag=self._camera_combo_tag,
                                callback=self._cb_camera_changed,
                                width=-1,
                            )
                            self._tip("Camera device. Locked while streaming.")
                            default_idx = int(default_label.split()[0])
                            default_fmts = self._camera_formats.get(default_idx, ["640x480"])
                            dpg.add_combo(
                                label="Format",
                                items=default_fmts,
                                default_value=default_fmts[0],
                                tag=self._format_combo_tag,
                                width=-1,
                            )
                            self._tip(
                                "Capture resolution. Higher resolutions give more accurate calibration\n"
                                "but corner detection is slower. Locked while streaming."
                            )
                            dpg.add_button(
                                label="Start",
                                tag=self._startstop_btn_tag,
                                callback=self._cb_start_stop,
                                width=80,
                            )
                            self._tip("Start or stop the camera stream.")

                        dpg.add_spacer(height=4)

                        # -- Capture --
                        with dpg.collapsing_header(label="Capture", default_open=True):
                            dpg.add_checkbox(
                                label="Auto-capture",
                                default_value=self._auto_capture,
                                callback=self._cb_auto_toggle,
                            )
                            self._tip(
                                "Automatically save a frame whenever the board has moved\n"
                                "enough from the previous capture. Recommended — move the\n"
                                "board slowly and vary position, angle, and distance."
                            )
                            dpg.add_spacer(height=4)
                            with dpg.group(horizontal=True):
                                dpg.add_button(
                                    label="Capture",
                                    tag=self._capture_btn_tag,
                                    callback=self._cb_manual_capture,
                                    width=80,
                                )
                                self._tip("Manually save the current frame (board must be visible).")
                                dpg.add_button(
                                    label="Reset",
                                    callback=self._cb_reset,
                                    width=80,
                                )
                                self._tip("Discard all captured frames and start over.")
                            dpg.add_spacer(height=4)
                            dpg.add_text(f"0 / {self._min_frames}", tag=self._frames_tag)
                            self._tip(f"Frames captured / target minimum ({self._min_frames}).")
                            dpg.add_text("Frames captured", color=(140, 140, 140, 255))

                        dpg.add_spacer(height=4)

                        # -- Reprojection error --
                        with dpg.collapsing_header(label="Reprojection Error", default_open=True):
                            dpg.add_text("Live", color=(140, 140, 140, 255))
                            self._tip(
                                "RMS reprojection error from the most recent incremental calibration.\n"
                                "This is a training-set error — it is expected to rise as more\n"
                                "diverse frames are added (a sign of better generalization, not worse).\n"
                                "Stop when the graph flattens out."
                            )
                            dpg.add_text("---", tag=self._error_tag, color=(180, 180, 180, 255))
                            dpg.add_spacer(height=4)
                            dpg.add_text("Best (exported on save)", color=(140, 140, 140, 255))
                            self._tip(
                                "Lowest RMS error seen once the minimum frame count is reached.\n"
                                "This is the calibration that will be written to disk.\n"
                                "Adding more frames may or may not improve it."
                            )
                            dpg.add_text("---", tag=self._best_error_tag, color=(180, 180, 180, 255))
                            dpg.add_spacer(height=4)
                            dpg.add_text(
                                "< 0.5 excellent   0.5-1.0 good   > 1.0 re-capture",
                                color=(110, 110, 110, 255),
                            )
                            self._tip(
                                "Rule of thumb for RMS reprojection error in pixels.\n"
                                "Calibrations with < 0.5 px error are considered excellent.\n"
                                "Note: a suspiciously low error (e.g. < 0.3 px) with few frames\n"
                                "likely indicates overfitting — collect more diverse poses."
                            )
                            dpg.add_spacer(height=4)
                            with dpg.plot(label="", height=_PLOT_H, width=-1, no_title=True):
                                dpg.add_plot_axis(dpg.mvXAxis, label="frames", tag="plot_x_axis")
                                with dpg.plot_axis(dpg.mvYAxis, label="RMS px", tag="plot_y_axis"):
                                    dpg.add_line_series(
                                        [], [], label="reprojection error",
                                        tag=self._plot_x_tag,
                                    )

                        dpg.add_spacer(height=4)

                        # -- Calibrate & Save --
                        with dpg.collapsing_header(label="Calibrate & Save", default_open=True):
                            dpg.add_button(
                                label="Calibrate",
                                tag=self._calibrate_btn_tag,
                                callback=self._cb_calibrate,
                                width=-1,
                                enabled=False,
                            )
                            self._tip(
                                "Run a final calibration pass using all captured frames.\n"
                                "Enables the undistorted preview and the save buttons.\n"
                                "Available once the minimum frame count is reached."
                            )
                            dpg.add_spacer(height=4)
                            dpg.add_button(
                                label="Save YAML",
                                tag=self._save_yaml_tag,
                                callback=self._cb_save_yaml,
                                width=-1,
                                enabled=False,
                            )
                            self._tip(
                                "Save the best calibration to a YAML file.\n"
                                "Filename encodes camera, resolution, and timestamp.\n"
                                "Contains: camera_matrix, dist_coeffs, image_size, rms_error."
                            )
                            dpg.add_button(
                                label="Save JSON",
                                tag=self._save_json_tag,
                                callback=self._cb_save_json,
                                width=-1,
                                enabled=False,
                            )
                            self._tip(
                                "Save the best calibration to a JSON file.\n"
                                "Filename encodes camera, resolution, and timestamp.\n"
                                "Contains: camera_matrix, dist_coeffs, image_size, rms_error."
                            )

        dpg.set_primary_window("main_window", True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the Dear PyGui application."""
        # Probe cameras before creating the window so the UI is fully
        # populated the moment it appears — no loading state needed.
        self._camera_labels, self._camera_formats = _enumerate_all()

        dpg.create_context()
        self._build_ui()

        dpg.create_viewport(
            title="calicam",
            width=_DISPLAY_W + _SIDEBAR_W + 40,
            height=_DISPLAY_H + 140,
            min_width=600,
            min_height=400,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            latest: _FrameResult | None = None
            while True:
                try:
                    latest = self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                self._process_frame(latest)

            dpg.render_dearpygui_frame()

        self._stop_camera()
        dpg.destroy_context()
