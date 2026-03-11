# calicam

Camera calibration tool for USB cameras.

## Features

- **Live streaming** from any USB/built-in camera (by index) or pre-recorded video file
- **Auto-capture mode** (default) — frames are accepted automatically when the board is detected and the camera has moved enough relative to the previous capture
- **Running reprojection error** — updates after every new frame so you can watch it stabilise in real time
- **Dear PyGui GUI** — GPU-accelerated, zero-dependency desktop UI
- **YAML output** matching OpenCV's `cv::FileStorage` format (compatible with Bonsai, FreeMoCap, etc.)
- **JSON output** for everything else
- CLI helpers: `convert`, `inspect`

## Installation

```bash
uv tool install calicam
```

Requires Python 3.10+.

## Quick start

```bash
calicam gui
```

```bash
# Convert between formats
calicam convert calibration.yaml            # -> calibration.json
calicam convert calibration.json -o cam0.yaml

# Print a summary
calicam inspect calibration.yaml
```

## GUI overview

| Panel | Description |
|---|---|
| Camera feed | Live frame with detected corners drawn |
| Source | Camera index or video file path; Start/Stop |
| Board | Inner corner grid size and square size |
| Capture | Auto/manual mode, frame counter, Reset |
| Reprojection Error | Current RMS (px) + live plot over captured frames |
| Calibrate & Save | Calibrate once target frame count reached; save YAML / JSON |

Reprojection error colour coding: **green** < 0.5 px · **yellow** 0.5–1.0 px · **red** > 1.0 px

## YAML format

```yaml
%YAML:1.0
image_width: 640
image_height: 480
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1.08e+03, 0., 1.93e+02, 0., 1.13e+03, 2.53e+02, 0., 0., 1. ]
distortion_coefficients: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [ -1.25, -4.12, -0.005, 0.163, 18.98 ]
reprojection_error: 8.39e-01
```

## Python API

```python
from calicam import Calibrator, CameraSource, save_yaml, load_yaml

# Headless calibration from a video file
with CameraSource("/path/to/video.mp4") as cam:
    calibrator = Calibrator(board_cols=9, board_rows=6, square_size_mm=25.0)
    for frame in cam.frames():
        found, corners, _ = calibrator.detect(frame)
        if found and calibrator.should_auto_capture(corners):
            error = calibrator.add_frame(corners, (cam.width, cam.height))
            if error is not None:
                print(f"  frames={calibrator.n_frames}  RMS={error:.4f}")
        if calibrator.n_frames >= 40:
            break

result = calibrator.calibrate((cam.width, cam.height))
save_yaml(result, "calibration.yaml")
```

## Generating calibration patterns

Print your own board at the correct physical size using the `generate` command.
The `--dpi` flag controls output resolution (300 is print-ready); the `--output` / `-o`
flag sets the filename — the extension determines the image format (`.png`, `.tiff`, `.pdf`, …).

```bash
# Checkerboard — 9x6 inner corners, 25 mm squares
calicam generate checkerboard --cols 9 --rows 6 --square-size 25 -o board.png

# ChArUco — 5x7 squares, embedded 5x5 ArUco markers
calicam generate charuco --cols 5 --rows 7 --square-size 30 --marker-size 22 -o charuco.png

# Asymmetric circle grid — 4 columns, 11 rows
calicam generate circles --cols 4 --rows 11 --spacing 20 --radius 5 -o circles.png
```

All commands report the output dimensions in both pixels and millimetres so you can
verify the print size before sending to a printer.

| Pattern | Best for |
|---|---|
| `checkerboard` | General use; the default for `calicam gui` |
| `charuco` | Partial occlusion, longer range, multi-camera |
| `circles` | Scenes with low texture; sub-pixel accuracy |

## Tips

- Use a **flat, rigid** checkerboard. Print on matte paper, mount on a hard surface.
- Aim for 20–40 frames covering the **full image area** — tilt and rotate the board.
- A reprojection error below **0.5 px** is excellent; below **1.0 px** is acceptable.
- If the error stays high, try removing outlier frames and recalibrating (Reset → recapture).
