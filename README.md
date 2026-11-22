# stray-scanner-dashboard

Turn a [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662) iPhone/iPad recording into **one dashboard MP4** that shows every stream in sync:

```
┌──────────────┬────────────────────┬──────────────────────┐
│     RGB      │  Depth (colourised) │  Confidence (0/1/2)  │   row 1: image streams, same timestamp
│  t, frame #  │  colourbar in m     │  legend + % per level│
├──────────────┴─────────┬──────────┴──────────────────────┤
│ IMU · accelerometer    │ IMU · gyroscope                 │   row 2: imu.csv
├────────────────────────┼─────────────────────────────────┤
│ Odometry · position    │ Odometry · orientation (Euler)  │   row 3: odometry.csv
└────────────────────────┴─────────────────────────────────┘
```

The time-series lines **grow as the video plays**: each frame plots data only up to "now", on axes fixed to
the whole recording's range (so nothing jumps), with a vertical cursor and a dot at the current value.

## Install

```powershell
python -m pip install -r requirements.txt
```

Python 3.10+. `imageio-ffmpeg` ships its own ffmpeg binary and writes H.264; without it the script falls back
to OpenCV's `mp4v` writer.

## Usage

```powershell
python stray_dashboard.py "D:\path\to\stray scanner videos\9c74aaa7f0"
```

Writes `9c74aaa7f0_dashboard.mp4` **next to** the recording folder (never inside it). A 12 s recording renders
in about 20 s at 1920 px wide.

| option | meaning |
| --- | --- |
| `folders ...` | one or more recording folders. The outer `<id>` folder or the inner `<id>\<id>` both work. |
| `-o out.mp4` | output path (single folder only) |
| `--fps N` | output frame rate. Default is the recording's native rate (~45–60 Hz, one output frame per depth frame, plays in real time). Lower values (e.g. `--fps 30`) still play in real time and drop frames. |
| `--width W` | output width in px (default 1920; height follows, about 0.59 × width). Rounded down to a multiple of 16. |
| `--depth-range MIN MAX` | fix the depth colour scale in mm, e.g. `--depth-range 200 1500`. Default: 1st–99th percentile of the recording. Use a fixed range to compare recordings. |
| `--min-confidence {0,1,2}` | black out depth pixels below this confidence level |
| `--max-frames N` | render only the first N frames (quick preview) |

Batch over several recordings:

```powershell
python stray_dashboard.py "D:\scans\9c74aaa7f0" "D:\scans\ec1e820f1b"
```

## Stray Scanner data format

A recording folder (as exported from the app) contains:

| file | content |
| --- | --- |
| `rgb.mp4` | colour video, 1920×1440 on iPhone Pro, one frame per odometry row |
| `depth/000000.png …` | LiDAR depth, 16-bit PNG, **millimetres**, 256×192 (0 = no data) |
| `confidence/000000.png …` | ARKit depth confidence, 8-bit PNG, `0` low, `1` medium, `2` high |
| `imu.csv` | `timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z`: accelerometer in g, gyro in rad/s, ~100 Hz |
| `odometry.csv` | `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, …`: ARKit camera pose (position in m, orientation as quaternion) and per-frame intrinsics |
| `camera_matrix.csv` | 3×3 intrinsics of the full-resolution RGB frame |

CSV headers have leading spaces after the commas; the script strips them. Conventions follow
[StrayVisualizer](https://github.com/kekeblom/StrayVisualizer) and [strayrobots/scanner](https://github.com/strayrobots/scanner).

### How the streams are synced

* **Master clock:** `odometry.csv`. Each row is one output frame (at the default `--fps`).
* **Depth/confidence:** the row's `frame` column is the PNG index (`frame 283` → `depth/000283.png`).
* **RGB:** video frame *i* ↔ odometry row *i*. The script prints both counts. On the sample recordings the
  container reports the same count as odometry, but OpenCV decodes one frame fewer, so the very last output frame
  reuses the previous RGB frame (reported as a note).
* **IMU / odometry plots:** IMU and odometry timestamps share the same device clock (seconds since boot); the
  plots show `timestamp − first timestamp` and draw every sample with `timestamp ≤ now`.
* **Orientation** is the odometry quaternion converted to ZYX (yaw-pitch-roll) Euler angles in degrees, unwrapped
  so it doesn't jump at ±180°. The absolute values depend on ARKit's world frame; the changes are what matter.

## Keeping recordings out of git

Recordings are large and can show private spaces. `.gitignore` excludes `*.mp4`, `*.csv`, `*.png`,
`depth/`, `confidence/`, `recordings/` and `data/`. Keep recordings outside the repo and point the script at them.

## How it renders quickly

* Row 1 is composed with OpenCV (depth via `COLORMAP_TURBO`, nearest-neighbour upscaling so depth pixels stay honest).
* Rows 2–3 are a single matplotlib figure drawn once; each frame restores the cached background and redraws only
  the line, marker and cursor artists (blitting).
* Frames go straight into ffmpeg (`libx264`, `yuv420p`) through `imageio-ffmpeg`.
