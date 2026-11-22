#!/usr/bin/env python3
"""Render a Stray Scanner recording into a single synced "dashboard" MP4.

Layout (top to bottom):
  row 1  RGB video | colourised depth (+ colourbar) | confidence (3 levels)
  row 2  IMU: accelerometer x/y/z | gyroscope x/y/z
  row 3  odometry: position x/y/z | orientation (roll/pitch/yaw)

The time-series lines grow with playback: each output frame shows data up to
"now" on axes fixed to the whole recording's range, with a cursor at "now".

Usage:
  python stray_dashboard.py <recording_folder> [<more folders> ...]
                            [-o out.mp4] [--fps N] [--width W]
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------
# Look & feel
# --------------------------------------------------------------------------
BG = "#14161a"
AX_BG = "#1c1f25"
FG = "#d8dde6"
GRID = "#2e333b"
XYZ_COLORS = ["#ff6b6b", "#51cf66", "#4dabf7"]  # x / y / z (also roll / pitch / yaw)
BG_BGR = (0x1A, 0x16, 0x14)
FG_BGR = (0xE6, 0xDD, 0xD8)
DIM_BGR = (0x9A, 0x92, 0x8A)

# Confidence levels as written by ARKit / Stray Scanner: 0 low, 1 medium, 2 high
CONF_COLORS_BGR = {0: (60, 60, 220), 1: (40, 190, 240), 2: (90, 200, 80)}
CONF_NAMES = {0: "low", 1: "medium", 2: "high"}

BASE_WIDTH = 1920  # all sizes below are designed for this width and scaled


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def find_recording_dir(path: Path) -> Path:
    """Return the folder that actually holds rgb.mp4 / odometry.csv.

    Stray Scanner exports often unzip to <id>/<id>/..., so accept either level.
    """
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"not a folder: {path}")

    def is_rec(p: Path) -> bool:
        return (p / "odometry.csv").is_file() and (p / "rgb.mp4").is_file()

    if is_rec(path):
        return path
    same = path / path.name
    if is_rec(same):
        return same
    hits = sorted(p.parent for p in path.glob("*/odometry.csv") if is_rec(p.parent))
    hits += sorted(p.parent for p in path.glob("*/*/odometry.csv") if is_rec(p.parent))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"no Stray Scanner recording (rgb.mp4 + odometry.csv) under {path}")
    raise RuntimeError(
        f"several recordings under {path}; pass one of them:\n  " + "\n  ".join(map(str, hits))
    )


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def quat_to_euler_deg(qx, qy, qz, qw):
    """Quaternion -> (roll, pitch, yaw) in degrees, ZYX (yaw-pitch-roll) convention, unwrapped."""
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return [np.degrees(np.unwrap(a)) for a in (roll, pitch, yaw)]


@dataclass
class Recording:
    root: Path
    odom: pd.DataFrame
    imu: pd.DataFrame | None
    t0: float
    t_end: float
    rgb_count: int
    rgb_fps: float
    has_conf: bool

    def depth_path(self, frame: int) -> Path:
        return self.root / "depth" / f"{frame:06d}.png"

    def conf_path(self, frame: int) -> Path:
        return self.root / "confidence" / f"{frame:06d}.png"


def load_recording(root: Path) -> Recording:
    odom = read_csv(root / "odometry.csv")
    odom = odom.dropna(subset=["timestamp"]).reset_index(drop=True)
    odom["frame"] = odom["frame"].astype(int)
    imu_path = root / "imu.csv"
    imu = read_csv(imu_path) if imu_path.is_file() else None

    cap = cv2.VideoCapture(str(root / "rgb.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {root / 'rgb.mp4'}")
    rgb_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    starts = [odom["timestamp"].iloc[0]]
    ends = [odom["timestamp"].iloc[-1]]
    if imu is not None and len(imu):
        starts.append(imu["timestamp"].iloc[0])
        ends.append(imu["timestamp"].iloc[-1])
    return Recording(
        root=root,
        odom=odom,
        imu=imu,
        t0=float(min(starts)),
        t_end=float(max(ends)),
        rgb_count=rgb_count,
        rgb_fps=rgb_fps,
        has_conf=(root / "confidence").is_dir(),
    )


def auto_depth_range(rec: Recording, samples: int = 40) -> tuple[float, float]:
    """Robust (1st..99th percentile) depth range in mm over a sample of frames."""
    frames = rec.odom["frame"].to_numpy()
    pick = frames[np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int)]
    vals = []
    for f in pick:
        d = cv2.imread(str(rec.depth_path(f)), cv2.IMREAD_UNCHANGED)
        if d is not None:
            vals.append(d[d > 0].ravel())
    if not vals or not sum(len(v) for v in vals):
        return 0.0, 5000.0
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, [1, 99])
    lo = np.floor(lo / 50) * 50
    hi = np.ceil(hi / 50) * 50
    if hi <= lo:
        hi = lo + 100
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Top row (OpenCV)
# --------------------------------------------------------------------------
class TopRow:
    def __init__(self, width: int, scale: float, rgb_aspect: float, depth_range, min_conf: int):
        self.width = width
        self.s = scale
        self.cell_w = width // 3
        self.img_w = self.cell_w - 2 * self.pad
        self.img_h = int(round(self.img_w / rgb_aspect))
        self.title_h = int(round(38 * scale))
        self.caption_h = int(round(62 * scale))
        self.height = self.title_h + self.img_h + self.caption_h
        self.dmin, self.dmax = depth_range
        self.min_conf = min_conf
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fs = 0.72 * scale
        self.fs_small = 0.55 * scale
        self.th = max(1, int(round(1.6 * scale)))
        self.template = self._make_template()

    @property
    def pad(self) -> int:
        return int(round(8 * self.s))

    def _text(self, img, text, org, fs=None, color=FG_BGR, th=None):
        cv2.putText(img, text, org, self.font, fs or self.fs, color, th or self.th, cv2.LINE_AA)

    def _make_template(self):
        img = np.full((self.height, self.width, 3), BG_BGR, np.uint8)
        titles = ["RGB", "Depth", "Confidence"]
        for k, title in enumerate(titles):
            x0 = k * self.cell_w + self.pad
            self._text(img, title, (x0, int(self.title_h * 0.72)))
        y_cap = self.title_h + self.img_h
        # depth colourbar (horizontal, under the depth panel)
        x0 = self.cell_w + self.pad
        bar_y = y_cap + int(round(10 * self.s))
        bar_h = int(round(16 * self.s))
        grad = np.linspace(0, 255, self.img_w).astype(np.uint8)[None, :]
        bar = cv2.applyColorMap(np.repeat(grad, bar_h, 0), cv2.COLORMAP_TURBO)
        img[bar_y : bar_y + bar_h, x0 : x0 + self.img_w] = bar
        n_ticks = 5
        for i in range(n_ticks):
            frac = i / (n_ticks - 1)
            x = x0 + int(round(frac * (self.img_w - 1)))
            cv2.line(img, (x, bar_y + bar_h), (x, bar_y + bar_h + int(5 * self.s)), FG_BGR, 1)
            val = (self.dmin + frac * (self.dmax - self.dmin)) / 1000.0
            label = f"{val:.2f} m"
            (tw, _), _ = cv2.getTextSize(label, self.font, self.fs_small, 1)
            tx = min(max(x - tw // 2, x0), x0 + self.img_w - tw)
            self._text(img, label, (tx, bar_y + bar_h + int(24 * self.s)), self.fs_small, FG_BGR, 1)
        # confidence legend
        x = 2 * self.cell_w + self.pad
        box = int(round(18 * self.s))
        y = y_cap + int(round(12 * self.s))
        for lvl in (0, 1, 2):
            cv2.rectangle(img, (x, y), (x + box, y + box), CONF_COLORS_BGR[lvl], -1)
            label = f"{lvl} = {CONF_NAMES[lvl]}"
            self._text(img, label, (x + box + int(8 * self.s), y + box - int(3 * self.s)), self.fs_small, FG_BGR, 1)
            (tw, _), _ = cv2.getTextSize(label, self.font, self.fs_small, 1)
            x += box + tw + int(30 * self.s)
        if self.min_conf > 0:
            x0 = self.cell_w + self.pad
            note = f"depth masked where confidence < {self.min_conf}"
            self._text(img, note, (x0 + int(200 * self.s), int(self.title_h * 0.72)), self.fs_small, DIM_BGR, 1)
        return img

    def _place(self, canvas, k, panel):
        x0 = k * self.cell_w + self.pad
        canvas[self.title_h : self.title_h + self.img_h, x0 : x0 + self.img_w] = panel

    def render(self, rgb_bgr, depth, conf, t_now: float, frame: int, n_frames: int):
        canvas = self.template.copy()
        # RGB
        if rgb_bgr is not None:
            self._place(canvas, 0, cv2.resize(rgb_bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA))
        # depth
        if depth is not None:
            valid = depth > 0
            if conf is not None and self.min_conf > 0 and conf.shape == depth.shape:
                valid &= conf >= self.min_conf
            norm = (depth.astype(np.float32) - self.dmin) / max(self.dmax - self.dmin, 1e-6)
            d8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
            dcol = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
            dcol[~valid] = 0
            self._place(canvas, 1, cv2.resize(dcol, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST))
        # confidence
        if conf is not None:
            ccol = np.zeros(conf.shape + (3,), np.uint8)
            for lvl, col in CONF_COLORS_BGR.items():
                ccol[conf == lvl] = col
            self._place(canvas, 2, cv2.resize(ccol, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST))
            share = [(conf == lvl).mean() * 100 for lvl in (0, 1, 2)]
            stats = "  ".join(f"{CONF_NAMES[l][0].upper()} {p:4.1f}%" for l, p in zip((0, 1, 2), share))
            (tw, _), _ = cv2.getTextSize(stats, self.font, self.fs_small, 1)
            self._text(canvas, stats, (3 * self.cell_w - self.pad - tw, int(self.title_h * 0.72)), self.fs_small, DIM_BGR, 1)
        else:
            x0 = 2 * self.cell_w + self.pad
            self._text(canvas, "no confidence data", (x0 + 20, self.title_h + self.img_h // 2), self.fs, DIM_BGR)
        # RGB caption: time and frame
        y = self.title_h + self.img_h + int(round(34 * self.s))
        self._text(canvas, f"t = {t_now:6.2f} s", (self.pad, y))
        self._text(canvas, f"frame {frame:06d}  ({frame + 1}/{n_frames})", (self.pad + int(190 * self.s), y), self.fs_small, DIM_BGR, 1)
        return canvas


# --------------------------------------------------------------------------
# Rows 2 + 3 (matplotlib with blitting)
# --------------------------------------------------------------------------
class PlotRows:
    def __init__(self, width: int, height: int, scale: float, rec: Recording):
        dpi = 100 * scale
        plt.rcParams.update(
            {
                "text.color": FG,
                "axes.labelcolor": FG,
                "xtick.color": FG,
                "ytick.color": FG,
                "axes.edgecolor": GRID,
                "font.size": 10,
            }
        )
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BG)
        gs = self.fig.add_gridspec(2, 2, left=0.045, right=0.99, top=0.93, bottom=0.10, hspace=0.42, wspace=0.10)
        span = (0.0, rec.t_end - rec.t0)
        self.series = []  # (t, [y arrays], [Line2D], [marker Line2D])
        self.cursors = []

        def make_axes(r, c, title, t, ys, labels, ylabel):
            ax = self.fig.add_subplot(gs[r, c])
            ax.set_facecolor(AX_BG)
            ax.set_title(title, loc="left", fontsize=11, color=FG, pad=4)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_xlim(*span)
            ax.set_ylabel(ylabel)
            if r == 1:
                ax.set_xlabel("time (s)")
            lo = min(np.nanmin(y) for y in ys)
            hi = max(np.nanmax(y) for y in ys)
            m = 0.06 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - m, hi + m)
            lines, marks = [], []
            for y, lab, col in zip(ys, labels, XYZ_COLORS):
                (ln,) = ax.plot([], [], color=col, lw=1.4, label=lab, animated=True)
                (mk,) = ax.plot([], [], "o", color=col, ms=4, animated=True)
                lines.append(ln)
                marks.append(mk)
            # legend sits above the axes (right of the title) so the growing lines never cover it
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=9, frameon=False,
                      labelcolor=FG, borderaxespad=0.2, handlelength=1.6, columnspacing=1.2)
            cur = ax.axvline(0, color="#ffffff", lw=0.9, alpha=0.55, animated=True)
            self.series.append((np.asarray(t), [np.asarray(y) for y in ys], lines, marks))
            self.cursors.append(cur)
            return ax

        if rec.imu is not None and len(rec.imu):
            ti = rec.imu["timestamp"].to_numpy() - rec.t0
            make_axes(0, 0, "IMU · accelerometer", ti, [rec.imu[c].to_numpy() for c in ("a_x", "a_y", "a_z")],
                      ["a_x", "a_y", "a_z"], "accel (g)")
            make_axes(0, 1, "IMU · gyroscope", ti, [rec.imu[c].to_numpy() for c in ("alpha_x", "alpha_y", "alpha_z")],
                      ["ω_x", "ω_y", "ω_z"], "rate (rad/s)")
        else:
            for c, name in ((0, "IMU · accelerometer"), (1, "IMU · gyroscope")):
                ax = self.fig.add_subplot(gs[0, c])
                ax.set_facecolor(AX_BG)
                ax.set_title(name + " (no imu.csv)", loc="left", fontsize=11, color=FG)
        o = rec.odom
        to = o["timestamp"].to_numpy() - rec.t0
        make_axes(1, 0, "Odometry · position", to, [o[c].to_numpy() for c in ("x", "y", "z")], ["x", "y", "z"], "position (m)")
        eul = quat_to_euler_deg(*(o[c].to_numpy() for c in ("qx", "qy", "qz", "qw")))
        make_axes(1, 1, "Odometry · orientation (ZYX Euler)", to, eul, ["roll", "pitch", "yaw"], "angle (deg)")

        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        w, h = self.fig.canvas.get_width_height()
        self.size = (w, h)

    def render(self, t_now: float) -> np.ndarray:
        canvas = self.fig.canvas
        canvas.restore_region(self.bg)
        for (t, ys, lines, marks), cur in zip(self.series, self.cursors):
            k = int(np.searchsorted(t, t_now, side="right"))
            for y, ln, mk in zip(ys, lines, marks):
                ln.set_data(t[:k], y[:k])
                if k:
                    mk.set_data([t[k - 1]], [y[k - 1]])
                else:
                    mk.set_data([], [])
                ln.axes.draw_artist(ln)
                mk.axes.draw_artist(mk)
            cur.set_xdata([t_now, t_now])
            cur.axes.draw_artist(cur)
        buf = np.asarray(canvas.buffer_rgba())
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


# --------------------------------------------------------------------------
# Video I/O
# --------------------------------------------------------------------------
class RgbReader:
    """Sequential reader with forward seeking; holds the last frame if the video runs short."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        self.pos = -1
        self.frame = None
        self.decoded = 0
        self.ended = False

    def get(self, idx: int):
        while self.pos < idx - 1 and not self.ended:  # skip frames we don't need
            if not self.cap.grab():
                self.ended = True
                break
            self.pos += 1
        if self.pos == idx - 1 and not self.ended:
            ok, fr = self.cap.read()
            if ok:
                self.pos += 1
                self.frame = fr
            else:
                self.ended = True
        self.decoded = self.pos + 1
        return self.frame

    def close(self):
        self.cap.release()


class Writer:
    def __init__(self, path: Path, size, fps: float):
        self.size = size
        try:
            import imageio_ffmpeg

            self.gen = imageio_ffmpeg.write_frames(
                str(path), size, fps=fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                quality=8, macro_block_size=16, ffmpeg_log_level="error",
            )
            self.gen.send(None)
            self.kind = "ffmpeg"
        except ImportError:
            self.gen = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            self.kind = "opencv"

    def write(self, frame_bgr: np.ndarray):
        if self.kind == "ffmpeg":
            self.gen.send(np.ascontiguousarray(frame_bgr))
        else:
            self.gen.write(frame_bgr)

    def close(self):
        if self.kind == "ffmpeg":
            self.gen.close()
        else:
            self.gen.release()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def default_output(root: Path) -> Path:
    # Next to the recording folder (never inside it), e.g. .../9c74aaa7f0/9c74aaa7f0_dashboard.mp4
    return root.parent / f"{root.name}_dashboard.mp4"


def render(folder: Path, out: Path | None, fps: float | None, width: int, depth_range, min_conf: int,
           max_frames: int | None) -> Path:
    root = find_recording_dir(folder)
    rec = load_recording(root)
    out = (out or default_output(root)).resolve()
    o = rec.odom
    n_odom = len(o)
    t_odom = o["timestamp"].to_numpy()
    native_fps = (n_odom - 1) / (t_odom[-1] - t_odom[0]) if n_odom > 1 else 30.0

    print(f"recording : {root}")
    print(f"odometry  : {n_odom} rows, {t_odom[-1] - t_odom[0]:.2f} s, ~{native_fps:.2f} Hz")
    if rec.imu is not None:
        ti = rec.imu["timestamp"].to_numpy()
        print(f"imu       : {len(ti)} rows, ~{(len(ti) - 1) / (ti[-1] - ti[0]):.1f} Hz")
    print(f"rgb.mp4   : {rec.rgb_count} frames (header), {rec.rgb_fps:.2f} fps")
    if rec.rgb_count != n_odom:
        print(f"  note: rgb frame count {rec.rgb_count} != odometry rows {n_odom}; "
              "mapping by frame index, holding the last decodable frame at the end")

    # output schedule: which odometry row to show at each output frame
    if fps is None:
        fps = native_fps
        rows = np.arange(n_odom)
    else:
        times = np.arange(t_odom[0], t_odom[-1] + 1e-9, 1.0 / fps)
        rows = np.clip(np.searchsorted(t_odom, times, side="right") - 1, 0, n_odom - 1)
    if max_frames:
        rows = rows[:max_frames]

    if depth_range is None:
        depth_range = auto_depth_range(rec)
    print(f"depth     : colour range {depth_range[0]:.0f}-{depth_range[1]:.0f} mm")

    cap = cv2.VideoCapture(str(root / "rgb.mp4"))
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    aspect = vw / vh if vw and vh else 4 / 3

    width = int(width) // 16 * 16  # H.264 macro blocks
    scale = width / BASE_WIDTH
    top = TopRow(width, scale, aspect, depth_range, min_conf)
    plot_h = int(round(560 * scale))
    total_h = top.height + plot_h
    total_h = int(np.ceil(total_h / 16) * 16)
    plot_h = total_h - top.height
    plots = PlotRows(width, plot_h, scale, rec)
    if plots.size != (width, plot_h):  # guard against DPI rounding
        print(f"  note: plot canvas {plots.size} resized to {(width, plot_h)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = Writer(out, (width, total_h), fps)
    reader = RgbReader(root / "rgb.mp4")
    print(f"output    : {out}  ({width}x{total_h}, {fps:.2f} fps, {len(rows)} frames, {writer.kind})")

    t_start = time.time()
    frames = o["frame"].to_numpy()
    last_pct = 0
    try:
        for i, r in enumerate(rows):
            f = int(frames[r])
            t_now = t_odom[r] - rec.t0
            rgb = reader.get(int(r))
            depth = cv2.imread(str(rec.depth_path(f)), cv2.IMREAD_UNCHANGED)
            conf = cv2.imread(str(rec.conf_path(f)), cv2.IMREAD_UNCHANGED) if rec.has_conf else None
            top_img = top.render(rgb, depth, conf, t_now, f, n_odom)
            plot_img = plots.render(t_now)
            if plot_img.shape[:2] != (plot_h, width):
                plot_img = cv2.resize(plot_img, (width, plot_h), interpolation=cv2.INTER_AREA)
            writer.write(np.vstack([top_img, plot_img]))
            pct = int(100 * (i + 1) / len(rows))
            if pct // 10 != last_pct // 10:
                last_pct = pct
                print(f"  {pct:3d}%  ({i + 1}/{len(rows)}, {time.time() - t_start:.1f} s)", flush=True)
    finally:
        writer.close()
        reader.close()
        plt.close(plots.fig)
    if reader.decoded < rows.max() + 1:
        print(f"  note: rgb.mp4 decoded {reader.decoded} frames; last {rows.max() + 1 - reader.decoded} "
              "output frame(s) reuse the final RGB frame")
    print(f"done in {time.time() - t_start:.1f} s -> {out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Turn a Stray Scanner recording into a synced dashboard MP4.")
    p.add_argument("folders", nargs="+", type=Path, help="recording folder(s); the outer <id> or inner <id>/<id> both work")
    p.add_argument("-o", "--output", type=Path, help="output .mp4 (only with a single folder); "
                   "default: <recording>_dashboard.mp4 next to the recording folder")
    p.add_argument("--fps", type=float, help="output frame rate; default = native odometry/RGB rate (real-time, 1 frame per depth frame)")
    p.add_argument("--width", type=int, default=1920, help="output width in px (rounded down to a multiple of 16); default 1920")
    p.add_argument("--depth-range", type=float, nargs=2, metavar=("MIN_MM", "MAX_MM"),
                   help="fix the depth colour scale in mm; default = 1st-99th percentile of the recording")
    p.add_argument("--min-confidence", type=int, default=0, choices=(0, 1, 2),
                   help="black out depth pixels below this confidence level (default 0 = show all)")
    p.add_argument("--max-frames", type=int, help="only render the first N output frames (quick preview)")
    a = p.parse_args(argv)
    if a.output and len(a.folders) > 1:
        p.error("-o/--output can only be used with a single folder")
    failed = 0
    for folder in a.folders:
        try:
            render(folder, a.output, a.fps, a.width, a.depth_range, a.min_confidence, a.max_frames)
        except Exception as e:  # keep going for batch runs
            failed += 1
            print(f"ERROR on {folder}: {e}", file=sys.stderr)
            if len(a.folders) == 1:
                raise
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
