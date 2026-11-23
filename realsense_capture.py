#!/usr/bin/env python3
"""Live view + start/stop recording for an Intel RealSense D435i / D435if.

Running the script opens a live RGB | depth window. Recording is toggled from the window
(Space / R) or from the terminal (Enter or "r"); while recording, the window gets a red border
and a blinking "REC" timer. Every start/stop makes a new take folder, saved in the background
so the live view keeps running.

Each take folder uses the same layout as a Stray Scanner export, so stray_dashboard.py renders it:
  rgb.mp4               colour video, H.264, one frame per row of frames.csv
  depth/000000.png ...  16-bit depth in millimetres, aligned to the colour image (0 = no data)
  imu.csv               timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z   (g, rad/s; gyro resampled to accel times)
  accel.csv, gyro.csv   raw motion samples: timestamp, x, y, z   (m/s^2, rad/s)
  frames.csv            timestamp, frame, color_timestamp, depth_timestamp, hw_frame_number, fx, fy, cx, cy
  camera_matrix.csv     3x3 colour intrinsics (depth is aligned to colour, so these apply to both)
  metadata.json         device, firmware, stream settings, intrinsics, depth scale, measured IMU rates, dropped frames

Timestamps are seconds on the RealSense global (host-synchronised) clock, shared by all streams.
No odometry/pose and no confidence maps: the D435i family does not produce them.

Usage:
  py realsense_capture.py                       live view; Space / R (window) or Enter (terminal) to start/stop
  py realsense_capture.py --record              start recording straight away
  py realsense_capture.py --duration 10         record one 10 s take, then exit
  py realsense_capture.py --no-window           terminal control only
Quit with Q / Esc in the window, "q" in the terminal, Ctrl+C, or by closing the window.
"""
from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    sys.exit("pyrealsense2 is not installed: py -m pip install pyrealsense2")

G = 9.80665
WINDOW = "RealSense live view"
PREVIEW_DEPTH_MM = (300, 4000)  # fixed colour scale for the live depth view
RED = (40, 40, 230)
GREEN = (90, 200, 80)
WHITE = (235, 235, 235)
GREY = (160, 160, 160)


def default_out_dir() -> Path:
    return Path.home() / "Videos" / "RealSense"


def intrinsics_dict(i) -> dict:
    return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
            "model": str(i.model), "coeffs": list(i.coeffs)}


def find_device(serial: str | None):
    devs = list(rs.context().query_devices())
    if not devs:
        sys.exit("no RealSense camera found: check the USB cable (use a USB 3 port)")
    if serial:
        for d in devs:
            if d.get_info(rs.camera_info.serial_number) == serial:
                return d
        sys.exit(f"no RealSense with serial {serial}; found: "
                 + ", ".join(d.get_info(rs.camera_info.serial_number) for d in devs))
    return devs[0]


def fmt_time(s: float) -> str:
    m, s = divmod(max(s, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


# --------------------------------------------------------------------------
# One recording ("take")
# --------------------------------------------------------------------------
class DiskWriter(threading.Thread):
    """Writes colour frames to rgb.mp4 and depth PNGs off the capture thread."""

    def __init__(self, out: Path, size, fps: float):
        super().__init__(daemon=True)
        import imageio_ffmpeg

        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.out = out
        self.video = imageio_ffmpeg.write_frames(
            str(out / "rgb.mp4"), size, fps=fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
            quality=8, macro_block_size=1, ffmpeg_log_level="error",
        )
        self.video.send(None)
        self.error: Exception | None = None

    def run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                idx, color, depth_mm = item
                self.video.send(np.ascontiguousarray(color))
                cv2.imwrite(str(self.out / "depth" / f"{idx:06d}.png"), depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        except Exception as e:  # surfaced via .error
            self.error = e
        finally:
            self.video.close()


class Take:
    def __init__(self, out: Path, size, fps: int, meta_base: dict, color_intr):
        (out / "depth").mkdir(parents=True)
        self.out = out
        self.fps = fps
        self.meta_base = meta_base
        self.color_intr = color_intr
        self.writer = DiskWriter(out, size, fps)
        self.writer.start()
        self.rows: list[tuple] = []
        self.accel: list[tuple] = []
        self.gyro: list[tuple] = []
        self.imu_lock = threading.Lock()
        self.hw_prev = None
        self.dropped = 0
        self.t_start = time.time()
        self.t_stop: float | None = None
        self.next_status = 5.0  # seconds into the take for the next terminal progress line

    @property
    def elapsed(self) -> float:
        return (self.t_stop or time.time()) - self.t_start

    def add_imu(self, is_accel: bool, row: tuple):
        with self.imu_lock:
            (self.accel if is_accel else self.gyro).append(row)

    def add_frame(self, cts: float, dts: float, hw: int, color: np.ndarray, depth_mm: np.ndarray):
        if self.hw_prev is not None and hw > self.hw_prev + 1:
            self.dropped += hw - self.hw_prev - 1
        self.hw_prev = hw
        idx = len(self.rows)
        self.rows.append((cts, idx, cts, dts, hw))
        self.writer.q.put((idx, color, depth_mm))
        if self.writer.error:
            raise self.writer.error

    def finish(self) -> str:
        """Flush video/depth, write the side files; returns a one-line summary. Runs in a background thread."""
        self.t_stop = self.t_stop or time.time()
        self.writer.q.put(None)
        self.writer.join()
        out, ci = self.out, self.color_intr
        with open(out / "frames.csv", "w", newline="") as f:
            f.write("timestamp, frame, color_timestamp, depth_timestamp, hw_frame_number, fx, fy, cx, cy\n")
            for ts, idx, cts, dts, hw in self.rows:
                f.write(f"{ts:.6f}, {idx:06d}, {cts:.6f}, {dts:.6f}, {hw}, {ci.fx}, {ci.fy}, {ci.ppx}, {ci.ppy}\n")
        K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1]])
        (out / "camera_matrix.csv").write_text("\n".join(", ".join(repr(float(v)) for v in r) for r in K))

        with self.imu_lock:
            accel, gyro = sorted(self.accel), sorted(self.gyro)
        imu_rows = 0
        if accel and gyro:
            A, Gy = np.array(accel), np.array(gyro)
            np.savetxt(out / "accel.csv", A, delimiter=", ", header="timestamp, x, y, z", comments="", fmt="%.6f")
            np.savetxt(out / "gyro.csv", Gy, delimiter=", ", header="timestamp, x, y, z", comments="", fmt="%.6f")
            # imu.csv in Stray Scanner units: accel in g, gyro in rad/s, gyro linearly resampled to accel times
            A = A[(A[:, 0] >= Gy[0, 0]) & (A[:, 0] <= Gy[-1, 0])]
            gi = np.column_stack([np.interp(A[:, 0], Gy[:, 0], Gy[:, k]) for k in (1, 2, 3)])
            imu = np.column_stack([A[:, 0], A[:, 1:4] / G, gi])
            np.savetxt(out / "imu.csv", imu, delimiter=", ", comments="", fmt="%.6f",
                       header="timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z")
            imu_rows = len(imu)

        def rate(samples):
            if len(samples) < 2:
                return None
            return round((len(samples) - 1) / (samples[-1][0] - samples[0][0]), 1)

        n = len(self.rows)
        meta = dict(self.meta_base)
        meta["recorded_at"] = datetime.fromtimestamp(self.t_start).isoformat(timespec="seconds")
        meta["duration_s"] = round(self.elapsed, 3)
        meta["frames"] = n
        meta["dropped_frames"] = self.dropped
        meta["imu"] = dict(meta["imu"], accel_hz_measured=rate(accel), gyro_hz_measured=rate(gyro),
                           accel_samples=len(accel), gyro_samples=len(gyro), imu_csv_rows=imu_rows)
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        span = self.rows[-1][0] - self.rows[0][0] if n > 1 else 0.0
        return (f"saved     : {out.name}  {n} frames, {span:.1f} s (~{(n - 1) / span if span else 0:.1f} fps), "
                f"{self.dropped} dropped, {len(accel)} accel + {len(gyro)} gyro samples\n"
                f"            dashboard: py stray_dashboard.py \"{out}\"")


# --------------------------------------------------------------------------
# Live view
# --------------------------------------------------------------------------
def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    lo, hi = PREVIEW_DEPTH_MM
    d8 = (np.clip((depth_mm.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    col = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    col[depth_mm == 0] = 0
    return col


def draw_view(color: np.ndarray, depth_mm: np.ndarray, st: dict) -> np.ndarray:
    """Compose the live window: RGB | depth, a status banner on top and a key-hint bar below.

    st: recording (bool), elapsed (s), frames, take (folder name), saving (int), last_saved (str), info (str)
    """
    pw = 640
    ph = int(round(color.shape[0] * pw / color.shape[1]))
    rgb = cv2.resize(color, (pw, ph), interpolation=cv2.INTER_AREA)
    dep = cv2.resize(colorize_depth(depth_mm), (pw, ph), interpolation=cv2.INTER_NEAREST)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(dep, f"depth {PREVIEW_DEPTH_MM[0] / 1000:.1f}-{PREVIEW_DEPTH_MM[1] / 1000:.0f} m", (pw - 170, ph - 12),
                font, 0.5, WHITE, 1, cv2.LINE_AA)
    banner_h, bar_h = 44, 54
    img = np.full((banner_h + ph + bar_h, 2 * pw, 3), 22, np.uint8)
    img[banner_h:banner_h + ph, :pw] = rgb
    img[banner_h:banner_h + ph, pw:] = dep

    if st["recording"]:
        img[:banner_h] = (30, 30, 150)  # dark red banner
        if int(time.time() * 2) % 2 == 0:  # blinking dot, 1 Hz
            cv2.circle(img, (24, banner_h // 2), 10, (60, 60, 255), -1, cv2.LINE_AA)
        cv2.putText(img, f"REC  {fmt_time(st['elapsed'])}", (44, 31), font, 0.85, WHITE, 2, cv2.LINE_AA)
        cv2.putText(img, f"{st['frames']} frames  ->  {st['take']}", (250, 30), font, 0.6, WHITE, 1, cv2.LINE_AA)
        t = 6  # thick red border around the whole view
        cv2.rectangle(img, (t // 2, t // 2), (img.shape[1] - 1 - t // 2, img.shape[0] - 1 - t // 2), RED, t)
    else:
        cv2.putText(img, "LIVE  (not recording)", (16, 30), font, 0.8, GREEN, 2, cv2.LINE_AA)
        if st.get("info"):
            cv2.putText(img, st["info"], (330, 29), font, 0.55, GREY, 1, cv2.LINE_AA)

    y = banner_h + ph
    hint = ("[Space / R] stop recording" if st["recording"] else "[Space / R] start recording") + \
           "    [Q / Esc] quit    terminal: Enter = start/stop, q = quit"
    cv2.putText(img, hint, (16, y + 22), font, 0.55, WHITE, 1, cv2.LINE_AA)
    status = []
    if st["saving"]:
        status.append(f"saving {st['saving']} take(s)...")
    if st.get("last_saved"):
        status.append(f"last saved: {st['last_saved']}")
    if status:
        cv2.putText(img, "   ".join(status), (16, y + 44), font, 0.5, GREY, 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Live view + start/stop recording for a RealSense D435i/D435if.")
    p.add_argument("-o", "--out-dir", type=Path, default=None,
                   help="folder that receives one rs_<date>_<time> folder per take (default: ~/Videos/RealSense)")
    p.add_argument("--record", action="store_true", help="start recording immediately")
    p.add_argument("--duration", type=float, help="record one take of this many seconds, then exit (implies --record)")
    p.add_argument("--no-window", action="store_true", help="no live window; control from the terminal only")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30, choices=(15, 30, 60))
    p.add_argument("--accel-hz", type=int, default=200, choices=(100, 200, 400))
    p.add_argument("--gyro-hz", type=int, default=200, choices=(200, 400))
    p.add_argument("--no-imu", action="store_true", help="record without the motion module")
    p.add_argument("--warmup", type=float, default=1.0, help="seconds to let auto-exposure settle before --record starts")
    p.add_argument("--serial", help="camera serial number, if several are connected")
    a = p.parse_args(argv)
    if a.duration:
        a.record = True
    show = not a.no_window
    base = (a.out_dir or default_out_dir()).expanduser().resolve()

    dev = find_device(a.serial)
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    firmware = dev.get_info(rs.camera_info.firmware_version)
    has_imu = any(s.get_info(rs.camera_info.name) == "Motion Module" for s in dev.query_sensors())
    use_imu = has_imu and not a.no_imu
    for s in dev.query_sensors():  # put all streams on the shared host-synchronised clock
        if s.supports(rs.option.global_time_enabled):
            s.set_option(rs.option.global_time_enabled, 1)

    # --- video pipeline (depth + colour) ---
    vcfg = rs.config()
    vcfg.enable_device(serial)
    vcfg.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, a.fps)
    vcfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, a.fps)
    vpipe = rs.pipeline()
    try:
        vprof = vpipe.start(vcfg)
    except RuntimeError as e:
        sys.exit(f"could not start depth+colour at {a.width}x{a.height}@{a.fps}: {e}")
    depth_scale = vprof.get_device().first_depth_sensor().get_depth_scale()  # metres per unit
    color_intr = vprof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_intr = vprof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)

    # --- current take, shared with the IMU callback ---
    cur_lock = threading.Lock()
    current: list[Take | None] = [None]

    def on_motion(frame):
        with cur_lock:
            take = current[0]
        if take is None:
            return
        m = frame.as_motion_frame()
        d = m.get_motion_data()
        take.add_imu(m.get_profile().stream_type() == rs.stream.accel, (m.get_timestamp() / 1000.0, d.x, d.y, d.z))

    mpipe = None
    imu_extr = None
    if use_imu:
        mcfg = rs.config()
        mcfg.enable_device(serial)
        mcfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, a.accel_hz)
        mcfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, a.gyro_hz)
        mpipe = rs.pipeline()
        try:
            mprof = mpipe.start(mcfg, on_motion)
        except RuntimeError as e:
            print(f"warning: IMU not started ({e}); recording without IMU", file=sys.stderr)
            mpipe, use_imu = None, False
        else:
            # informational only; not every firmware exposes IMU->camera extrinsics across pipelines
            for dst in (rs.stream.color, rs.stream.depth):
                try:
                    e = mprof.get_stream(rs.stream.accel).get_extrinsics_to(vprof.get_stream(dst))
                    imu_extr = {"to": str(dst).split(".")[-1], "rotation": list(e.rotation),
                                "translation": list(e.translation)}
                    break
                except RuntimeError:
                    pass

    meta_base = {
        "device": name, "serial": serial, "firmware": firmware,
        "pyrealsense2": getattr(rs, "__version__", "?"),
        "width": a.width, "height": a.height, "fps": a.fps,
        "depth_scale_m_per_unit": depth_scale, "depth_png_units": "mm", "depth_aligned_to": "color",
        "color_intrinsics": intrinsics_dict(color_intr), "depth_intrinsics_unaligned": intrinsics_dict(depth_intr),
        # requested rates; the D435i/D435if firmware may deliver accel at ~400 Hz regardless (measured added per take)
        "imu": {"enabled": use_imu, "accel_hz_requested": a.accel_hz, "gyro_hz_requested": a.gyro_hz,
                "imu_csv_units": {"accel": "g", "gyro": "rad/s"}, "raw_units": {"accel": "m/s^2", "gyro": "rad/s"},
                "axes": "RealSense IMU frame: x right, y down, z forward (out of the lens)",
                "extrinsics_accel_to_camera": imu_extr},
        "timestamps": "seconds, RealSense global (host-synchronised) time domain",
    }

    print(f"camera    : {name}  serial {serial}  fw {firmware}")
    print(f"streams   : colour + depth {a.width}x{a.height}@{a.fps}"
          + (f", accel {a.accel_hz} Hz, gyro {a.gyro_hz} Hz" if use_imu else ", no IMU"))
    print(f"takes go  : {base}\\rs_<date>_<time>")

    # --- start / stop ---
    finishers: list[threading.Thread] = []
    ui = {"last_saved": "", "info": ""}

    def saving_count() -> int:
        return sum(t.is_alive() for t in finishers)

    def start_take():
        with cur_lock:
            if current[0] is not None:
                return
        out = base / datetime.now().strftime("rs_%Y%m%d_%H%M%S")
        k = 1
        while out.exists():
            k += 1
            out = base / datetime.now().strftime(f"rs_%Y%m%d_%H%M%S_{k}")
        take = Take(out, (a.width, a.height), a.fps, meta_base, color_intr)
        with cur_lock:
            current[0] = take
        print(f"REC start : {out}", flush=True)

    def stop_take():
        with cur_lock:
            take, current[0] = current[0], None
        if take is None:
            return
        take.t_stop = time.time()
        print(f"REC stop  : {take.out.name} after {take.elapsed:.1f} s, saving in the background ...", flush=True)

        def finish():
            try:
                print(take.finish(), flush=True)
                ui["last_saved"] = take.out.name
            except Exception as e:
                print(f"ERROR saving {take.out}: {e}", file=sys.stderr, flush=True)

        t = threading.Thread(target=finish)
        t.start()
        finishers.append(t)

    def toggle():
        with cur_lock:
            rec = current[0] is not None
        stop_take() if rec else start_take()

    # --- terminal commands (Enter / r = toggle, start, stop, q = quit) ---
    cmds: queue.Queue = queue.Queue()

    def read_stdin():
        try:
            for line in sys.stdin:
                cmds.put(line.strip().lower())
        except Exception:
            pass

    threading.Thread(target=read_stdin, daemon=True).start()
    quit_ev = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: quit_ev.set())

    print("controls  : " + ("window: Space / R = start/stop, Q / Esc = quit;  " if show else "")
          + "terminal: Enter (or r) = start/stop, q = quit, Ctrl+C = quit", flush=True)

    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    t_ready = time.time() + (a.warmup if a.record else 0.0)
    pending_record = a.record
    try:
        while not quit_ev.is_set():
            # terminal commands
            while not cmds.empty():
                c = cmds.get()
                if c in ("", "r", "rec", "t", "toggle"):
                    toggle()
                elif c in ("s", "start"):
                    start_take()
                elif c == "stop":
                    stop_take()
                elif c in ("q", "quit", "exit"):
                    quit_ev.set()
                else:
                    print(f"unknown command {c!r}: Enter/r = start/stop, start, stop, q = quit", flush=True)
            if quit_ev.is_set():
                break

            try:
                fs = vpipe.wait_for_frames(5000)
            except RuntimeError:
                print("warning: no frames for 5 s", file=sys.stderr)
                continue
            fs = align.process(fs)
            cf, df = fs.get_color_frame(), fs.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data()).copy()
            raw = np.asanyarray(df.get_data())
            if abs(depth_scale - 0.001) < 1e-9:
                depth_mm = raw.copy()
            else:
                depth_mm = np.clip(raw.astype(np.float64) * depth_scale * 1000.0, 0, 65535).astype(np.uint16)

            if pending_record and time.time() >= t_ready:
                pending_record = False
                start_take()
            with cur_lock:
                take = current[0]
            if take is not None:
                take.add_frame(cf.get_timestamp() / 1000.0, df.get_timestamp() / 1000.0, cf.get_frame_number(),
                               color, depth_mm)
                if a.duration and take.elapsed >= a.duration:
                    stop_take()
                    quit_ev.set()
                elif take.elapsed >= take.next_status:  # progress line every 5 s of the take
                    take.next_status += 5.0
                    print(f"  REC {fmt_time(take.elapsed)}  {len(take.rows)} frames  "
                          f"{len(take.accel)} accel / {len(take.gyro)} gyro  dropped {take.dropped}  "
                          f"queue {take.writer.q.qsize()}", flush=True)

            if show:
                st = {"recording": take is not None, "elapsed": take.elapsed if take else 0.0,
                      "frames": len(take.rows) if take else 0, "take": take.out.name if take else "",
                      "saving": saving_count(), "last_saved": ui["last_saved"],
                      "info": "starting in a moment ..." if pending_record else ""}
                cv2.imshow(WINDOW, draw_view(color, depth_mm, st))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord(" "), ord("r"), ord("R")):
                    toggle()
                elif key in (ord("q"), ord("Q"), 27):
                    quit_ev.set()
                elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # window closed with X
                    quit_ev.set()
    finally:
        stop_take()
        vpipe.stop()
        if mpipe:
            mpipe.stop()
        if show:
            cv2.destroyAllWindows()
        if saving_count():
            print("finishing : waiting for takes to be saved ...", flush=True)
        for t in finishers:
            t.join()
    print("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
