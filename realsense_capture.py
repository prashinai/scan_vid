#!/usr/bin/env python3
"""Record RGB + depth + IMU from an Intel RealSense D435i / D435if into a Stray-Scanner-like folder.

Output folder (same layout as a Stray Scanner export, so stray_dashboard.py can render it):
  rgb.mp4               colour video, H.264, one frame per row of frames.csv
  depth/000000.png ...  16-bit depth in millimetres, aligned to the colour image (0 = no data)
  imu.csv               timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z   (g, rad/s; gyro resampled to accel times)
  accel.csv, gyro.csv   raw motion samples: timestamp, x, y, z   (m/s^2, rad/s)
  frames.csv            timestamp, frame, color_timestamp, depth_timestamp, hw_frame_number, fx, fy, cx, cy
  camera_matrix.csv     3x3 colour intrinsics (depth is aligned to colour, so these apply to both)
  metadata.json         device, firmware, stream settings, intrinsics, depth scale, IMU extrinsics, dropped frames

Timestamps are seconds on the RealSense global (host-synchronised) clock, shared by all streams.
No odometry/pose and no confidence maps: the D435i family does not produce them.

Usage:
  python realsense_capture.py [-o OUT_DIR] [--duration S] [--preview] [--width 640 --height 480 --fps 30]
Stop with q / Esc in the preview window, Ctrl+C, or --duration.
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


def default_out_dir() -> Path:
    return Path.home() / "Videos" / "RealSense" / datetime.now().strftime("rs_%Y%m%d_%H%M%S")


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
        except Exception as e:  # surface in main thread
            self.error = e
        finally:
            self.video.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Record RGB + depth + IMU from a RealSense D435i/D435if.")
    p.add_argument("-o", "--out", type=Path, help="output folder (default: ~/Videos/RealSense/rs_<date>_<time>)")
    p.add_argument("--duration", type=float, help="stop after this many seconds (default: until q/Esc/Ctrl+C)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30, choices=(15, 30, 60))
    p.add_argument("--accel-hz", type=int, default=200, choices=(100, 200, 400))
    p.add_argument("--gyro-hz", type=int, default=200, choices=(200, 400))
    p.add_argument("--no-imu", action="store_true", help="record without the motion module")
    p.add_argument("--warmup", type=float, default=1.0, help="seconds to discard at start while auto-exposure settles")
    p.add_argument("--preview", action="store_true", help="show a live RGB | depth window (q / Esc to stop)")
    p.add_argument("--serial", help="camera serial number, if several are connected")
    a = p.parse_args(argv)

    dev = find_device(a.serial)
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    has_imu = any(s.get_info(rs.camera_info.name) == "Motion Module" for s in dev.query_sensors())
    use_imu = has_imu and not a.no_imu
    for s in dev.query_sensors():  # put all streams on the shared host-synchronised clock
        if s.supports(rs.option.global_time_enabled):
            s.set_option(rs.option.global_time_enabled, 1)

    out = (a.out or default_out_dir()).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"output folder is not empty: {out}")
    (out / "depth").mkdir(parents=True, exist_ok=True)

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

    # --- motion pipeline (separate pipeline + callback so no IMU samples are lost) ---
    accel, gyro = [], []
    imu_lock = threading.Lock()
    recording = threading.Event()
    mpipe = None
    imu_extr = None
    if use_imu:
        mcfg = rs.config()
        mcfg.enable_device(serial)
        mcfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, a.accel_hz)
        mcfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, a.gyro_hz)

        def on_motion(frame):
            if not recording.is_set():
                return
            m = frame.as_motion_frame()
            d = m.get_motion_data()
            row = (m.get_timestamp() / 1000.0, d.x, d.y, d.z)
            with imu_lock:
                (accel if m.get_profile().stream_type() == rs.stream.accel else gyro).append(row)

        mpipe = rs.pipeline()
        try:
            mprof = mpipe.start(mcfg, on_motion)
        except RuntimeError as e:
            print(f"warning: IMU not started ({e}); recording without IMU", file=sys.stderr)
            mpipe, use_imu = None, False
        else:
            # informational only; not every firmware exposes IMU->colour extrinsics across pipelines
            for src, dst in ((rs.stream.accel, rs.stream.color), (rs.stream.accel, rs.stream.depth)):
                try:
                    e = mprof.get_stream(src).get_extrinsics_to(vprof.get_stream(dst))
                    imu_extr = {"to": str(dst).split(".")[-1], "rotation": list(e.rotation),
                                "translation": list(e.translation)}
                    break
                except RuntimeError:
                    pass

    print(f"camera    : {name}  serial {serial}  fw {dev.get_info(rs.camera_info.firmware_version)}")
    print(f"streams   : colour + depth {a.width}x{a.height}@{a.fps}"
          + (f", accel {a.accel_hz} Hz, gyro {a.gyro_hz} Hz" if use_imu else ", no IMU"))
    print(f"output    : {out}")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    # warm-up: let auto-exposure settle
    t_warm = time.time() + a.warmup
    while time.time() < t_warm:
        vpipe.wait_for_frames()

    writer = DiskWriter(out, (a.width, a.height), a.fps)
    writer.start()
    rows = []
    hw_prev = None
    dropped = 0
    t_start = time.time()
    recording.set()
    print("recording : press q / Esc in the preview window or Ctrl+C to stop" if a.preview
          else "recording : Ctrl+C to stop" + (f" (or wait {a.duration:g} s)" if a.duration else ""))
    try:
        while not stop.is_set():
            if a.duration and time.time() - t_start >= a.duration:
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
            hw = cf.get_frame_number()
            if hw_prev is not None and hw > hw_prev + 1:
                dropped += hw - hw_prev - 1
            hw_prev = hw
            idx = len(rows)
            rows.append((cf.get_timestamp() / 1000.0, idx, cf.get_timestamp() / 1000.0,
                         df.get_timestamp() / 1000.0, hw))
            writer.q.put((idx, color, depth_mm))
            if writer.error:
                raise writer.error
            if a.preview:
                dcol = cv2.applyColorMap(cv2.convertScaleAbs(depth_mm, alpha=255 / 4000), cv2.COLORMAP_TURBO)
                dcol[depth_mm == 0] = 0
                view = np.hstack([color, dcol])
                cv2.putText(view, f"REC {time.time() - t_start:5.1f} s  frame {idx}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("RealSense capture (q / Esc to stop)", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if idx and idx % (a.fps * 5) == 0:
                print(f"  {time.time() - t_start:6.1f} s  {idx} frames  "
                      f"{len(accel)} accel / {len(gyro)} gyro  dropped {dropped}  queue {writer.q.qsize()}", flush=True)
    finally:
        recording.clear()
        elapsed = time.time() - t_start
        vpipe.stop()
        if mpipe:
            mpipe.stop()
        if a.preview:
            cv2.destroyAllWindows()
        print("finishing : writing remaining frames ...", flush=True)
        writer.q.put(None)
        writer.join()

    # ---- side files ----
    n = len(rows)
    with open(out / "frames.csv", "w", newline="") as f:
        f.write("timestamp, frame, color_timestamp, depth_timestamp, hw_frame_number, fx, fy, cx, cy\n")
        for ts, idx, cts, dts, hw in rows:
            f.write(f"{ts:.6f}, {idx:06d}, {cts:.6f}, {dts:.6f}, {hw}, "
                    f"{color_intr.fx}, {color_intr.fy}, {color_intr.ppx}, {color_intr.ppy}\n")
    K = np.array([[color_intr.fx, 0, color_intr.ppx], [0, color_intr.fy, color_intr.ppy], [0, 0, 1]])
    (out / "camera_matrix.csv").write_text("\n".join(", ".join(repr(float(v)) for v in r) for r in K))

    imu_rows = 0
    if use_imu and accel and gyro:
        A = np.array(sorted(accel))
        Gy = np.array(sorted(gyro))
        np.savetxt(out / "accel.csv", A, delimiter=", ", header="timestamp, x, y, z", comments="", fmt="%.6f")
        np.savetxt(out / "gyro.csv", Gy, delimiter=", ", header="timestamp, x, y, z", comments="", fmt="%.6f")
        # imu.csv in Stray Scanner units: accel in g, gyro in rad/s, gyro linearly resampled to accel times
        m = (A[:, 0] >= Gy[0, 0]) & (A[:, 0] <= Gy[-1, 0])
        A = A[m]
        gi = np.column_stack([np.interp(A[:, 0], Gy[:, 0], Gy[:, k]) for k in (1, 2, 3)])
        imu = np.column_stack([A[:, 0], A[:, 1:4] / G, gi])
        np.savetxt(out / "imu.csv", imu, delimiter=", ", comments="", fmt="%.6f",
                   header="timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z")
        imu_rows = len(imu)

    def rate(samples):
        if len(samples) < 2:
            return None
        ts = sorted(s[0] for s in samples)
        return round((len(ts) - 1) / (ts[-1] - ts[0]), 1)

    meta = {
        "device": name, "serial": serial, "firmware": dev.get_info(rs.camera_info.firmware_version),
        "pyrealsense2": getattr(rs, "__version__", "?"),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(elapsed, 3), "frames": n, "dropped_frames": dropped,
        "width": a.width, "height": a.height, "fps": a.fps,
        "depth_scale_m_per_unit": depth_scale, "depth_png_units": "mm", "depth_aligned_to": "color",
        "color_intrinsics": intrinsics_dict(color_intr), "depth_intrinsics_unaligned": intrinsics_dict(depth_intr),
        # requested rates; the D435i/D435if firmware may deliver accel at ~400 Hz regardless, so measured rates too
        "imu": {"enabled": use_imu, "accel_hz_requested": a.accel_hz, "gyro_hz_requested": a.gyro_hz,
                "accel_hz_measured": rate(accel), "gyro_hz_measured": rate(gyro),
                "accel_samples": len(accel), "gyro_samples": len(gyro), "imu_csv_rows": imu_rows,
                "imu_csv_units": {"accel": "g", "gyro": "rad/s"}, "raw_units": {"accel": "m/s^2", "gyro": "rad/s"},
                "axes": "RealSense IMU frame: x right, y down, z forward (out of the lens)",
                "extrinsics_accel_to_color": imu_extr},
        "timestamps": "seconds, RealSense global (host-synchronised) time domain",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    span = rows[-1][0] - rows[0][0] if n > 1 else 0
    print(f"done      : {n} frames in {span:.2f} s (~{(n - 1) / span if span else 0:.1f} fps), "
          f"{dropped} dropped by the camera/USB, {len(accel)} accel + {len(gyro)} gyro samples")
    print(f"            {out}")
    print(f"dashboard : py stray_dashboard.py \"{out}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
