#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive


def _default_calib_path(arm: str) -> Path:
    filename = "realsense_T_world_camera_right.json" if arm == "right" else "realsense_T_world_camera.json"
    return Path(__file__).resolve().parent / "outputs" / filename


def _load_T_world_camera(calib_path: Path, serial: str) -> np.ndarray:
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with calib_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cams = data.get("cameras", {})
    if serial not in cams:
        raise KeyError(f"Serial {serial} not found in {calib_path}")
    return np.array(cams[serial]["T_world_camera"], dtype=np.float64)


def _pixel_to_cam(u: int, v: int, depth_m: float, intr) -> np.ndarray:
    x = (u - intr.ppx) / intr.fx * depth_m
    y = (v - intr.ppy) / intr.fy * depth_m
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def _depth_at(depth_m: np.ndarray, u: int, v: int) -> Optional[float]:
    h, w = depth_m.shape
    if u < 0 or v < 0 or u >= w or v >= h:
        return None
    d = float(depth_m[v, u])
    if d > 0:
        return d
    # Fallback: median in a small window if center depth is missing.
    u0, u1 = max(0, u - 2), min(w, u + 3)
    v0, v1 = max(0, v - 2), min(h, v + 3)
    window = depth_m[v0:v1, u0:u1].reshape(-1)
    window = window[window > 0]
    if window.size == 0:
        return None
    return float(np.median(window))


def main():
    parser = argparse.ArgumentParser(description="Click a pixel and print world point using T_world_camera.")
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default="left",
        help="Arm side used for default calibration file naming.",
    )
    parser.add_argument("--serial", required=True, help="RealSense serial to use")
    parser.add_argument(
        "--calib",
        default=None,
        help="Path to calibration JSON (default depends on --arm)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-ip", default=None, help="Optional: print TCP pose for comparison")
    args = parser.parse_args()

    calib_path = Path(args.calib) if args.calib else _default_calib_path(args.arm)
    T_world_camera = _load_T_world_camera(calib_path, args.serial)

    rtde = None
    if args.robot_ip:
        rtde = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    window = "click-to-world"
    last_depth = None

    def on_mouse(event, x, y, flags, param):
        nonlocal last_depth
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if last_depth is None:
            print("No depth frame yet.")
            return
        d = _depth_at(last_depth, x, y)
        if d is None:
            print(f"Pixel ({x},{y}): no valid depth")
            return
        p_cam = _pixel_to_cam(x, y, d, intr)
        p_world = (T_world_camera[:3, :3] @ p_cam) + T_world_camera[:3, 3]
        print(f"Pixel ({x},{y}) depth {d:.4f} m")
        print(f"Camera point: [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}] m")
        print(f"World point : [{p_world[0]:.4f}, {p_world[1]:.4f}, {p_world[2]:.4f}] m")
        if rtde is not None:
            pose = rtde.getActualTCPPose()
            print(
                "TCP pose   : "
                f"[{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}, "
                f"{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}]"
            )
            delta = np.array(pose[:3]) - p_world
            print(f"Delta (TCP - point): [{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}] m")

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            last_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            color = np.asanyarray(color_frame.get_data())
            cv2.imshow(window, color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
