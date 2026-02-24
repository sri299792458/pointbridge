#!/usr/bin/env python3
import argparse
import time
from typing import Optional

import numpy as np
import rtde_control
import rtde_receive


def main():
    parser = argparse.ArgumentParser(
        description="Measure TCP workspace bounds by moving the robot in freedrive."
    )
    parser.add_argument("--robot-ip", required=True, help="Robot IP address")
    parser.add_argument("--hz", type=float, default=20.0, help="Sampling rate (Hz)")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to sample (default: until Ctrl+C)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Expand bounds by this margin (meters)",
    )
    parser.add_argument(
        "--freedrive",
        action="store_true",
        help="Enable freedrive via RTDE control (requires Remote Control)",
    )
    args = parser.parse_args()

    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    rtde_c = None
    if args.freedrive:
        rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
        rtde_c.freedriveMode()
    period = 1.0 / max(args.hz, 1e-3)

    min_xyz = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    max_xyz = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    print("Move the robot through the intended workspace (freedrive).")
    print("Press Ctrl+C when done.")

    start = time.time()
    try:
        while True:
            pose = rtde_r.getActualTCPPose()
            xyz = np.array(pose[:3], dtype=np.float64)
            min_xyz = np.minimum(min_xyz, xyz)
            max_xyz = np.maximum(max_xyz, xyz)
            if args.duration is not None and (time.time() - start) >= args.duration:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        if rtde_c is not None:
            try:
                rtde_c.endFreedriveMode()
            except Exception as exc:
                print(f"[warn] Failed to end Freedrive mode cleanly: {exc}")

    if not np.isfinite(min_xyz).all() or not np.isfinite(max_xyz).all():
        raise RuntimeError("No valid TCP samples collected.")

    min_xyz -= args.margin
    max_xyz += args.margin

    print("\nMeasured workspace bounds (meters):")
    print(f"workspace_min = [{min_xyz[0]:.4f}, {min_xyz[1]:.4f}, {min_xyz[2]:.4f}]")
    print(f"workspace_max = [{max_xyz[0]:.4f}, {max_xyz[1]:.4f}, {max_xyz[2]:.4f}]")
    print("\nSuggested config snippet:")
    print(
        "config.safety = SafetyLimits(\n"
        "    max_translation=0.01,\n"
        "    max_rotation=np.deg2rad(3.0),\n"
        ")"
    )


if __name__ == "__main__":
    main()
