#!/usr/bin/env python3
"""
Inspect a recorded demo.pkl and print which frames are selected as L waypoints.

This is a lightweight debug tool to sanity-check:
- gripper state recording (measured vs commanded)
- waypoint extraction quality (L=10 by default)
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from ip.utils.data_proc import extract_waypoints


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a demo.pkl and print selected waypoint frames.")
    parser.add_argument("--demo", required=True, help="Path to demo .pkl")
    args = parser.parse_args()

    demo = _load(Path(args.demo))
    print("keys:", demo.keys())

    T_w_es = demo.get("T_w_es", [])
    grips = demo.get("grips", [])
    grip_cmds = demo.get("grip_cmds", [])
    grip_raws = demo.get("grip_raws", [])
    grip_objs = demo.get("grip_objs", [])
    print("frames:", len(T_w_es))
    if not T_w_es:
        raise RuntimeError("demo has no T_w_es")
    if not grips:
        print("warning: demo has no grips; waypoint extraction will ignore gripper flips")

    if grips:
        arr = np.asarray(grips, dtype=np.float64)
        uniq, cnt = np.unique(arr, return_counts=True)
        print("grips unique + counts:", (uniq, cnt))
        print("first 20 grips:", arr[:20].tolist())
    if grip_cmds:
        non_none = [x for x in grip_cmds if x is not None]
        print("grip_cmds non-null:", len(non_none), "/", len(grip_cmds))
        if non_none:
            u, c = np.unique(np.asarray(non_none, dtype=np.float64), return_counts=True)
            print("grip_cmds unique + counts:", (u, c))
    if grip_raws:
        non_none = [x for x in grip_raws if x is not None]
        print("grip_raws non-null:", len(non_none), "/", len(grip_raws))
        if non_none:
            vals = np.asarray(non_none, dtype=np.float64)
            print("grip_raws range:", float(vals.min()), float(vals.max()))
    if grip_objs:
        non_none = [x for x in grip_objs if x is not None]
        print("grip_objs non-null:", len(non_none), "/", len(grip_objs))
        if non_none:
            u, c = np.unique(np.asarray(non_none, dtype=np.int64), return_counts=True)
            print("grip_objs unique + counts:", (u, c))

    idx = extract_waypoints(
        np.asarray(T_w_es, dtype=np.float64),
        np.asarray(grips, dtype=np.float64) if grips else np.zeros(len(T_w_es), dtype=np.float64),
        num_waypoints=10,
    )
    print("\nSelected waypoint frames:")
    for wp_i, frame_idx in enumerate(idx):
        grip = None
        grip_cmd = None
        grip_raw = None
        grip_obj = None
        if grips and 0 <= frame_idx < len(grips):
            grip = float(grips[frame_idx])
        if grip_cmds and 0 <= frame_idx < len(grip_cmds):
            grip_cmd = grip_cmds[frame_idx]
        if grip_raws and 0 <= frame_idx < len(grip_raws):
            grip_raw = grip_raws[frame_idx]
        if grip_objs and 0 <= frame_idx < len(grip_objs):
            grip_obj = grip_objs[frame_idx]
        T = np.asarray(T_w_es[frame_idx], dtype=np.float64)
        t = T[:3, 3].tolist() if T.shape == (4, 4) else None
        print(
            f"  [{wp_i:02d}] frame={int(frame_idx):04d} grip={grip} cmd={grip_cmd} raw={grip_raw} obj={grip_obj} "
            f"t_w_e_xyz={[float(x) for x in t] if t else None}"
        )


if __name__ == "__main__":
    main()
