#!/usr/bin/env python3
"""Minimal demo point-cloud viewer in end-effector frame."""

import argparse
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import viser


def _to_ee_frame(points: np.ndarray, T_w_e: np.ndarray) -> np.ndarray:
    R = T_w_e[:3, :3]
    t = T_w_e[:3, 3]
    return (R.T @ (points - t).T).T


def _subsample(points: np.ndarray, num_points: int = 2048) -> np.ndarray:
    if points.shape[0] <= num_points:
        return points
    idx = np.random.choice(points.shape[0], num_points, replace=False)
    return points[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show demo object point clouds in end-effector frame."
    )
    parser.add_argument("--demo", required=True, help="Path to demo .pkl")
    args = parser.parse_args()

    with Path(args.demo).open("rb") as f:
        data = pickle.load(f)

    pcds = data.get("pcds", [])
    T_w_es = data.get("T_w_es", [])
    if not pcds or not T_w_es:
        raise RuntimeError("Demo must contain non-empty 'pcds' and 'T_w_es'.")

    valid_indices = [
        i for i, p in enumerate(pcds)
        if p is not None and len(p) > 0 and i < len(T_w_es)
    ]
    if not valid_indices:
        raise RuntimeError("No valid frames found with both pcd and T_w_e.")

    def _frame_points(frame_idx: int) -> np.ndarray:
        pts_w = np.asarray(pcds[frame_idx], dtype=np.float32)
        T_w_e = np.asarray(T_w_es[frame_idx], dtype=np.float64)
        pts_ee = _to_ee_frame(pts_w, T_w_e).astype(np.float32)
        return _subsample(pts_ee, num_points=2048)

    server = viser.ViserServer()
    print(f"[viewer] Loaded demo with {len(valid_indices)} valid frames.")
    print(f"[viewer] Open http://localhost:{server.get_port()}")
    server.scene.world_axes.visible = False
    server.initial_camera.position = np.array([0.0, 0.0, -0.3], dtype=np.float64)
    server.initial_camera.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    server.initial_camera.up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    server.scene.add_frame(
        "/ee",
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        axes_length=0.08,
        axes_radius=0.004,
    )

    first_idx = valid_indices[0]
    pcd_handle = server.scene.add_point_cloud(
        "/demo/pcd",
        points=_frame_points(first_idx),
        colors=(180, 180, 180),
        point_size=0.003,
        point_shape="square",
        precision="float32",
    )

    with server.gui.add_folder("Playback"):
        play = server.gui.add_checkbox("Play", initial_value=False)
        fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=20)
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(valid_indices) - 1,
            step=1,
            initial_value=0,
        )

    state = {"frame": 0, "lock": threading.Lock(), "updating": False}

    def _set_frame(frame_list_idx: int, from_slider: bool = False) -> None:
        with state["lock"]:
            frame_list_idx = int(np.clip(frame_list_idx, 0, len(valid_indices) - 1))
            state["frame"] = frame_list_idx
            frame_idx = valid_indices[frame_list_idx]
            pcd_handle.points = _frame_points(frame_idx)
            if not from_slider:
                state["updating"] = True
                frame_slider.value = frame_list_idx
                state["updating"] = False

    @frame_slider.on_update
    def _(_evt):
        if state["updating"]:
            return
        _set_frame(frame_slider.value, from_slider=True)

    def _playback_loop() -> None:
        while True:
            if play.value:
                _set_frame((state["frame"] + 1) % len(valid_indices))
            time.sleep(1.0 / max(1.0, float(fps.value)))

    thread = threading.Thread(target=_playback_loop, daemon=True)
    thread.start()

    server.sleep_forever()


if __name__ == "__main__":
    main()
