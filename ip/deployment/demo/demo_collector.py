import json
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from ip.utils.data_proc import extract_waypoints, sample_to_cond_demo
from ip.deployment.control.spark_input import SparkDemoInput


class DemoCollectionCancelled(RuntimeError):
    pass


class DemoCollector:
    def __init__(self, perception, state, control):
        self.perception = perception
        self.state = state
        self.control = control

    @staticmethod
    def _binarize_grips_from_position(grip_raws, open_threshold: float = 0.9) -> list:
        """Convert measured gripper openness to RLBench-style binary state.

        Input `grip_raws` uses model convention from URRTDEState.get_gripper_state():
          1.0 = fully open, 0.0 = fully closed.

        RLBench-style state:
          open (1.0)  if open_amount > 0.9
          closed (0.0) otherwise
        """
        if not grip_raws:
            raise RuntimeError("No gripper state samples were recorded.")
        grips = []
        for idx, raw in enumerate(grip_raws):
            if raw is None:
                raise RuntimeError(f"Missing gripper state at frame {idx}.")
            value = float(raw)
            if not np.isfinite(value):
                raise RuntimeError(f"Non-finite gripper state at frame {idx}: {value}")
            grips.append(1.0 if value > open_threshold else 0.0)
        return grips

    def collect_kinesthetic(
        self,
        task_name: str,
        rate_hz: float = 10.0,
        use_segmentation: bool = False,
        debug_waypoints: bool = False,
        control_mode: str = "keyboard",
        spark_input: Optional[SparkDemoInput] = None,
    ) -> Dict:
        try:
            from pynput import keyboard
        except Exception as exc:
            raise ImportError("pynput is required for demo hotkeys. Install with `pip install pynput`.") from exc

        print(f"Collecting demo for: {task_name}")
        mode = str(control_mode).lower()
        if mode not in {"keyboard", "spark"}:
            raise ValueError(f"Unsupported control_mode={control_mode!r}. Expected 'keyboard' or 'spark'.")
        if mode == "spark" and spark_input is None:
            raise ValueError("spark_input must be provided when control_mode='spark'.")

        if mode == "keyboard":
            print("Move robot to start position, press ENTER to begin recording...")
            input()

        frames = {"pcds": [], "T_w_es": [], "grips": [], "grip_cmds": [], "grip_raws": []}
        debug_frames_all = [] if debug_waypoints else None
        stop_event = threading.Event()
        grip_cmd = {"value": None}
        spark_gui = None
        spark_cancelled = False

        if mode == "keyboard":
            def on_press(key):
                if stop_event.is_set():
                    return False
                if key == keyboard.Key.esc:
                    stop_event.set()
                    return False
                try:
                    char = key.char.lower()
                except AttributeError:
                    return None
                if char == "q":
                    stop_event.set()
                    return False
                if char == "o":
                    if hasattr(self.control, "execute_gripper"):
                        self.control.execute_gripper(1.0)
                    grip_cmd["value"] = 1.0
                elif char == "c":
                    if hasattr(self.control, "execute_gripper"):
                        self.control.execute_gripper(0.0)
                    grip_cmd["value"] = 0.0
                return None
        else:
            def on_press(key):
                if stop_event.is_set():
                    return False
                if key == keyboard.Key.esc:
                    stop_event.set()
                    return False
                try:
                    char = key.char.lower()
                except AttributeError:
                    return None
                if char == "q":
                    stop_event.set()
                    return False
                return None

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        if mode == "keyboard":
            print("Recording... hotkeys: o=open, c=close, q/esc=stop")
            if hasattr(self.control, "enable_freedrive"):
                self.control.enable_freedrive()
        else:
            from ip.deployment.control.spark_alignment_gui import SparkAlignmentGui

            print("Starting Spark input...")
            spark_input.start()
            print("Opening Spark alignment GUI...")
            spark_gui = SparkAlignmentGui(
                spark_input=spark_input,
                state=self.state,
                profile_name=spark_input.profile.name,
            )
            print("Spark GUI ready. Click 'Start Recording' to begin.")
            proceed = spark_gui.wait_for_start(external_stop=lambda: stop_event.is_set())
            if not proceed:
                raise DemoCollectionCancelled("Spark demo cancelled before recording started.")
            print("Recording... Spark input active (GUI controls stop)")

        try:
            period = 1.0 / rate_hz
            while not stop_event.is_set():
                start = time.time()
                if mode == "spark" and spark_gui is not None:
                    spark_gui.pump()
                    if spark_gui.should_stop():
                        spark_cancelled = not spark_gui.should_save()
                        stop_event.set()
                        break
                pcd_w = self.perception.capture_pcd_world(
                    use_segmentation=use_segmentation,
                    capture_debug_frames=debug_waypoints,
                )
                if debug_waypoints:
                    debug_frames = self.perception.get_last_debug_frames()
                    frame_pack = []
                    for frame in debug_frames:
                        rgb = frame.get("rgb")
                        if rgb is None:
                            continue
                        # Copy RGB buffers: RealSense frames can reuse underlying memory, so without
                        # a copy, many stored frames may end up identical.
                        rgb = np.asanyarray(rgb).copy()
                        mask = frame.get("mask")
                        if mask is not None:
                            mask = np.asanyarray(mask).copy()
                        entry = {"rgb": rgb, "mask": mask}
                        if "serial" in frame:
                            entry["serial"] = frame["serial"]
                        elif "camera_index" in frame:
                            entry["serial"] = f"cam{frame['camera_index']}"
                        frame_pack.append(entry)
                    debug_frames_all.append(frame_pack)
                T_w_e = self.state.get_T_w_e()
                grip_raw_f = float(self.state.get_gripper_state())
                if not np.isfinite(grip_raw_f):
                    raise RuntimeError(f"Non-finite gripper feedback during demo collection: {grip_raw_f}")

                frames["pcds"].append(pcd_w)
                frames["T_w_es"].append(T_w_e)
                frames["grips"].append(None)
                if mode == "spark":
                    spark_err = spark_input.get_last_error()
                    if spark_err:
                        raise RuntimeError(f"Spark input error during collection: {spark_err}")
                    cmd = spark_input.get_last_open_command()
                else:
                    cmd = grip_cmd["value"]
                frames["grip_cmds"].append(None if cmd is None else float(cmd))
                frames["grip_raws"].append(grip_raw_f)

                if mode == "spark" and spark_gui is not None:
                    spark_gui.pump()
                    if spark_gui.should_stop():
                        spark_cancelled = not spark_gui.should_save()
                        stop_event.set()
                        break

                elapsed = time.time() - start
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            listener.stop()
            if mode == "keyboard":
                if hasattr(self.control, "disable_freedrive"):
                    self.control.disable_freedrive()
            else:
                if spark_gui is not None:
                    spark_gui.close()
                spark_input.stop()

        if spark_cancelled:
            raise DemoCollectionCancelled("Spark demo cancelled; recording discarded.")

        # Post-process measured gripper openness into binary RLBench-style state.
        grips = self._binarize_grips_from_position(frames["grip_raws"])
        frames["grips"] = grips

        print(f"Recorded {len(frames['pcds'])} frames")
        if debug_waypoints and debug_frames_all is not None:
            frames["_debug_frames"] = debug_frames_all
        return frames

    def prepare_for_model(self, raw_demo: Dict, num_traj_wp: int = 10) -> Dict:
        return sample_to_cond_demo(raw_demo, num_traj_wp)

    def save_waypoint_debug_images(self, raw_demo: Dict, out_dir: str, num_traj_wp: int = 10) -> None:
        if "_debug_frames" not in raw_demo:
            print("No debug RGB+mask frames stored; re-run with debug_waypoints=True.")
            return

        T_w_es = raw_demo.get("T_w_es", [])
        grips = raw_demo.get("grips", [])
        if not T_w_es or not grips:
            print("No trajectory data found in demo.")
            return

        waypoints = extract_waypoints(
            np.array(T_w_es),
            np.array(grips),
            num_waypoints=num_traj_wp,
        )

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with (out_path / "waypoints.json").open("w", encoding="utf-8") as f:
            payload = {
                "num_waypoints": int(num_traj_wp),
                # Cast to plain ints for JSON (may contain numpy scalars).
                "indices": [int(x) for x in waypoints],
                "waypoints": [],
            }
            for wp_i, frame_idx in enumerate(waypoints):
                entry = {
                    "wp": int(wp_i),
                    "frame": int(frame_idx),
                    "grip": None,
                    "grip_cmd": None,
                    "grip_raw": None,
                    "t_w_e_xyz": None,
                }
                if 0 <= int(frame_idx) < len(grips):
                    grip_value = grips[int(frame_idx)]
                    entry["grip"] = None if grip_value is None else float(grip_value)
                grip_cmds = raw_demo.get("grip_cmds", [])
                if 0 <= int(frame_idx) < len(grip_cmds):
                    grip_cmd_value = grip_cmds[int(frame_idx)]
                    entry["grip_cmd"] = None if grip_cmd_value is None else float(grip_cmd_value)
                grip_raws = raw_demo.get("grip_raws", [])
                if 0 <= int(frame_idx) < len(grip_raws):
                    grip_raw_value = grip_raws[int(frame_idx)]
                    entry["grip_raw"] = None if grip_raw_value is None else float(grip_raw_value)
                if 0 <= int(frame_idx) < len(T_w_es):
                    T = np.asarray(T_w_es[int(frame_idx)], dtype=np.float64)
                    if T.shape == (4, 4):
                        entry["t_w_e_xyz"] = [float(x) for x in T[:3, 3].tolist()]
                payload["waypoints"].append(entry)
            json.dump(payload, f, indent=2)

        debug_frames = raw_demo["_debug_frames"]
        for wp_i, frame_idx in enumerate(waypoints):
            if frame_idx < 0 or frame_idx >= len(debug_frames):
                continue
            for cam_idx, cam in enumerate(debug_frames[frame_idx]):
                rgb = cam.get("rgb")
                if rgb is None:
                    continue
                overlay = rgb.copy()
                mask = cam.get("mask")
                if mask is not None and mask.shape == overlay.shape[:2]:
                    green = np.zeros_like(overlay)
                    green[..., 1] = 255
                    overlay = np.where(
                        mask[..., None].astype(bool),
                        (0.3 * overlay + 0.7 * green).astype(overlay.dtype),
                        overlay,
                    )
                serial = cam.get("serial", f"cam{cam_idx}")
                safe_serial = "".join(
                    ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(serial)
                )
                filename = f"wp_{wp_i:02d}_frame_{frame_idx:04d}_{safe_serial}.png"
                bgr = overlay[:, :, ::-1] if overlay.ndim == 3 and overlay.shape[2] == 3 else overlay
                bgr = np.ascontiguousarray(bgr)
                bgr = bgr.copy()
                grip = raw_demo.get("grips", [None])[frame_idx] if raw_demo.get("grips") else None
                grip_raw = raw_demo.get("grip_raws", [None])[frame_idx] if raw_demo.get("grip_raws") else None
                label = f"wp={wp_i} frame={frame_idx} grip={grip} raw={grip_raw}"
                cv2.putText(
                    bgr,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(out_path / filename), bgr)

    def save_demo(self, demo: Dict, path: str) -> None:
        parent = Path(path).expanduser().resolve().parent
        if parent.name:
            parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(demo, f)

    def load_demo(self, path: str) -> Dict:
        with open(path, "rb") as f:
            return pickle.load(f)
