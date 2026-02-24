#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Deployment-only runner for using a PointBridge checkpoint on UR + RealSense setups.

This script is intentionally independent of PointBridge training pipelines. It:
1) Loads a PointBridge checkpoint + normalization stats
2) Builds live observations from UR RTDE state and segmented RealSense point clouds
3) Runs policy inference and executes actions through the vendored IP deployment stack
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from point_bridge.agent.pb import BCAgent
from point_bridge.robot_utils.common.franka_gripper_points import extrapoints
from point_bridge.robot_utils.common.utils import (
    farthest_point_sampling,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _nested_get(cfg: Optional[Dict[str, Any]], path: str, default: Any) -> Any:
    if cfg is None:
        return default
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_hydra_config_from_checkpoint(ckpt_path: Path) -> Optional[Path]:
    for parent in [ckpt_path.parent] + list(ckpt_path.parents):
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.exists():
            return candidate
    return None


def _build_obs_shape(pixel_keys, width: int, height: int, proprio_key: str) -> Dict[str, Any]:
    obs_shape: Dict[str, Any] = {}
    for key in pixel_keys:
        obs_shape[key] = (height, width, 3)
    obs_shape[proprio_key] = (10,)
    obs_shape["features"] = (10,)
    return obs_shape


def _sample_points(points: np.ndarray, num_points: int, use_fps: bool) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    pts = points.astype(np.float32, copy=False)
    if use_fps and pts.shape[0] > num_points:
        try:
            return farthest_point_sampling(pts, num_points).astype(np.float32)
        except Exception:
            pass

    replace = pts.shape[0] < num_points
    idx = np.random.choice(pts.shape[0], size=num_points, replace=replace)
    return pts[idx].astype(np.float32)


def _compute_robot_points(
    T_w_e: np.ndarray,
    gripper_open: float,
    finger_span_open_m: float,
    finger_span_closed_m: float,
) -> np.ndarray:
    points = []
    half_span = 0.5 * (
        finger_span_closed_m + gripper_open * (finger_span_open_m - finger_span_closed_m)
    )
    for idx, base_T in enumerate(extrapoints[:8]):
        Tp = base_T.copy()
        if idx == 0:
            Tp[1, 3] = half_span
        elif idx == 1:
            Tp[1, 3] = -half_span
        pt = T_w_e @ Tp
        points.append(pt[:3, 3])
    return np.asarray(points, dtype=np.float32)


def _load_camera_configs(
    calib_path: Path,
    width: int,
    height: int,
    fps: int,
    selected_serials: Optional[set[str]] = None,
):
    from ip.deployment.config import CameraConfig

    with calib_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cameras = data.get("cameras", {})
    if not cameras:
        raise ValueError(f"No cameras found in calibration: {calib_path}")

    camera_cfgs = []
    for serial, entry in cameras.items():
        if selected_serials and serial not in selected_serials:
            continue
        T = np.asarray(entry.get("T_world_camera"), dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Camera {serial} has invalid T_world_camera shape: {T.shape}")
        camera_cfgs.append(
            CameraConfig(
                serial=serial,
                T_world_camera=T,
                width=width,
                height=height,
                fps=fps,
                align_to_color=True,
            )
        )

    if not camera_cfgs:
        raise ValueError(
            f"No camera configs selected from {calib_path}. "
            f"Available serials: {sorted(cameras.keys())}"
        )
    return camera_cfgs


def _filter_workspace(
    pcd: np.ndarray,
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
    z_min: Optional[float],
    z_max: Optional[float],
) -> np.ndarray:
    if pcd.size == 0:
        return pcd
    mask = np.ones((pcd.shape[0],), dtype=bool)
    if x_min is not None:
        mask &= pcd[:, 0] >= x_min
    if x_max is not None:
        mask &= pcd[:, 0] <= x_max
    if y_min is not None:
        mask &= pcd[:, 1] >= y_min
    if y_max is not None:
        mask &= pcd[:, 1] <= y_max
    if z_min is not None:
        mask &= pcd[:, 2] >= z_min
    if z_max is not None:
        mask &= pcd[:, 2] <= z_max
    return pcd[mask]


def _resolve_agent_runtime(
    payload: Dict[str, Any],
    cfg: Optional[Dict[str, Any]],
    device: str,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    policy_head = _nested_get(cfg, "policy_head", "deterministic")
    obs_type = list(_nested_get(cfg, "suite.obs_type", ["points"]))
    action_mode = _nested_get(cfg, "suite.action_mode", "pose")
    hidden_dim = int(_nested_get(cfg, "suite.hidden_dim", 256))
    pixel_keys = list(_nested_get(cfg, "suite.pixel_keys", ["pixels"]))
    proprio_key = _nested_get(cfg, "suite.proprio_key", "proprioceptive")
    use_proprio = bool(payload.get("use_proprio", _nested_get(cfg, "use_proprio", True)))
    use_language = bool(_nested_get(cfg, "use_language", False))
    history_len = int(_nested_get(cfg, "suite.history_len", 1))
    eval_history_len = int(_nested_get(cfg, "suite.eval_history_len", history_len))
    action_chunking = bool(_nested_get(cfg, "action_chunking", True))
    num_queries = int(_nested_get(cfg, "num_queries", 40))
    temporal_agg_strategy = _nested_get(cfg, "temporal_agg_strategy", "exponential_average")
    max_episode_len = int(payload.get("max_episode_len", 300))
    robot_points_key = _nested_get(cfg, "suite.robot_points_key", "robot_tracks")
    object_points_key = _nested_get(cfg, "suite.object_points_key", "object_tracks")
    num_points_per_obj = int(_nested_get(cfg, "suite.num_points_per_obj", 128))

    if action_mode != "pose":
        raise ValueError(
            f"deploy_ur_ip.py currently supports action_mode='pose' only, got {action_mode!r}"
        )
    if "points" not in obs_type:
        raise ValueError(
            f"deploy_ur_ip.py requires points in obs_type, got {obs_type}. "
            "Use a points-based PointBridge checkpoint."
        )

    obs_shape = _build_obs_shape(pixel_keys, image_width, image_height, proprio_key)
    action_shape = (10,)
    return {
        "obs_shape": obs_shape,
        "action_shape": action_shape,
        "device": device,
        "lr": 1e-4,
        "hidden_dim": hidden_dim,
        "stddev_schedule": "0.1",
        "use_tb": False,
        "policy_head": policy_head,
        "use_language": use_language,
        "pixel_keys": pixel_keys,
        "proprio_key": proprio_key,
        "use_proprio": use_proprio,
        "history_len": history_len,
        "eval_history_len": eval_history_len,
        "action_chunking": action_chunking,
        "num_queries": num_queries,
        "temporal_agg_strategy": temporal_agg_strategy,
        "max_episode_len": max_episode_len,
        "film": True,
        "obs_type": obs_type,
        "action_mode": action_mode,
        "robot_points_key": robot_points_key,
        "object_points_key": object_points_key,
        "num_points_per_obj": num_points_per_obj,
    }


def _load_agent(
    checkpoint: Path,
    hydra_config: Optional[Path],
    device: str,
    image_width: int,
    image_height: int,
):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _load_yaml(hydra_config) if hydra_config else None
    runtime_cfg = _resolve_agent_runtime(payload, cfg, device, image_width, image_height)

    agent = BCAgent(**runtime_cfg)
    agent.load_snapshot(payload, eval=True)
    stats = payload.get("stats")
    if stats is None:
        raise ValueError(
            f"Checkpoint {checkpoint} does not contain normalization stats. "
            "Use a snapshot generated by PointBridge train.py"
        )
    return agent, stats, runtime_cfg


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy PointBridge on UR + RealSense setup")
    parser.add_argument("--checkpoint", required=True, type=Path, help="PointBridge snapshot .pt path")
    parser.add_argument(
        "--hydra-config",
        default=None,
        type=Path,
        help="Optional Hydra config.yaml from training run (.hydra/config.yaml).",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="Calibration JSON with cameras.{serial}.T_world_camera (from IP calibration).",
    )
    parser.add_argument("--robot-ip", default="10.33.55.90")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-serial", action="append", default=None, help="Filter camera serial(s)")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Extra sleep between control cycles")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Point cloud voxel downsample (m)")
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument(
        "--default-gripper-open",
        type=float,
        default=1.0,
        help="Fallback normalized gripper openness if feedback is unavailable.",
    )
    parser.add_argument("--open-gripper-at-start", action="store_true")
    parser.add_argument("--finger-span-open-m", type=float, default=0.085)
    parser.add_argument("--finger-span-closed-m", type=float, default=0.010)
    parser.add_argument("--object-points", type=int, default=128)
    parser.add_argument("--no-fps-object-sampling", action="store_true")
    parser.add_argument("--max-translation-step", type=float, default=0.01)
    parser.add_argument("--max-rotation-step-deg", type=float, default=3.0)
    parser.add_argument(
        "--flange-to-policy-origin-m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.088),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--task-text", default=None, help="Required only for language-conditioned checkpoints.")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    if args.hydra_config is None:
        args.hydra_config = _find_hydra_config_from_checkpoint(args.checkpoint)
    if args.hydra_config:
        print(f"Using hydra config: {args.hydra_config}")

    # Vendored IP deployment stack (pointbridge/ip/deployment)
    from ip.deployment.config import DeploymentConfig
    from ip.deployment.control.action_executor import ActionExecutor, SafetyLimits
    from ip.deployment.control.robotiq_gripper import RobotiqGripper
    from ip.deployment.control.ur_rtde_control import URRTDEControl
    from ip.deployment.perception.realsense_perception import RealSensePerception
    from ip.deployment.state.ur_rtde_state import URRTDEState

    agent, stats, runtime_cfg = _load_agent(
        checkpoint=args.checkpoint,
        hydra_config=args.hydra_config,
        device=args.device,
        image_width=args.camera_width,
        image_height=args.camera_height,
    )
    print(
        "Loaded checkpoint with config: "
        f"obs_type={runtime_cfg['obs_type']} action_mode={runtime_cfg['action_mode']} "
        f"use_proprio={runtime_cfg['use_proprio']} use_language={runtime_cfg['use_language']}"
    )

    task_emb = None
    if runtime_cfg["use_language"]:
        if not args.task_text:
            raise ValueError("--task-text is required for language-conditioned checkpoint deployment")
        from sentence_transformers import SentenceTransformer

        lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        task_emb = lang_model.encode(args.task_text)
        print(f"Language conditioning enabled. task_text={args.task_text!r}")

    selected_serials = set(args.camera_serial) if args.camera_serial else None
    camera_cfgs = _load_camera_configs(
        calib_path=args.calibration,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        selected_serials=selected_serials,
    )
    print(f"Camera serials: {[cfg.serial for cfg in camera_cfgs]}")

    dep_cfg = DeploymentConfig(camera_configs=camera_cfgs)
    dep_cfg.robot_ip = args.robot_ip
    dep_cfg.device = args.device
    dep_cfg.pcd_voxel_size = args.voxel_size if args.voxel_size > 0 else None
    dep_cfg.gripper.enable = not args.disable_gripper
    dep_cfg.tcp_offset_in_code = True
    dep_cfg.tcp_offset_m = np.array(args.flange_to_policy_origin_m, dtype=np.float64)
    dep_cfg.safety = SafetyLimits(
        max_translation=float(args.max_translation_step),
        max_rotation=np.deg2rad(float(args.max_rotation_step_deg)),
    )

    # Keep deployment path self-contained and avoid non-PointBridge segmentation
    # dependencies for now. Object points are sampled from workspace-filtered
    # fused RealSense point cloud.
    dep_cfg.segmentation.enable = False
    print("Segmentation disabled: using workspace-filtered fused RealSense point cloud.")

    segmenter = None
    perception = RealSensePerception(
        dep_cfg.camera_configs,
        segmenter=segmenter,
        voxel_size=dep_cfg.pcd_voxel_size,
    )

    gripper = None
    if dep_cfg.gripper.enable:
        gripper = RobotiqGripper(
            host=dep_cfg.gripper.host or dep_cfg.robot_ip,
            port=dep_cfg.gripper.port,
            open_position=dep_cfg.gripper.open_position,
            closed_position=dep_cfg.gripper.closed_position,
        )
        gripper.connect()
        gripper.activate()

    rtde_ctrl = URRTDEControl.connect(dep_cfg.robot_ip, dep_cfg.rtde)
    rtde_recv = URRTDEState.connect(dep_cfg.robot_ip)
    state = URRTDEState(
        rtde_recv,
        gripper=gripper,
        tcp_offset_in_code=dep_cfg.tcp_offset_in_code,
        tcp_offset_m=dep_cfg.tcp_offset_m,
    )
    control = URRTDEControl(
        rtde_ctrl,
        control_config=dep_cfg.rtde,
        gripper=gripper,
        gripper_config=dep_cfg.gripper,
        tcp_offset_in_code=dep_cfg.tcp_offset_in_code,
        tcp_offset_m=dep_cfg.tcp_offset_m,
    )
    executor = ActionExecutor(control, state, dep_cfg.safety)

    if args.open_gripper_at_start and dep_cfg.gripper.enable:
        control.execute_gripper(1.0)

    robot_key = f"{runtime_cfg['robot_points_key']}_3d"
    object_key = f"{runtime_cfg['object_points_key']}_{runtime_cfg['num_points_per_obj']}_3d"
    use_fps = not args.no_fps_object_sampling
    object_points_count = int(runtime_cfg["num_points_per_obj"])
    if int(args.object_points) != object_points_count:
        print(
            f"[warn] --object-points={args.object_points} ignored; "
            f"checkpoint expects {object_points_count} points/object."
        )

    print("Starting deployment loop. Press Ctrl+C to stop.")
    print(f"robot_key={robot_key}, object_key={object_key}")

    try:
        for step in range(args.max_steps):
            loop_t0 = time.time()

            T_w_e = state.get_T_w_e()
            try:
                gripper_open = float(state.get_gripper_state())
            except Exception:
                gripper_open = float(np.clip(args.default_gripper_open, 0.0, 1.0))

            pcd_w = perception.capture_pcd_world(use_segmentation=dep_cfg.segmentation.enable)
            pcd_w = _filter_workspace(
                pcd_w,
                x_min=args.x_min,
                x_max=args.x_max,
                y_min=args.y_min,
                y_max=args.y_max,
                z_min=args.z_min,
                z_max=args.z_max,
            )
            obj_pts = _sample_points(
                pcd_w,
                num_points=max(1, object_points_count),
                use_fps=use_fps,
            )
            obj_pts = obj_pts[None, ...]  # one object slot

            robot_pts = _compute_robot_points(
                T_w_e=T_w_e,
                gripper_open=gripper_open,
                finger_span_open_m=args.finger_span_open_m,
                finger_span_closed_m=args.finger_span_closed_m,
            )

            rot6 = matrix_to_rotation_6d(T_w_e[:3, :3]).astype(np.float32)
            proprio = np.concatenate(
                [T_w_e[:3, 3].astype(np.float32), rot6, np.array([gripper_open], dtype=np.float32)],
                axis=0,
            )

            obs: Dict[str, Any] = {
                robot_key: robot_pts,
                object_key: obj_pts,
                runtime_cfg["proprio_key"]: proprio,
            }
            if "image" in runtime_cfg["obs_type"]:
                dummy = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
                for key in runtime_cfg["pixel_keys"]:
                    obs[key] = dummy
            if runtime_cfg["use_language"]:
                obs["task_emb"] = task_emb

            action = agent.act(obs, stats, step=step, global_step=0)
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape[0] != 10:
                raise RuntimeError(f"Expected 10-dim pose action, got shape {action.shape}")
            if not np.all(np.isfinite(action)):
                raise RuntimeError(f"Non-finite action at step {step}: {action}")

            T_target = np.eye(4, dtype=np.float64)
            T_target[:3, 3] = action[:3]
            T_target[:3, :3] = rotation_6d_to_matrix(action[3:9])

            grip_open_cmd = float(np.clip(action[9], 0.0, 1.0))
            grip_model_cmd = grip_open_cmd * 2.0 - 1.0  # ActionExecutor convention

            T_rel = np.linalg.inv(T_w_e) @ T_target
            success, substeps, error = executor.execute_actions(
                actions=np.array([T_rel], dtype=np.float64),
                grips=np.array([grip_model_cmd], dtype=np.float64),
                T_w_e_initial=T_w_e,
                horizon=1,
            )
            if not success:
                raise RuntimeError(f"Execution failed at step {step}: {error}")

            elapsed = time.time() - loop_t0
            print(
                f"[step {step:04d}] action_pos={np.round(action[:3], 4).tolist()} "
                f"grip_open={grip_open_cmd:.3f} exec_substeps={substeps} dt={elapsed:.3f}s"
            )
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

    except KeyboardInterrupt:
        print("Stopping deployment (Ctrl+C).")
    finally:
        try:
            control.stop_motion()
        except Exception:
            pass
        try:
            if hasattr(control, "_rtde"):
                control._rtde.stopScript()
                control._rtde.disconnect()
        except Exception:
            pass
        try:
            if hasattr(state, "_rtde"):
                state._rtde.disconnect()
        except Exception:
            pass
        try:
            if gripper is not None:
                gripper.disconnect()
        except Exception:
            pass
        try:
            perception.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
