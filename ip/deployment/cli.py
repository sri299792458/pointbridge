"""
Example deployment entrypoint for Instant Policy on UR5e (RTDE, ROS-free).
"""
import argparse
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path

import numpy as np
import rtde_control

from ip.deployment.config import CameraConfig, DeploymentConfig
from ip.deployment.control.action_executor import SafetyLimits
from ip.deployment.control.spark_input import SparkDemoInput
from ip.deployment.demo.demo_collector import DemoCollector, DemoCollectionCancelled
from ip.deployment.perception.manual_seed_xmem import manual_seed_xmem
from ip.deployment.orchestrator import InstantPolicyDeployment

HOME_MOVE_SPEED_RAD_S = 1.05
HOME_MOVE_ACCEL_RAD_S2 = 1.4
DEPLOYMENT_DIR = Path(__file__).resolve().parent
HOME_JOINTS_PATH = DEPLOYMENT_DIR / "assets" / "home_joint.json"
LIVE_OUT_PATH = DEPLOYMENT_DIR / "live.pkl"
DEBUG_LIVE_FRAMES_DIR = DEPLOYMENT_DIR / "debug_live_frames"
DEBUG_DEMO_WAYPOINTS_DIR = DEPLOYMENT_DIR / "debug_waypoints"


def _default_calibration_path(arm: str) -> Path:
    filename = "realsense_T_world_camera_right.json" if arm == "right" else "realsense_T_world_camera.json"
    return DEPLOYMENT_DIR / "calibration" / "outputs" / filename


def build_default_config() -> DeploymentConfig:
    config = DeploymentConfig(camera_configs=[])
    config.robot_ip = "10.33.55.90"
    config.model_path = "./checkpoints/ip"
    config.segmentation.backend = "xmem"
    config.segmentation.sam_checkpoint_path = "./checkpoints/sam/sam_vit_b_01ec64.pth"
    config.segmentation.xmem_checkpoint_path = "./checkpoints/xmem/XMem.pth"
    config.segmentation.enable = True
    config.device = "cuda:0"
    config.safety = SafetyLimits(
        max_translation=0.01,
        max_rotation=np.deg2rad(3.0),
    )
    return config


def _frame_spec_from_config(config: DeploymentConfig) -> dict:
    return {
        "robot_tcp_frame": "flange",
        "flange_to_policy_origin_m": [
            float(x) for x in np.asarray(config.tcp_offset_m, dtype=np.float64).reshape(3)
        ],
    }


def _offset_from_demo_frame_spec(spec: dict, demo_path: str) -> np.ndarray:
    if "flange_to_policy_origin_m" not in spec:
        raise ValueError(
            f"Demo {demo_path} frame_spec is missing required key 'flange_to_policy_origin_m'."
        )
    offset = np.asarray(spec["flange_to_policy_origin_m"], dtype=np.float64)
    if offset.shape != (3,):
        raise ValueError(
            f"Demo {demo_path} has invalid flange_to_policy_origin_m shape {offset.shape}; expected (3,)."
        )
    return offset


def _validate_demo_frame_spec(demo: dict, demo_path: str, expected_spec: dict) -> None:
    spec = demo.get("frame_spec")
    if spec is None:
        raise ValueError(
            f"Demo {demo_path} is missing required frame_spec metadata."
        )

    robot_tcp = str(spec.get("robot_tcp_frame", "")).lower()
    if robot_tcp != "flange":
        raise ValueError(
            f"Demo {demo_path} has unsupported robot_tcp_frame={robot_tcp!r}. Expected 'flange'."
        )

    demo_offset = _offset_from_demo_frame_spec(spec, demo_path)
    expected_offset = np.asarray(expected_spec["flange_to_policy_origin_m"], dtype=np.float64)
    if not np.allclose(demo_offset, expected_offset, atol=1e-6):
        raise ValueError(
            f"Frame mismatch for demo {demo_path}: demo flange_to_policy_origin_m="
            f"{demo_offset.tolist()} != current run {expected_offset.tolist()}."
        )


def _load_demos(paths, expected_frame_spec=None):
    demos = []
    for path in paths:
        with open(path, "rb") as f:
            demo = pickle.load(f)
        if expected_frame_spec is not None:
            _validate_demo_frame_spec(demo, str(path), expected_frame_spec)
        demos.append(demo)
    return demos


def _apply_calibration_json(config: DeploymentConfig, calib_path: Path) -> None:
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    cameras = calib.get("cameras", {})
    if not cameras:
        raise ValueError(f"No cameras found in calibration file: {calib_path}")

    existing_by_serial = {cfg.serial: cfg for cfg in config.camera_configs}
    new_configs = []
    for serial, cam in cameras.items():
        if "T_world_camera" not in cam:
            raise KeyError(
                f"Calibration for serial {serial} is missing required key 'T_world_camera'."
            )
        T = np.array(cam["T_world_camera"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(
                f"Calibration for serial {serial} has invalid T_world_camera shape {T.shape}; expected (4, 4)."
            )
        if serial in existing_by_serial:
            base = existing_by_serial[serial]
            new_configs.append(
                CameraConfig(
                    serial=serial,
                    T_world_camera=T,
                    width=base.width,
                    height=base.height,
                    fps=base.fps,
                    align_to_color=base.align_to_color,
                )
            )
        else:
            new_configs.append(CameraConfig(serial=serial, T_world_camera=T))
    if not new_configs:
        raise ValueError(f"No valid T_world_camera entries in calibration file: {calib_path}")
    config.camera_configs = new_configs


def _load_home_joints() -> np.ndarray:
    with HOME_JOINTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "joints_rad" in data:
        return np.array(data["joints_rad"], dtype=np.float64)
    if "joints_deg" in data:
        return np.deg2rad(np.array(data["joints_deg"], dtype=np.float64))
    raise ValueError(f"{HOME_JOINTS_PATH} must contain joints_rad or joints_deg")


def _go_home(robot_ip: str) -> None:
    joints_rad = _load_home_joints()
    rtde = rtde_control.RTDEControlInterface(robot_ip)
    try:
        rtde.moveJ(joints_rad.tolist(), HOME_MOVE_SPEED_RAD_S, HOME_MOVE_ACCEL_RAD_S2)
    finally:
        # Release remote control so pendant Freedrive is available if needed.
        try:
            rtde.stopScript()
        except Exception as exc:
            print(f"[warn] Failed to stop RTDE script after homing: {exc}")
        try:
            rtde.disconnect()
        except Exception as exc:
            print(f"[warn] Failed to disconnect RTDE control after homing: {exc}")


def _open_gripper(control, config: DeploymentConfig) -> None:
    if not config.gripper.enable:
        return
    if hasattr(control, "execute_gripper"):
        control.execute_gripper(1.0)


def main():
    parser = argparse.ArgumentParser(description="Instant Policy deployment on UR5e (RTDE)")
    parser.add_argument("--robot-ip", default=None, help="UR5e IP address (default: config.robot_ip)")
    parser.add_argument(
        "--demo",
        action="append",
        nargs="+",
        default=[],
        metavar="DEMO",
        help=(
            "One or more demo .pkl paths. Can be repeated. "
            "Tip: you can use shell globs, e.g. --demo demos/task1_demo*.pkl"
        ),
    )
    parser.add_argument("--collect-demo", action="store_true", help="Collect a kinesthetic demo and exit")
    parser.add_argument("--demo-out", default="demo.pkl", help="Output path for collected demo")
    parser.add_argument(
        "--demo-control",
        choices=["keyboard", "spark"],
        default="keyboard",
        help="Demo control source: pendant freedrive + keyboard hotkeys, or Spark teleop input.",
    )
    parser.add_argument(
        "--spark-serial",
        default=None,
        help="Spark serial device (required when --demo-control spark), e.g. /dev/ttyUSB0",
    )
    parser.add_argument(
        "--spark-profile",
        choices=["lightning", "thunder"],
        default="lightning",
        help="Spark mapping profile (offsets/gripper map).",
    )
    parser.add_argument(
        "--spark-offsets-pickle",
        default=None,
        help="Optional path override for Spark offsets pickle.",
    )
    parser.add_argument(
        "--debug-demo-waypoints",
        action="store_true",
        help="Save RGB+mask images for selected waypoint frames during demo collection.",
    )
    parser.add_argument(
        "--camera-serial",
        action="append",
        default=None,
        help="Use only these camera serials (repeatable).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Max execution steps")
    parser.add_argument("--manual-seed", action="store_true", help="Manually seed XMem masks before running")
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default="left",
        help="Arm side used for default calibration file naming.",
    )
    parser.add_argument(
        "--calib",
        default=None,
        help="Calibration JSON to auto-load (default depends on --arm).",
    )
    parser.add_argument(
        "--no-home",
        action="store_false",
        dest="go_home",
        help="Skip moving robot to home joints before starting",
    )
    parser.add_argument(
        "--no-open-gripper",
        action="store_false",
        dest="open_gripper",
        help="Skip opening the gripper before starting",
    )
    parser.add_argument(
        "--save-live",
        action="store_true",
        help="Save live rollout to a demo-like .pkl (pcds, T_w_es, grips, frame_spec, recorded_at_utc).",
    )
    parser.add_argument(
        "--debug-live-frames",
        action="store_true",
        help="Save per-step RGB+mask snapshots during live deployment.",
    )
    parser.add_argument(
        "--flange-to-policy-origin-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Policy origin offset from flange in meters (applied in code to RTDE flange pose).",
    )
    parser.set_defaults(open_gripper=True)
    parser.set_defaults(go_home=True)
    args = parser.parse_args()

    # Flatten --demo which is a list of lists due to nargs="+".
    demo_paths = []
    for item in args.demo:
        if isinstance(item, (list, tuple)):
            demo_paths.extend(item)
        else:
            demo_paths.append(item)
    args.demo = demo_paths

    config = build_default_config()
    if args.robot_ip:
        config.robot_ip = args.robot_ip
    calib_path = Path(args.calib) if args.calib else _default_calibration_path(args.arm)
    _apply_calibration_json(config, calib_path)
    config.execute_until_grip_change = True
    config.tcp_offset_in_code = True
    if args.flange_to_policy_origin_m is not None:
        config.tcp_offset_m = np.array(args.flange_to_policy_origin_m, dtype=np.float64)
    if args.camera_serial:
        serials = set(args.camera_serial)
        filtered = [cfg for cfg in config.camera_configs if cfg.serial in serials]
        if not filtered:
            raise ValueError(
                "Requested camera serials not found in default deployment config. "
                f"Available: {[cfg.serial for cfg in config.camera_configs]}"
            )
        config.camera_configs = filtered
    print(f"Loaded calibration: {calib_path}")
    print(f"Camera serials: {[cfg.serial for cfg in config.camera_configs]}")
    if args.manual_seed:
        config.segmentation.xmem_init_with_sam = False
        if config.segmentation.backend.lower() != "xmem":
            raise ValueError("--manual-seed requires segmentation.backend == 'xmem'")
    frame_spec = _frame_spec_from_config(config)
    print("ROBOT RTDE TCP FRAME = FLANGE (fixed)")
    print(f"POLICY ORIGIN OFFSET FROM FLANGE (m) = {config.tcp_offset_m.tolist()}")
    if any(cfg.serial.startswith("CAMERA_SERIAL") for cfg in config.camera_configs):
        raise ValueError("Please update camera serials and T_world_camera in the default deployment config")

    if args.collect_demo:
        task_name = Path(args.demo_out).stem or "task"
        if args.go_home:
            _go_home(config.robot_ip)
        deployment = InstantPolicyDeployment(config, load_model=False, debug_gripper=False)
        if args.open_gripper:
            _open_gripper(deployment.control, config)
        if args.manual_seed and config.segmentation.enable:
            manual_seed_xmem(
                deployment.perception,
                [cfg.serial for cfg in config.camera_configs],
                out_dir=None,
            )
        collector = DemoCollector(deployment.perception, deployment.state, deployment.control)
        spark_input = None
        if args.demo_control == "spark":
            if not args.spark_serial:
                raise ValueError("--spark-serial is required when --demo-control spark")
            spark_input = SparkDemoInput(
                state=deployment.state,
                control=deployment.control,
                serial_device=args.spark_serial,
                profile_name=args.spark_profile,
                offsets_pickle=args.spark_offsets_pickle,
            )
        try:
            raw_demo = collector.collect_kinesthetic(
                task_name,
                use_segmentation=config.segmentation.enable,
                debug_waypoints=args.debug_demo_waypoints,
                control_mode=args.demo_control,
                spark_input=spark_input,
            )
        except DemoCollectionCancelled as exc:
            print(f"[collect-demo] {exc}")
            return
        raw_demo["frame_spec"] = frame_spec
        raw_demo["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
        if args.debug_demo_waypoints:
            collector.save_waypoint_debug_images(
                raw_demo,
                str(DEBUG_DEMO_WAYPOINTS_DIR),
                num_traj_wp=config.num_traj_wp,
            )
            raw_demo.pop("_debug_frames", None)
        collector.save_demo(raw_demo, args.demo_out)
        print(f"Saved demo to {args.demo_out}")
        return

    if args.go_home:
        _go_home(config.robot_ip)
    deployment = InstantPolicyDeployment(config, debug_gripper=False)
    if args.open_gripper:
        _open_gripper(deployment.control, config)
    if args.manual_seed and config.segmentation.enable:
        manual_seed_xmem(
            deployment.perception,
            [cfg.serial for cfg in config.camera_configs],
            out_dir=None,
        )
    demos = _load_demos(args.demo, expected_frame_spec=frame_spec)
    deployment.run(
        demos,
        max_steps=args.max_steps,
        save_live=args.save_live,
        live_out=LIVE_OUT_PATH,
        debug_live_frames=args.debug_live_frames,
        debug_live_frames_dir=DEBUG_LIVE_FRAMES_DIR,
    )


if __name__ == "__main__":
    main()
